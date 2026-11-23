# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness tests for the fused routing-map padding operator.

Two oracles, because they fail differently:

* ``pad_routing_map_ref`` -- the plain-torch transcription of Megatron's
  ``moe_utils.pad_routing_map``. It asks nothing of the platform, so the
  operator is testable off NVIDIA, and it is checked on every tier.
* Megatron-LM's ``fused_pad_routing_map`` -- the kernel this port is a port *of*,
  imported from the installed ``megatron-core`` package the way the blocked_flash
  suite imports DeepSpeed's kernel. On a backend in ``_NV_BASELINE_VENDORS`` it
  is *required*: the loader raises rather than letting the tier disappear
  quietly. That kernel is also the baseline the benchmark times, so holding the
  two oracles to each other is what keeps the reported speedup from being
  measured against a contract nothing had checked.

The tier decides which oracle a case is *held to*; it does not suppress the
torch reference, which runs everywhere.
"""

import importlib.metadata
import sys
import warnings

import numpy as np
import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# ``[num_tokens, num_experts]``. 8193 crosses the 8192-token tile boundary, so the
# two-pass path is exercised; 16384 walks two tiles, where upstream's one-block
# layout starts spilling; 4097 is a non-power-of-two row that still fits one
# masked tile.
_SHAPES = [
    (1, 1),
    (1, 64),
    (7, 64),
    (64, 1),
    (257, 64),
    (4097, 64),
    (8193, 64),
    (16384, 64),
]

# 256 is what Transformer Engine's grouped kernels align to; 32 is the other
# common multiple. 1 leaves every column untouched.
_PAD_MULTIPLES = [1, 8, 32, 256]

_DTYPES = utils.BOOL_TYPES + [torch.int32]
if utils.int64_is_supported:
    _DTYPES.append(torch.int64)

# Expert counts that are not powers of two: the transpose passes tile the expert
# axis, so a ragged last tile is a distinct path from every power-of-two case.
_RAGGED_EXPERT_COUNTS = [3, 17, 33, 48, 100]

# Dtypes the Triton kernels do not load; upstream casts every input with .int()
# before its kernel, so these reach the operator too. Complex always exists; the
# fp8 names do not on older torch.
_CAST_DTYPES = [torch.complex64]
for _name in ("float8_e4m3fn", "float8_e5m2"):
    if hasattr(torch, _name):
        _CAST_DTYPES.append(getattr(torch, _name))
del _name

_TOP_K = 8


# ---------------------------------------------------------------------------
# Megatron oracle
#
# Imported from the installed ``megatron-core`` package -- the same kernel the
# benchmark times, rather than a copy that could drift from it. Importing
# ``megatron.core`` runs its ``__init__`` and pulls in parallel_state and the
# rest of the core tree, which is why it is loaded once here, behind the vendor
# gate, instead of per case.
#
# Like the blocked_flash suite's DeepSpeed oracle, the import is *required* on
# the backends in ``_NV_BASELINE_VENDORS``: a package missing where the suite
# says it is the oracle is reported, not quietly dropped.
# ---------------------------------------------------------------------------
_NV_BASELINE_VENDORS = {"nvidia"}


def _load_mcore_fused_pad_routing_map():
    """Megatron-LM's ``fused_pad_routing_map``, or ``None`` on another backend."""
    if flag_train.vendor_name not in _NV_BASELINE_VENDORS:
        return None

    try:
        from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

        return fused_pad_routing_map
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use Megatron-LM's fused_pad_routing_map "
            f"as its second oracle, but megatron-core could not provide it: "
            f"{exc!r}. Install megatron-core for this platform, or drop the "
            f"backend from _NV_BASELINE_VENDORS."
        ) from exc


# Resolved once, at module import time.
_mcore_fused_pad_routing_map = _load_mcore_fused_pad_routing_map()

# The eager body under torch.compile's wrapper. A CPU map with an empty expert
# axis never launches a kernel upstream, but the wrapper still tries to compile
# the Triton kernel for the CPU and fails; the eager body is what returns.
_mcore_fused_pad_routing_map_eager = (
    getattr(_mcore_fused_pad_routing_map, "_torchdynamo_orig_callable", None)
    if _mcore_fused_pad_routing_map is not None
    else None
)

# The reference implementation's version, recorded as tests/megatron/README.md
# asks for. Only meaningful on the backends where the oracle applies.
_MCORE_VERSION = (
    importlib.metadata.version("megatron-core")
    if _mcore_fused_pad_routing_map is not None
    else None
)

# The tier, named. It is the benchmark's baseline on the same backends.
_ORACLE = "megatron" if _mcore_fused_pad_routing_map is not None else "torch_ref"

needs_mcore_oracle = pytest.mark.skipif(
    _mcore_fused_pad_routing_map is None,
    reason="the Megatron oracle is only wired up on nvidia",
)


# ---------------------------------------------------------------------------
# Reference implementation
#
# Megatron's ``megatron.core.transformer.moe.moe_utils.pad_routing_map``,
# transcribed. The fused kernel is its kernel form, so this is the oracle for
# both. It preserves the input dtype; the operator returns int32 (upstream
# materialises an ``.int()`` copy), so the result is cast to int32 at the end --
# comparing there rather than casting the operator's output down keeps the
# operator's contract under test.
#
# The pad pass flips zeros only: upstream's kernel masks with
# ``(zero_ranks <= num_to_pad) & is_zero`` and keeps every other value as it
# came, so the reference must do the same. For a boolean or 0/1 map the extra
# ``is_zero`` guard is a no-op, which is why it is invisible in the dense cases;
# with any other value (an integer map, a float map before ``.int()``) it is the
# difference between preserving the value and overwriting it with 1.
# ---------------------------------------------------------------------------


def pad_routing_map_ref(routing_map, pad_multiple):
    """Reference for ``fused_pad_routing_map``, composed from torch ops."""
    if routing_map.shape[0] == 0:
        return routing_map

    work = routing_map.clone().transpose(0, 1)  # [num_experts, num_tokens]

    num_ones = work.sum(dim=1)
    num_to_pad = (-num_ones) % pad_multiple

    is_zero = work == 0
    zero_ranks = torch.cumsum(is_zero.int(), dim=1)
    work[(zero_ranks <= num_to_pad.unsqueeze(1)) & is_zero] = 1

    return work.transpose(0, 1).int()


def _topk_map(num_tokens, num_experts, dtype, top_k=_TOP_K):
    """A routing map with exactly ``top_k`` set entries per token."""
    routing_map = torch.zeros(
        (num_tokens, num_experts), dtype=torch.bool, device=flag_train.device
    )
    indices = (
        torch.rand(
            (num_tokens, num_experts), dtype=torch.float32, device=flag_train.device
        )
        .topk(min(top_k, num_experts), dim=-1)
        .indices
    )
    routing_map.scatter_(1, indices, True)
    return routing_map.to(dtype)


def _bernoulli_map(num_tokens, num_experts, dtype, density=0.2):
    routing_map = (
        torch.rand(
            (num_tokens, num_experts), dtype=torch.float32, device=flag_train.device
        )
        < density
    )
    return routing_map.to(dtype)


def _assert_matches_mcore(res_out, routing_map, pad_multiple):
    """Hold the operator to Megatron's kernel on the backends that have it.

    Both sides go through ``utils.to_reference``: under ``--ref cpu`` the
    comparison helper requires its reference on the CPU, and both operands here
    are produced on the device.
    """
    if _mcore_fused_pad_routing_map is None:
        return

    mcore_out = _mcore_fused_pad_routing_map(routing_map, pad_multiple)
    utils.train_assert_equal(utils.to_reference(res_out), utils.to_reference(mcore_out))


def _check(routing_map, pad_multiple):
    ref_out = pad_routing_map_ref(utils.to_reference(routing_map), pad_multiple)
    res_out = flag_train.megatron.fused_pad_routing_map(routing_map, pad_multiple)
    utils.train_assert_equal(res_out, ref_out)
    _assert_matches_mcore(res_out, routing_map, pad_multiple)


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize("pad_multiple", _PAD_MULTIPLES)
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_pad_routing_map_topk(shape, pad_multiple, dtype):
    _check(_topk_map(*shape, dtype), pad_multiple)


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize("pad_multiple", _PAD_MULTIPLES)
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_pad_routing_map_random(shape, pad_multiple, dtype):
    _check(_bernoulli_map(*shape, dtype), pad_multiple)


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize("pad_multiple", [8, 32])
@pytest.mark.parametrize("num_experts", _RAGGED_EXPERT_COUNTS)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_fused_pad_routing_map_ragged_experts(num_experts, pad_multiple, dtype):
    """A non-power-of-two expert axis leaves the expert tile's last part masked.

    Every shape in ``_SHAPES`` and every model config uses a power-of-two expert
    count, where that mask folds away at compile time; these counts keep it live.
    """
    _check(_topk_map(257, num_experts, dtype), pad_multiple)


@pytest.mark.fused_pad_routing_map
@needs_mcore_oracle
@pytest.mark.parametrize("pad_multiple", _PAD_MULTIPLES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_matches_mcore_oracle(shape, pad_multiple):
    """Pin the Megatron kernel -- the benchmark's NVIDIA baseline -- explicitly.

    A tier that quietly stopped being reached would thin the suite without
    saying so. On a backend in ``_NV_BASELINE_VENDORS`` the load itself cannot
    fail quietly, because the module raises at import; what this catches is the
    *comparison* ceasing to happen. It skips on the backends where the tier does
    not apply at all. Every other case compares against this oracle as well, via
    ``_check``.
    """
    _check(_topk_map(*shape, torch.bool), pad_multiple)


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize("pad_multiple", [8, 32, 256])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_pad_routing_map_saturates(pad_multiple, dtype):
    """Fewer zeros than the alignment needs: every zero is flipped, none is invented."""
    num_tokens, num_experts = 8, 32
    routing_map = torch.ones(
        (num_tokens, num_experts), dtype=torch.bool, device=flag_train.device
    )
    # One zero per expert column, which cannot reach an 8/32/256 multiple.
    routing_map[3, :] = False
    _check(routing_map.to(dtype), pad_multiple)


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_pad_routing_map_no_zeros(dtype):
    """An all-ones map has nothing to flip and must come back unchanged."""
    routing_map = torch.ones((64, 16), dtype=torch.bool, device=flag_train.device)
    _check(routing_map.to(dtype), 32)


@pytest.mark.fused_pad_routing_map
def test_fused_pad_routing_map_all_zero():
    """A column already at zero is aligned with every alignment: no zeros flip."""
    routing_map = torch.zeros((64, 16), dtype=torch.bool, device=flag_train.device)
    _check(routing_map, 32)


@pytest.mark.fused_pad_routing_map
def test_fused_pad_routing_map_fully_dense_saturates():
    """No zero exists to reach the alignment: the column stays short, not invented."""
    routing_map = torch.ones((64, 16), dtype=torch.bool, device=flag_train.device)
    # 64 % 7 == 1, so six flips are requested; there is no zero to flip.
    _check(routing_map, 7)


# Real deployments, as ``(num_tokens, num_experts, top_k, pad_multiple)``.
#
# ``num_experts`` is the routing map's expert dimension: the global count with
# the all-to-all dispatcher, the local slice with flex/DeepEP. ``pad_multiple``
# is what Megatron's ``get_align_size_for_quantization`` returns for the
# deployment: 256 for TE's grouped-tensor / fused grouped-MLP kernels, 16 for
# blockwise FP8, 32 for MXFP8 and FP4. ``num_tokens`` covers one micro-batch
# (4096) through a few, which is when the map gets large enough for the tiled
# path to matter.
_MODEL_CONFIGS = [
    # name, num_tokens, num_experts, top_k, pad_multiple
    ("mixtral-8x7b", 4096, 8, 2, 256),
    ("dbrx-132b", 8192, 16, 4, 16),
    ("olmoe-1b-7b", 4096, 64, 8, 32),
    ("qwen3-235b-a22b", 8192, 128, 8, 256),
    # DeepSeek-V3 routes to 256 experts; on EP=8 a rank's map holds its slice.
    ("deepseek-v3-ep8", 8192, 32, 8, 16),
    # A large dispatch batch of the unrouted full map: 32K tokens, 256 experts.
    ("deepseek-v3-full", 32768, 256, 8, 256),
]

_MODEL_CASES = [config[1:] for config in _MODEL_CONFIGS]
_MODEL_IDS = [config[0] for config in _MODEL_CONFIGS]


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, pad_multiple", _MODEL_CASES, ids=_MODEL_IDS
)
def test_fused_pad_routing_map_model_shape(
    num_tokens, num_experts, top_k, pad_multiple
):
    """A real deployment's map, held to both oracles and to the alignment it exists for."""
    routing_map = _topk_map(num_tokens, num_experts, torch.bool, top_k=top_k)
    res = flag_train.megatron.fused_pad_routing_map(routing_map, pad_multiple)

    utils.train_assert_equal(
        res,
        pad_routing_map_ref(utils.to_reference(routing_map), pad_multiple),
    )
    _assert_matches_mcore(res, routing_map, pad_multiple)

    # The point of the operator: every expert's token count lands on the multiple.
    counts = res.sum(dim=0)
    assert torch.all(counts % pad_multiple == 0), counts
    # Padding only flips zeros to ones; it never drops a routed token.
    assert torch.all(res >= routing_map.to(res.dtype))


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, pad_multiple", _MODEL_CASES, ids=_MODEL_IDS
)
def test_fused_pad_routing_map_dropped_tokens(
    num_tokens, num_experts, top_k, pad_multiple
):
    """Rows whose tokens were all dropped, as a capacity-limited router leaves them.

    A row with no expert set contributes only zeros, so it is the first place
    padding draws from -- the same behaviour as the dense case, on the map a
    real router produces after its drop policy.
    """
    routing_map = _topk_map(num_tokens, num_experts, torch.bool, top_k=top_k)
    routing_map[::10] = False
    _check(routing_map, pad_multiple)

    res = flag_train.megatron.fused_pad_routing_map(routing_map, pad_multiple)
    counts_before = routing_map.sum(dim=0)
    counts_after = res.sum(dim=0)
    # These maps have far more zeros than any column needs, so each column gains
    # exactly the padding its own count asks for -- no more, no less.
    utils.train_assert_equal(
        utils.to_reference(counts_after - counts_before),
        utils.to_reference((-counts_before) % pad_multiple),
    )


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_pad_routing_map_non_contiguous(dtype):
    """A transposed view is padded by content, not by memory order."""
    routing_map = _topk_map(257, 64, dtype).t()
    assert not routing_map.is_contiguous()

    _check(routing_map, 32)

    # Same values as the contiguous argument, in the same [tokens, experts] order.
    res_out = flag_train.megatron.fused_pad_routing_map(routing_map, 32)
    contiguous_out = flag_train.megatron.fused_pad_routing_map(
        routing_map.contiguous(), 32
    )
    utils.train_assert_equal(
        utils.to_reference(res_out), utils.to_reference(contiguous_out)
    )


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize(
    "shape, dtype",
    [((0, 64), torch.bool), ((0, 0), torch.bool), ((0, 64), torch.int32)],
)
def test_fused_pad_routing_map_empty_tokens(shape, dtype):
    """Upstream returns the argument itself when there are no tokens."""
    routing_map = torch.zeros(shape, dtype=dtype, device=flag_train.device)
    res = flag_train.megatron.fused_pad_routing_map(routing_map, 32)

    assert res.dtype == dtype
    utils.train_assert_equal(
        res, pad_routing_map_ref(utils.to_reference(routing_map), 32)
    )
    _assert_matches_mcore(res, routing_map, 32)


@pytest.mark.fused_pad_routing_map
def test_empty_tokens_return_before_device_and_pad_validation():
    """Upstream returns the empty argument before checking device or pad_multiple."""
    routing_map = torch.zeros((0, 8), dtype=torch.bool)  # CPU on purpose
    for pad in (0, None):
        assert (
            flag_train.megatron.fused_pad_routing_map(routing_map, pad) is routing_map
        )


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_fused_pad_routing_map_no_experts(dtype):
    """num_experts == 0 is legal; the result is empty and still int32."""
    routing_map = torch.zeros((16, 0), dtype=dtype, device=flag_train.device)
    res = flag_train.megatron.fused_pad_routing_map(routing_map, 32)

    assert res.shape == (16, 0)
    assert res.dtype == torch.int32
    # Upstream builds this buffer as ``empty_like(x.transpose(0, 1).contiguous()
    # .int()).transpose(0, 1)``, and torch does not canonicalise a zero-size
    # dimension's stride the same way on every backend (CPU: (1, 1); Ascend:
    # (1, num_tokens)). Pin the composition rather than a literal, so the
    # assertion follows whichever stride upstream itself would return here.
    expected = torch.empty_like(
        routing_map.transpose(0, 1).contiguous().int()
    ).transpose(0, 1)
    assert res.stride() == expected.stride(), (res.stride(), expected.stride())
    utils.train_assert_equal(
        res, pad_routing_map_ref(utils.to_reference(routing_map), 32)
    )
    _assert_matches_mcore(res, routing_map, 32)


@pytest.mark.fused_pad_routing_map
def test_no_experts_returns_before_device_and_pad_validation():
    """An empty expert axis never reaches a kernel upstream.

    Its grid is empty, so neither the device nor ``pad_multiple`` is consulted;
    a CPU map, or a zero alignment, still returns the empty int32 result with
    upstream's zero-size strides.
    """
    routing_map = torch.zeros((16, 0), dtype=torch.bool)  # CPU on purpose
    res = flag_train.megatron.fused_pad_routing_map(routing_map, 0)

    assert res.shape == (16, 0)
    assert res.dtype == torch.int32
    assert res.stride() == (1, 1)

    if _mcore_fused_pad_routing_map_eager is not None:
        expected = _mcore_fused_pad_routing_map_eager(routing_map, 0)
        utils.train_assert_equal(res, expected)
        assert res.stride() == expected.stride()


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize(
    "pad_multiple",
    [
        True,
        np.int8(8),
        np.int16(8),
        np.int32(8),
        np.int64(8),
        8.0,
        np.float32(8.0),
        np.float64(8.0),
    ],
    ids=["bool", "int8", "int16", "int32", "int64", "float", "float32", "float64"],
)
def test_fused_pad_routing_map_pad_forms_match_integer(pad_multiple):
    """Upstream's constexpr kernel accepts bools, numpy integers and whole floats.

    The alignment is the same integer in every case, so the result must be.
    """
    routing_map = _topk_map(257, 64, torch.bool)
    res = flag_train.megatron.fused_pad_routing_map(routing_map, pad_multiple)
    expected = flag_train.megatron.fused_pad_routing_map(routing_map, int(pad_multiple))
    utils.train_assert_equal(utils.to_reference(res), utils.to_reference(expected))

    if _mcore_fused_pad_routing_map is not None:
        # Upstream's acceptance of these forms depends on Triton's argument
        # specialisation, so compare through the integer form it always takes.
        utils.train_assert_equal(
            utils.to_reference(res),
            utils.to_reference(
                _mcore_fused_pad_routing_map(routing_map, int(pad_multiple))
            ),
        )


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize(
    "dtype", _CAST_DTYPES, ids=lambda dtype: str(dtype).split(".")[-1]
)
def test_fused_pad_routing_map_casts_like_upstream(dtype):
    """Dtypes Triton cannot load are cast with ``.int()`` first, as upstream does.

    Upstream materialises an ``.int()`` working copy of every input, so complex
    (dropping its imaginary part) and fp8 maps reach its kernel. The fused path
    keeps the input dtype only when Triton can load it and takes the same cast
    otherwise.
    """
    routing_map = _topk_map(64, 16, torch.bool)
    try:
        routing_map = routing_map.to(dtype)
    except (RuntimeError, TypeError) as exc:
        pytest.skip(f"{dtype} is not available on this device: {exc}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = flag_train.megatron.fused_pad_routing_map(routing_map, 8)
        ref_map = utils.to_reference(routing_map).int()
        utils.train_assert_equal(res, pad_routing_map_ref(ref_map, 8))

        if _mcore_fused_pad_routing_map is not None:
            utils.train_assert_equal(
                utils.to_reference(res),
                utils.to_reference(_mcore_fused_pad_routing_map(routing_map, 8)),
            )


# Values that make the input's conversion to int32 visible: a fraction that
# truncates toward zero, a value that is nonzero but truncates to 0, and
# magnitudes beyond a narrow dtype's range. Upstream casts the whole map with
# ``.int()`` before its kernel, so the operator has to agree with that cast --
# a rounding load, a zero-extended int8 load or a wrapping float-to-int would
# change the per-expert sum and with it how many zeros get padded.
#
# Negative values are deliberately absent: upstream computes
# ``num_ones % pad_multiple`` with Triton's C-style remainder (the result keeps
# the dividend's sign) while torch's ``%`` does not, so a negative row sum has
# no portable answer and is outside the boolean routing map's contract.
_INTEGER_CONVERSION_VALUES = {
    torch.uint8: [0, 1, 2, 255],
    torch.int8: [0, 1, 2, 7],
    torch.int16: [0, 1, 2, 1000],
    torch.int32: [0, 1, 2, 1 << 20],
    torch.int64: [0, 1, 2, 1 << 40],
    torch.float16: [0.0, 0.6, 1.0, 1.9],
    torch.bfloat16: [0.0, 0.6, 1.0, 1.9],
    torch.float32: [0.0, 0.6, 1.0, 2.9, 1e-8],
    torch.float64: [0.0, 0.6, 1.0, 2.9, 1e-8],
}

# The widest set this device can actually hold and load, in the capability terms
# the rest of the suite uses.
_INTEGER_CONVERSION_DTYPES = [
    dtype
    for dtype in _INTEGER_CONVERSION_VALUES
    if (dtype != torch.float64 or utils.fp64_is_supported)
    and (dtype != torch.int64 or utils.int64_is_supported)
    and (dtype != torch.bfloat16 or utils.bf16_is_supported)
]


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize(
    "dtype", _INTEGER_CONVERSION_DTYPES, ids=lambda d: str(d).split(".")[-1]
)
def test_fused_pad_routing_map_matches_int_conversion(dtype):
    """A map of any accepted dtype reaches the kernel as ``.int()`` would.

    Upstream materialises ``routing_map.transpose(0, 1).contiguous().int()``
    before its kernel, so the operator's job for a non-boolean map is to agree
    with that cast element by element -- and, per the reference above, to flip
    zeros without rewriting the values it keeps.
    """
    values = _INTEGER_CONVERSION_VALUES[dtype]
    index = torch.randint(0, len(values), (257, 64))
    routing_map = torch.tensor(values, dtype=dtype)[index].to(flag_train.device)

    res = flag_train.megatron.fused_pad_routing_map(routing_map, 32)

    ref_map = utils.to_reference(routing_map).int()
    utils.train_assert_equal(res, pad_routing_map_ref(ref_map, 32))
    assert res.dtype == torch.int32
    _assert_matches_mcore(res, routing_map, 32)


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize("shape", [(4096, 64), (4097, 64)])
def test_fused_pad_routing_map_contract(shape):
    """int32 out, [tokens, experts] view of an [experts, tokens] buffer, input intact."""
    num_tokens, _ = shape
    routing_map = _topk_map(*shape, torch.bool)
    original = routing_map.clone()

    res = flag_train.megatron.fused_pad_routing_map(routing_map, 32)

    assert res.dtype == torch.int32
    assert res.shape == shape
    assert res.stride() == (1, num_tokens), res.stride()
    utils.train_assert_equal(
        utils.to_reference(routing_map), utils.to_reference(original)
    )


@pytest.mark.fused_pad_routing_map
def test_fused_pad_routing_map_replays_deterministically():
    """The kernel has no atomics, so repeated calls are bit-identical.

    Upstream pins the same property in its determinism suite; a future change
    that introduced a racy count or flip would be visible here even if every
    single-call comparison still passed.
    """
    routing_map = _topk_map(4097, 64, torch.bool)
    _check(routing_map, 32)

    first = flag_train.megatron.fused_pad_routing_map(routing_map, 32)
    for _ in range(2):
        utils.train_assert_equal(
            utils.to_reference(first),
            utils.to_reference(
                flag_train.megatron.fused_pad_routing_map(routing_map, 32)
            ),
        )


@pytest.mark.fused_pad_routing_map
@pytest.mark.parametrize(
    "pad_multiple", [32, 256], ids=["short_alignment", "hierarchical_alignment"]
)
def test_fused_pad_routing_map_many_partials(pad_multiple):
    """A row whose count kernel emits more partial tiles than the flip sums at once.

    The flip pass cannot know a row's total until it has summed every partial
    the count pass wrote, and it does that in blocks (Ascend's ``PARTIAL_BLOCK``
    is 1024 tiles), so a row with more tiles than one block takes the
    block-at-a-time loop. One expert keeps the map small while its token count
    crosses that bound: at a 2048-lane tile, 2 097 153 tokens is 1025 partials.
    Both the short-alignment scan and the hierarchical one are exercised.

    The Megatron oracle is not consulted here: upstream's kernel sizes its block
    as ``next_power_of_2(num_tokens)``, which does not compile at this length.
    """
    num_tokens = 2048 * 1024 + 1
    routing_map = _bernoulli_map(num_tokens, 1, torch.bool)

    # Pin the coverage on a backend that exposes its tiling: the plan must
    # really emit more tiles than the flip sums in one block, or this case
    # would silently stop testing the loop.
    op_module = sys.modules[flag_train.megatron.fused_pad_routing_map.__module__]
    run_config = getattr(op_module, "_run_config", None)
    if run_config is not None:
        try:
            config = run_config(num_tokens, 1)
        except TypeError:  # the generic plan also takes a device index
            config = None
        if isinstance(config, dict) and "num_tiles" in config:
            assert config["num_tiles"] > 1024, config

    res = flag_train.megatron.fused_pad_routing_map(routing_map, pad_multiple)
    ref_out = pad_routing_map_ref(utils.to_reference(routing_map), pad_multiple)
    utils.train_assert_equal(res, ref_out)


@pytest.mark.fused_pad_routing_map
def test_fused_pad_routing_map_rejects_bad_arguments():
    routing_map = torch.zeros((16, 8), dtype=torch.bool, device=flag_train.device)

    for bad_shape in ((16,), (2, 16, 8)):
        with pytest.raises(ValueError):
            flag_train.megatron.fused_pad_routing_map(
                torch.zeros(bad_shape, dtype=torch.bool, device=flag_train.device), 8
            )
    if flag_train.device != "cpu":
        with pytest.raises(ValueError, match="only supports"):
            flag_train.megatron.fused_pad_routing_map(routing_map.cpu(), 8)

    # Upstream has no check here and its kernel's behaviour is undefined for
    # these (a modulo by zero, or a negative alignment); reject them instead.
    for bad_pad in (0, -8, 8.5, np.float32(8.5), None, "8"):
        with pytest.raises(AssertionError):
            flag_train.megatron.fused_pad_routing_map(routing_map, bad_pad)
