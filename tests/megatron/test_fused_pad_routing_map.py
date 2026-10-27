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
# !! KEEP IN SYNC with benchmark/megatron/test_fused_pad_routing_map.py !!
#
# The benchmark carries a byte-identical copy. A reference belongs to the tests
# that check against it, not to the operator package, so the duplication is
# deliberate; the cost is that the two copies can drift.
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
    work[zero_ranks <= num_to_pad.unsqueeze(1)] = 1

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

    Both sides stay on the device, so this is a plain compare rather than one
    through ``train_assert_equal`` (whose ``--ref cpu`` path requires a CPU ref).
    """
    if _mcore_fused_pad_routing_map is None:
        return

    mcore_out = _mcore_fused_pad_routing_map(routing_map, pad_multiple)
    assert torch.equal(res_out, mcore_out), (
        f"the Megatron oracle disagrees with the operator on "
        f"shape={tuple(routing_map.shape)} pad_multiple={pad_multiple}"
    )


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
    assert torch.equal(counts_after - counts_before, (-counts_before) % pad_multiple), (
        counts_before,
        counts_after,
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
    assert torch.equal(res_out, contiguous_out)


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
def test_fused_pad_routing_map_no_experts():
    """num_experts == 0 is legal; the result is empty and still int32."""
    routing_map = torch.zeros((16, 0), dtype=torch.bool, device=flag_train.device)
    res = flag_train.megatron.fused_pad_routing_map(routing_map, 32)

    assert res.shape == (16, 0)
    assert res.dtype == torch.int32
    utils.train_assert_equal(
        res, pad_routing_map_ref(utils.to_reference(routing_map), 32)
    )
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
    assert torch.equal(routing_map, original)


@pytest.mark.fused_pad_routing_map
def test_fused_pad_routing_map_rejects_bad_arguments():
    routing_map = torch.zeros((16, 8), dtype=torch.bool, device=flag_train.device)

    with pytest.raises(AssertionError):
        flag_train.megatron.fused_pad_routing_map(
            torch.zeros((16,), dtype=torch.bool, device=flag_train.device), 8
        )
    with pytest.raises(AssertionError):
        flag_train.megatron.fused_pad_routing_map(routing_map, 0)
    with pytest.raises(AssertionError):
        flag_train.megatron.fused_pad_routing_map(routing_map, -8)
