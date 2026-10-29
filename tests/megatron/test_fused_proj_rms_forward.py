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
"""Correctness tests for the fused projection + RMS normalization (forward).

The oracle is Megatron-LM's ``native_proj_rms`` -- the reference implementation
this operator is a port of -- taken from the installed ``megatron-core``, the
same way ``test_fused_pad_routing_map.py`` imports Megatron's kernel. The
function arrived with the mHC proj_rms stage in megatron-core 0.20, so on an
older install the oracle is ``native_proj_rms_ref``: Megatron-LM's
``hyper_connection.py`` transcribed into this file, operation for operation,
the way the functional suites here carry the references they check against.
Either way the operator is held to Megatron's arithmetic rather than the tests
skipping, and the performance suite times the same implementation they check.
Where megatron-core is new enough to have the function but cannot import it,
the loader raises instead of quietly substituting the transcription.

``native_proj_rms`` is wrapped in ``@torch.compile`` upstream; the eager
function underneath (``_torchdynamo_orig_callable``) is the same code without a
dynamo compile in front of every shape in the sweep, and that is what is
compared against. The benchmark keeps the wrapper, because that is what Megatron
pays; the vendored fallback is already the eager body, so there the benchmark
compiles the copy.

Two ambient settings make the *reference* the inaccurate side here, so the
oracle is run with them pinned -- see ``_reference_arithmetic``.

Comparison is a fraction of the reference's RMS, not a per-element ``rtol``; see
``_assert_close`` for why the house table does not fit a matmul output.
"""

import importlib.metadata
import math
import re
import warnings
from contextlib import contextmanager

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import accuracy_utils as utils

_EPS = 1e-6

# Floor for ``|x|`` rows in the short-reduction cases; see ``_operands``.
_MIN_ROW_NORM = 0.1

# ``(M, N, K)``: ``x`` is ``[M, K]``, ``weight`` is ``[N, K]``, ``proj`` is
# ``[M, N]`` and ``r`` is ``[M, 1]``.
#
# 24 is the mHC mapping height, i.e. the width the operator exists for; 80, 288
# and 1024 walk the N tile (one masked tile, several tiles, and a grid where the
# output tile is the work). M covers a single row, masked row tiles (7, 4097),
# the row-block switch at 1024, and a full micro-batch. 8192 x 8192 keeps the
# reduction long enough for split-K behaviour to show up, and 4097 is the
# non-power-of-two row and reduction length.
_SHAPES = [
    (1, 24, 4096),
    (7, 24, 4096),
    (128, 24, 4096),
    (256, 24, 4096),
    (1024, 24, 4096),
    (4096, 24, 4096),
    (8192, 24, 8192),
    (4096, 80, 4096),
    (4096, 100, 4096),
    (4096, 288, 4096),
    (4096, 1024, 4096),
    (8192, 4096, 8192),
    (4097, 24, 4097),
]

# ``--quick`` (the second CI pass) keeps one masked row block and one full tile.
if utils.QUICK_MODE:
    _SHAPES = [(7, 24, 512), (128, 24, 512)]

# Every dimension below its tile: one row, one output, one reduction term, and
# an N under ``tl.dot``'s 16-wide floor.
_TINY_SHAPES = [
    (1, 1, 1),
    (3, 5, 7),
    (7, 24, 1),
    (17, 17, 17),
    (5, 1, 33),
]

# The sweep the performance suite reports latency for.
# !! KEEP IN SYNC with benchmark/megatron/test_fused_proj_rms_forward.py !!
_BENCHMARK_SHAPES = [
    (1, 24, 4096),
    (128, 24, 4096),
    (256, 24, 4096),
    (1024, 24, 4096),
    (4096, 24, 4096),
    (8192, 24, 8192),
    (4096, 80, 4096),
    (4096, 288, 4096),
    (4096, 1024, 4096),
    (8192, 4096, 8192),
]

# FP32 exercises the exact-arithmetic path; FP16 is always available, BF16 only
# where the runtime reports it.
_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)
_BENCHMARK_DTYPES = [dtype for dtype in _DTYPES if dtype is not torch.float32]

# Mixed dtypes are a separate contract test; see
# ``test_fused_proj_rms_forward_mixed_dtype``.
_MIXED_DTYPES = [(torch.float16, torch.float32), (torch.float32, torch.float16)]
if utils.bf16_is_supported:
    _MIXED_DTYPES += [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ]

# ---------------------------------------------------------------------------
# Megatron oracle
#
# The kernel this port is a port of, resolved from the installed megatron-core
# when that is new enough to have it, and otherwise from ``native_proj_rms_ref``
# below. ``native_proj_rms`` and the mHC proj_rms stage arrived in megatron-core
# 0.20; 0.19 and earlier have no ``hyper_connection`` module at all, so without
# the transcription an older container could only skip these tests.
#
# The rule is the same on every supported backend: megatron-core when it has
# the function, the transcription otherwise. A megatron-core that is new enough
# to have the function but cannot import it is a broken installation, and that
# is raised rather than papered over with the transcription.
# ---------------------------------------------------------------------------
_MCORE_MIN_VERSION = (0, 20)


def _mcore_version():
    try:
        return importlib.metadata.version("megatron-core")
    except Exception:
        return "not installed"


_MCORE_VERSION = _mcore_version()


def _version_tuple(version):
    """``(major, minor)`` of a distribution version, or ``None`` without one."""
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", version)
    return None if match is None else (int(match.group(1)), int(match.group(2) or 0))


def native_proj_rms_ref(x, weight, eps=1e-6):
    """Megatron-LM's ``native_proj_rms``, transcribed.

    From ``megatron/core/transformer/hyper_connection.py`` at Megatron-LM 0.20,
    kept operation for operation, because the operator is checked against
    exactly this arithmetic. Upstream it carries a ``@torch.compile``
    decorator; this is the eager body, which is what the correctness suite
    compares against when the function comes from an installed megatron-core
    too -- the benchmarks add the compile back.
    """
    proj = torch.matmul(x, weight.t())
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    v = norm / math.sqrt(K) + eps
    r = 1.0 / v
    return proj, r


def _load_native_proj_rms():
    """Megatron-LM's ``native_proj_rms``, or the transcription when older."""
    try:
        from megatron.core.transformer.hyper_connection import native_proj_rms
    except Exception as exc:
        version = _version_tuple(_MCORE_VERSION)
        if version is not None and version >= _MCORE_MIN_VERSION:
            raise RuntimeError(
                f"megatron-core {_MCORE_VERSION} should provide "
                f"native_proj_rms, but importing it failed: {exc!r}. This "
                f"operator is the proj_rms stage of "
                f"megatron/core/fusions/fused_mhc_kernels.py, so it must be "
                f"checked against Megatron's own implementation; the vendored "
                f"transcription in this file only stands in for megatron-core "
                f"< 0.20."
            ) from exc
        warnings.warn(
            f"megatron-core {_MCORE_VERSION} does not provide native_proj_rms "
            f"({exc!r}); fused_proj_rms_forward is checked against "
            f"native_proj_rms_ref, this file's transcription of Megatron-LM's "
            f"implementation",
            stacklevel=2,
        )
        return native_proj_rms_ref

    # Megatron wraps it in ``@torch.compile``; compare against the code beneath.
    return getattr(native_proj_rms, "_torchdynamo_orig_callable", native_proj_rms)


_native_proj_rms = _load_native_proj_rms()


@contextmanager
def _reference_arithmetic():
    """Pin the reference's matmul precision while the oracle runs.

    Two ambient settings decide how accurate Megatron's own arithmetic is:

    * ``allow_tf32``. This build enables TF32 for fp32 matmuls, which puts
      Megatron's projection about 1e-3 off the exact product -- three orders of
      magnitude worse than the IEEE dot the kernel issues. Pinning the oracle to
      IEEE compares two accurate implementations instead of measuring the
      container's default.
    * ``allow_{fp16,bf16}_reduced_precision_reduction``. With these on, cuBLAS
      may accumulate split-K partials in the operand dtype; at K=4097/8192 that
      is a measurable error in the *reference* (its ``grad_weight`` moves by
      1e-3 of the output RMS in fp16, and by more in bf16), which would be
      charged to the kernel.

    Both are restored on the way out, so the operator is still launched under
    whatever the environment sets.
    """
    matmul = torch.backends.cuda.matmul
    saved = (
        matmul.allow_tf32,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction,
    )
    matmul.allow_tf32 = False
    matmul.allow_fp16_reduced_precision_reduction = False
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        (
            matmul.allow_tf32,
            matmul.allow_fp16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction,
        ) = saved


# Tolerances, as a fraction of the reference's RMS.
#
# The error that separates two correct implementations of this operator is fp32
# *accumulation* error: absolute, and not proportional to each output element.
# Measured over every shape in ``_SHAPES``, the worst element is 1.7e-5 of the
# RMS in fp32 (our dot is IEEE and Megatron's is cuBLAS, so only the summation
# order differs), 4.5e-3 in fp16 and 2.6e-2 in bf16 -- the last two dominated by
# the reference rounding its own intermediates into the narrow dtype. Each
# entry is that band with an order of magnitude (fp32) or 2-4x of headroom.
_TOL = {
    torch.float32: 1e-4,
    torch.float16: 2e-2,
    torch.bfloat16: 5e-2,
}


def _assert_close(res, ref, dtype, tol):
    """Compare at ``tol`` of the reference's RMS, elementwise.

    ``torch.testing.assert_close``'s per-element ``rtol`` is the wrong shape of
    tolerance for a matmul output: on ``grad_weight`` (RMS ~ sqrt(M)) it would
    demand sub-ULP agreement on the small elements of a large tensor and pass
    anything at all on the large ones. Scaling by the reference's RMS is the
    elementwise form of the relative Frobenius error, so every element is held
    to the same meaningful budget, and the compare stays honest about which side
    is rounding.

    ``equal_nan`` covers the degenerate ``K == 0`` row, where both sides divide
    zero by zero; matching infinities (an fp16 row whose norm is zero) compare
    equal on their own.
    """
    assert res.dtype == dtype, (res.dtype, dtype)
    assert res.shape == ref.shape, (res.shape, ref.shape)

    res = utils.to_reference(res)
    ref = utils.to_reference(ref)
    if res.numel() == 0:
        return

    # float() is exact for every dtype in play and keeps the comparison itself
    # out of the output dtype's rounding.
    res32 = res.float()
    ref32 = ref.float()
    finite = torch.isfinite(ref32)
    scale = ref32[finite].pow(2).mean().sqrt().item() if finite.any() else 0.0
    torch.testing.assert_close(
        res32, ref32, atol=tol * scale, rtol=tol, equal_nan=True, check_dtype=False
    )


def _oracle_forward(x, weight, eps):
    """Megatron's ``native_proj_rms`` on the reference device."""
    with _reference_arithmetic():
        return _native_proj_rms(utils.to_reference(x), utils.to_reference(weight), eps)


def _operands(shape, dtype):
    """``(x, weight)`` for one case.

    ``weight`` is scaled by ``1/sqrt(K)`` so the projection stays O(1) instead of
    growing with the reduction length; a scale that drifted with K would make
    every case the same test of the same rounding.

    With a short reduction, a row whose norm is near zero is lifted to
    ``_MIN_ROW_NORM``: ``r`` is ``1 / (|x|/sqrt(K) + eps)``, which at fp16 leaves
    the dtype's range once ``|x|`` drops below ~1.5e-5, and the two sides then
    differ by an inf only because their rounding of ``|x|`` landed on either side
    of the overflow. Rows at the floor are what that test is about, and they are
    covered deliberately by ``test_fused_proj_rms_forward_zero_rows``.
    """
    M, N, K = shape
    device = flag_train.device
    x = torch.randn((M, K), dtype=dtype, device=device)
    if dtype == torch.float16 and K < 16:
        # fp16 -> fp32 -> fp16 is exact, so rows above the floor come back
        # bit-identical and only the degenerate ones move.
        norm = x.float().norm(dim=1, keepdim=True)
        x = (x.float() * (_MIN_ROW_NORM / norm.clamp_min(1e-30)).clamp(min=1.0)).to(
            dtype
        )
    weight = torch.randn((N, K), dtype=torch.float32, device=device)
    weight = (weight / math.sqrt(max(K, 1))).to(dtype)
    return x, weight


def _check(x, weight, dtype, eps=_EPS):
    """Run the operator against the oracle and return its outputs."""
    proj, r = flag_train.megatron.fused_proj_rms_forward(x, weight, eps)
    proj_ref, r_ref = _oracle_forward(x, weight, eps)
    _assert_close(proj, proj_ref, dtype, _TOL[dtype])
    _assert_close(r, r_ref, dtype, _TOL[dtype])
    return proj, r


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_forward(shape, dtype):
    """Both outputs of the forward match Megatron, in every dtype."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("shape", _TINY_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_forward_tiny_shapes(shape, dtype):
    """Shapes under one tile: masked loads, one reduction step, N below 16."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("shape", _BENCHMARK_SHAPES)
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
def test_fused_proj_rms_forward_benchmark_shapes(shape, dtype):
    """Every benchmarked shape has a correctness check behind it."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_forward_zero_rows(dtype):
    """An all-zero row has no direction: ``r`` degenerates to ``1/eps``.

    Megatron computes ``1 / (0 / sqrt(K) + eps)``, so the row is legal and the
    reference defines the answer -- ``_check`` has already compared it. In fp16
    that value saturates to inf, on both sides.
    """
    shape = (7, 24, 512) if utils.QUICK_MODE else (1024, 24, 4096)
    x, weight = _operands(shape, dtype)
    x[::3] = 0.0

    proj, r = _check(x, weight, dtype)

    assert torch.all(proj[::3] == 0)
    # 1/eps in the output dtype. fp16 saturates to inf there, which is what the
    # reference's own cast returns; ``_check`` has already compared the two.
    expected = torch.full(
        r[::3].shape, 1.0 / _EPS, dtype=torch.float32, device=r.device
    )
    expected = expected.to(dtype).float()
    torch.testing.assert_close(r[::3].float(), expected, rtol=1e-3, atol=0.0)


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("eps", [0.0, 1e-6, 1e-5, 0.5])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_forward_eps(eps, dtype):
    """eps is a parameter of the contract, not just the default 1e-6."""
    shape = (7, 24, 512) if utils.QUICK_MODE else (257, 24, 1024)
    _check(*_operands(shape, dtype), dtype, eps)


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_forward_non_contiguous(dtype):
    """A strided argument is read by content, not by memory order."""
    M, N, K = 257, 24, 1024
    device = flag_train.device
    x = torch.randn((K, M), dtype=dtype, device=device).t()
    weight = torch.randn((N, K), dtype=torch.float32, device=device)
    weight = (weight / math.sqrt(K)).t().contiguous().t().to(dtype)
    assert not x.is_contiguous()
    assert not weight.is_contiguous()

    proj, r = _check(x, weight, dtype)

    # Same values, same memory order as the transposed views: bit-identical.
    proj_c, r_c = flag_train.megatron.fused_proj_rms_forward(
        x.contiguous(), weight.contiguous()
    )
    assert torch.equal(proj, proj_c)
    assert torch.equal(r, r_c)


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("dtype, weight_dtype", _MIXED_DTYPES)
def test_fused_proj_rms_forward_mixed_dtype(dtype, weight_dtype):
    """Two dtypes promote to one, and the kernel computes in it.

    Megatron's native path has no single answer for the mixed case: its
    projection is promoted by ``torch.matmul`` but ``r`` is computed from the
    unpromoted ``x`` and stays in ``x``'s dtype. The fused mHC path, which this
    operator follows, computes both in one dtype. That contract is checked here
    as self-consistency -- the mixed call must be bit-identical to the call with
    both operands already promoted.
    """
    shape = (7, 24, 512) if utils.QUICK_MODE else (257, 24, 1024)
    x, weight = _operands(shape, dtype)
    weight = weight.to(weight_dtype)
    promoted = torch.promote_types(dtype, weight_dtype)

    proj, r = flag_train.megatron.fused_proj_rms_forward(x, weight)
    proj_ref, r_ref = flag_train.megatron.fused_proj_rms_forward(
        x.to(promoted), weight.to(promoted)
    )

    assert proj.dtype == promoted and r.dtype == promoted
    assert torch.equal(proj, proj_ref)
    assert torch.equal(r, r_ref)


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("shape", [(0, 24, 4096), (16, 0, 4096), (16, 24, 0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_proj_rms_forward_empty(shape, dtype):
    """Empty M, N and K are legal; the reference defines each of them.

    ``K == 0`` leaves ``0 / sqrt(0)`` on both sides, which is NaN -- the same
    NaN, and ``_assert_close`` compares them as equal. ``N == 0`` is the path
    that computes ``r`` without a projection to hang it off.
    """
    x, weight = _operands(shape, dtype)
    _check(x, weight, dtype)


@pytest.mark.fused_proj_rms_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_forward_contract(dtype):
    """Shapes, dtypes, and the arguments left alone."""
    x, weight = _operands((129, 24, 512), dtype)
    x_copy, weight_copy = x.clone(), weight.clone()

    proj, r = flag_train.megatron.fused_proj_rms_forward(x, weight)

    assert proj.shape == (129, 24) and proj.dtype == dtype
    assert r.shape == (129, 1) and r.dtype == dtype
    assert proj.is_contiguous() and r.is_contiguous()
    assert torch.equal(x, x_copy) and torch.equal(weight, weight_copy)


@pytest.mark.fused_proj_rms_forward
def test_fused_proj_rms_forward_rejects_bad_arguments():
    device = flag_train.device
    x = torch.randn((16, 32), dtype=torch.float32, device=device)
    weight = torch.randn((8, 32), dtype=torch.float32, device=device)
    op = flag_train.megatron.fused_proj_rms_forward

    with pytest.raises(ValueError):
        op(x.unsqueeze(0), weight)
    with pytest.raises(ValueError):
        op(x, torch.randn((8, 16), dtype=torch.float32, device=device))
    with pytest.raises(ValueError):
        op(x.to(torch.int32), weight)
    with pytest.raises(ValueError):
        op(x, weight, -1e-6)
    with pytest.raises(TypeError):
        op(x, weight.tolist())
    with pytest.raises(ValueError):
        op(x.cpu(), weight.cpu())
