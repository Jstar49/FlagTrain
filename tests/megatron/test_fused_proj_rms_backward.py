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
"""Correctness tests for the fused projection + RMS normalization (backward).

The oracle is ``torch.autograd.grad`` through Megatron-LM's ``native_proj_rms``
-- the same reference implementation the forward suite compares against, so the
gradients under test are the reference's own, derived by autograd, rather than a
hand-written transcription of the chain rule. As in the forward suite, the
function is taken from the installed megatron-core when that provides it and
otherwise from ``native_proj_rms_ref``, this file's transcription of
``hyper_connection.py`` (the function arrived in megatron-core 0.20); a
megatron-core new enough to have it but unable to import it is raised rather
than substituted.

The saved forward scale the operator takes (``r``) comes from *our* forward,
which is how the operator is used; the reference recomputes its own from ``x``
inside the graph. The two can differ by a rounding step in the narrow dtypes,
which is part of what the tolerances cover.

See ``tests/megatron/test_fused_proj_rms_forward.py`` for ``_reference_arithmetic``
and ``_assert_close``; they are repeated here because a reference belongs to the
tests that check against it, not to a shared helper -- the same reason
``test_fused_pad_routing_map.py`` and its benchmark carry byte-identical copies.
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

# ``(M, N, K)``, as in the forward suite: ``x`` is ``[M, K]``, ``weight`` is
# ``[N, K]``, ``grad_proj`` is ``[M, N]``, ``grad_r`` is ``[M, 1]``; the outputs
# are ``grad_x`` ``[M, K]`` and ``grad_weight`` ``[N, K]``.
#
# The three products in the backward parallelise differently -- ``grad_weight``
# reduces over M while ``grad_x`` reduces over N -- so the sweep walks both
# reduction lengths: M from 0-sized tiles to a full micro-batch, N at mHC's 24
# up to a tile-filling 1024, and K from one tile to 8192.
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

if utils.QUICK_MODE:
    _SHAPES = [(7, 24, 512), (128, 24, 512)]

_TINY_SHAPES = [
    (1, 1, 1),
    (3, 5, 7),
    (7, 24, 1),
    (17, 17, 17),
    (5, 1, 33),
]

# The sweep the performance suite reports latency for.
# !! KEEP IN SYNC with benchmark/megatron/test_fused_proj_rms_backward.py !!
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

_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)
_BENCHMARK_DTYPES = [dtype for dtype in _DTYPES if dtype is not torch.float32]

_MIXED_DTYPES = [(torch.float16, torch.float32), (torch.float32, torch.float16)]
if utils.bf16_is_supported:
    _MIXED_DTYPES += [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ]

# ---------------------------------------------------------------------------
# Megatron oracle
#
# Resolved from the installed megatron-core when that is new enough to have
# ``native_proj_rms`` (0.20), and otherwise from ``native_proj_rms_ref`` below.
# 0.19 and earlier have no ``hyper_connection`` module at all, so without the
# transcription an older container could only skip these tests.
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
    kept operation for operation, because the gradients under test are this
    function's own. Upstream it carries a ``@torch.compile`` decorator; this is
    the eager body, which is what the autograd oracle differentiates when the
    function comes from an installed megatron-core too.
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
                f"megatron/core/fusions/fused_mhc_kernels.py, so its gradients "
                f"must be checked against Megatron's own implementation; the "
                f"vendored transcription in this file only stands in for "
                f"megatron-core < 0.20."
            ) from exc
        warnings.warn(
            f"megatron-core {_MCORE_VERSION} does not provide native_proj_rms "
            f"({exc!r}); fused_proj_rms_backward is checked against "
            f"native_proj_rms_ref, this file's transcription of Megatron-LM's "
            f"implementation",
            stacklevel=2,
        )
        return native_proj_rms_ref

    # Megatron wraps it in ``@torch.compile``; differentiate the code beneath.
    return getattr(native_proj_rms, "_torchdynamo_orig_callable", native_proj_rms)


_native_proj_rms = _load_native_proj_rms()


@contextmanager
def _reference_arithmetic():
    """Pin the reference's matmul precision while the oracle runs.

    Same reasoning as the forward suite: TF32 would put Megatron's fp32
    projection ~1e-3 off the exact product, and the reduced-precision reduction
    flags let cuBLAS accumulate split-K partials in the operand dtype, which at
    K=M=4097/8192 moves the *reference's* gradients by up to 1e-3 of their RMS
    in fp16 and more in bf16. Both are restored on the way out.
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


# Tolerances, as a fraction of the reference's RMS. Measured over every shape in
# ``_SHAPES`` with random ``grad_proj`` and ``grad_r``, the worst element is
# 1.7e-5 of the RMS in fp32, 4.5e-3 in fp16 and 2.6e-2 in bf16. The narrow-dtype
# bands are the reference's own rounding, not the kernel's: it casts every
# matmul and every norm-backward step into the activation dtype, while the
# kernel accumulates in fp32 and rounds once. 5e-2 for bf16 is also the band
# Megatron-LM's own mHC kernel tests allow for a backward.
_TOL = {
    torch.float32: 1e-4,
    torch.float16: 2e-2,
    torch.bfloat16: 5e-2,
}


def _assert_close(res, ref, dtype, tol):
    """Compare at ``tol`` of the reference's RMS, elementwise.

    ``torch.testing.assert_close``'s per-element ``rtol`` would demand sub-ULP
    agreement on the small elements of ``grad_weight`` (RMS ~ sqrt(M)) and pass
    anything on the large ones; see the forward suite for the full argument.
    ``equal_nan`` covers the degenerate ``K == 0`` case, where both sides are
    empty and the coefficient is zero over zero.
    """
    assert res.dtype == dtype, (res.dtype, dtype)
    assert res.shape == ref.shape, (res.shape, ref.shape)

    res = utils.to_reference(res)
    ref = utils.to_reference(ref)
    if res.numel() == 0:
        return

    res32 = res.float()
    ref32 = ref.float()
    finite = torch.isfinite(ref32)
    scale = ref32[finite].pow(2).mean().sqrt().item() if finite.any() else 0.0
    torch.testing.assert_close(
        res32, ref32, atol=tol * scale, rtol=tol, equal_nan=True, check_dtype=False
    )


def _oracle_backward(x, weight, grad_proj, grad_r, eps):
    """``torch.autograd.grad`` through Megatron's ``native_proj_rms``.

    The graph is the reference's own forward, so ``grad_x`` carries the norm
    backward exactly as PyTorch derives it -- including the zero at a row whose
    norm is zero, which the operator reproduces rather than special-cases away.
    """
    x = utils.to_reference(x).detach().clone().requires_grad_(True)
    weight = utils.to_reference(weight).detach().clone().requires_grad_(True)
    grad_proj = utils.to_reference(grad_proj)
    grad_r = utils.to_reference(grad_r)

    with _reference_arithmetic():
        proj, r = _native_proj_rms(x, weight, eps)
        grad_x, grad_weight = torch.autograd.grad(
            (proj, r), (x, weight), (grad_proj, grad_r)
        )
    return grad_x.detach(), grad_weight.detach()


def _operands(shape, dtype):
    """``(x, weight, grad_proj, grad_r)`` for one case.

    With a short reduction, a row whose norm is near zero is lifted to
    ``_MIN_ROW_NORM``. The gradient's coefficient term goes as
    ``grad_r * x / |x|^3``, i.e. ``grad_x`` grows as ``1 / |x|^2``; below
    ``|x| ~ 0.01`` that leaves fp16's range, and the two sides then differ by an
    inf only because their rounding of ``|x|`` landed on either side of the
    overflow. Rows at the floor are what that test is about, and they are covered
    deliberately by ``test_fused_proj_rms_backward_zero_rows``.
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
    grad_proj = torch.randn((M, N), dtype=dtype, device=device)
    grad_r = torch.randn((M, 1), dtype=dtype, device=device)
    return x, weight, grad_proj, grad_r


def _check(x, weight, grad_proj, grad_r, eps=_EPS):
    """Run the operator against the oracle and return its gradients."""
    promoted = torch.promote_types(x.dtype, weight.dtype)
    _, r = flag_train.megatron.fused_proj_rms_forward(x, weight, eps)
    grad_x, grad_weight = flag_train.megatron.fused_proj_rms_backward(
        x, weight, grad_proj, grad_r, r
    )
    grad_x_ref, grad_weight_ref = _oracle_backward(x, weight, grad_proj, grad_r, eps)

    _assert_close(grad_x, grad_x_ref, x.dtype, _TOL[promoted])
    _assert_close(grad_weight, grad_weight_ref, weight.dtype, _TOL[promoted])
    return grad_x, grad_weight


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_backward(shape, dtype):
    """Both gradients match autograd through Megatron, in every dtype."""
    _check(*_operands(shape, dtype))


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("shape", _TINY_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_backward_tiny_shapes(shape, dtype):
    """Shapes under one tile: masked loads, one reduction step, N below 16."""
    _check(*_operands(shape, dtype))


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("shape", _BENCHMARK_SHAPES)
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
def test_fused_proj_rms_backward_benchmark_shapes(shape, dtype):
    """Every benchmarked shape has a correctness check behind it."""
    _check(*_operands(shape, dtype))


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_backward_without_grad_r(dtype):
    """``grad_r == 0`` leaves the pure matmul path.

    This is the half of the operator that has no coefficient term, so a
    difference here is a difference in ``grad_proj @ weight`` or in the
    ``grad_proj^T @ x`` product -- not in the norm backward.
    """
    x, weight, grad_proj, grad_r = _operands(
        (7, 24, 512) if utils.QUICK_MODE else (4096, 24, 4096), dtype
    )
    grad_r = torch.zeros((x.shape[0], 1), dtype=dtype, device=x.device)
    _check(x, weight, grad_proj, grad_r)


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_proj_rms_backward_coefficient_only(dtype):
    """``grad_proj == 0`` leaves only the coefficient term, ``coef * x``.

    Narrow dtypes are not checked here: the term is ~0.3% of a full ``grad_x``,
    and the reference computes its coefficient through several casts into the
    activation dtype, so at bf16 its own error is ~10% of *this* term's RMS.
    That is a property of the reference, not of the operator -- which is why the
    dtype sweep above, where the term is in proportion, is the real check.

    The shapes are small because the term is one elementwise multiply by ``x``
    and a row reduction: there is no matmul to exhaust.
    """
    shape = (7, 24, 512) if utils.QUICK_MODE else (1024, 24, 4096)
    x, weight, grad_proj, grad_r = _operands(shape, dtype)
    grad_proj = torch.zeros_like(grad_proj)
    _check(x, weight, grad_proj, grad_r)


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_backward_zero_rows(dtype):
    """A zero row has no norm direction: the coefficient term contributes zero.

    PyTorch's norm backward returns zero there rather than NaN, and so does the
    operator; the row's ``grad_proj`` term is untouched.

    fp16 is the exception, and it is the *reference* that fails there: ``1/eps``
    saturates ``r`` to inf, and autograd's ``inf * 0`` turns the whole row into
    NaN. The operator's guard keeps the row finite, so the fp16 case is held to
    the value the guard promises instead of to a NaN.
    """
    x, weight, grad_proj, grad_r = _operands(
        (7, 24, 512) if utils.QUICK_MODE else (1024, 24, 4096), dtype
    )
    x[::3] = 0.0
    grad_proj_zeroed = grad_proj.clone()
    grad_proj_zeroed[::3] = 0.0

    if 1.0 / _EPS <= torch.finfo(dtype).max:
        _check(x, weight, grad_proj, grad_r)
        grad_x, _ = _check(x, weight, grad_proj_zeroed, grad_r)
    else:
        _, r = flag_train.megatron.fused_proj_rms_forward(x, weight, _EPS)
        op = flag_train.megatron.fused_proj_rms_backward
        grad_x, _ = op(x, weight, grad_proj, grad_r, r)
        grad_x_no_r, _ = op(x, weight, grad_proj, torch.zeros_like(grad_r), r)
        # The coefficient term is exactly absent from those rows...
        assert torch.equal(grad_x[::3], grad_x_no_r[::3])
        # ...and nothing on them is NaN.
        assert torch.isfinite(grad_x).all()
        grad_x, _ = op(x, weight, grad_proj_zeroed, grad_r, r)

    # With grad_proj zeroed on those rows, all that is left is the coefficient
    # term, which must be exactly zero -- not NaN.
    assert torch.all(grad_x[::3] == 0)


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_backward_flat_vectors(dtype):
    """``r`` and ``grad_r`` are accepted as ``[M]`` as well as ``[M, 1]``."""
    x, weight, grad_proj, grad_r = _operands(
        (7, 24, 512) if utils.QUICK_MODE else (257, 24, 1024), dtype
    )
    _, r = flag_train.megatron.fused_proj_rms_forward(x, weight, _EPS)
    op = flag_train.megatron.fused_proj_rms_backward

    grad_x_2d, grad_weight_2d = op(x, weight, grad_proj, grad_r, r)
    grad_x_1d, grad_weight_1d = op(
        x, weight, grad_proj, grad_r.reshape(-1), r.reshape(-1)
    )
    assert torch.equal(grad_x_2d, grad_x_1d)
    assert torch.equal(grad_weight_2d, grad_weight_1d)

    _check(x, weight, grad_proj, grad_r)


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_backward_non_contiguous(dtype):
    """A strided ``grad_proj`` is read by content, not by memory order."""
    M, N, K = 257, 24, 1024
    device = flag_train.device
    x, weight, _, grad_r = _operands((M, N, K), dtype)
    grad_proj = torch.randn((N, M), dtype=dtype, device=device).t()
    assert not grad_proj.is_contiguous()

    _, r = flag_train.megatron.fused_proj_rms_forward(x, weight, _EPS)
    op = flag_train.megatron.fused_proj_rms_backward

    grad_x, grad_weight = op(x, weight, grad_proj, grad_r, r)
    grad_x_c, grad_weight_c = op(x, weight, grad_proj.contiguous(), grad_r, r)
    assert torch.equal(grad_x, grad_x_c)
    assert torch.equal(grad_weight, grad_weight_c)

    _check(x, weight, grad_proj, grad_r)


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("dtype, weight_dtype", _MIXED_DTYPES)
def test_fused_proj_rms_backward_mixed_dtype(dtype, weight_dtype):
    """Mixed operands promote to one dtype for the whole computation.

    As in the forward suite, Megatron's native path has no single mixed-dtype
    answer (its gradients follow the unpromoted ``x`` for the norm term), so the
    fused contract -- one dtype throughout -- is checked as self-consistency:
    the mixed call must be bit-identical to the call with both operands already
    promoted. The promoted call returns the promoted dtype, so its gradients are
    cast back to each operand's dtype before comparing -- that cast is the only
    difference between the two calls.
    """
    x, weight, grad_proj, grad_r = _operands(
        (7, 24, 512) if utils.QUICK_MODE else (257, 24, 1024), dtype
    )
    weight = weight.to(weight_dtype)
    promoted = torch.promote_types(x.dtype, weight.dtype)
    grad_proj = grad_proj.to(promoted)

    _, r = flag_train.megatron.fused_proj_rms_forward(x, weight, _EPS)
    op = flag_train.megatron.fused_proj_rms_backward

    grad_x, grad_weight = op(x, weight, grad_proj, grad_r, r)
    grad_x_ref, grad_weight_ref = op(
        x.to(promoted),
        weight.to(promoted),
        grad_proj.to(promoted),
        grad_r.to(promoted),
        r.to(promoted),
    )

    assert grad_x.dtype == x.dtype and grad_weight.dtype == weight.dtype
    assert torch.equal(grad_x, grad_x_ref.to(grad_x.dtype))
    assert torch.equal(grad_weight, grad_weight_ref.to(grad_weight.dtype))


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("shape", [(0, 24, 4096), (16, 0, 4096), (16, 24, 0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_proj_rms_backward_empty(shape, dtype):
    """Empty M, N and K are legal, and autograd defines each of them.

    ``N == 0`` is the path where ``grad_x`` is the coefficient term alone and
    ``grad_weight`` is empty; ``K == 0`` returns two empties before any kernel
    runs.
    """
    _check(*_operands(shape, dtype))


@pytest.mark.fused_proj_rms_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_proj_rms_backward_contract(dtype):
    """Shapes, dtypes, and the arguments left alone."""
    x, weight, grad_proj, grad_r = _operands((129, 24, 512), dtype)
    _, r = flag_train.megatron.fused_proj_rms_forward(x, weight, _EPS)
    copies = [t.clone() for t in (x, weight, grad_proj, grad_r, r)]

    grad_x, grad_weight = flag_train.megatron.fused_proj_rms_backward(
        x, weight, grad_proj, grad_r, r
    )

    assert grad_x.shape == (129, 512) and grad_x.dtype == dtype
    assert grad_weight.shape == (24, 512) and grad_weight.dtype == dtype
    assert grad_x.is_contiguous() and grad_weight.is_contiguous()
    for tensor, copy in zip((x, weight, grad_proj, grad_r, r), copies):
        assert torch.equal(tensor, copy)


@pytest.mark.fused_proj_rms_backward
def test_fused_proj_rms_backward_rejects_bad_arguments():
    device = flag_train.device
    x = torch.randn((16, 32), dtype=torch.float32, device=device)
    weight = torch.randn((8, 32), dtype=torch.float32, device=device)
    grad_proj = torch.randn((16, 8), dtype=torch.float32, device=device)
    grad_r = torch.randn((16, 1), dtype=torch.float32, device=device)
    r = torch.ones((16, 1), dtype=torch.float32, device=device)
    op = flag_train.megatron.fused_proj_rms_backward

    with pytest.raises(ValueError):
        op(x.unsqueeze(0), weight, grad_proj, grad_r, r)
    with pytest.raises(ValueError):
        op(
            x,
            torch.randn((8, 16), dtype=torch.float32, device=device),
            grad_proj,
            grad_r,
            r,
        )
    with pytest.raises(ValueError):
        op(
            x,
            weight,
            torch.randn((16, 9), dtype=torch.float32, device=device),
            grad_r,
            r,
        )
    with pytest.raises(ValueError):
        op(x.to(torch.int32), weight, grad_proj, grad_r, r)
    with pytest.raises(ValueError):
        op(
            x,
            weight,
            grad_proj,
            torch.randn((15,), dtype=torch.float32, device=device),
            r,
        )
    with pytest.raises(ValueError):
        op(
            x,
            weight,
            grad_proj,
            grad_r,
            torch.ones((17,), dtype=torch.float32, device=device),
        )
    with pytest.raises(TypeError):
        op(x, weight, grad_proj, grad_r, r.tolist())
