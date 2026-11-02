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
"""Correctness tests for the fused n-stream weighted aggregation (backward).

The oracle is ``torch.autograd`` through Megatron-LM's ``native_h_aggregate`` --
the derivative Megatron's own training step takes of this stage -- from the
installed megatron-core when it provides the function, and otherwise from
``native_h_aggregate_ref``, the same code transcribed below.

The two gradients are checked separately because they are different arithmetic:
``grad_x`` is a broadcast product, which the reference evaluates in the output
dtype and the kernel in fp32 -- both round the same exact product -- while
``grad_h`` is a reduction the reference rounds term by term, making it the less
accurate side. ``native_h_aggregate`` is wrapped in ``@torch.compile`` upstream;
the eager body underneath is what is differentiated here.
"""

import importlib.metadata
import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import accuracy_utils as utils

# ``(s, b, n, C)``: ``x`` is ``[s, b, n, C]``, ``h_pre`` is ``[s, b, n]``,
# ``grad_output`` is ``[s, b, C]``, and the outputs are ``grad_x`` of
# ``[s, b, n, C]`` and ``grad_h`` of ``[s, b, n]``. The first three are the
# shapes Megatron's own suite checks this stage at; the rest walk the stream
# count (1, the non-powers-of-two 3/5/7 that exercise the padded lane, and 8),
# the column tile (300, 4097, 7168) and the row count.
_SHAPES = [
    (2, 4, 4, 1024),
    (1, 1, 2, 256),
    (64, 8, 4, 4096),
    (1, 1, 4, 4096),
    (8, 16, 4, 4096),
    (32, 32, 4, 4096),
    (128, 32, 4, 4096),
    (256, 32, 2, 4096),
    (256, 32, 8, 4096),
    (256, 32, 4, 7168),
    (3, 5, 1, 300),
    (1, 3, 5, 64),
    (7, 11, 3, 4097),
]

# ``--quick`` (the second CI pass) keeps one masked row block and one full tile.
if utils.QUICK_MODE:
    _SHAPES = [(2, 4, 4, 1024), (1, 1, 2, 256)]

# Every dimension below its tile: one row, one stream, one column.
_TINY_SHAPES = [
    (1, 1, 1, 1),
    (1, 1, 3, 1),
    (1, 1, 1, 7),
    (2, 3, 5, 17),
    (1, 1, 8, 33),
]

# The sweep the performance suite reports latency for.
# !! KEEP IN SYNC with benchmark/megatron/test_fused_h_aggregate_backward.py !!
_BENCHMARK_SHAPES = [
    (1, 1, 2, 256),
    (1, 1, 4, 4096),
    (8, 16, 4, 4096),
    (64, 8, 4, 4096),
    (32, 32, 4, 4096),
    (128, 32, 4, 4096),
    (256, 32, 4, 4096),
    (256, 32, 2, 4096),
    (256, 32, 8, 4096),
    (256, 32, 4, 7168),
]

# FP32 exercises the exact-arithmetic path; FP16 is always available, BF16 only
# where the runtime reports it. The benchmark sweep is measured in the narrow
# dtypes a training step runs.
_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)
_BENCHMARK_DTYPES = [dtype for dtype in _DTYPES if dtype is not torch.float32]

# The mHC mapping parameters are held in fp32 while the activations are narrow.
_MIXED_DTYPES = [(torch.float16, torch.float32), (torch.float32, torch.float16)]
if utils.bf16_is_supported:
    _MIXED_DTYPES += [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ]

# ---------------------------------------------------------------------------
# Megatron oracle
#
# The oracle is the installed ``native_h_aggregate`` when it can be imported,
# and ``native_h_aggregate_ref`` otherwise: an older megatron-core has no
# ``hyper_connection`` module at all, so the two paths are exactly import
# success and import failure.
# ---------------------------------------------------------------------------


def native_h_aggregate_ref(x, h_pre):
    """Megatron-LM's ``native_h_aggregate`` (``hyper_connection.py``, 0.20)."""
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


def _mcore_version():
    try:
        return importlib.metadata.version("megatron-core")
    except Exception:
        return "not installed"


def _load_native_h_aggregate():
    """Megatron-LM's ``native_h_aggregate``, or the transcription when absent."""
    version = _mcore_version()
    try:
        from megatron.core.transformer.hyper_connection import native_h_aggregate
    except Exception:
        warnings.warn(
            f"megatron-core {version} has no native_h_aggregate; checking "
            f"against native_h_aggregate_ref, Megatron-LM's implementation "
            f"transcribed below",
            stacklevel=2,
        )
        return native_h_aggregate_ref
    # Megatron wraps it in ``@torch.compile``; differentiate the code beneath.
    return getattr(native_h_aggregate, "_torchdynamo_orig_callable", native_h_aggregate)


_native_h_aggregate = _load_native_h_aggregate()

# Tolerances, as a fraction of the reference's RMS.
#
# ``grad_x``: the measured difference is zero at every dtype and shape, because
# both sides round the same exact product; the entries are two ULPs of the
# output dtype, which leaves room for a backend that reorders the multiply.
#
# ``grad_h``: a reduction the reference rounds term by term; the measured worst
# case over the sweep is 1.1e-6 of the RMS in fp32, 3.0e-3 in fp16 and 3.1e-2 in
# bf16, and each entry is that band with 3-4x of headroom.
_TOL_GRAD_X = {
    torch.float32: 1e-5,
    torch.float16: 2e-3,
    torch.bfloat16: 1e-2,
}
_TOL_GRAD_H = {
    torch.float32: 1e-4,
    torch.float16: 2e-2,
    torch.bfloat16: 1e-1,
}


def _assert_close(res, ref, dtype, tol):
    """Compare at ``tol`` of the reference's RMS, elementwise.

    A per-element ``rtol`` is the wrong shape of tolerance for a reduction:
    scaling by the reference's RMS holds every element to the same budget, which
    is the elementwise form of the relative Frobenius error.
    """
    assert res.dtype == dtype, (res.dtype, dtype)
    assert res.shape == ref.shape, (res.shape, ref.shape)

    res = utils.to_reference(res).float()
    ref = utils.to_reference(ref).float()
    if res.numel() == 0:
        return
    finite = torch.isfinite(ref)
    scale = ref[finite].pow(2).mean().sqrt().item() if finite.any() else 0.0
    torch.testing.assert_close(
        res, ref, atol=tol * scale, rtol=tol, equal_nan=True, check_dtype=False
    )


def _oracle_backward(grad_output, x, h_pre):
    """``torch.autograd`` through Megatron's ``native_h_aggregate``."""
    x_ref = utils.to_reference(x).detach().clone().requires_grad_(True)
    h_ref = utils.to_reference(h_pre).detach().clone().requires_grad_(True)
    out = _native_h_aggregate(x_ref, h_ref)
    # ``backward`` wants the incoming gradient in the output's dtype; the cast is
    # exact (narrow -> wide) when the reference promoted mixed inputs.
    out.backward(utils.to_reference(grad_output).to(out.dtype))
    return x_ref.grad, h_ref.grad


def _operands(shape, dtype, h_dtype=None):
    """``(grad_output, x, h_pre)`` for one case."""
    device = flag_train.device
    s, b, _, C = shape
    x = torch.randn(shape, dtype=dtype, device=device)
    h_pre = torch.randn(shape[:3], dtype=h_dtype or dtype, device=device)
    grad_output = torch.randn((s, b, C), dtype=dtype, device=device)
    return grad_output, x, h_pre


def _check(grad_output, x, h_pre, dtype, h_dtype=None):
    """Run the operator against the oracle and return both gradients."""
    grad_x, grad_h = flag_train.megatron.fused_h_aggregate_backward(
        grad_output, x, h_pre
    )
    grad_x_ref, grad_h_ref = _oracle_backward(grad_output, x, h_pre)
    _assert_close(grad_x, grad_x_ref, dtype, _TOL_GRAD_X[dtype])
    _assert_close(grad_h, grad_h_ref, h_dtype or dtype, _TOL_GRAD_H[h_dtype or dtype])
    return grad_x, grad_h


@pytest.mark.fused_h_aggregate_backward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_backward(shape, dtype):
    """Both gradients match Megatron's autograd, in every dtype."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_h_aggregate_backward
@pytest.mark.parametrize("shape", _TINY_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_backward_tiny_shapes(shape, dtype):
    """Shapes under one tile: masked rows, columns, and padded stream lanes."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_h_aggregate_backward
@pytest.mark.parametrize("shape", _BENCHMARK_SHAPES)
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
def test_fused_h_aggregate_backward_benchmark_shapes(shape, dtype):
    """Every benchmarked shape has a correctness check behind it."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_h_aggregate_backward
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_backward_stream_counts(n, dtype):
    """The stream axis is padded to a power of two; the lanes above ``n`` stay out."""
    shape = (4, 8, n, 512) if utils.QUICK_MODE else (32, 8, n, 4096)
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_h_aggregate_backward
@pytest.mark.parametrize("dtype, h_dtype", _MIXED_DTYPES)
def test_fused_h_aggregate_backward_mixed_dtype(dtype, h_dtype):
    """Each gradient takes its own input's dtype, and the casting is the kernel's.

    The oracle covers the dtypes; what is checked in addition is that the mixed
    call matches the call with both operands already promoted -- bit for bit for
    ``grad_x``, and at tolerance for the ``grad_h`` reduction, which the two
    calls sum in different orders.
    """
    shape = (2, 4, 4, 1024) if utils.QUICK_MODE else (32, 32, 4, 4096)
    grad_output, x, h_pre = _operands(shape, dtype, h_dtype)

    grad_x, grad_h = _check(grad_output, x, h_pre, dtype, h_dtype)

    promoted = torch.promote_types(dtype, h_dtype)
    grad_x_p, grad_h_p = flag_train.megatron.fused_h_aggregate_backward(
        grad_output, x.to(promoted), h_pre.to(promoted)
    )
    assert grad_x.dtype == dtype and grad_h.dtype == h_dtype
    assert torch.equal(grad_x, grad_x_p.to(dtype))
    _assert_close(grad_h, grad_h_p, h_dtype, _TOL_GRAD_H[h_dtype])


@pytest.mark.fused_h_aggregate_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_backward_non_contiguous(dtype):
    """A strided argument is read by content, not by memory order."""
    s, b, n, C = 32, 16, 4, 1024
    device = flag_train.device
    x = torch.randn((b, s, n, C), dtype=dtype, device=device).transpose(0, 1)
    h_pre = torch.randn((b, s, n), dtype=dtype, device=device).transpose(0, 1)
    grad_output = torch.randn((b, s, C), dtype=dtype, device=device).transpose(0, 1)
    assert not x.is_contiguous() and not h_pre.is_contiguous()

    grad_x, grad_h = _check(grad_output, x, h_pre, dtype)

    # Same values in the same order as the contiguous copies: bit-identical.
    grad_x_c, grad_h_c = flag_train.megatron.fused_h_aggregate_backward(
        grad_output.contiguous(), x.contiguous(), h_pre.contiguous()
    )
    assert torch.equal(grad_x, grad_x_c)
    assert torch.equal(grad_h, grad_h_c)


@pytest.mark.fused_h_aggregate_backward
@pytest.mark.parametrize(
    "shape", [(0, 4, 4, 1024), (2, 0, 4, 1024), (2, 3, 0, 64), (2, 3, 4, 0)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_h_aggregate_backward_empty(shape, dtype):
    """Empty axes are legal; ``C == 0`` reduces to zeros, not uninitialised memory."""
    grad_output, x, h_pre = _operands(shape, dtype)
    grad_x, grad_h = _check(grad_output, x, h_pre, dtype)

    assert grad_x.shape == x.shape and grad_h.shape == h_pre.shape
    if shape[3] == 0:
        assert torch.count_nonzero(grad_h) == 0


@pytest.mark.fused_h_aggregate_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_backward_contract(dtype):
    """Shapes, dtypes, and the arguments left alone."""
    grad_output, x, h_pre = _operands((129, 7, 4, 300), dtype)
    copies = [tensor.clone() for tensor in (grad_output, x, h_pre)]

    grad_x, grad_h = flag_train.megatron.fused_h_aggregate_backward(
        grad_output, x, h_pre
    )

    assert grad_x.shape == (129, 7, 4, 300) and grad_x.dtype == dtype
    assert grad_h.shape == (129, 7, 4) and grad_h.dtype == dtype
    assert grad_x.is_contiguous() and grad_h.is_contiguous()
    for tensor, copy in zip((grad_output, x, h_pre), copies):
        assert torch.equal(tensor, copy)


@pytest.mark.fused_h_aggregate_backward
def test_fused_h_aggregate_backward_rejects_bad_arguments():
    device = flag_train.device
    grad_output = torch.randn((2, 4, 64), dtype=torch.float32, device=device)
    x = torch.randn((2, 4, 4, 64), dtype=torch.float32, device=device)
    h_pre = torch.randn((2, 4, 4), dtype=torch.float32, device=device)
    op = flag_train.megatron.fused_h_aggregate_backward

    with pytest.raises(ValueError):
        op(x, x, h_pre)  # 4-D grad_output
    with pytest.raises(ValueError):
        op(grad_output, grad_output, h_pre)  # 3-D x
    with pytest.raises(ValueError):
        op(grad_output, x, x)  # 4-D h_pre
    with pytest.raises(ValueError):
        op(grad_output, x, torch.randn((2, 4, 5), dtype=torch.float32, device=device))
    with pytest.raises(ValueError):
        op(torch.randn((2, 4, 63), dtype=torch.float32, device=device), x, h_pre)
    with pytest.raises(ValueError):
        op(x.to(torch.int32), x.to(torch.int32), h_pre)
    with pytest.raises(TypeError):
        op(grad_output.tolist(), x, h_pre)
    with pytest.raises(ValueError):
        op(grad_output.cpu(), x.cpu(), h_pre.cpu())
