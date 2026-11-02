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
"""Correctness tests for the fused n-stream weighted aggregation (forward).

The oracle is Megatron-LM's ``native_h_aggregate`` -- the function this operator
ports -- from the installed megatron-core when it provides the function, and otherwise
from ``native_h_aggregate_ref``, the same code transcribed below.
Megatron wraps it in ``@torch.compile``; the eager body is what is compared
against. The kernel accumulates in fp32 and rounds once, while the reference
rounds each of the ``n`` products into the input dtype first, so at the narrow
dtypes the reference is the less accurate side and the tolerances below cover
its rounding.
"""

import importlib.metadata
import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import accuracy_utils as utils

# ``(s, b, n, C)``: ``x`` is ``[s, b, n, C]``, ``h_pre`` is ``[s, b, n]`` and the
# output is ``[s, b, C]``. The first three are the shapes Megatron's own suite
# checks this stage at; the rest walk the stream count (1, the non-powers-of-two
# 3/5/7, and 8), the column tile (300, 4097, 7168) and the row count.
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
# !! KEEP IN SYNC with benchmark/megatron/test_fused_h_aggregate_forward.py !!
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
    # Megatron wraps it in ``@torch.compile``; compare against the code beneath.
    return getattr(native_h_aggregate, "_torchdynamo_orig_callable", native_h_aggregate)


_native_h_aggregate = _load_native_h_aggregate()

# Tolerances, as a fraction of the reference's RMS. Measured over the sweep, the
# worst element is 2.0e-6 of the RMS in fp32 (both sides add the same products),
# 7.8e-3 in fp16 and 8.9e-2 in bf16 -- the reference's own rounding. Each entry
# is that band with 2-4x of headroom.
_TOL = {
    torch.float32: 1e-4,
    torch.float16: 2e-2,
    torch.bfloat16: 2e-1,
}


def _assert_close(res, ref, dtype, tol):
    """Compare at ``tol`` of the reference's RMS, elementwise.

    A per-element ``rtol`` is the wrong shape of tolerance for a sum of ``n``
    products: scaling by the reference's RMS holds every element to the same
    budget, which is the elementwise form of the relative Frobenius error.
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


def _operands(shape, dtype, h_dtype=None):
    """``(x, h_pre)`` for one case."""
    device = flag_train.device
    x = torch.randn(shape, dtype=dtype, device=device)
    h_pre = torch.randn(shape[:3], dtype=h_dtype or dtype, device=device)
    return x, h_pre


def _check(x, h_pre, dtype):
    """Run the operator against the oracle and return its output."""
    out = flag_train.megatron.fused_h_aggregate_forward(x, h_pre)
    ref = _native_h_aggregate(utils.to_reference(x), utils.to_reference(h_pre))
    _assert_close(out, ref, dtype, _TOL[dtype])
    return out


@pytest.mark.fused_h_aggregate_forward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_forward(shape, dtype):
    """The aggregation matches Megatron, in every dtype."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_h_aggregate_forward
@pytest.mark.parametrize("shape", _TINY_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_forward_tiny_shapes(shape, dtype):
    """Shapes under one tile: masked rows, columns, and streams."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_h_aggregate_forward
@pytest.mark.parametrize("shape", _BENCHMARK_SHAPES)
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
def test_fused_h_aggregate_forward_benchmark_shapes(shape, dtype):
    """Every benchmarked shape has a correctness check behind it."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_h_aggregate_forward
@pytest.mark.parametrize("dtype, h_dtype", _MIXED_DTYPES)
def test_fused_h_aggregate_forward_mixed_dtype(dtype, h_dtype):
    """Mixed inputs promote, and the promotion is the kernel's.

    The oracle covers the arithmetic; what is checked in addition is that the
    mixed call is bit-identical to the call with both operands already promoted,
    which a host-side copy would not be.
    """
    shape = (2, 4, 4, 1024) if utils.QUICK_MODE else (32, 32, 4, 4096)
    promoted = torch.promote_types(dtype, h_dtype)
    x, h_pre = _operands(shape, dtype, h_dtype)

    out = _check(x, h_pre, promoted)

    out_ref = flag_train.megatron.fused_h_aggregate_forward(
        x.to(promoted), h_pre.to(promoted)
    )
    assert out.dtype == promoted
    assert torch.equal(out, out_ref)


@pytest.mark.fused_h_aggregate_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_forward_non_contiguous(dtype):
    """A strided argument is read by content, not by memory order."""
    s, b, n, C = 32, 16, 4, 1024
    device = flag_train.device
    x = torch.randn((b, s, n, C), dtype=dtype, device=device).transpose(0, 1)
    h_pre = torch.randn((b, s, n), dtype=dtype, device=device).transpose(0, 1)
    assert not x.is_contiguous() and not h_pre.is_contiguous()

    out = _check(x, h_pre, dtype)

    # Same values in the same order as the contiguous copies: bit-identical.
    out_c = flag_train.megatron.fused_h_aggregate_forward(
        x.contiguous(), h_pre.contiguous()
    )
    assert torch.equal(out, out_c)


@pytest.mark.fused_h_aggregate_forward
@pytest.mark.parametrize(
    "shape", [(0, 4, 4, 1024), (2, 0, 4, 1024), (2, 3, 0, 64), (2, 3, 4, 0)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_h_aggregate_forward_empty(shape, dtype):
    """Empty axes are legal; ``n == 0`` is the empty sum, i.e. zeros."""
    x, h_pre = _operands(shape, dtype)
    out = _check(x, h_pre, dtype)

    if shape[2] == 0:
        assert torch.count_nonzero(out) == 0


@pytest.mark.fused_h_aggregate_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_h_aggregate_forward_contract(dtype):
    """Shapes, dtypes, and the arguments left alone."""
    x, h_pre = _operands((129, 7, 4, 300), dtype)
    x_copy, h_pre_copy = x.clone(), h_pre.clone()

    out = flag_train.megatron.fused_h_aggregate_forward(x, h_pre)

    assert out.shape == (129, 7, 300) and out.dtype == dtype
    assert out.is_contiguous()
    assert torch.equal(x, x_copy) and torch.equal(h_pre, h_pre_copy)


@pytest.mark.fused_h_aggregate_forward
def test_fused_h_aggregate_forward_rejects_bad_arguments():
    device = flag_train.device
    x = torch.randn((2, 4, 4, 64), dtype=torch.float32, device=device)
    h_pre = torch.randn((2, 4, 4), dtype=torch.float32, device=device)
    op = flag_train.megatron.fused_h_aggregate_forward

    with pytest.raises(ValueError):
        op(x[0], h_pre[0])  # 3-D activations
    with pytest.raises(ValueError):
        op(x, h_pre.unsqueeze(0))  # 4-D weights
    with pytest.raises(ValueError):
        op(x, torch.randn((2, 4, 5), dtype=torch.float32, device=device))
    with pytest.raises(ValueError):
        op(x, h_pre[:, :3])
    with pytest.raises(ValueError):
        op(x.to(torch.int32), h_pre)
    with pytest.raises(TypeError):
        op(x, h_pre.tolist())
    with pytest.raises(ValueError):
        op(x.cpu(), h_pre.cpu())
