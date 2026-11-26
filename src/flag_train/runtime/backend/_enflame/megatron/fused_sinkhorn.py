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

"""Fuse Sinkhorn normalization and its VJP on GCU300."""

import torch
import triton
import triton.language as tl

from flag_train.megatron.fused_sinkhorn import _prepare_sinkhorn, _TorchSinkhorn
from flag_train.megatron.fused_sinkhorn import fused_sinkhorn as _generic
from flag_train.runtime import torch_device_fn

from .. import arch_version


@triton.jit
def _round(value, dtype: tl.constexpr):
    return value.to(dtype).to(tl.float32)


@triton.jit
def _normalize(matrix, eps, axis: tl.constexpr, dtype: tl.constexpr):
    total = _round(tl.sum(matrix, axis, keep_dims=True), dtype)
    denominator = _round(total + eps, dtype)
    return _round(tl.div_rn(matrix, denominator), dtype)


@triton.jit
def _normalize_matrices(
    SOFTMAX,
    OUT,
    count,
    iterations,
    eps,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batches = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = tl.arange(0, TILE)
    cols = tl.arange(0, TILE)
    offsets = (
        batches[:, None, None] * WIDTH * WIDTH
        + rows[None, :, None] * WIDTH
        + cols[None, None, :]
    )
    keep = (
        (batches[:, None, None] < count)
        & (rows[None, :, None] < WIDTH)
        & (cols[None, None, :] < WIDTH)
    )
    dtype: tl.constexpr = SOFTMAX.dtype.element_ty
    matrix = tl.load(SOFTMAX + offsets, keep, 0).to(tl.float32)
    matrix = tl.where(keep, _round(matrix + eps, dtype), 0.0)
    matrix = tl.where(keep, _normalize(matrix, eps, 1, dtype), 0.0)
    for _ in range(iterations - 1):
        matrix = tl.where(keep, _normalize(matrix, eps, 2, dtype), 0.0)
        matrix = tl.where(keep, _normalize(matrix, eps, 1, dtype), 0.0)
    tl.store(OUT + offsets, matrix, keep)


@triton.jit
def _normalization_state(matrix, eps, axis: tl.constexpr, dtype: tl.constexpr):
    total = _round(tl.sum(matrix, axis, keep_dims=True), dtype)
    denom = _round(total + eps, dtype)
    return _round(tl.div_rn(matrix, denom), dtype), denom


@triton.jit
def _normalization_vjp(
    gradient, output, denom, keep, axis: tl.constexpr, dtype: tl.constexpr
):
    direct = _round(tl.div_rn(gradient, denom), dtype)
    ratio = _round(tl.div_rn(output, denom), dtype)
    local_grad = _round(-gradient * ratio, dtype)
    reduced = _round(
        tl.sum(tl.where(keep, local_grad, 0.0), axis, keep_dims=True), dtype
    )
    return tl.where(keep, _round(direct + reduced, dtype), 0.0)


@triton.jit
def _backprop_matrices(
    SOFTMAX,
    DOUT,
    DS,
    OUTPUTS,
    DENOMS,
    count,
    steps,
    eps,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batches = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = tl.arange(0, TILE)
    cols = tl.arange(0, TILE)
    offsets = (
        batches[:, None, None] * WIDTH * WIDTH
        + rows[None, :, None] * WIDTH
        + cols[None, None, :]
    )
    keep = (
        (batches[:, None, None] < count)
        & (rows[None, :, None] < WIDTH)
        & (cols[None, None, :] < WIDTH)
    )
    dtype: tl.constexpr = SOFTMAX.dtype.element_ty
    matrix = tl.load(SOFTMAX + offsets, keep, 0).to(tl.float32)
    matrix = tl.where(keep, _round(matrix + eps, dtype), 0.0)
    stride = count * WIDTH * WIDTH

    # Recreate the rounded states on demand instead of saving them in forward.
    for step in range(steps):
        if step % 2 == 0:
            normalized, col_denom = _normalization_state(matrix, eps, 1, dtype)
            denominator = tl.broadcast_to(col_denom, (BLOCK, TILE, TILE))
        else:
            normalized, row_denom = _normalization_state(matrix, eps, 2, dtype)
            denominator = tl.broadcast_to(row_denom, (BLOCK, TILE, TILE))
        matrix = tl.where(keep, normalized, 0.0)
        tl.store(OUTPUTS + step * stride + offsets, matrix, keep)
        tl.store(DENOMS + step * stride + offsets, denominator, keep)

    gradient = tl.load(DOUT + offsets, keep, 0).to(tl.float32)
    for j in range(steps):
        step = steps - j - 1
        output = tl.load(OUTPUTS + step * stride + offsets, keep, 0).to(tl.float32)
        denominator = tl.load(DENOMS + step * stride + offsets, keep, 1).to(tl.float32)
        if step % 2 == 0:
            gradient = _normalization_vjp(gradient, output, denominator, keep, 1, dtype)
        else:
            gradient = _normalization_vjp(gradient, output, denominator, keep, 2, dtype)
    tl.store(DS + offsets, gradient, keep)


class _SmallSinkhorn(_TorchSinkhorn):
    @staticmethod
    def forward(ctx, logits, num_iterations, eps):
        ctx.save_for_backward(logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        out = torch.empty_like(logits)
        if not out.numel():
            return out
        width = logits.shape[-1]
        count = logits.numel() // (width * width)
        batch_tile = 8
        probabilities = logits.softmax(dim=-1)
        with torch_device_fn.device(logits.device):
            _normalize_matrices[(triton.cdiv(count, batch_tile),)](
                probabilities,
                out,
                count,
                num_iterations,
                eps,
                width,
                triton.next_power_of_2(width),
                batch_tile,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        if not logits.numel():
            return torch.empty_like(logits), None, None
        steps = 2 * ctx.num_iterations - 1
        if 2 * steps * logits.numel() >= 2**26:
            return _TorchSinkhorn.backward(ctx, grad_output)
        probabilities = logits.softmax(dim=-1)
        dsoftmax = torch.empty_like(logits)
        width = logits.shape[-1]
        count = logits.numel() // (width * width)
        batch_tile = 8
        stage_outputs = torch.empty(
            (steps, *logits.shape), dtype=logits.dtype, device=logits.device
        )
        denominators = torch.empty_like(stage_outputs)
        with torch_device_fn.device(logits.device):
            _backprop_matrices[(triton.cdiv(count, batch_tile),)](
                probabilities,
                grad_output.contiguous(),
                dsoftmax,
                stage_outputs,
                denominators,
                count,
                steps,
                ctx.eps,
                width,
                triton.next_power_of_2(width),
                batch_tile,
                num_warps=4,
                enable_fp_fusion=False,
            )
        dx = torch.ops.aten._softmax_backward_data(
            dsoftmax, probabilities, -1, logits.dtype
        )
        return dx, None, None


def fused_sinkhorn(input_logits, num_iterations, eps=1e-6):
    """Project small batched square matrices on GCU300."""
    if arch_version != 300:
        return _generic(input_logits, num_iterations, eps)
    logits, num_iterations, eps = _prepare_sinkhorn(input_logits, num_iterations, eps)
    if logits.shape[-1] > 16 or num_iterations > 32 or logits.numel() >= 2**31:
        return _TorchSinkhorn.apply(logits, num_iterations, eps)
    return _SmallSinkhorn.apply(logits, num_iterations, eps)
