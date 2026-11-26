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

"""Stage small Sinkhorn normalizations on Kunlunxin XPU."""

import torch
import triton
import triton.language as tl

from flag_train.megatron.fused_sinkhorn import _prepare_sinkhorn, _TorchSinkhorn
from flag_train.runtime import torch_device_fn


@triton.jit
def _round(value, dtype: tl.constexpr):
    if dtype == tl.bfloat16:
        bits = value.to(tl.uint32, bitcast=True)
        finite = (bits & 0x7F800000) != 0x7F800000
        truncated = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return tl.where(finite, truncated, bits).to(tl.float32, bitcast=True)
    return value.to(dtype).to(tl.float32)


@triton.jit
def _normalize_once(
    SOURCE,
    OUT,
    num_lines,
    eps,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    LINES: tl.constexpr,
    ROWS: tl.constexpr,
    FIRST: tl.constexpr,
):
    line = tl.program_id(0) * LINES + tl.arange(0, LINES)
    t = tl.arange(0, TILE)
    mat = line // WIDTH
    fixed = line % WIDTH
    base = mat[:, None] * WIDTH * WIDTH
    if ROWS:
        offsets = base + fixed[:, None] * WIDTH + t[None, :]
    else:
        offsets = base + t[None, :] * WIDTH + fixed[:, None]
    keep = (line[:, None] < num_lines) & (t[None, :] < WIDTH)
    dtype: tl.constexpr = SOURCE.dtype.element_ty
    matrix = tl.load(SOURCE + offsets, keep, 0).to(tl.float32)
    if FIRST:
        matrix = tl.where(keep, _round(matrix + eps, dtype), 0.0)
    total = _round(tl.sum(matrix, 1, keep_dims=True), dtype)
    denominator = _round(total + eps, dtype)
    tl.store(OUT + offsets, _round(matrix / denominator, dtype), keep)


class _StagedXPUSinkhorn(_TorchSinkhorn):
    @staticmethod
    def forward(ctx, logits, num_iterations, eps):
        ctx.save_for_backward(logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        out = torch.empty_like(logits)
        if not out.numel():
            return out
        width = logits.shape[-1]
        num_lines = logits.numel() // width
        tile = triton.next_power_of_2(width)
        block = 16
        probabilities = logits.softmax(dim=-1)
        tempor = torch.empty_like(logits) if num_iterations > 1 else out
        grid = (triton.cdiv(num_lines, block),)
        with torch_device_fn.device(logits.device):
            _normalize_once[grid](
                probabilities,
                out,
                num_lines,
                eps,
                width,
                tile,
                block,
                ROWS=False,
                FIRST=True,
                num_warps=1,
                enable_fp_fusion=False,
            )
            for _ in range(1, num_iterations):
                _normalize_once[grid](
                    out,
                    tempor,
                    num_lines,
                    eps,
                    width,
                    tile,
                    block,
                    ROWS=True,
                    FIRST=False,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
                _normalize_once[grid](
                    tempor,
                    out,
                    num_lines,
                    eps,
                    width,
                    tile,
                    block,
                    ROWS=False,
                    FIRST=False,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
        return out


def fused_sinkhorn(input_logits, num_iterations, eps=1e-6):
    """Project small power-of-two tiles with short normalization chains."""
    logits, num_iterations, eps = _prepare_sinkhorn(input_logits, num_iterations, eps)
    width = logits.shape[-1]
    # The staged kernel is useful for a single matrix and compact FP32 chains.
    # Batched low-precision casts and larger grids are faster in native Torch.
    matrices = logits.numel() // (width * width)
    compact_batch = (
        logits.dtype == torch.float32 and num_iterations > 1 and matrices * width <= 256
    )
    if (
        width > 16
        or width & (width - 1)
        or num_iterations > 3
        or (matrices > 1 and not compact_batch)
    ):
        return _TorchSinkhorn.apply(logits, num_iterations, eps)
    return _StagedXPUSinkhorn.apply(logits, num_iterations, eps)
