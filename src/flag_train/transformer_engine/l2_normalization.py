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

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

import flag_train
from flag_train.runtime import torch_device_fn


@triton.jit
def _l2_forward(X, Y, RN, eps, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    x_row = X + row * WIDTH
    y_row = Y + row * WIDTH

    if WIDTH <= BLOCK:
        x = tl.load(x_row + offsets, offsets < WIDTH, other=0).to(tl.float32)
        r = tl.rsqrt(tl.sum(x * x, axis=0) + eps)
        tl.store(y_row + offsets, x * r, offsets < WIDTH)
    else:
        squared = tl.zeros([BLOCK], tl.float32)
        for start in range(0, WIDTH, BLOCK):
            cols = start + offsets
            x = tl.load(x_row + cols, cols < WIDTH, other=0).to(tl.float32)
            squared += x * x
        r = tl.rsqrt(tl.sum(squared, axis=0) + eps)
        for start in range(0, WIDTH, BLOCK):
            cols = start + offsets
            x = tl.load(x_row + cols, cols < WIDTH, other=0).to(tl.float32)
            tl.store(y_row + cols, x * r, cols < WIDTH)
    if RN is not None:
        tl.store(RN + row, r)


def _forward(x, eps, save_norm):
    x = x.contiguous()
    y = torch.empty_like(x)
    width = x.shape[-1]
    rows = x.numel() // width if width else 0
    inverse_norm = (
        torch.empty(rows, device=x.device, dtype=torch.float32) if save_norm else None
    )
    if rows:
        block = min(triton.next_power_of_2(width), 2048)
        with torch_device_fn.device(x.device):
            _l2_forward[(rows,)](x, y, inverse_norm, eps, width, block)
    return y, inverse_norm


class _L2Normalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        y, inverse_norm = _forward(x, eps, save_norm=True)
        ctx.save_for_backward(x, inverse_norm)
        ctx.eps = eps
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, inverse_norm = ctx.saved_tensors
        grad_x = flag_train.transformer_engine.l2_normalization_backward(
            grad_output, x, inverse_norm, ctx.eps
        )
        return grad_x, None


def l2_normalization(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize the last dimension of x and support first-order gradients.

    The sum of squares, inverse norm, and normalization are computed in FP32.
    The output has the input dtype and shape.
    """
    if x.ndim == 0 or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("x must be a non-scalar FP16, BF16, or FP32 tensor")
    if torch.is_grad_enabled() and x.requires_grad:
        return _L2Normalization.apply(x, eps)
    y, _ = _forward(x, eps, save_norm=False)
    return y
