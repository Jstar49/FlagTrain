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
import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_NUM_WARPS = 4

_BLOCK_SIZE = 1024

_QUICK_GELU_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Megatron-LM ``quick_gelu`` multiplies the sigmoid argument by this.
_ALPHA = 1.702

# fp32 sigmoid is already 0 or 1 outside this range, and exp stays finite.
_SIGMOID_CLAMP = 80.0


def _check_input(tensor, name, dtypes):
    if tensor.dtype not in dtypes:
        wanted = " / ".join(str(dtype) for dtype in dtypes)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


def _flatten(tensor):
    """Logical elements in a contiguous 1-D buffer.

    A non-contiguous argument is copied, the same way ``packbits`` moves an axis
    that is not last before the kernel sees it. The caller views the result
    back onto the original shape.
    """
    return tensor.contiguous().view(-1)


@libentry()
@triton.jit
def _quick_gelu_forward_kernel(x_ptr, y_ptr, n, alpha, clamp, BLOCK_SIZE: tl.constexpr):
    """``y = x * sigmoid(alpha * x)``, Megatron-LM ``quick_gelu``.

    The product is computed in fp32 and stored in the pointer dtype. ``alpha * x``
    is clamped to ``[-clamp, clamp]`` first: beyond that, fp32 sigmoid is 0 or 1,
    and ``exp`` of the unclamped value overflows.
    """
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.minimum(tl.maximum(alpha * x, -clamp), clamp)
    s = 1.0 / (1.0 + tl.exp(-z))
    tl.store(y_ptr + offsets, (x * s).to(y_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _quick_gelu_backward_kernel(
    x_ptr, dy_ptr, dx_ptr, n, alpha, clamp, BLOCK_SIZE: tl.constexpr
):
    """Gradient of Megatron-LM ``quick_gelu``.

    With ``z = alpha * x`` and ``s = sigmoid(z)``,
    ``dx = dy * s * (1 + z * (1 - s))``. The same clamp as the forward kernel
    is applied before ``exp``. At the clamp boundary ``s`` is already 0 or 1,
    so the extra ``z * (1 - s)`` term is zero either way.
    """
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.minimum(tl.maximum(alpha * x, -clamp), clamp)
    s = 1.0 / (1.0 + tl.exp(-z))
    dx = dy * s * (1.0 + z * (1.0 - s))
    tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _launch(kernel, args, n, device):
    grid = (triton.cdiv(n, _BLOCK_SIZE),)
    with torch_device_fn.device(device):
        kernel[grid](
            *args,
            n,
            _ALPHA,
            _SIGMOID_CLAMP,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )


def _quick_gelu_forward(x):
    _check_input(x, "quick_gelu", _QUICK_GELU_DTYPES)
    if x.numel() == 0:
        # No launch when there is nothing to activate.
        return torch.empty_like(x)

    flat = _flatten(x)
    out = torch.empty_like(flat)
    _launch(_quick_gelu_forward_kernel, (flat, out), flat.numel(), x.device)
    return out.view(x.shape)


def _quick_gelu_backward(x, grad_output):
    _check_input(x, "quick_gelu", _QUICK_GELU_DTYPES)
    _check_input(grad_output, "quick_gelu grad", _QUICK_GELU_DTYPES)
    if grad_output.shape != x.shape or grad_output.dtype != x.dtype:
        raise ValueError(
            "quick_gelu grad needs the same shape and dtype as the input, got "
            f"{tuple(grad_output.shape)} {grad_output.dtype} for "
            f"{tuple(x.shape)} {x.dtype}"
        )
    if x.numel() == 0:
        return torch.empty_like(x)

    flat_x = _flatten(x)
    flat_dy = _flatten(grad_output)
    dx = torch.empty_like(flat_x)
    _launch(
        _quick_gelu_backward_kernel,
        (flat_x, flat_dy, dx),
        flat_x.numel(),
        x.device,
    )
    return dx.view(x.shape)


class QuickGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _quick_gelu_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("TRAIN QUICK_GELU_BACKWARD")
        (x,) = ctx.saved_tensors
        return _quick_gelu_backward(x, grad_output)


def quick_gelu(x):
    """``x * sigmoid(1.702 * x)``, Megatron-LM Quick GELU.

    Args:
        x (Tensor): fp16, bf16 or fp32 tensor on the active device. A
            non-contiguous input is copied. An empty input is returned without
            a launch.

    Returns:
        Tensor: same shape and dtype as ``x``. Backward goes through autograd.
    """
    logger.debug("TRAIN QUICK_GELU")
    return QuickGelu.apply(x)
