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
import math

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_NUM_WARPS = 4

# 16-bit inputs move half the bytes per element, so H20 needs a wider block.
_BLOCK_SIZE = 1024
_BLOCK_SIZE_16BIT = 2048

_WEIGHTED_SQUARED_RELU_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_input(tensor, name):
    if tensor.dtype not in _WEIGHTED_SQUARED_RELU_DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _WEIGHTED_SQUARED_RELU_DTYPES)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


@triton.jit
def _clamped_square(value, scale, neg_two_over_scale):
    """``(scale * tanh(relu(value) / scale)) ** 2`` for a non-negative scale.

    Folding ``-2 / scale`` into the exponent replaces the per-element division
    with a multiply, and ``exp(-2 * relu / scale)`` underflows to 0, so large
    magnitudes yield exactly ``scale ** 2``. The caller passes ``|clamp_scale|``,
    which leaves the squared result unchanged for a negative scale.
    """
    e = tl.exp(tl.maximum(value, 0.0) * neg_two_over_scale)
    c = scale * ((1.0 - e) / (1.0 + e))
    return c * c


@libentry()
@triton.jit
def _weighted_forward_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    hidden,
    abs_scale,
    neg_two_over_scale,
    HAS_CLAMP: tl.constexpr,
    EVEN_HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """``y = squared_relu(x) * weight[row]``, optionally tanh soft-clamped.

    One program handles a ``BLOCK_SIZE`` slice of one row, so the per-token
    weight is a scalar load. The activation is computed in fp32, matching the
    reference; the bounds check is dropped when ``hidden`` is block-aligned.
    """
    row = tle.program_id(0)
    base = row * hidden
    columns = tle.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN_HIDDEN:
        x = tl.load(x_ptr + base + columns).to(tl.float32)
    else:
        mask = columns < hidden
        x = tl.load(x_ptr + base + columns, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + row).to(tl.float32)

    if HAS_CLAMP:
        act = _clamped_square(x, abs_scale, neg_two_over_scale)
    else:
        r = tl.maximum(x, 0.0)
        act = r * r

    if EVEN_HIDDEN:
        tl.store(y_ptr + base + columns, (act * weight).to(y_ptr.dtype.element_ty))
    else:
        tl.store(
            y_ptr + base + columns,
            (act * weight).to(y_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def _clamped_forward_kernel(
    x_ptr,
    y_ptr,
    n,
    abs_scale,
    neg_two_over_scale,
    EVEN_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """``y = (clamp_scale * tanh(relu(x) / clamp_scale)) ** 2`` without weights."""
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN_N:
        x = tl.load(x_ptr + offsets).to(tl.float32)
    else:
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    act = _clamped_square(x, abs_scale, neg_two_over_scale)

    if EVEN_N:
        tl.store(y_ptr + offsets, act.to(y_ptr.dtype.element_ty))
    else:
        tl.store(y_ptr + offsets, act.to(y_ptr.dtype.element_ty), mask=mask)


def _launch_config(dtype):
    """Block/warp pair measured fastest for ``dtype`` on this device."""
    if runtime_device.vendor_name == "nvidia" and dtype != torch.float32:
        return _BLOCK_SIZE_16BIT, _NUM_WARPS
    return _BLOCK_SIZE, _NUM_WARPS


def _clamp_operands(clamp_scale):
    """``(|scale|, -2 / |scale|)`` for the folded tanh soft clamp."""
    abs_scale = abs(clamp_scale)
    # scale == 0 behaves like the original division: exp(-inf) is 0, so
    # tanh -> 1 and the result is 0 for relu(x) > 0 (nan at relu(x) == 0).
    neg_two_over_scale = float("-inf") if abs_scale == 0.0 else -2.0 / abs_scale
    return abs_scale, neg_two_over_scale


def _launch_weighted(x, weights, out, hidden, num_tokens, clamp_scale):
    block_size, num_warps = _launch_config(x.dtype)
    if clamp_scale is None:
        abs_scale, neg_two_over_scale = 0.0, 0.0
    else:
        abs_scale, neg_two_over_scale = _clamp_operands(clamp_scale)
    grid = (num_tokens, triton.cdiv(hidden, block_size))
    with torch_device_fn.device(x.device):
        _weighted_forward_kernel[grid](
            x,
            weights,
            out,
            hidden,
            abs_scale,
            neg_two_over_scale,
            HAS_CLAMP=clamp_scale is not None,
            EVEN_HIDDEN=hidden % block_size == 0,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )


def _launch_clamped(x, out, n, clamp_scale):
    block_size, num_warps = _launch_config(x.dtype)
    abs_scale, neg_two_over_scale = _clamp_operands(clamp_scale)
    grid = (triton.cdiv(n, block_size),)
    with torch_device_fn.device(x.device):
        _clamped_forward_kernel[grid](
            x,
            out,
            n,
            abs_scale,
            neg_two_over_scale,
            EVEN_N=n % block_size == 0,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )


def weighted_squared_relu_impl(input, weights=None, clamp_scale=None):
    """Squared-ReLU with optional per-token weights and optional tanh soft clamp.

    Forward-only port of ``megatron.core.fusions.fused_weighted_squared_relu
    .weighted_squared_relu_impl``. The input is viewed as ``(num_tokens, hidden)``,
    so ``weights`` is one scalar per row. With ``clamp_scale`` the pre-activation
    is ``clamp_scale * tanh(relu(x) / clamp_scale)``; both arguments ``None`` is
    not a fused case (mirrors the reference's assertion).

    Args:
        input (Tensor): fp16, bf16 or fp32 tensor of shape ``(B, hidden)`` or
            ``(S, B, hidden)`` on the active device. Non-contiguous input is
            copied; empty input is returned without a launch.
        weights (Tensor, optional): per-token weights of shape ``(B, 1)``
            (``(S * B, 1)`` for a 3-D input), usually fp32.
        clamp_scale (float, optional): the tanh soft-clamp scale.

    Returns:
        Tensor: same shape and dtype as ``input``.
    """
    logger.debug("TRAIN WEIGHTED_SQUARED_RELU_IMPL")
    assert weights is not None or clamp_scale is not None, (
        "weighted_squared_relu_impl needs weights and/or clamp_scale; "
        "use squared_relu directly otherwise."
    )
    assert input.dim() in (
        2,
        3,
    ), f"weighted_squared_relu_impl expects a 2-D or 3-D input, got {input.dim()}-D"
    _check_input(input, "weighted_squared_relu_impl")

    hidden = input.shape[-1]
    num_tokens = math.prod(input.shape[:-1])

    if weights is not None:
        _check_input(weights, "weighted_squared_relu_impl weights")
        expected = (num_tokens, 1)
        if tuple(weights.shape) != expected:
            raise ValueError(
                "weighted_squared_relu_impl weights must be broadcast per token, "
                f"i.e. shape {expected} for an input of shape "
                f"{tuple(input.shape)}, got {tuple(weights.shape)}"
            )
        if weights.device != input.device:
            raise ValueError(
                "weighted_squared_relu_impl weights must be on the same device as the "
                f"input, got {weights.device} and {input.device}"
            )

    if input.numel() == 0:
        # No launch when there is nothing to activate.
        return torch.empty_like(input)

    if clamp_scale is not None:
        clamp_scale = float(clamp_scale)

    flat = input.contiguous().view(-1, hidden)
    out = torch.empty_like(flat)
    if weights is None:
        _launch_clamped(flat, out, flat.numel(), clamp_scale)
    else:
        _launch_weighted(
            flat,
            weights.contiguous().view(-1),
            out,
            hidden,
            num_tokens,
            clamp_scale,
        )
    return out.view(input.shape)
