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

"""Fused bias + GeGLU.

Forward only. One token is one row; the last dimension is the concatenated
``[gate | linear]`` and the output keeps the gated half. Bias is added inside
the kernel.

The backward is a separate operator with its own implementation file, so this
module saves nothing for autograd and defines no ``torch.autograd.Function``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_NUM_WARPS = 4
_BLOCK_SIZE = 1024
_MIN_BLOCK_SIZE = 16
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# ``tl_extra_shim.tanh`` maps to libdevice and falls back to an exp-based
# form on backends without it.
_tanh = tl_extra_shim.tanh


_BIAS_NONE = 0
_BIAS_VECTOR = 1
_BIAS_FULL = 2

# sqrt(2/pi) and the cubic coefficient of the tanh GELU approximation.
_GELU_SQRT_2_OVER_PI = tl.constexpr(0.79788456)
_GELU_COEFF = tl.constexpr(0.044715)


def _prepare_bias(bias, flat):
    """Prepare a bias for the validated Triton fast path."""
    if bias is None:
        return None, _BIAS_NONE

    n_rows, hidden = flat.shape
    expanded = torch.broadcast_to(bias, (n_rows, hidden))
    if n_rows == 0 or hidden == 0:
        return bias, _BIAS_VECTOR
    if expanded.stride(0) == 0:
        # All tokens share one row; do not materialize the full broadcast.
        return expanded[0].contiguous(), _BIAS_VECTOR
    return expanded.contiguous(), _BIAS_FULL


def _validate_inputs(flat, bias):
    """Validate operands supported by the fused Triton forward kernel."""
    if flat.device.type != runtime_device.name:
        raise ValueError("bias_geglu input must be on the active device")
    supported_dtype = flat.dtype in _DTYPES or (
        flat.dtype == torch.float64 and runtime_device.support_fp64
    )
    if not supported_dtype:
        raise ValueError("bias_geglu input has an unsupported dtype")
    if flat.shape[-1] % 2 != 0:
        raise ValueError("bias_geglu input's last dimension must be even")

    if bias is None:
        return
    if not isinstance(bias, torch.Tensor):
        raise TypeError("bias_geglu bias must be a Tensor or None")
    if bias.device != flat.device:
        raise ValueError("bias_geglu bias must be on the input device")
    if bias.dtype != flat.dtype:
        raise ValueError("bias_geglu bias must have the input dtype")

    try:
        torch.broadcast_to(bias, flat.shape)
    except RuntimeError as exc:
        raise ValueError(
            "bias_geglu bias must be broadcastable to the flattened input"
        ) from exc


@triton.jit
def _activate(gate, linear):
    """Forward value of one gate/linear pair, in fp32 or fp64."""
    # tanh approximation of GELU:
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
    t = _tanh(_GELU_SQRT_2_OVER_PI * gate * (1.0 + _GELU_COEFF * gate * gate))
    return 0.5 * gate * (1.0 + t) * linear


@libentry()
@triton.jit
def _bias_geglu_forward_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_cols,
    BIAS_MODE: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    COMPUTE_DTYPE: tl.constexpr = tl.float64 if IS_FP64 else tl.float32
    row = tle.program_id(0)
    hidden = n_cols * 2
    x_row = x_ptr + row * hidden
    y_row = y_ptr + row * n_cols
    bias_row = bias_ptr + row * hidden

    for start in tl.range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        gate = tl.load(x_row + cols, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        linear = tl.load(x_row + n_cols + cols, mask=mask, other=0.0).to(COMPUTE_DTYPE)
        if BIAS_MODE == 1:
            gate += tl.load(bias_ptr + cols, mask=mask, other=0.0).to(COMPUTE_DTYPE)
            linear += tl.load(bias_ptr + n_cols + cols, mask=mask, other=0.0).to(
                COMPUTE_DTYPE
            )
        elif BIAS_MODE == 2:
            gate += tl.load(bias_row + cols, mask=mask, other=0.0).to(COMPUTE_DTYPE)
            linear += tl.load(bias_row + n_cols + cols, mask=mask, other=0.0).to(
                COMPUTE_DTYPE
            )
        out = _activate(gate, linear)
        tl.store(y_row + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


def _launch_config(n_cols):
    """Power-of-two column tile and the warps needed to cover it."""
    block = max(_MIN_BLOCK_SIZE, min(_BLOCK_SIZE, triton.next_power_of_2(n_cols)))
    if block <= 64:
        return block, 1
    if block <= 256:
        return block, 2
    return block, _NUM_WARPS


def _flatten_input(x):
    """Use Megatron's rank check and view semantics exactly."""
    ori_shape = x.shape
    assert len(ori_shape) in [2, 3]
    return x.view(-1, ori_shape[-1]), ori_shape


def _forward(flat, bias):
    """Run the kernel over a flattened ``[rows, hidden]`` view."""
    n_rows, hidden = flat.shape
    n_cols = hidden // 2
    bias_tensor, bias_mode = _prepare_bias(bias, flat)
    if n_rows == 0 or n_cols == 0:
        return flat.new_empty((n_rows, n_cols))

    block, num_warps = _launch_config(n_cols)
    y = flat.new_empty((n_rows, n_cols))
    with torch_device_fn.device(flat.device):
        _bias_geglu_forward_kernel[(n_rows,)](
            flat,
            # The constexpr mode stops the kernel from reading this pointer,
            # but Triton still type-checks the argument, so reuse ``flat``.
            flat if bias_tensor is None else bias_tensor,
            y,
            n_cols,
            BIAS_MODE=bias_mode,
            IS_FP64=flat.dtype == torch.float64,
            BLOCK=block,
            num_warps=num_warps,
        )
    return y


def bias_geglu_impl(input, bias):
    """Apply Megatron-compatible bias + GEGLU forward.

    The input must be contiguous and on the active device. Supported dtypes
    are fp16/bf16/fp32, plus fp64 on devices with FP64 support. Bias must be
    None or a same-device, same-dtype tensor broadcastable to the flattened
    input. All activation arithmetic runs in the Triton kernel.
    """
    logger.debug("TRAIN BIAS_GEGLU_IMPL")
    # Judge contiguity on the caller's tensor, before ``view`` can touch it:
    # some runtimes monkeypatch ``torch.Tensor.view`` to copy non-contiguous
    # tensors first (Megatron-LM-FL does this on NPU), which would make the
    # flattened view contiguous and hide the layout from ``_validate_inputs``.
    if not input.is_contiguous():
        raise ValueError("bias_geglu input must be contiguous")
    flat, ori_shape = _flatten_input(input)

    _validate_inputs(flat, bias)
    output = _forward(flat, bias)

    return (
        output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
    )
