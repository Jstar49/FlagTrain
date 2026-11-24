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

"""Fused bias + SwiGLU, including the clamped and SiTU-GLU variants.

Forward only. One token is one row; the last dimension is the concatenated
``[gate | linear]`` and the output keeps the gated half. Bias is added inside
the kernel.

The backward is a separate operator with its own implementation file, so this
module saves nothing for autograd and defines no ``torch.autograd.Function``.
``fp8_input_store`` and ``cpu_offload_input`` only affected activations saved
for the backward; they are accepted and ignored here, as the forward output is
independent of them.

The activation is computed by a Triton kernel for fp16/bf16/fp32/fp64 tensors.
A CPU input is copied to the active accelerator for the kernel and the result
is copied back; those are data copies, not a torch-compute fallback. Bias may
have a different floating dtype; the kernel reproduces PyTorch's promotion and
writes the promoted dtype. Non-positive, NaN and infinite clamps are handled by
the same kernel arithmetic as Megatron-LM. Non-contiguous inputs are moved to a
contiguous buffer (a layout copy, not a compute fallback); layouts
``view(-1, hidden)`` cannot express are rejected, as upstream does.
"""

import logging
import numbers

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
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

# Both helpers are provided by the platform layer. ``tl.sigmoid`` is a core
# Triton builtin; ``tl_extra_shim.tanh`` maps to libdevice and falls back to an
# exp-based form only on backends without it. Neither needs the extra argument
# clamp this file used to carry: exp overflow in the sigmoid denominator lands
# on inf, and ``1 / (1 + inf)`` is exactly 0 rather than NaN.
_sigmoid = tl.sigmoid
_tanh = tl_extra_shim.tanh

_MODE_PLAIN = 0
_MODE_CLAMPED = 1
_MODE_SITU = 2

_BIAS_NONE = 0
_BIAS_VECTOR = 1
_BIAS_FULL = 2


def _scalar(value, name):
    """A Python float for a real scalar or a one-element tensor."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"{name} needs a scalar, got shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} needs a real scalar, got {type(value).__name__}")
    return float(value)


def _variant(clamp_value, gate_clamp_scale, linear_clamp_scale):
    """``(mode, clamp, gate_scale, linear_scale)`` for the Triton kernel.

    This mirrors Megatron-LM's Python-level selection exactly: a SiTU scale
    wins, otherwise a positive clamp selects the clamped kernel, and a
    non-positive or NaN clamp is the unclamped kernel.
    """
    if gate_clamp_scale is not None:
        gate_scale = _scalar(gate_clamp_scale, "gate_clamp_scale")
        linear_scale = (
            None
            if linear_clamp_scale is None
            else _scalar(linear_clamp_scale, "linear_clamp_scale")
        )
        return _MODE_SITU, 0.0, gate_scale, linear_scale

    if clamp_value is not None:
        clamp = _scalar(clamp_value, "clamp_value")
        if clamp > 0.0:
            return _MODE_CLAMPED, clamp, 1.0, None
        return _MODE_PLAIN, 0.0, 1.0, None

    return _MODE_PLAIN, 0.0, 1.0, None


def _prepare_bias(bias, flat):
    """Return a kernel bias tensor and a mode.

    Mode vector is one row of ``hidden`` values shared by every token, which
    is the ``bias`` produced by ``skip_bias_add``. Mode full is a different
    bias per element and is only used when the tensor really depends on the
    token index.
    """
    if bias is None:
        return None, _BIAS_NONE
    if bias.device != flat.device:
        raise ValueError(
            f"bias_swiglu bias is on {bias.device}, expected {flat.device}"
        )
    n_rows, hidden = flat.shape
    flat_shape = (n_rows, hidden)
    expanded = torch.broadcast_to(bias, flat_shape)
    # Empty inputs still reach this point so a bad bias is rejected. Keep the
    # guard before indexing a row: with no rows there is no row 0 to copy.
    if n_rows == 0 or hidden == 0:
        return bias, _BIAS_VECTOR
    if expanded.stride(0) == 0:
        # Same values on every token. Copy one row, not the whole broadcast.
        return expanded[0].contiguous(), _BIAS_VECTOR
    return expanded.contiguous(), _BIAS_FULL


@triton.jit
def _activate(
    gate,
    linear,
    clamp_value,
    gate_scale,
    linear_scale,
    MODE: tl.constexpr,
    HAS_LINEAR_CLAMP: tl.constexpr,
):
    """Forward value of one gate/linear pair, in the kernel's compute dtype."""
    if MODE == 0:
        return gate * _sigmoid(gate) * linear
    if MODE == 1:
        gate_c = tl.minimum(gate, clamp_value)
        linear_c = tl.minimum(tl.maximum(linear, -clamp_value), clamp_value)
        # torch.clamp propagates NaN; tl.minimum/maximum default to
        # minnum/maxnum, which replace NaN with the clamp bound. Restore the
        # NaN so the clamped variant matches Megatron-LM's clamped_swiglu.
        gate_c = tl.where(gate != gate, gate, gate_c)
        linear_c = tl.where(linear != linear, linear, linear_c)
        return gate_c * _sigmoid(gate_c) * linear_c
    t = _tanh(gate / gate_scale)
    gate_v = (gate_scale * t) * _sigmoid(gate)
    if HAS_LINEAR_CLAMP:
        lin = linear_scale * _tanh(linear / linear_scale)
    else:
        lin = linear
    return gate_v * lin


@libentry()
@triton.jit
def _bias_swiglu_forward_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    n_cols,
    clamp_value,
    gate_scale,
    linear_scale,
    BIAS_MODE: tl.constexpr,
    MODE: tl.constexpr,
    HAS_LINEAR_CLAMP: tl.constexpr,
    ADD_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tle.program_id(0)
    hidden = n_cols * 2
    x_row = x_ptr + row * hidden
    y_row = y_ptr + row * n_cols
    bias_row = bias_ptr + row * hidden

    for start in tl.range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        gate = tl.load(x_row + cols, mask=mask, other=0.0).to(ADD_DTYPE)
        linear = tl.load(x_row + n_cols + cols, mask=mask, other=0.0).to(ADD_DTYPE)
        if BIAS_MODE == 1:
            gate += tl.load(bias_ptr + cols, mask=mask, other=0.0).to(ADD_DTYPE)
            linear += tl.load(bias_ptr + n_cols + cols, mask=mask, other=0.0).to(
                ADD_DTYPE
            )
        elif BIAS_MODE == 2:
            gate += tl.load(bias_row + cols, mask=mask, other=0.0).to(ADD_DTYPE)
            linear += tl.load(bias_row + n_cols + cols, mask=mask, other=0.0).to(
                ADD_DTYPE
            )
        out = _activate(
            gate.to(COMPUTE_DTYPE),
            linear.to(COMPUTE_DTYPE),
            clamp_value,
            gate_scale,
            linear_scale,
            MODE,
            HAS_LINEAR_CLAMP,
        )
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
    hidden = x.shape[-1]
    if hidden % 2 != 0:
        raise RuntimeError(
            f"bias_swiglu input's last dimension must be even, got {hidden}"
        )
    # Megatron uses view(), not reshape(): a layout it cannot express is an
    # error there too. The copy that follows is a layout operation, not a
    # compute fallback.
    flat = x.view(-1, hidden)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    out_shape = (*x.shape[:-1], hidden // 2)
    return flat, out_shape


def _forward(flat, bias, mode, clamp, gate_scale, linear_scale, out_dtype):
    """Run the kernel over a flattened ``[rows, hidden]`` view."""
    n_rows, hidden = flat.shape
    n_cols = hidden // 2
    bias_tensor, bias_mode = _prepare_bias(bias, flat)
    if n_rows == 0 or n_cols == 0:
        return torch.empty((n_rows, n_cols), dtype=out_dtype, device=flat.device)

    block, num_warps = _launch_config(n_cols)
    y = torch.empty((n_rows, n_cols), dtype=out_dtype, device=flat.device)
    # Narrow outputs keep the add in fp32; this matches the rounding the
    # compiled Megatron path exhibits for the same fp16/bf16 inputs.
    add_dtype = tl.float64 if out_dtype == torch.float64 else tl.float32
    compute_dtype = (
        tl.float64 if mode == _MODE_PLAIN and out_dtype == torch.float64 else tl.float32
    )
    with torch_device_fn.device(flat.device):
        _bias_swiglu_forward_kernel[(n_rows,)](
            flat,
            # An absent bias borrows the input pointer; BIAS_MODE stops the
            # kernel from reading it.
            flat if bias_tensor is None else bias_tensor,
            y,
            n_cols,
            clamp,
            gate_scale,
            1.0 if linear_scale is None else linear_scale,
            BIAS_MODE=bias_mode,
            MODE=mode,
            HAS_LINEAR_CLAMP=linear_scale is not None,
            ADD_DTYPE=add_dtype,
            COMPUTE_DTYPE=compute_dtype,
            BLOCK=block,
            num_warps=num_warps,
        )
    return y


def bias_swiglu_impl(
    input,
    bias,
    fp8_input_store=False,
    cpu_offload_input=False,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    """Bias plus SwiGLU, or SiTU-GLU when ``gate_clamp_scale`` is set.

    Forward only: the backward is a separate operator.

    Args:
        input (Tensor): rank-2 or rank-3 fp16/bf16/fp32/fp64 tensor whose last
            dimension is ``[gate | linear]``. A CPU tensor is copied to the
            active device for the kernel and the result is copied back; a
            non-contiguous view is copied before the kernel.
        bias (Tensor or None): added to ``input`` before the activation.
            ``None`` skips the add. Broadcasting follows PyTorch, so a bias
            with a different dtype promotes the result exactly as upstream.
        fp8_input_store (bool): accepted for Megatron-LM signature
            compatibility and ignored; it only affected the saved activations.
        cpu_offload_input (bool): same as ``fp8_input_store``.
        clamp_value (float or None): hard-clamp the gate at this upper bound
            and the linear half to ``[-clamp_value, clamp_value]``. As
            upstream, a non-positive or NaN value selects the unclamped path.
        gate_clamp_scale (float or None): select SiTU-GLU. The gate's linear
            factor becomes ``scale * tanh(x / scale)``.
        linear_clamp_scale (float or None): also soft-clamp the linear half.
            Requires ``gate_clamp_scale``.

    Returns:
        Tensor: same rank as ``input`` and the last dimension halved. Its dtype
            is ``input.dtype`` unless a bias promotes it, as in PyTorch.

    Raises:
        ValueError: If the input/bias is on different devices, has an
            unsupported dtype, or a CPU tensor is given without an active
            accelerator.
        RuntimeError: If a 3-D input has zero elements. Megatron-LM restores
            the 3-D shape with ``view(..., -1)``, which is ambiguous for a
            0-element tensor; this operator mirrors that behavior.
    """
    logger.debug("TRAIN BIAS_SWIGLU_IMPL")
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"bias_swiglu input needs a tensor, got {type(input).__name__}")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(
            f"bias_swiglu bias needs a tensor or None, got {type(bias).__name__}"
        )
    assert input.ndim in (2, 3), (
        f"bias_swiglu input must have rank 2 or 3, got rank {input.ndim} "
        f"shape {tuple(input.shape)}"
    )
    assert gate_clamp_scale is not None or linear_clamp_scale is None
    assert gate_clamp_scale is None or clamp_value is None

    if input.dtype not in _DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _DTYPES)
        raise ValueError(
            f"bias_swiglu input needs a {wanted} tensor, got {input.dtype}"
        )
    if bias is not None:
        if bias.dtype not in _DTYPES:
            wanted = " / ".join(str(dtype) for dtype in _DTYPES)
            raise ValueError(
                f"bias_swiglu bias needs a {wanted} tensor, got {bias.dtype}"
            )
        if bias.device != input.device:
            raise ValueError("bias_swiglu input and bias must be on the same device")

    cpu_input = input.device.type == "cpu"
    if cpu_input and runtime_device.name == "cpu":
        raise ValueError(
            "bias_swiglu needs an active accelerator to serve a CPU tensor"
        )
    if not cpu_input and input.device.type != runtime_device.name:
        raise ValueError(
            "bias_swiglu needs a tensor on the active device, got "
            f"{input.device.type}"
        )
    out_dtype = (
        input.dtype if bias is None else torch.promote_types(input.dtype, bias.dtype)
    )

    mode, clamp, gate_scale, linear_scale = _variant(
        clamp_value, gate_clamp_scale, linear_clamp_scale
    )
    flat, out_shape = _flatten_input(input)
    if cpu_input:
        # Serve CPU tensors by moving the operands to the accelerator and the
        # result back. These are copies, not torch activation arithmetic.
        with torch_device_fn.device(runtime_device.name):
            flat = flat.to(runtime_device.name)
            bias_for_kernel = None if bias is None else bias.to(runtime_device.name)
            y = _forward(
                flat,
                bias_for_kernel,
                mode,
                clamp,
                gate_scale,
                linear_scale,
                out_dtype,
            )
        y = y.to("cpu")
    else:
        y = _forward(flat, bias, mode, clamp, gate_scale, linear_scale, out_dtype)

    # Megatron-LM restores the 3-D shape with ``view(ori_shape[0],
    # ori_shape[1], -1)``. The ``-1`` is ambiguous for a 0-element tensor, so a
    # 3-D input whose leading dimensions contain zero raises RuntimeError.
    # Keep that behavior: this operator mirrors the upstream entry point.
    return y if input.ndim == 2 else y.view(out_shape[0], out_shape[1], -1)
