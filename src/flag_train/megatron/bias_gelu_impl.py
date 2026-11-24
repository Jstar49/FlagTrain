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
"""Forward-only bias + tanh GELU using the common Triton/FlagTree backend.

Dtype promotion and output dtypes follow Megatron's torch expression; integer
and bool sums become FP32, while complex inputs are not supported. The output
is a plain tensor: this operator does not propagate gradients.

This module is the portable implementation. Ascend replaces the launch with a
persistent, core-count-bounded grid and gives ``_bias_gelu_tile`` its libdevice
tanh; see ``runtime/backend/_ascend/megatron/bias_gelu_impl.py``.
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

_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_INTEGRAL_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)
# uint16/32/64 are accepted as operands only when promotion moves the sum to a
# supported dtype. torch.promote_types rejects mixed unsigned pairs, and eager
# torch add does not implement them either.
_OPERAND_DTYPES = (
    _FLOAT_DTYPES + _INTEGRAL_DTYPES + (torch.uint16, torch.uint32, torch.uint64)
)
_SUM_DTYPES = _FLOAT_DTYPES + (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)
_BLOCK_SIZE = 256
_NUM_WARPS = 4
# Only this instruction capability differs; GPU scheduling and tiling are shared.
_HAS_NATIVE_TANH = tl.constexpr(runtime_device.vendor_name == "nvidia")


def _accelerator_device():
    name = runtime_device.name
    if name == "cpu":
        raise RuntimeError(
            "bias_gelu_impl cannot run CPU inputs without an accelerator: "
            "no device Triton backend is available"
        )
    is_available = getattr(torch_device_fn, "is_available", None)
    if callable(is_available) and not is_available():
        raise RuntimeError(
            f"bias_gelu_impl cannot run CPU inputs: {name} is unavailable"
        )
    return name


def _resolve_dtypes(input, bias):
    """Megatron's torch promotion plus the output dtype of the tanh GELU."""
    if input.dtype not in _OPERAND_DTYPES:
        raise ValueError(f"unsupported input dtype {input.dtype}")
    if bias.dtype not in _OPERAND_DTYPES:
        raise ValueError(f"unsupported bias dtype {bias.dtype}")

    try:
        sum_dtype = torch.promote_types(input.dtype, bias.dtype)
    except RuntimeError as exc:
        raise ValueError(
            "input and bias promote to an unsupported dtype "
            f"({input.dtype}, {bias.dtype})"
        ) from exc
    if sum_dtype not in _SUM_DTYPES:
        raise ValueError(f"input and bias promote to an unsupported dtype {sum_dtype}")

    if sum_dtype in _INTEGRAL_DTYPES or sum_dtype == torch.uint8:
        # Python float constants in the GELU expression promote integer/bool
        # tensors to the default float dtype.
        output_dtype = torch.float32
    else:
        output_dtype = sum_dtype
    return sum_dtype, output_dtype


def _check_devices(input, bias):
    """Apply torch's 0-dim CPU scalar rule and return the result device."""
    if input.device == bias.device:
        return input, bias, input.device
    if input.dim() == 0 and input.device.type == "cpu":
        return input.to(bias.device), bias, bias.device
    if bias.dim() == 0 and bias.device.type == "cpu":
        return input, bias.to(input.device), input.device
    raise ValueError("input and bias must be on the same device")


def _as_scalar_tensor(value, tensor, name):
    """Convert a Python number using the tensor's weak scalar promotion."""
    try:
        dtype = torch.result_type(tensor, value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"{name} scalar does not promote with dtype {tensor.dtype}"
        ) from exc
    try:
        return torch.as_tensor(value, dtype=dtype, device=tensor.device)
    except (RuntimeError, OverflowError, TypeError):
        # torch's integer scalar add wraps an out-of-range Python integer in
        # the tensor dtype instead of rejecting or promoting it.
        if dtype in _INTEGRAL_DTYPES:
            return torch.tensor(value, device=tensor.device).to(dtype)
        raise


@triton.jit
def _gpu_gelu(z, argument, NATIVE_TANH: tl.constexpr):
    if NATIVE_TANH:
        t = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[argument],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return 0.5 * z * (1.0 + t)
    # Portable sigmoid identity: no CUDA libdevice or foreign inline assembly.
    return z * (1.0 / (1.0 + tl.exp(-2.0 * argument)))


@triton.jit
def _bias_gelu_tile(
    X,
    B,
    Y,
    N,
    H: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    tile,
    ROWWISE: tl.constexpr,
    EVEN: tl.constexpr,
    INTEGRAL: tl.constexpr,
    BOOL_SUM: tl.constexpr,
    FP64: tl.constexpr,
    SHIM_TANH: tl.constexpr,
):
    if (ROWWISE and BIAS_MODE == 1) and not (H <= BLOCK and H & (H - 1) == 0):
        # Explicitly contiguous bias columns avoid Ascend's scalar gather
        # lowering of a vector remainder. Row tails have their own mask.
        columns = tile % tl.cdiv(H, BLOCK) * BLOCK + tl.arange(0, BLOCK)
        offsets = tile // tl.cdiv(H, BLOCK) * H + columns
        mask = (offsets < N) & (columns < H)
    else:
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
    if EVEN:
        mask = True
    x = tl.load(X + offsets, mask, other=0)
    if BIAS_MODE == 0:
        b = tl.load(B)
    elif BIAS_MODE == 1:
        if (ROWWISE and H <= BLOCK) and H & (H - 1) == 0:
            row = tl.load(B + tl.arange(0, H))
            b = tl.broadcast_to(row[None, :], (BLOCK // H, H)).reshape((BLOCK,))
        elif ROWWISE:
            b = tl.load(B + columns, mask, other=0)
        else:
            b = tl.load(B + offsets % H, mask, other=0)
    else:
        b = tl.load(B + offsets, mask, other=0)
    if BOOL_SUM:
        # torch bool addition is logical OR; the kernel stores 0/1 as uint8.
        z = x | b
        zf = z.to(tl.float32)
    elif INTEGRAL:
        # Ascend lowers sub-word adds at 32 bits -- an int8/int16 sum does not
        # wrap the way torch's add does -- and converts a uint8 arithmetic
        # result to float by sign. Wrap at the operand width and widen through
        # int32 before the float conversion; on vendors that already wrap both
        # casts are identities.
        z = (x + b).to(X.dtype.element_ty)
        if X.dtype.element_ty == tl.int64:
            zf = z.to(tl.float32)
        else:
            zf = z.to(tl.int32).to(tl.float32)
    elif FP64:
        z = x + b
        zf = z
    else:
        # Round the bias add to the operand dtype before FP32 activation.
        # Ascend BF16 legalization can otherwise retain an unrounded FP32
        # intermediate; FP16 rows keep their native-add path.
        if not ROWWISE or X.dtype.element_ty != tl.float16:
            x = x.to(tl.float32)
            b = b.to(tl.float32)
        z = (x + b).to(X.dtype.element_ty).to(tl.float32)
        zf = z

    # Megatron's expression evaluates ``0.044715 * z`` before ``* z``, and
    # that Python float promotes integer/bool sums to FP32 before the square.
    # Keep the grouping: squaring in the integer dtype would wrap above the
    # dtype's square root.
    argument = 0.79788456 * zf * (1.0 + 0.044715 * zf * zf)
    if ROWWISE:
        # 0.5*z*(1+tanh(u)) == z/(1+exp(-2*u)).
        result = zf * (1.0 / (1.0 + tl.exp(-2.0 * argument)))
    elif FP64 or SHIM_TANH:
        t = tl_extra_shim.tanh(argument)
        result = 0.5 * zf * (1.0 + t)
    else:
        result = _gpu_gelu(zf, argument, _HAS_NATIVE_TANH)
    tl.store(Y + offsets, result, mask)


@libentry()
@triton.jit
def _bias_gelu_forward_kernel(
    X,
    B,
    Y,
    N,
    H: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr = False,
    INTEGRAL: tl.constexpr = False,
    BOOL_SUM: tl.constexpr = False,
    FP64: tl.constexpr = False,
):
    _bias_gelu_tile(
        X,
        B,
        Y,
        N,
        H,
        BIAS_MODE,
        BLOCK,
        tle.program_id(0),
        False,
        EVEN,
        INTEGRAL,
        BOOL_SUM,
        FP64,
        False,
    )


def bias_gelu_impl(input, bias):
    """Return tanh GELU(input + bias), without gradient propagation.

    The forward semantics and dtype promotion are Megatron's: operands are
    broadcast, added in their promoted dtype, and the GELU runs in that dtype;
    integer/bool sums produce FP32. One operand may be a Python number: it
    follows torch's weak scalar promotion, as Megatron's expression does.
    Complex inputs are rejected. CPU inputs are staged through the accelerator
    and the result is copied back. Invalid types raise TypeError; unsupported
    dtypes, devices or shapes raise ValueError.
    """
    logger.debug("TRAIN BIAS_GELU_IMPL")
    for name, value in (("input", input), ("bias", bias)):
        if isinstance(value, torch.Tensor):
            if value.dtype not in _OPERAND_DTYPES:
                raise ValueError(f"{name} has unsupported dtype {value.dtype}")
        elif isinstance(value, numbers.Number):
            # Complex was explicitly left unsupported; every other Python
            # number follows Megatron's weak scalar promotion below.
            if isinstance(value, complex):
                raise ValueError(f"{name} has unsupported dtype complex")
        else:
            raise TypeError(f"{name} must be a tensor or a number")

    input_is_tensor = isinstance(input, torch.Tensor)
    bias_is_tensor = isinstance(bias, torch.Tensor)
    if not input_is_tensor and not bias_is_tensor:
        # Megatron's torch.tanh also needs one tensor operand.
        raise TypeError("at least one of input and bias must be a tensor")
    if not input_is_tensor:
        input = _as_scalar_tensor(input, bias, "input")
    elif not bias_is_tensor:
        bias = _as_scalar_tensor(bias, input, "bias")

    input, bias, result_device = _check_devices(input, bias)
    sum_dtype, out_dtype = _resolve_dtypes(input, bias)
    kernel_dtype = torch.uint8 if sum_dtype == torch.bool else sum_dtype
    bool_sum = sum_dtype == torch.bool

    vector_bias = input.ndim and bias.ndim == 1 and bias.numel() == input.shape[-1]
    if vector_bias:
        x, b = input, bias
    else:
        try:
            x, b = torch.broadcast_tensors(input, bias)
        except RuntimeError:
            raise ValueError(
                "input and bias must have broadcast-compatible shapes"
            ) from None
    x = x.to(kernel_dtype)
    b = b.to(kernel_dtype)
    n = x.numel()

    if not n:
        return torch.empty(x.shape, dtype=out_dtype, device=result_device)

    compute_device = result_device
    if result_device.type == "cpu":
        # Megatron accepts CPU operands; Triton has no CPU backend, so stage
        # the already-promoted operands and copy the result back.
        compute_device = _accelerator_device()
        x = x.to(compute_device)
        b = b.to(compute_device)

    h = x.shape[-1] if x.ndim else 1
    if vector_bias:
        b, mode = b.contiguous(), 1
    elif b.numel() == 1:
        b, mode = b.contiguous(), 0
    elif b.stride(-1) == 1 and all(s == 0 for s in b.stride()[:-1]):
        # Preserve a shared bias row instead of copying its full broadcast.
        b, mode = b.reshape(-1, h)[0].contiguous(), 1
    else:
        b, mode = b.contiguous(), 2
    x = x.contiguous()
    out = torch.empty(x.shape, dtype=out_dtype, device=compute_device)

    block = _BLOCK_SIZE
    if n >= 65536:
        # Share GPU tiling across vendors; FP32 favors a smaller tile.
        block = 1024 if sum_dtype == torch.float32 else 2048
    grid = (triton.cdiv(n, block),)
    integral = sum_dtype in _INTEGRAL_DTYPES
    fp64 = sum_dtype == torch.float64
    with torch_device_fn.device(compute_device):
        _bias_gelu_forward_kernel[grid](
            x,
            b,
            out,
            n,
            h,
            mode,
            BLOCK=block,
            num_warps=_NUM_WARPS,
            EVEN=n % block == 0,
            INTEGRAL=integral,
            BOOL_SUM=bool_sum,
            FP64=fp64,
        )
    return out.to(result_device) if result_device.type == "cpu" else out
