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
"""Megatron-LM Tanh Soft Clamp activation, implemented entirely in Triton.

The arithmetic is Megatron's ``(scale * tanh(x.float() / scale)).to(x.dtype)``:
operands widen to FP32, the tanh runs there, and the result is cast back to the
input dtype. Every supported tensor is processed by a Triton kernel; the host
only validates, normalizes/broadcasts operands (copies), and stages CPU inputs
through the accelerator.
"""

import logging
import math
import numbers
from contextlib import nullcontext
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle
from flag_train.utils.codegen_config_utils import (
    get_codegen_config,
    get_heuristics_for_num_warps,
)

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INTEGRAL_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)
_KERNEL_DTYPES = _FLOAT_DTYPES + _INTEGRAL_DTYPES
_NUM_WARPS = 4

_CODE_CONFIG = get_codegen_config()
_MAX_TILE = getattr(_CODE_CONFIG, "max_tile_size", 0) or 2048
_MAX_GRID = (getattr(_CODE_CONFIG, "max_grid_size", None) or (65536,))[0]


@triton.jit
def _tanh_soft_clamp_tile(
    X,
    S,
    Y,
    N,
    tile,
    SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask, other=0).to(tl.float32)
    if SCALE_MODE == 0:
        # Megatron's expression divides; dividing here too keeps the same
        # rounding, which the integer casts below can otherwise expose.
        # ``SCALE`` is a compile-time constant on purpose: an extra runtime
        # scalar measured ~40% slower on Ascend, and a production scale is
        # fixed per model. A new scale value costs one compile (a few seconds),
        # which Triton's on-disk cache keeps for later calls.
        y = SCALE * tl_extra_shim.tanh(x / SCALE)
    elif SCALE_MODE == 1:
        # A one-element scale tensor: loaded once per tile, not per lane.
        s = tl.load(S).to(tl.float32)
        y = s * tl_extra_shim.tanh(x / s)
    else:
        s = tl.load(S + offsets, mask, other=0).to(tl.float32)
        y = s * tl_extra_shim.tanh(x / s)
    tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask)


@libentry()
@triton.jit
def _tanh_soft_clamp_kernel(
    X,
    S,
    Y,
    N,
    SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _tanh_soft_clamp_tile(X, S, Y, N, tle.program_id(0), SCALE, SCALE_MODE, BLOCK)


@libentry()
@triton.jit
def _tanh_soft_clamp_walk_kernel(
    X,
    S,
    Y,
    N,
    SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program walks several tiles; the runtime trip count keeps the loop
    # from unrolling. Used when the flat grid would exceed the vendor's
    # ``max_grid_size`` (portable) or the vector-core count (Ascend).
    tiles = tl.cdiv(N, BLOCK)
    for tile in range(tle.program_id(0), tiles, tl.num_programs(0)):
        _tanh_soft_clamp_tile(X, S, Y, N, tile, SCALE, SCALE_MODE, BLOCK)


def _accelerator_device():
    name = runtime_device.name
    if name == "cpu":
        raise RuntimeError(
            "tanh_soft_clamp cannot run CPU inputs without an accelerator: "
            "no device Triton backend is available"
        )
    is_available = getattr(torch_device_fn, "is_available", None)
    if callable(is_available) and not is_available():
        raise RuntimeError(
            f"tanh_soft_clamp cannot run CPU inputs: {name} is unavailable"
        )
    return name


def _check_dtype(tensor, name):
    if tensor.is_complex():
        raise TypeError(
            f"tanh_soft_clamp does not support complex {name}, got {tensor.dtype}"
        )
    if tensor.dtype not in _KERNEL_DTYPES:
        raise ValueError(f"tanh_soft_clamp does not support dtype {tensor.dtype}")
    if tensor.dtype is torch.float64 and not runtime_device.support_fp64:
        raise ValueError("tanh_soft_clamp needs a device with FP64 support for float64")


def _validate(x, scale):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"tanh_soft_clamp expects a tensor, got {type(x).__name__}")
    _check_dtype(x, "input")
    if x.device.type not in ("cpu", runtime_device.name):
        raise ValueError(
            f"tanh_soft_clamp supports CPU or {runtime_device.name} tensors, "
            f"got {x.device.type}"
        )
    if isinstance(scale, torch.Tensor):
        _check_dtype(scale, "scale")
        if scale.dtype is torch.float64 and scale.dim() != 0:
            # torch promotes ``x.float() / scale`` to float64 for a shaped
            # float64 scale (a 0-dim one follows the wrapped-number rule and
            # stays fp32); the kernel computes in fp32, so reject the case
            # rather than silently rounding it.
            raise ValueError(
                "tanh_soft_clamp does not support a non-scalar float64 scale "
                "tensor: torch promotes x.float() / scale to float64"
            )
        return
    if isinstance(scale, numbers.Real):
        return
    raise TypeError(
        "tanh_soft_clamp scale must be a real number or a tensor, got "
        f"{type(scale).__name__}"
    )


def _dense_storage(tensor):
    """Whether ``tensor``'s elements form a dense permutation of its memory.

    Decided from the strides, not from the storage size: a device allocator may
    round an allocation up (torch-npu does), and a view may start at a non-zero
    storage offset. Both are still dense permutations, and ``_flat_storage``
    handles them. Width-one dimensions carry no information and are skipped.
    """
    expected = 1
    for stride, size in sorted(
        (stride, size)
        for stride, size in zip(tensor.stride(), tensor.shape)
        if size > 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _flat_storage(tensor):
    """1-D view over a dense tensor's storage (no copy)."""
    return torch.as_strided(tensor, (tensor.numel(),), (1,))


def _copy_back(tensor):
    """Move a result to the CPU keeping its strides (``.to`` flattens them)."""
    cpu = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cpu"
    )
    cpu.copy_(tensor)
    return cpu


def _finish(prep):
    """The operator's return value: a staged result comes back on the CPU."""
    return _copy_back(prep.out) if prep.staged else prep.out


class _Prepared(NamedTuple):
    """Operands and launch parameters after validation, staging and copies."""

    x_flat: torch.Tensor
    s_arg: torch.Tensor
    out_flat: torch.Tensor
    out: torch.Tensor
    n: int
    scalar: float
    mode: int
    result_device: torch.device
    staged: bool
    compute_device: object
    out_dtype: torch.dtype


def _launch_context(result_device, compute_device):
    """Device context for a launch on ``compute_device``."""
    no_switch = (
        result_device.type != "cpu"
        and torch_device_fn.current_device() == result_device.index
    )
    return nullcontext() if no_switch else torch_device_fn.device(compute_device)


def _prepare(x, scale):
    """Validate/normalize operands and return what a launch needs.

    The host work is shared by the portable implementation and the Ascend
    override; only the launch geometry and kernels differ.
    """
    _validate(x, scale)
    scale_is_tensor = isinstance(scale, torch.Tensor)

    result_device = x.device
    staged = result_device.type == "cpu"
    compute_device = _accelerator_device() if staged else result_device
    if staged:
        x = x.to(compute_device)
        if scale_is_tensor:
            scale = scale.to(compute_device)

    if scale_is_tensor:
        if scale.dim() == 0 and scale.device.type == "cpu":
            # torch's 0-dimensional CPU-scalar rule, as Megatron's expression
            # relies on. (A CPU input was staged together with the scale above,
            # so reaching this point means the input is on the accelerator.)
            scale = scale.to(compute_device)
        if scale.device != x.device:
            # The full device, not just its type: two accelerators of the same
            # vendor are different devices, and the kernel would dereference a
            # pointer from the other one.
            raise ValueError("tanh_soft_clamp scale and input must share a device")
        out_shape = torch.broadcast_shapes(x.shape, scale.shape)
    else:
        out_shape = x.shape

    if scale_is_tensor:
        if scale.numel() == 1:
            mode = 1
            scale_arg = scale.reshape(1).contiguous()
        elif out_shape != x.shape:
            # A scale that expands the input: materialize both operands at the
            # broadcast shape (copies, no arithmetic on the host).
            x = torch.broadcast_to(x, out_shape).contiguous()
            scale_arg = torch.broadcast_to(scale, out_shape).contiguous().view(-1)
            mode = 2
        elif scale.numel() == x.numel() and scale.is_contiguous():
            mode = 2
            scale_arg = scale.view(-1)
        else:
            mode = 2
            scale_arg = torch.broadcast_to(scale, x.shape).contiguous().view(-1)
    else:
        mode = 0
        scale_arg = None

    if mode == 2 and not x.is_contiguous():
        # A per-element scale is read in logical order, so the input has to be
        # in logical order too: a storage-order flat view would pair each lane
        # with the wrong scale element. (The broadcast branch above has
        # already materialized x.)
        x = x.contiguous()

    out_dtype = x.dtype
    if out_shape == x.shape:
        dense = x.is_contiguous() or _dense_storage(x)
        x_flat = (
            x.view(-1)
            if x.is_contiguous()
            else (_flat_storage(x) if dense else x.contiguous().view(-1))
        )
        if dense:
            # Same layout as the input, so the strides survive a transposed or
            # permuted view -- which is what Megatron's eager expression does.
            # The strides are requested explicitly rather than through
            # ``empty_like(memory_format=torch.preserve_format)``: torch-npu
            # does not guarantee preserve-format strides for a permuted view
            # and falls back to a contiguous allocation (measured on the
            # 2.10 stack, where a transposed input got a contiguous output).
            out = torch.empty_strided(
                x.shape, x.stride(), dtype=out_dtype, device=x.device
            )
        else:
            out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        # ``out`` is a fresh dense allocation (strided or contiguous), so its
        # storage view is always the flat one.
        out_flat = _flat_storage(out)
    else:
        out = torch.empty(out_shape, dtype=out_dtype, device=x.device)
        # A one-element scale can still raise the output's rank, and then x
        # need not be contiguous (only a broadcast view is materialized above).
        x_flat = x.contiguous().view(-1)
        out_flat = out.view(-1)

    n = x_flat.numel()

    if mode == 0:
        scalar = float(scale)
        s_arg = x_flat  # never dereferenced under SCALE_MODE == 0
        if scalar == 0.0 or not math.isfinite(scalar):
            # A compile-time division by zero is not IEEE, and a NaN/inf scale
            # constexpr lets the compiler fold a NaN-to-integer cast into a
            # poison value. Hand the kernel a one-element runtime scale so the
            # division and cast happen at runtime, as Megatron's expression
            # does.
            s_arg = torch.full((1,), scalar, dtype=torch.float32, device=x.device)
            mode = 1
            scalar = 0.0  # unused under SCALE_MODE == 1
    else:
        scalar = 0.0
        s_arg = scale_arg

    return _Prepared(
        x_flat=x_flat,
        s_arg=s_arg,
        out_flat=out_flat,
        out=out,
        n=n,
        scalar=scalar,
        mode=mode,
        result_device=result_device,
        staged=staged,
        compute_device=compute_device,
        out_dtype=out_dtype,
    )


def tanh_soft_clamp(x: torch.Tensor, scale) -> torch.Tensor:
    """Return Megatron's ``(scale * tanh(x.float() / scale)).to(x.dtype)``.

    ``scale`` is a real Python number or a real tensor broadcastable against
    ``x``. The output has ``x``'s dtype and the broadcast shape of ``x`` and a
    tensor ``scale``. CPU operands are staged through the accelerator and the
    result is copied back. No backward operator is provided.

    This portable launch uses one program per ``BLOCK``-sized tile. Ascend
    overrides the launch geometry only; validation, staging and copies are
    shared through ``_prepare``.
    """
    logger.debug("TRAIN TANH_SOFT_CLAMP")
    prep = _prepare(x, scale)
    if not prep.n:
        return _finish(prep)

    block = min(_MAX_TILE, triton.next_power_of_2(prep.n))
    tiles = triton.cdiv(prep.n, block)
    if tiles > _MAX_GRID:
        grid = (_MAX_GRID,)
        kernel = _tanh_soft_clamp_walk_kernel
    else:
        grid = (tiles,)
        kernel = _tanh_soft_clamp_kernel
    with _launch_context(prep.result_device, prep.compute_device):
        kernel[grid](
            prep.x_flat,
            prep.s_arg,
            prep.out_flat,
            prep.n,
            SCALE=prep.scalar,
            SCALE_MODE=prep.mode,
            BLOCK=block,
            num_warps=get_heuristics_for_num_warps(block),
        )
    return _finish(prep)
