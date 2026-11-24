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
"""Ascend tuned ``bias_gelu_impl``.

Same contract, promotion and arithmetic as
``flag_train.megatron.bias_gelu_impl`` -- the per-tile kernel is imported from
there, including the integer-width wrap that Ascend needs. What differs is the
launch:

* **Persistence, not one program per tile.** The grid is bounded by the vector
  core count (``CORE_NUM``) and each program walks its logical tiles with a
  runtime loop, so a large tensor does not pay the per-program floor once per
  256-lane tile (the portable launch's ``_BLOCK_SIZE``).
* **Tile width follows the tensor size**, not a fixed 256 lanes: ``n/CORE_NUM``
  rounded up to a power of two, clamped to [128, 16384]. 16384 is the measured
  ceiling of a low-precision tile; fp32 tiles stay below it.
* **Low-precision non-power-of-two rows get a 4096-lane tile.** At e.g. H=11008
  a larger tile would spend nearly half its lanes on a masked remainder, so the
  wider tile's UB saving is worth the extra programs.
* **libdevice tanh** (``tl_extra_shim.tanh``) instead of the portable
  sigmoid-identity/native-assembly pair: the identity costs an extra ``exp``
  chain on this backend.
* **No device exchange for a tensor already on the current NPU.** Staging a CPU
  operand still needs one.

The host-side validation, promotion and staging helpers are imported from the
generic module rather than copied; the runtime's registration runs after the
generic package is fully imported (see ``flag_train/__init__.py``), so the
import cannot re-enter the operator scan.
"""

import logging
import numbers
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_train.megatron.bias_gelu_impl import (
    _INTEGRAL_DTYPES,
    _NUM_WARPS,
    _OPERAND_DTYPES,
    _accelerator_device,
    _as_scalar_tensor,
    _bias_gelu_tile,
    _check_devices,
    _resolve_dtypes,
)
from flag_train.runtime import torch_device_fn
from flag_train.runtime.backend._ascend.utils import CORE_NUM
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _bias_gelu_persistent_kernel(
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
    # Bound physical programs to vector cores; stride over arbitrarily many
    # logical tiles. The runtime trip count avoids compile-time unrolling.
    if (BIAS_MODE == 1 and H >= 128) and not (H <= BLOCK and H & (H - 1) == 0):
        tiles = (N // H) * tl.cdiv(H, BLOCK)
    else:
        tiles = tl.cdiv(N, BLOCK)
    for tile in range(tle.program_id(0), tiles, tl.num_programs(0)):
        _bias_gelu_tile(
            X,
            B,
            Y,
            N,
            H,
            BIAS_MODE,
            BLOCK,
            tile,
            H >= 128,
            EVEN,
            INTEGRAL,
            BOOL_SUM,
            FP64,
            True,
        )


def bias_gelu_impl(input, bias):
    """Return tanh GELU(input + bias), without gradient propagation.

    Same signature, promotion and output as the generic operator; only the
    launch shape differs (see this module's docstring).
    """
    logger.debug("TRAIN_ASCEND BIAS_GELU_IMPL")
    for name, value in (("input", input), ("bias", bias)):
        if isinstance(value, torch.Tensor):
            if value.dtype not in _OPERAND_DTYPES:
                raise ValueError(f"{name} has unsupported dtype {value.dtype}")
        elif isinstance(value, numbers.Number):
            if isinstance(value, complex):
                raise ValueError(f"{name} has unsupported dtype complex")
        else:
            raise TypeError(f"{name} must be a tensor or a number")

    input_is_tensor = isinstance(input, torch.Tensor)
    bias_is_tensor = isinstance(bias, torch.Tensor)
    if not input_is_tensor and not bias_is_tensor:
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

    tile_dtype = sum_dtype if sum_dtype != torch.bool else torch.uint8
    # FlagTree wraps its integer helpers in JIT dispatch; these launch
    # dimensions are host integers and need only ordinary Python arithmetic.
    lanes = (n + CORE_NUM - 1) // CORE_NUM
    block = min(16384, max(128, 1 << (lanes - 1).bit_length()))
    # Low-precision non-power-of-two rows can waste nearly half a large tile
    # (e.g. H=11008). A 4096-lane tile reduces masked UB work.
    if (
        tile_dtype != torch.float32
        and mode == 1
        and h >= 128
        and block > 4096
        and not (h <= block and h & (h - 1) == 0)
        and ((h + block - 1) // block) * block * 8 > h * 9
    ):
        block = 4096
    if mode == 1 and h >= 128 and not (h <= block and h & (h - 1) == 0):
        tiles = (n // h) * ((h + block - 1) // block)
    else:
        tiles = (n + block - 1) // block
    grid = (min(tiles, CORE_NUM),)

    # A tensor on the current NPU needs no device exchange. CPU staging always
    # needs one.
    no_switch = (
        result_device.type != "cpu"
        and torch_device_fn.current_device() == result_device.index
    )
    integral = sum_dtype in _INTEGRAL_DTYPES
    fp64 = sum_dtype == torch.float64
    context = nullcontext() if no_switch else torch_device_fn.device(compute_device)
    with context:
        # Same LibEntry.run path as kernel[grid], without its temporary
        # closure/dict merge. warmup=False is required on cold JIT misses.
        _bias_gelu_persistent_kernel.run(
            x,
            b,
            out,
            n,
            h,
            mode,
            grid=grid,
            warmup=False,
            BLOCK=block,
            num_warps=_NUM_WARPS,
            num_stages=1,
            EVEN=n % block == 0 and (mode != 1 or h % block == 0),
            INTEGRAL=integral,
            BOOL_SUM=bool_sum,
            FP64=fp64,
        )
    return out.to(result_device) if result_device.type == "cpu" else out
