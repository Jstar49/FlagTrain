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
"""Triton port of Megatron-LM's scaled masked softmax backward.

Megatron-LM's ``megatron/core/fusions/fused_softmax.py`` implements
``ScaledMaskedSoftmax.backward`` as

    import scaled_masked_softmax_cuda
    softmax_results, scale_t = ctx.saved_tensors
    input_grads = scaled_masked_softmax_cuda.backward(
        output_grads, softmax_results, scale_t[0]
    )

This module replaces that call with a Triton kernel. The name keeps the Apex
spelling (``scaled_masked_softmax_cuda_backward`` is
``scaled_masked_softmax_cuda.backward`` with the dot flattened to an
underscore) so a caller can replace one call site at a time.

The gradient is ``dx = scale * y * (dy - sum(dy * y))`` over the key axis,
accumulated in fp32 and rounded once to the output dtype. The mask is not an
operand: positions masked in the forward have ``y == 0`` and therefore zero
gradient, which is why the upstream signature does not take it either. The
upstream forward replaces masked scores with the finite sentinel ``-10000``
(and zeroes a fully masked row) rather than producing NaNs; the backward here
only consumes the resulting probabilities, so it needs no special case.

Megatron only selects the fused kernels for ``16 < sk <= 4096``
(``FusedScaleMaskSoftmax.is_kernel_available``); the same upper bound is
enforced here because a program reduces a row at a time.

The operator moves three streams and computes almost nothing, so it is bound
by how much of the memory bus it can keep busy: measured against a plain copy
kernel it runs at the same rate or above, so there is no arithmetic left worth
hiding behind the traffic. The tile is a block of rows of one key length --
short rows are batched so a program has enough bytes to move, long ones are
not -- and a power-of-two key length takes an unpredicated path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# ``FusedScaleMaskSoftmax.is_kernel_available`` rejects ``sk > 4096``; above
# that Megatron falls back to the torch path. One program reduces one row, so
# the same cap keeps the register tile bounded.
_MAX_KEY_LEN = 4096

# 32-bit index arithmetic halves the offset tile. It is only safe when no
# addressed element can pass 2**31; above that the logical row count times the
# key length crosses the boundary.
_I64_THRESHOLD = 1 << 31


# ``(max padded key length, rows per program, warps)``. A short row is a few
# lanes wide -- far less than one program should hold -- so those rows are
# batched, the way the upstream kernel groups batches per warp; from 2048 keys
# on a single row already fills the tile, and a taller one only costs registers
# (two rows at 4096 keys measured 5x slower, i.e. spilling). Swept on an H20-3e
# over 1..16 rows per program and 1..16 warps, each entry measured against the
# others in one process: cross-process numbers on this part move by 10% or
# more, which is larger than the differences the sweep looks for, so the
# entries are only ranked by their best-of-21. Only the 2048 row moved -- two
# rows there cost 5%; the rest of the table is a plateau. The forward reads a
# mask as well, so its table differs.
_TILING = ((128, 8, 4), (256, 4, 2), (512, 4, 2), (1024, 2, 2), (2048, 1, 2))
_DEFAULT_TILE = (1, 4)


def _tile(block_n: int) -> tuple:
    """Rows per program and warps for one tile, from the key length."""
    for max_key_len, rows, warps in _TILING:
        if block_n <= max_key_len:
            return rows, warps
    return _DEFAULT_TILE


def _check_scale(scale):
    """Accept the Python float or 0-dim tensor that Megatron passes."""
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(
                f"fused softmax scale needs one element, got {tuple(scale.shape)}"
            )
        scale = scale.item()
    return float(scale)


def _check_tensor(tensor, name):
    if tensor.ndim != 4:
        raise ValueError(f"{name} needs a 4-D tensor, got {tensor.ndim}-D")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


@libentry()
@triton.jit
def _scaled_masked_softmax_backward_kernel(
    dy_ptr,
    y_ptr,
    dx_ptr,
    n,
    total_rows,
    scale,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
):
    """``dx = scale * y * (dy - sum(dy * y))``, ``ROWS`` rows per program.

    Masked positions carry ``y == 0``, so this is the upstream backward
    formula without needing the mask: both terms vanish there.

    ``EVEN`` marks a key length that fills ``BLOCK_N`` and a row count the tile
    divides: no lane is ever out of range, and dropping the predicates lets the
    loads and stores issue unpredicated.
    """
    # ``tle.program_id`` hands back an int64; only widen the index arithmetic
    # when the offsets can actually pass 2**31, since 64-bit addresses cost a
    # second register per element of the tile.
    row_start = tle.program_id(0)
    if not I64:
        row_start = row_start.to(tl.int32)
    rows = row_start * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * n + col[None, :]

    if EVEN:
        dy = tl.load(dy_ptr + offsets).to(tl.float32)
        y = tl.load(y_ptr + offsets).to(tl.float32)
    else:
        valid = (rows[:, None] < total_rows) & (col[None, :] < n)
        dy = tl.load(dy_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    dot = tl.sum(dy * y, 1)
    dx = scale * y * (dy - dot[:, None])
    out = dx.to(dx_ptr.dtype.element_ty)
    if EVEN:
        tl.store(dx_ptr + offsets, out)
    else:
        tl.store(dx_ptr + offsets, out, mask=valid)


@torch.compiler.disable
def scaled_masked_softmax_cuda_backward(output_grads, softmax_results, scale):
    """Fused scaled-softmax gradient, Megatron's masked-attention backward.

    This is the operator behind ``ScaledMaskedSoftmax.backward`` in
    ``megatron/core/fusions/fused_softmax.py``: it corresponds to
    ``scaled_masked_softmax_cuda.backward(output_grads, softmax_results,
    scale)``. The mask is not an operand because positions that were masked in
    the forward have ``softmax_results == 0`` and therefore zero gradient.

    Args:
        output_grads (Tensor): fp16, bf16 or fp32 upstream gradient of shape
            ``[b, np, sq, sk]``. A non-contiguous tensor is copied.
        softmax_results (Tensor): forward output, same shape and dtype as
            ``output_grads``.
        scale (float | Tensor): the forward scaling factor; Megatron passes a
            0-dim tensor (``scale_t[0]``).

    Returns:
        Tensor: contiguous input gradient with the input's shape and dtype.

    Raises:
        ValueError: if the operands are not matching 4-D tensors of a supported
            dtype on the active device, or the key length exceeds 4096.
    """
    logger.debug("TRAIN SCALED_MASKED_SOFTMAX_CUDA_BACKWARD")
    _check_tensor(output_grads, "scaled masked softmax backward grad")
    _check_tensor(softmax_results, "scaled masked softmax backward results")
    if output_grads.shape != softmax_results.shape:
        raise ValueError(
            "scaled masked softmax backward needs matching shapes, got "
            f"{tuple(output_grads.shape)} and {tuple(softmax_results.shape)}"
        )
    if output_grads.dtype != softmax_results.dtype:
        raise ValueError(
            "scaled masked softmax backward needs matching dtypes, got "
            f"{output_grads.dtype} and {softmax_results.dtype}"
        )
    if output_grads.device != softmax_results.device:
        raise ValueError(
            "scaled masked softmax backward needs both operands on one device"
        )
    sk = output_grads.shape[-1]
    if sk > _MAX_KEY_LEN:
        raise ValueError(
            f"scaled masked softmax backward: key length {sk} exceeds "
            f"Megatron's fused-softmax limit of {_MAX_KEY_LEN}"
        )
    if sk == 0:
        raise ValueError("scaled masked softmax backward: key length must be positive")
    scale = _check_scale(scale)
    if output_grads.numel() == 0:
        return torch.empty_like(output_grads, memory_format=torch.contiguous_format)

    dy = output_grads.contiguous()
    y = softmax_results.contiguous()
    dx = torch.empty_like(dy)
    rows = dy.numel() // sk
    block_n = triton.next_power_of_2(sk)
    rows_per_program, num_warps = _tile(block_n)
    grid = (triton.cdiv(rows, rows_per_program),)
    with torch_device_fn.device(output_grads.device):
        _scaled_masked_softmax_backward_kernel[grid](
            dy,
            y,
            dx,
            sk,
            rows,
            scale,
            BLOCK_N=block_n,
            ROWS=rows_per_program,
            EVEN=sk == block_n and rows % rows_per_program == 0,
            I64=rows * sk >= _I64_THRESHOLD,
            num_warps=num_warps,
        )
    return dx
