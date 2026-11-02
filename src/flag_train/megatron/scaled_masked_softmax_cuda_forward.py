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
"""Triton port of Megatron-LM's scaled masked softmax forward.

Megatron-LM's ``megatron/core/fusions/fused_softmax.py`` implements
``ScaledMaskedSoftmax.forward`` as

    import scaled_masked_softmax_cuda
    scale_t = torch.tensor([scale])
    softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])

This module replaces that call with a Triton kernel. The name keeps the Apex
spelling (``scaled_masked_softmax_cuda_forward`` is
``scaled_masked_softmax_cuda.forward`` with the dot flattened to an
underscore), so it pairs with ``scaled_masked_softmax_cuda_backward`` and a
caller can replace one call site at a time.

The scores are scaled, the keys selected by ``mask`` are pushed to ``-inf``, and
the softmax is accumulated in fp32. Upstream pushes them to the finite sentinel
``-10000`` and zeroes the masked positions afterwards; either way a masked key
ends up as an exact zero, and a row whose keys are all masked comes out as
zeros rather than as the uniform distribution ``exp(0) / sk`` would give.

The mask is a bool tensor of shape ``[b or 1, 1, sq, sk]`` where ``True`` marks
a key that must not be attended to; a leading dimension of one is shared across
the batch. Megatron only selects the fused kernels for ``16 < sk <= 4096``
(``FusedScaleMaskSoftmax.is_kernel_available``); the same upper bound is
enforced here.

One program reduces one query position for ``ROWS`` attention heads: the heads
of a query share their mask row, so loading it once per program is what keeps
the mask from being re-read once per head.

For a long row the program walks every head block of its query position in a
pipelined loop, which is what keeps the exponential off the critical path: the
tile is small enough that the next block's load can be in flight while the
current block exponentiates, and that overlap is worth more than the register
tile a whole query would need. Short rows are launch-bound instead and take the
single-tile path.
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

# log2(e), folded into the scale for the 16-bit dtypes so the kernel exponentiates
# with ``ex2``.
_LOG2E = 1.4426950408889634


# ``(max padded key length, heads per program, warps, pipeline stages)``. A
# short row is a few lanes wide -- far less than one program should hold -- so
# heads are added to the tile until it is worth a program; a long row already
# fills it, and a tile of one head keeps the register tile small enough for the
# pipeliner to hold several in flight (see ``_loop_depth``). Swept on an
# H20-3e, measuring each entry with ``do_bench``.
_TILING = (
    (128, 16, 4, 1),
    (256, 4, 2, 1),
    (512, 4, 2, 1),
    (1024, 2, 2, 4),
    (2048, 1, 2, 4),
    (4096, 1, 2, 4),
)
_DEFAULT_TILE = (1, 2, 4)

# A program walks ``LOOP`` head blocks of one query position when the row is
# long enough for the walk to pay: the pipeliner can then issue the next
# block's load behind the current block's exponential, which is what keeps the
# arithmetic off the critical path. Short rows are latency-bound instead, and
# walking them only costs the extra mask and index setup.
_LOOP_MIN_BLOCK = 1024
# Two floors on the walk: it needs enough blocks to overlap in the first place,
# and it stops short of leaving the grid below ``_MIN_GRID`` programs, because
# the device has to stay fed.
_LOOP_MIN_WORK = 16384
_MIN_GRID = 2048


def _tile(block_n: int, heads: int) -> tuple:
    """Heads per program, warps and pipeline stages for one block."""
    # ``ROWS`` is the extent of a ``tl.arange``, so it must be a power of two;
    # the table holds powers of two, and the head count is rounded down to one.
    limit = 1 << (heads.bit_length() - 1)
    for max_key_len, rows, warps, stages in _TILING:
        if block_n <= max_key_len:
            return min(rows, limit), warps, stages
    rows, warps, stages = _DEFAULT_TILE
    return min(rows, limit), warps, stages


def _loop_depth(block_n: int, batch: int, sq: int, head_blocks: int) -> int:
    """Head blocks one program walks, as a divisor of ``head_blocks``."""
    if block_n < _LOOP_MIN_BLOCK or batch * sq * head_blocks < _LOOP_MIN_WORK:
        return 1
    loop = 1
    while (
        loop * 2 <= head_blocks
        and head_blocks % (loop * 2) == 0
        and batch * sq * head_blocks // (loop * 2) >= _MIN_GRID
    ):
        loop *= 2
    return loop


def _check_scale(scale):
    """Accept the Python float or 0-dim tensor that Megatron passes."""
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(
                f"fused softmax scale needs one element, got {tuple(scale.shape)}"
            )
        scale = scale.item()
    return float(scale)


def _check_input(inputs):
    if inputs.ndim != 4:
        raise ValueError(
            f"scaled masked softmax needs a 4-D tensor, got {inputs.ndim}-D"
        )
    if inputs.dtype not in _SUPPORTED_DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise ValueError(
            f"scaled masked softmax needs a {wanted} tensor, got {inputs.dtype}"
        )
    if inputs.device.type == "cpu":
        raise ValueError(
            "scaled masked softmax needs a device tensor, not one on the CPU"
        )
    if inputs.device.type != runtime_device.name:
        raise ValueError(
            "scaled masked softmax needs a "
            f"{runtime_device.name} tensor, got {inputs.device.type}"
        )


def _check_mask(mask, inputs):
    """A bool ``[b or 1, 1, sq, sk]`` mask over the scores' batch and key axes."""
    b, _, sq, sk = inputs.shape
    if mask.ndim != 4:
        raise ValueError(
            f"scaled masked softmax mask needs a 4-D tensor, got {mask.ndim}-D"
        )
    if mask.dtype != torch.bool:
        raise ValueError(
            f"scaled masked softmax mask needs a bool tensor, got {mask.dtype}"
        )
    if mask.shape[0] not in (1, b) or mask.shape[1] != 1:
        raise ValueError(
            "scaled masked softmax mask must be [b or 1, 1, sq, sk], got "
            f"{tuple(mask.shape)}"
        )
    if mask.shape[2] != sq or mask.shape[3] != sk:
        raise ValueError(
            "scaled masked softmax mask must match the scores on the query and "
            f"key axes, got {tuple(mask.shape[2:])} for scores {sq} x {sk}"
        )
    if mask.device != inputs.device:
        raise ValueError(
            "scaled masked softmax needs the scores and the mask on one device"
        )


@libentry()
@triton.jit
def _scaled_masked_softmax_forward_kernel(
    x_ptr,
    mask_ptr,
    y_ptr,
    sq,
    heads,
    n,
    groups,
    scale,
    mask_stride_b,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    LOOP: tl.constexpr,
    STAGES: tl.constexpr,
    EVEN_COL: tl.constexpr,
    EVEN_HEAD: tl.constexpr,
    I64: tl.constexpr,
    EXP2: tl.constexpr,
):
    """Scale, mask, and softmax ``ROWS`` heads of one query position, ``LOOP`` times.

    The tile is ``[ROWS, BLOCK_N]``: the same query position in ``ROWS``
    consecutive heads, which all read the same mask row. A program walks
    ``LOOP`` head blocks of that query position -- all of them when the grid
    allows -- so the mask row is loaded once and ``tl.range`` can pipeline the
    next block's load behind the current block's exponential. Programs are
    ordered with the head group innermost: neighbouring programs either share a
    mask row, or walk consecutive query positions of one head, which sit next
    to each other in memory.
    """
    # ``tle.program_id`` hands back an int64; only widen the index arithmetic
    # when the offsets can actually pass 2**31, since 64-bit addresses cost a
    # second register per element of the tile.
    program = tle.program_id(0)
    if not I64:
        program = program.to(tl.int32)
    head_block = (program % groups) * LOOP
    rest = program // groups
    query = rest % sq
    batch = rest // sq

    col = tl.arange(0, BLOCK_N)
    # One mask row for the whole program: every head of a query position shares
    # it. Padding past a key length that is not a power of two loads as True --
    # "do not attend" -- because ``other`` is the masked value.
    mask_row = batch * mask_stride_b + query * n
    if EVEN_COL:
        attended = tl.load(mask_ptr + mask_row + col) == 0
    else:
        attended = (tl.load(mask_ptr + mask_row + col, mask=col < n, other=1) == 0) & (
            col < n
        )

    # First row of the first head block; ``step`` advances by whole blocks.
    row_base = (batch * heads + head_block * ROWS) * sq + query
    for step in tl.range(0, LOOP, num_stages=STAGES):
        head = (head_block + step) * ROWS + tl.arange(0, ROWS)
        rows = row_base + (step * ROWS + tl.arange(0, ROWS)) * sq
        offsets = rows[:, None] * n + col[None, :]

        if EVEN_COL and EVEN_HEAD:
            x = tl.load(x_ptr + offsets).to(tl.float32)
        else:
            valid = (col[None, :] < n) & (head < heads)[:, None]
            x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

        # A non-attended lane -- masked, or padding of a key length that is not
        # a power of two -- is pushed to ``-inf``, so its exponential is an
        # exact zero once the row maximum is subtracted and no second select is
        # needed. Upstream uses the finite sentinel ``-10000`` for the same
        # purpose, which underflows to the same zero.
        scores = tl.where(attended[None, :], x * scale, -float("inf"))
        max_value = tl.max(scores, 1)
        # A row with no attended key is ``-inf`` all the way across. Pivot it at
        # zero: its exponentials then underflow to zero and the row comes out as
        # zeros, which is what upstream answers, rather than as ``0 / 0`` NaNs.
        max_value = tl.where(max_value == -float("inf"), 0.0, max_value)
        # ``scale`` arrives pre-folded with log2(e) in the exp2 path, which turns
        # the exponential into the hardware ``ex2`` instruction instead of
        # libdevice's ``exp``. fp32 keeps ``exp``: it is the dtype whose
        # tolerance is tight enough for the extra rounding in the folded
        # argument to matter.
        if EXP2:
            numerator = tl.exp2(scores - max_value[:, None])
        else:
            numerator = tl.exp(scores - max_value[:, None])
        denominator = tl.sum(numerator, 1)

        # Only an all-masked row can sum to zero -- any attended lane
        # contributes exp(0) -- and it has nothing to normalise.
        inv = tl.where(denominator == 0, 0.0, 1.0 / denominator)

        out = (numerator * inv[:, None]).to(y_ptr.dtype.element_ty)
        if EVEN_COL and EVEN_HEAD:
            tl.store(y_ptr + offsets, out)
        else:
            tl.store(y_ptr + offsets, out, mask=valid)


@torch.compiler.disable
def scaled_masked_softmax_cuda_forward(inputs, mask, scale):
    """Fused scale + mask + softmax, Megatron's masked-attention forward.

    This is the operator behind ``ScaledMaskedSoftmax.forward`` in
    ``megatron/core/fusions/fused_softmax.py``: it corresponds to
    ``scaled_masked_softmax_cuda.forward(inputs, mask, scale)``.

    Args:
        inputs (Tensor): fp16, bf16 or fp32 scores of shape ``[b, np, sq, sk]``.
            A non-contiguous tensor is copied.
        mask (Tensor): bool mask of shape ``[b or 1, 1, sq, sk]`` where ``True``
            marks a key that must not be attended to. A leading dimension of 1
            is shared across the batch.
        scale (float | Tensor): multiplier applied before the softmax; Megatron
            passes a 0-dim tensor (``scale_t[0]``).

    Returns:
        Tensor: contiguous softmax results with the input's shape and dtype.
        Masked keys, and every key of a row that masks all of them, are exactly
        zero; every other row sums to one.

    Raises:
        ValueError: if the operands are not a supported 4-D tensor and a
            matching bool mask on the active device, or the key length exceeds
            4096.
    """
    logger.debug("TRAIN SCALED_MASKED_SOFTMAX_CUDA_FORWARD")
    _check_input(inputs)
    _check_mask(mask, inputs)
    b, np_, sq, sk = inputs.shape
    if sk > _MAX_KEY_LEN:
        raise ValueError(
            f"scaled masked softmax: key length {sk} exceeds Megatron's "
            f"fused-softmax limit of {_MAX_KEY_LEN}"
        )
    if sk == 0:
        raise ValueError("scaled masked softmax: key length must be positive")
    scale = _check_scale(scale)
    if inputs.numel() == 0:
        return torch.empty_like(inputs, memory_format=torch.contiguous_format)

    # Fold log2(e) into the scale so the kernel can use ``ex2``; fp32 keeps the
    # natural exponential (see the kernel).
    exp2 = inputs.dtype != torch.float32
    if exp2:
        scale *= _LOG2E

    x = inputs.contiguous()
    mask = mask.contiguous()
    out = torch.empty_like(x)
    block_n = triton.next_power_of_2(sk)
    rows, num_warps, stages = _tile(block_n, np_)
    head_blocks = triton.cdiv(np_, rows)
    loop = _loop_depth(block_n, b, sq, head_blocks)
    groups = head_blocks // loop
    grid = (b * sq * groups,)
    with torch_device_fn.device(inputs.device):
        _scaled_masked_softmax_forward_kernel[grid](
            x,
            mask,
            out,
            sq,
            np_,
            sk,
            groups,
            scale,
            0 if mask.shape[0] == 1 else mask.stride(0),
            BLOCK_N=block_n,
            ROWS=rows,
            LOOP=loop,
            STAGES=stages if loop > 1 else 1,
            EVEN_COL=sk == block_n,
            EVEN_HEAD=np_ % rows == 0,
            I64=b * np_ * sq * sk >= _I64_THRESHOLD,
            EXP2=exp2,
            num_warps=num_warps,
        )
    return out
