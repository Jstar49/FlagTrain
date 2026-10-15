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

logger = logging.getLogger(__name__)

_NUM_WARPS = 4

_UNPACK_DTYPES = (torch.uint8,)

_UNPACK_OUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_UNPACK_BLOCK_BYTES = (16, 256)
_UNPACK_ROWS_MIN = 16384
_UNPACK_TILE_BYTES = 1024
_UNPACK_ROWS_MAX = 128


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


def _check_out_dtype(dtype):
    if dtype not in _UNPACK_OUT_DTYPES:
        wanted = " / ".join(str(dtype) for dtype in _UNPACK_OUT_DTYPES)
        raise ValueError(f"unpackbits needs a {wanted} output dtype, got {dtype}")


def _normalize_dim(dim, ndim, name):
    if ndim == 0:
        raise ValueError(f"{name} needs at least a 1-D tensor, got a 0-d tensor")
    if not -ndim <= dim < ndim:
        raise ValueError(f"dim {dim} is out of range for a {ndim}-D tensor")
    return dim % ndim


def _packed_axes(tensor, dim, name):
    dim = _normalize_dim(dim, tensor.ndim, name)
    if dim == tensor.ndim - 1:
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} needs a contiguous tensor for the last dimension; call "
                f".contiguous() on the argument"
            )
        return tensor, dim
    return tensor.movedim(dim, -1).contiguous(), dim


def _block_bytes(n_bytes_row):
    """Bytes per program: the row length, rounded up to a power of two and clamped."""
    return min(
        max(triton.next_power_of_2(n_bytes_row), _UNPACK_BLOCK_BYTES[0]),
        _UNPACK_BLOCK_BYTES[1],
    )


def _rows_per_prog(n_rows, block_bytes):
    """Rows one batched program carries."""
    by_bytes = max(1, _UNPACK_TILE_BYTES // block_bytes)
    rows = min(n_rows, _UNPACK_ROWS_MAX, by_bytes)
    return 1 << (rows.bit_length() - 1)


@libentry()
@triton.jit
def _unpackbits_flat_kernel(
    in_ptr,
    out_ptr,
    n_bytes,
    BLOCK_BYTES: tl.constexpr,
):
    """Expand a byte stream into 8 floats each: bit 1 -> +1.0, bit 0 -> -1.0.

    Not row by row. Without a ``count`` to trim it, a row's output is exactly
    ``8 * n_bytes_row`` long, so the rows are contiguous in both tensors and where
    one ends changes nothing: a program takes one wide run of bytes and fills the
    matching run of the result.
    """
    pid = tl.program_id(0)
    byte_off = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)

    # A byte past the end loads as zero, i.e. -1.0 after decoding; the store mask is
    # what keeps it out of the result.
    b = tl.load(in_ptr + byte_off, mask=byte_off < n_bytes, other=0).to(tl.int32)

    # Element ``8i+j`` is bit ``7-j`` of byte ``i``.
    bits = (b[:, None] >> (7 - tl.arange(0, 8))[None, :]) & 1
    values = tl.reshape(bits.to(tl.float32) * 2.0 - 1.0, (BLOCK_BYTES * 8,))

    out_off = pid * (BLOCK_BYTES * 8) + tl.arange(0, BLOCK_BYTES * 8)
    tl.store(
        out_ptr + out_off,
        values.to(out_ptr.dtype.element_ty),
        mask=out_off < n_bytes * 8,
    )


@libentry()
@triton.jit
def _unpackbits_row_kernel(
    in_ptr,
    out_ptr,
    n_bytes_row,
    n_out_row,
    n_blocks,
    BLOCK_BYTES: tl.constexpr,
):
    """The flat kernel per row, for the ``count`` that trims each row on its own."""
    pid = tl.program_id(0)
    row = pid // n_blocks
    byte_off = (pid % n_blocks) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)

    b = tl.load(
        in_ptr + row * n_bytes_row + byte_off, mask=byte_off < n_bytes_row, other=0
    ).to(tl.int32)

    bits = (b[:, None] >> (7 - tl.arange(0, 8))[None, :]) & 1
    values = tl.reshape(bits.to(tl.float32) * 2.0 - 1.0, (BLOCK_BYTES * 8,))

    out_off = (pid % n_blocks) * (BLOCK_BYTES * 8) + tl.arange(0, BLOCK_BYTES * 8)
    tl.store(
        out_ptr + row * n_out_row + out_off,
        values.to(out_ptr.dtype.element_ty),
        mask=out_off < n_out_row,
    )


@libentry()
@triton.jit
def _unpackbits_rows_kernel(
    in_ptr,
    out_ptr,
    n_bytes_row,
    n_out_row,
    n_rows,
    BLOCK_BYTES: tl.constexpr,
    ROWS: tl.constexpr,
):
    """The row kernel for rows short enough that one program covers several of them."""
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    byte_off = tl.arange(0, BLOCK_BYTES)
    row_mask = rows < n_rows

    b = tl.load(
        in_ptr + rows[:, None] * n_bytes_row + byte_off[None, :],
        mask=row_mask[:, None] & (byte_off[None, :] < n_bytes_row),
        other=0,
    ).to(tl.int32)

    bits = (b[:, :, None] >> (7 - tl.arange(0, 8))[None, None, :]) & 1
    values = tl.reshape(bits.to(tl.float32) * 2.0 - 1.0, (ROWS, BLOCK_BYTES * 8))

    out_off = tl.arange(0, BLOCK_BYTES * 8)
    tl.store(
        out_ptr + rows[:, None] * n_out_row + out_off[None, :],
        values.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None] & (out_off[None, :] < n_out_row),
    )


def unpackbits(packed, dim=-1, count=None, dtype=torch.float32):
    """Expand packed bits back to floats: bit 1 becomes +1.0, bit 0 becomes -1.0.

    Inverse only up to the padding: the zero bits ``packbits`` writes past the end
    of a row decode to -1.0, so pass ``count`` to trim the tail back to the element
    count that was packed.

    Args:
        packed (Tensor): uint8 device tensor, as returned by ``packbits``.
        dim (int): axis to expand along, growing to 8 times its length in place.
        count (int, optional): keep only the first ``count`` values along ``dim``.
            ``None`` keeps all of them, and is the fastest: nothing then has to
            know where one row stops and the next starts.
        dtype (torch.dtype): dtype of the result, fp16 / bf16 / fp32.

    Returns:
        Tensor: ``packed``'s shape with ``dim`` grown to ``8 * n`` (or ``count``).
    """
    logger.debug("TRAIN UNPACKBITS")

    _check_input(packed, "unpackbits", _UNPACK_DTYPES)
    _check_out_dtype(dtype)
    rows, dim = _packed_axes(packed, dim, "unpackbits")

    n_bytes_row = rows.shape[-1]
    n_out_row = n_bytes_row * 8 if count is None else count
    if not 0 <= n_out_row <= n_bytes_row * 8:
        raise ValueError(
            f"count {n_out_row} is outside 0..{n_bytes_row * 8}, the values packed "
            f"along dim {dim}"
        )
    if rows.numel() == 0 or n_out_row == 0:
        # No launch when there is nothing to expand.
        return torch.empty(
            *rows.shape[:-1], n_out_row, dtype=dtype, device=packed.device
        ).movedim(-1, dim)

    out = torch.empty(*rows.shape[:-1], n_out_row, dtype=dtype, device=packed.device)

    with torch_device_fn.device(packed.device):
        if count is None:
            # Nothing trims a row, so nothing needs the rows: ``rows.numel()`` is the
            # whole stream and the output is its expansion. One program per run of
            # bytes instead of one per row, which is where the short rows lose.
            block_bytes = _UNPACK_BLOCK_BYTES[1]
            _unpackbits_flat_kernel[(triton.cdiv(rows.numel(), block_bytes),)](
                rows,
                out,
                rows.numel(),
                BLOCK_BYTES=block_bytes,
                num_warps=_NUM_WARPS,
            )
        else:
            block_bytes = _block_bytes(n_bytes_row)
            n_blocks = triton.cdiv(n_bytes_row, block_bytes)
            n_rows = rows.numel() // n_bytes_row
            # Only a row that fits in one block, and only once there are rows
            # enough for the program count to be what the kernel is bound by,
            # leaves anything for a wider program to pick up.
            if n_blocks == 1 and n_rows >= _UNPACK_ROWS_MIN:
                # One block holds the whole row, so the block the row kernel would
                # have taken is already the width to spread rows over.
                rows_per_prog = _rows_per_prog(n_rows, block_bytes)
                _unpackbits_rows_kernel[(triton.cdiv(n_rows, rows_per_prog),)](
                    rows,
                    out,
                    n_bytes_row,
                    n_out_row,
                    n_rows,
                    BLOCK_BYTES=block_bytes,
                    ROWS=rows_per_prog,
                    num_warps=_NUM_WARPS,
                )
            else:
                _unpackbits_row_kernel[(n_rows * n_blocks,)](
                    rows,
                    out,
                    n_bytes_row,
                    n_out_row,
                    n_blocks,
                    BLOCK_BYTES=block_bytes,
                    num_warps=_NUM_WARPS,
                )

    return out if rows is packed else out.movedim(-1, dim)
