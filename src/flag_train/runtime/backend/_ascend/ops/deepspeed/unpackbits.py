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

_BLOCK_BYTES = 1024

_ROW_CHUNK = 65535

_UNPACK_DTYPES = (torch.uint8,)
_UNPACK_OUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_ROW_BLOCK_BYTES = (8, 1024)


def _row_block_bytes(n_bytes_row):
    wanted = triton.next_power_of_2(max(n_bytes_row, 1))
    return min(max(wanted, _ROW_BLOCK_BYTES[0]), _ROW_BLOCK_BYTES[1])


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


@libentry()
@triton.jit
def _unpackbits_flat_kernel(
    in_ptr,
    out_ptr,
    n_bytes,
    BLOCK_BYTES: tl.constexpr,
):
    """Expand a whole byte stream into eight floats each."""
    pid = tl.program_id(0)
    byte_off = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_valid = byte_off < n_bytes

    # A byte past the end loads as zero; the store mask keeps its expansion out.
    b = tl.load(in_ptr + byte_off, mask=byte_valid, other=0).to(tl.int32)

    # Element ``8i + j`` is bit ``7 - j`` of byte ``i``.
    j = tl.arange(0, 8)[None, :]
    acc = tl.zeros((BLOCK_BYTES, 8), tl.float32)
    for i in tl.static_range(8):
        acc += ((b >> (7 - i)) & 1).to(tl.float32)[:, None] * (j == i).to(tl.float32)

    vals = tl.reshape(acc * 2.0 - 1.0, (BLOCK_BYTES * 8,))
    out_off = pid * (BLOCK_BYTES * 8) + tl.arange(0, BLOCK_BYTES * 8)
    tl.store(
        out_ptr + out_off,
        vals.to(out_ptr.dtype.element_ty),
        mask=out_off < n_bytes * 8,
    )


@libentry()
@triton.jit
def _unpackbits_row_kernel(
    in_ptr,
    out_ptr,
    n_bytes_row,
    n_out_row,
    BLOCK_BYTES: tl.constexpr,
):
    """The flat kernel per row, for the ``count`` that trims each row on its own."""
    byte_off = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    row = tl.program_id(1)
    byte_valid = byte_off < n_bytes_row

    b = tl.load(in_ptr + row * n_bytes_row + byte_off, mask=byte_valid, other=0).to(
        tl.int32
    )

    j = tl.arange(0, 8)[None, :]
    acc = tl.zeros((BLOCK_BYTES, 8), tl.float32)
    for i in tl.static_range(8):
        acc += ((b >> (7 - i)) & 1).to(tl.float32)[:, None] * (j == i).to(tl.float32)

    vals = tl.reshape(acc * 2.0 - 1.0, (BLOCK_BYTES * 8,))
    out_off = tl.program_id(0) * (BLOCK_BYTES * 8) + tl.arange(0, BLOCK_BYTES * 8)
    tl.store(
        out_ptr + row * n_out_row + out_off,
        vals.to(out_ptr.dtype.element_ty),
        mask=out_off < n_out_row,
    )


def unpackbits(packed, dim=-1, count=None, dtype=torch.float32):
    """Expand packed bits back to floats: bit 1 becomes +1.0, bit 0 becomes -1.0."""
    logger.debug("TRAIN_ASCEND UNPACKBITS")

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
            # Nothing trims a row, so nothing needs the rows: the whole tensor is
            # one stream and the output is its expansion, one program per run of
            # bytes rather than one per row.
            flat_in = rows.reshape(-1)
            flat_out = out.reshape(-1)
            for start in range(0, rows.numel(), _ROW_CHUNK * _BLOCK_BYTES):
                take = min(_ROW_CHUNK * _BLOCK_BYTES, rows.numel() - start)
                _unpackbits_flat_kernel[(triton.cdiv(take, _BLOCK_BYTES),)](
                    flat_in[start:],
                    flat_out[start * 8 :],
                    take,
                    BLOCK_BYTES=_BLOCK_BYTES,
                    num_warps=_NUM_WARPS,
                )
        else:
            block = _row_block_bytes(n_bytes_row)
            n_blocks = triton.cdiv(n_bytes_row, block)
            n_rows = rows.numel() // n_bytes_row
            for row0 in range(0, n_rows, _ROW_CHUNK):
                chunk = min(_ROW_CHUNK, n_rows - row0)
                _unpackbits_row_kernel[(n_blocks, chunk)](
                    rows[row0 : row0 + chunk],
                    out[row0 : row0 + chunk],
                    n_bytes_row,
                    n_out_row,
                    BLOCK_BYTES=block,
                    num_warps=_NUM_WARPS,
                )

    return out if rows is packed else out.movedim(-1, dim)
