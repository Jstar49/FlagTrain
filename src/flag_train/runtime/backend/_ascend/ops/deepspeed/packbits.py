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

try:
    from triton.language.extra.cann import extension as _cann
except ImportError:
    _cann = None

logger = logging.getLogger(__name__)

_NUM_WARPS = 4

_PACK_BLOCK_GROUPS = (8, 512)
_PACK_BLOCK_BYTES = (8, 2048)
_PACK_BLOCK_TARGET = 8

_PACK_TILES_FROM = 65536
_PACK_TILES_WIDE_FROM = 262144

_ROW_CHUNK = 65535

_PACK_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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


def _block_bytes(n_bytes_row, rows_axis):
    """Block width for this backend. See ``_PACK_BLOCK_BYTES``."""
    blocks_per_row = max(-(-_PACK_BLOCK_TARGET // rows_axis), 1)
    wanted = triton.next_power_of_2(triton.cdiv(n_bytes_row, blocks_per_row))
    return min(max(wanted, _PACK_BLOCK_BYTES[0]), _PACK_BLOCK_BYTES[1])


def _pack_tiles(n_pairs):
    """Tiles per program, or 1 for the unlooped kernel."""
    if n_pairs >= _PACK_TILES_WIDE_FROM:
        return 8
    if n_pairs >= _PACK_TILES_FROM:
        return 4
    return 1


@libentry()
@triton.jit
def _packbits_kernel(
    x_ptr,
    out_ptr,
    n,
    n_bytes_row,
    IS_FP32: tl.constexpr,
    ALIGNED: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Pack every 8 consecutive elements of a row into one byte."""
    byte_off = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    row = tl.program_id(1)
    byte_valid = byte_off < n_bytes_row

    # The block is byte-major, so the reduction runs along the *outer* axis: a
    # 32-bit integer ``tl.sum`` along the last one is serialized here -- a fixed
    # ~2.6 ns per output byte, 51x slower at 16M elements.
    j = tl.arange(0, 8)
    column = byte_off[None, :] * 8 + j[:, None]

    if ALIGNED:
        x = tl.load(x_ptr + row * n + column, mask=byte_valid[None, :], other=0.0)
    else:
        elem_valid = byte_valid[None, :] & (column < n)
        x = tl.load(x_ptr + row * n + column, mask=elem_valid, other=0.0)

    # Unsigned bitcast so the shift is logical, i.e. ``!signbit``.
    if IS_FP32:
        neg = (x.to(tl.uint32, bitcast=True) >> 31) & 1
    else:
        neg = (x.to(tl.uint16, bitcast=True) >> 15) & 1
    bit = (1 - neg) & 1

    if not ALIGNED:
        # Mask the bit, not just the load: ``other=0.0`` pads with ones, since
        # +0.0 is not negative.
        bit = tl.where(elem_valid, bit, 0)

    packed = tl.sum(bit * (1 << (7 - j))[:, None], axis=0).to(tl.uint8)
    tl.store(out_ptr + row * n_bytes_row + byte_off, packed, mask=byte_valid)


@libentry()
@triton.jit
def _packbits_16bit_kernel(
    x_ptr,
    out_ptr,
    n,
    n_pairs,
    out_stride,
    PARTIAL: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """Pack a 16-bit row two bytes at a time."""
    grp = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    row = tl.program_id(1)
    r = tl.arange(0, 16)
    pair_valid = grp < n_pairs

    # Element ``16g + r`` of the row lands in bit ``7 - (r % 8)`` of byte ``2g``
    # for ``r < 8`` and of byte ``2g + 1`` otherwise, i.e. in bit
    # ``7 - (r % 8) + 8 * (r // 8)`` of the pair.
    column = grp[None, :] * 16 + r[:, None]
    if PARTIAL:
        valid = pair_valid[None, :] & (column < n)
        x = tl.load(x_ptr + row * n + column, mask=valid, other=0.0)
        # Unsigned bitcast so the shift is logical, i.e. ``!signbit``; ``other=0.0``
        # would pad with ones, since +0.0 is not negative.
        bit = (1 - ((x.to(tl.uint16, bitcast=True) >> 15) & 1)) & 1
        bit = tl.where(valid, bit, 0)
    else:
        x = tl.load(x_ptr + row * n + column, mask=pair_valid[None, :], other=0.0)
        bit = (1 - ((x.to(tl.uint16, bitcast=True) >> 15) & 1)) & 1

    weight = (1 << (7 - (r % 8) + 8 * (r // 8))).to(tl.uint32)
    packed = tl.sum(bit * weight[:, None], axis=0).to(tl.uint16)
    tl.store(out_ptr + row * out_stride + grp, packed, mask=pair_valid)


@libentry()
@triton.jit
def _packbits_16bit_tail_kernel(
    x_ptr,
    out_ptr,
    n,
    n_bytes_row,
    first_byte,
    BLOCK_BYTES: tl.constexpr,
):
    """Pack the bytes past the last whole pair, one program per row."""
    row = tl.program_id(0)
    b = first_byte + tl.arange(0, BLOCK_BYTES)
    byte_valid = b < n_bytes_row
    j = tl.arange(0, 8)
    column = b[:, None] * 8 + j[None, :]
    valid = byte_valid[:, None] & (column < n)
    x = tl.load(x_ptr + row * n + column, mask=valid, other=0.0)
    bit = (1 - ((x.to(tl.uint16, bitcast=True) >> 15) & 1)) & 1
    bit = tl.where(valid, bit, 0)
    packed = tl.sum(bit * (1 << (7 - j)).to(tl.uint32)[None, :], axis=1).to(tl.uint8)
    tl.store(out_ptr + row * n_bytes_row + b, packed, mask=byte_valid)


@libentry()
@triton.jit
def _packbits_16bit_pipelined_kernel(
    x_ptr,
    out_ptr,
    n,
    TILES: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """The pair kernel, with ``TILES`` tiles per program under a two-stage pipeline."""
    pid = tl.program_id(0)
    row = tl.program_id(1)
    r = tl.arange(0, 16)
    n_pairs = n // 16
    weight = (1 << (7 - (r % 8) + 8 * (r // 8))).to(tl.uint32)

    for i in _cann.parallel(0, TILES, num_stages=2, bind_sub_block=True):
        grp = (pid * TILES + i) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        pair_valid = grp < n_pairs
        column = grp[None, :] * 16 + r[:, None]
        x = tl.load(x_ptr + row * n + column, mask=pair_valid[None, :], other=0.0)
        _cann.multibuffer(x, 2)
        bit = (1 - ((x.to(tl.uint16, bitcast=True) >> 15) & 1)) & 1
        packed = tl.sum(bit * weight[:, None], axis=0).to(tl.uint16)
        tl.store(out_ptr + row * n_pairs + grp, packed, mask=pair_valid)


def _pack16_blocks(n_groups):
    """Block width for the 16-bit kernels, clamped to what compiles."""
    return min(
        max(triton.next_power_of_2(n_groups), _PACK_BLOCK_GROUPS[0]),
        _PACK_BLOCK_GROUPS[1],
    )


def packbits(x, dim=-1):
    """Pack one bit per element: bit 1 where the element is not negative."""
    logger.debug("TRAIN_ASCEND PACKBITS")

    _check_input(x, "packbits", _PACK_DTYPES)
    rows, dim = _packed_axes(x, dim, "packbits")

    n = rows.shape[-1]
    n_bytes_row = (n + 7) // 8
    if rows.numel() == 0:
        # No launch when there is nothing to pack.
        return torch.empty(
            *rows.shape[:-1], n_bytes_row, dtype=torch.uint8, device=x.device
        ).movedim(-1, dim)

    # Shaped rather than flattened-then-viewed: the shape is known here.
    out = torch.empty(*rows.shape[:-1], n_bytes_row, dtype=torch.uint8, device=x.device)
    rows_axis = rows.numel() // n
    moved = rows is not x

    # Rows that end on a byte boundary pack as one flat stream: element ``i``
    # lands in bit ``7 - i % 8`` of byte ``i // 8`` either way, so the bytes come
    # out the same and the output reshapes back.
    if rows_axis > 1 and n % 8 == 0:
        rows = rows.reshape(1, -1)
        n = rows.shape[-1]
        n_bytes_row = n // 8
        rows_axis = 1

    # 16-bit elements take the pair kernel whenever the output row views as uint16.
    whole = n // 16
    tail = n % 16
    odd_row = bool(tail) and n_bytes_row % 2 == 1
    partial = bool(tail) and not odd_row
    n_out_pairs = whole + 1 if partial else whole
    take_pair = (
        x.element_size() == 2 and n_out_pairs > 0 and not (odd_row and rows_axis > 1)
    )

    rows2 = rows.reshape(rows_axis, n)
    out2 = out.reshape(rows_axis, n_bytes_row)

    # One launch per ``_ROW_CHUNK`` rows: a grid axis holds 65535 programs, and
    # every kernel here keeps rows on their own axis.
    chunks = [
        (row0, min(_ROW_CHUNK, rows_axis - row0))
        for row0 in range(0, rows_axis, _ROW_CHUNK)
    ]

    with torch_device_fn.device(x.device):
        if not take_pair:
            block_bytes = _block_bytes(n_bytes_row, rows_axis)
            grid_x = triton.cdiv(n_bytes_row, block_bytes)
            for row0, chunk in chunks:
                _packbits_kernel[(grid_x, chunk)](
                    rows2[row0 : row0 + chunk],
                    out2[row0 : row0 + chunk],
                    n,
                    n_bytes_row,
                    IS_FP32=x.element_size() == 4,
                    ALIGNED=n % 8 == 0,
                    BLOCK_BYTES=block_bytes,
                    num_warps=_NUM_WARPS,
                )
        else:
            prefix_bytes = 2 * whole
            block_groups = _pack16_blocks(n_out_pairs)
            tiles = (
                1
                if (partial or _cann is None or x.dtype == torch.bfloat16)
                else _pack_tiles(n_out_pairs)
            )
            if odd_row:
                pairs = out2.view(-1)[:prefix_bytes].view(torch.uint16)[None, :]
                out_stride = whole
            else:
                pairs = out2.view(torch.uint16)
                out_stride = n_bytes_row // 2
            for row0, chunk in chunks:
                if tiles > 1:
                    _packbits_16bit_pipelined_kernel[
                        (triton.cdiv(n_out_pairs, block_groups * tiles), chunk)
                    ](
                        rows2[row0 : row0 + chunk],
                        pairs[row0 : row0 + chunk],
                        n,
                        TILES=tiles,
                        BLOCK_GROUPS=block_groups,
                        num_warps=_NUM_WARPS,
                    )
                else:
                    _packbits_16bit_kernel[
                        (triton.cdiv(n_out_pairs, block_groups), chunk)
                    ](
                        rows2[row0 : row0 + chunk],
                        pairs[row0 : row0 + chunk],
                        n,
                        n_out_pairs,
                        out_stride,
                        PARTIAL=partial,
                        BLOCK_GROUPS=block_groups,
                        num_warps=_NUM_WARPS,
                    )
            if odd_row:
                # Never more than the one row (see above), so no chunking here.
                _packbits_16bit_tail_kernel[(rows_axis,)](
                    rows2,
                    out2,
                    n,
                    n_bytes_row,
                    prefix_bytes,
                    BLOCK_BYTES=2,
                    num_warps=1,
                )

    return out.movedim(-1, dim) if moved else out
