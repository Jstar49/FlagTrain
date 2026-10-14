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

_PACK_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_PACK_BLOCK_BYTES = (8, 256)


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


def _block_bytes(n_bytes_row):
    """Bytes per program: the row length, rounded up to a power of two and clamped."""
    return min(
        max(triton.next_power_of_2(n_bytes_row), _PACK_BLOCK_BYTES[0]),
        _PACK_BLOCK_BYTES[1],
    )


@libentry()
@triton.jit
def _packbits_kernel(
    x_ptr,
    out_ptr,
    n,
    n_bytes_row,
    n_blocks,
    IS_FP32: tl.constexpr,
    ALIGNED: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Pack every 8 consecutive elements of a row into one byte."""
    pid = tl.program_id(0)
    row = pid // n_blocks
    byte_off = (pid % n_blocks) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_valid = byte_off < n_bytes_row

    # Element ``8i+j`` of the row lands in bit ``7-j`` of byte ``i``.
    j = tl.arange(0, 8)
    column = byte_off[None, :] * 8 + j[:, None]

    if ALIGNED:
        # ``n`` is a multiple of 8, so a byte within the row is always a whole
        # byte: masking per byte is enough and nothing loaded is padding.
        x = tl.load(x_ptr + row * n + column, mask=byte_valid[None, :], other=0.0)
    else:
        elem_valid = byte_valid[None, :] & (column < n)
        x = tl.load(x_ptr + row * n + column, mask=elem_valid, other=0.0)

    # Unsigned bitcast so the shift is logical, i.e. ``!signbit``; ``x >= 0`` would
    # pack -0.0 as a 1.
    if IS_FP32:
        neg = (x.to(tl.uint32, bitcast=True) >> 31) & 1
    else:
        neg = (x.to(tl.uint16, bitcast=True) >> 15) & 1
    bit = (1 - neg) & 1

    if not ALIGNED:
        # Mask the bit, not just the load: ``other=0.0`` would pad with ones, since
        # +0.0 is not negative.
        bit = tl.where(elem_valid, bit, 0)

    packed = tl.sum(bit * (1 << (7 - j))[:, None], axis=0).to(tl.uint8)
    tl.store(out_ptr + row * n_bytes_row + byte_off, packed, mask=byte_valid)


def packbits(x, dim=-1):
    """Pack one bit per element: bit 1 where the element is not negative.

    A row of ``n`` elements becomes ``ceil(n / 8)`` bytes, MSB-first, last byte
    zero-padded. Lossy by construction: only the sign survives.

    Args:
        x (Tensor): fp16, bf16 or fp32 device tensor.
        dim (int): axis to pack along, reduced to ``ceil(n / 8)`` in place as in
            ``numpy.packbits``. Defaults to the last one, which for a 1-D input is
            the flat stream DeepSpeed's caller packed. Not the last axis means a
            move and a copy.

    Returns:
        Tensor: uint8, ``x``'s shape with ``dim`` reduced to ``ceil(n / 8)``.
    """
    logger.debug("TRAIN PACKBITS")

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
    block_bytes = _block_bytes(n_bytes_row)
    n_blocks = triton.cdiv(n_bytes_row, block_bytes)
    grid = ((rows.numel() // n) * n_blocks,)

    with torch_device_fn.device(x.device):
        _packbits_kernel[grid](
            rows,
            out,
            n,
            n_bytes_row,
            n_blocks,
            IS_FP32=x.element_size() == 4,
            ALIGNED=n % 8 == 0,
            BLOCK_BYTES=block_bytes,
            num_warps=_NUM_WARPS,
        )

    return out if rows is x else out.movedim(-1, dim)
