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
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

_NUM_WARPS = 1

# Bytes one program may hold live.
_TILE_BYTES = 24 * 1024

# Programs the grid should keep in flight when the token count allows it.
_MIN_PROGRAMS = 32

_INDEX_DTYPES = (torch.int32, torch.int64)

_PROBS_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


# Elements the (ROWS, TOPK_BLOCK, EXPERTS_BLOCK) selection tile may hold.
_TOP_K_TILE_ELEMS = 8192


def _rows_per_prog(n_rows, topk_block, experts_block, elem_bytes, index_bytes):
    """Rows one program carries, floored to a power of 2.

    Bounded by the per-row byte budget, the minimum program count, and the
    top-k-wide selection tile's element budget.
    """
    per_row_bytes = experts_block * elem_bytes + 2 * index_bytes + elem_bytes
    rows = max(1, min(n_rows, _TILE_BYTES // per_row_bytes))
    rows = min(rows, max(1, n_rows // _MIN_PROGRAMS))
    rows = min(rows, max(1, _TOP_K_TILE_ELEMS // (topk_block * experts_block)))
    return 1 << (rows.bit_length() - 1)


@lru_cache(maxsize=64)
def _launch_plan(num_tokens, num_experts, topk, elem_bytes, index_bytes):
    """Block sizes and grid for one shape, cached because routing repeats it."""
    experts_block = triton.next_power_of_2(num_experts)
    topk_block = triton.next_power_of_2(topk)
    rows_per_prog = _rows_per_prog(
        num_tokens, topk_block, experts_block, elem_bytes, index_bytes
    )
    return (
        experts_block,
        topk_block,
        rows_per_prog,
        (triton.cdiv(num_tokens, rows_per_prog),),
    )


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


@libentry()
@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_ptr,
    multihot_ptr,
    multihot_probs_ptr,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    EXPERTS_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Turn ``ROWS`` tokens' top-k expert ids into ``num_experts``-wide rows.

    One (ROWS, TOPK_BLOCK, EXPERTS_BLOCK) tile is compared against the expert
    rows and reduced along top-k, so every output element is written once
    without the overlapping zero-then-scatter stores.
    """
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_mask = row < num_tokens

    expert_col = tl.arange(0, EXPERTS_BLOCK)
    expert_mask = expert_col < NUM_EXPERTS

    probs_dtype = multihot_probs_ptr.dtype.element_ty
    topk_col = tl.arange(0, TOPK_BLOCK)
    in_off = row[:, None] * TOPK + topk_col[None, :]
    in_mask = row_mask[:, None] & (topk_col[None, :] < TOPK)
    indices = tl.load(indices_ptr + in_off, mask=in_mask, other=NUM_EXPERTS)
    probs = tl.load(probs_ptr + in_off, mask=in_mask, other=0.0)
    # -1 (Megatron's padding) and other out-of-range ids map to the
    # never-matching column NUM_EXPERTS. fp32 compares avoid this backend's
    # scalar int64 path and stay exact below 2**24, INT64_MIN included.
    indices_f = indices.to(tl.float32)
    in_range = (indices_f >= 0.0) & (indices_f < NUM_EXPERTS * 1.0)
    indices = tl.where(in_range, indices.to(tl.int32), NUM_EXPERTS)

    selected = indices[:, :, None] == expert_col[None, None, :]
    hit = tl.sum(selected.to(tl.int32), axis=1) > 0
    probs_out = tl.sum(tl.where(selected, probs[:, :, None], 0.0), axis=1)

    out_off = row[:, None] * NUM_EXPERTS + expert_col[None, :]
    out_mask = row_mask[:, None] & expert_mask[None, :]
    tl.store(multihot_ptr + out_off, hit.to(tl.int1), mask=out_mask)
    tl.store(multihot_probs_ptr + out_off, probs_out.to(probs_dtype), mask=out_mask)


def _indices_to_multihot_forward(indices, probs, num_of_local_experts):
    """Allocate the outputs and launch the conversion kernel."""
    num_tokens, topk = indices.shape
    device = indices.device

    multihot = torch.empty(
        (num_tokens, num_of_local_experts), dtype=torch.bool, device=device
    )
    multihot_probs = torch.empty(
        (num_tokens, num_of_local_experts), dtype=probs.dtype, device=device
    )

    if num_tokens > 0 and topk > 0 and num_of_local_experts > 0:
        experts_block, topk_block, rows_per_prog, grid = _launch_plan(
            num_tokens,
            num_of_local_experts,
            topk,
            probs.element_size(),
            indices.element_size(),
        )
        with torch_device_fn.device(device):
            _indices_to_multihot_kernel[grid](
                indices,
                probs,
                multihot,
                multihot_probs,
                num_tokens,
                NUM_EXPERTS=num_of_local_experts,
                TOPK=topk,
                TOPK_BLOCK=topk_block,
                EXPERTS_BLOCK=experts_block,
                ROWS=rows_per_prog,
                num_warps=_NUM_WARPS,
            )
    elif num_tokens > 0 and topk == 0 and num_of_local_experts > 0:
        # Nothing to scatter, but every expert column is still unselected.
        multihot.zero_()
        multihot_probs.zero_()

    return multihot, multihot_probs


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    """Convert MoE top-k routing indices to a multihot expert map.

    Each row of ``indices`` marks its ``topk`` experts in a
    ``num_of_local_experts``-wide row and puts each probability in its expert's
    column. ``-1`` padding and indices outside ``[0, num_of_local_experts)`` are
    ignored. Forward-only: neither result carries a gradient.

    Args:
        indices (Tensor): int32 / int64 device tensor of shape ``(tokens, topk)``.
        probs_indices (Tensor): fp16 / bf16 / fp32 / fp64 tensor of the same
            shape, holding the routing probability of each index.
        num_of_local_experts (int): width of the expert map.

    Returns:
        tuple[Tensor, Tensor]: the bool map ``(tokens, num_of_local_experts)`` and
        the probabilities laid out in the same shape.
    """
    logger.debug("TRAIN_ASCEND FUSED_INDICES_TO_MULTIHOT")

    _check_input(indices, "fused_indices_to_multihot (indices)", _INDEX_DTYPES)
    _check_input(
        probs_indices, "fused_indices_to_multihot (probs_indices)", _PROBS_DTYPES
    )

    if indices.ndim != 2:
        raise ValueError(
            f"fused_indices_to_multihot needs a 2-D indices tensor, got "
            f"{indices.ndim}-D"
        )
    if probs_indices.shape != indices.shape:
        raise ValueError(
            "fused_indices_to_multihot needs probs_indices to have the same shape "
            f"as indices, got {tuple(probs_indices.shape)} and "
            f"{tuple(indices.shape)}"
        )
    if probs_indices.device != indices.device:
        raise ValueError(
            "fused_indices_to_multihot needs both tensors on the same device, got "
            f"{indices.device} and {probs_indices.device}"
        )
    if not isinstance(num_of_local_experts, int) or num_of_local_experts < 0:
        raise ValueError(
            "fused_indices_to_multihot needs a non-negative integer "
            f"num_of_local_experts, got {num_of_local_experts!r}"
        )

    # Row addressing assumes contiguous strides.
    indices = indices.contiguous()
    probs_indices = probs_indices.contiguous()

    return _indices_to_multihot_forward(indices, probs_indices, num_of_local_experts)
