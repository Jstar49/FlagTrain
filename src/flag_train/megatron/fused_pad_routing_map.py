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
"""Fused padding of a token-to-expert routing map.

Port of Megatron-LM's ``megatron.core.fusions.fused_pad_routing_map``
(``megatron/core/fusions/fused_pad_routing_map.py``). It is the kernel form of
``megatron.core.transformer.moe.moe_utils.pad_routing_map``: for every expert
column of a ``[num_tokens, num_experts]`` routing map, flip the earliest zeros
to ones until the column's set-bit count is a multiple of ``pad_multiple``. The
GroupedMLP kernels that consume the map need that alignment.

Contract, matching upstream:

* the result is ``int32`` whatever the input dtype was -- upstream materialises
  an ``.int()`` working copy -- and it is returned as a ``[num_tokens,
  num_experts]`` transposed view of a contiguous ``[num_experts, num_tokens]``
  buffer, so ``result.stride() == (1, num_tokens)``;
* a ``num_tokens == 0`` input is returned unchanged (the same object);
* padding saturates: if a column holds fewer zeros than the alignment needs,
  every zero is flipped and the column stays short of the multiple.

Upstream spends three launches on this -- a transposed bool copy, an int32 cast
and the kernel that walks the copy -- moving about 15 bytes per map element.
The copy and the cast exist only to give the kernel a row-major
``[num_experts, num_tokens]`` int32 input, so this port fuses them:

* a tile-transpose kernel reads the map coalesced along experts, writes the
  contiguous ``[num_experts, num_tokens]`` int32 buffer coalesced along tokens,
  and leaves one partial count per tile per expert;
* a per-expert kernel sums those partials for ``num_to_pad``, then walks its own
  row in order, flipping zeros *in place* and stopping as soon as it has flipped
  ``num_to_pad`` of them. The top-k maps the dispatcher feeds are mostly zeros,
  so this is typically one 512-token chunk per expert, not a second full pass
  over the row.

On Hopper-and-newer NVIDIA the flip pass is a programmatic dependent launch
(PDL): it may be dispatched before the transpose kernel drains and then waits
on the device for it, which removes the launch gap between the two. On other
vendors it is launched normally.
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

try:
    # Only the CUDA backend has PDL; another vendor's Triton may not ship it.
    from triton.language.extra.cuda import gdc_wait as _gdc_wait
except ImportError:  # pragma: no cover - exercised by non-CUDA Triton builds
    _gdc_wait = None

# These kernels are launched directly rather than through ``libentry()``. They
# carry no tunable configuration, and this operator is two dependent launches
# on the critical path of an MoE step: the entry point's per-call bookkeeping
# would cost more host time than the kernels themselves (measured at ~8 us of
# the ~19 us between their first and last launch), while the GPU work they
# dispatch is a handful of microseconds.

# Tile geometry for the transpose pass: one program owns a [BLOCK_T, BLOCK_E]
# tile of the [tokens, experts] map. The tile is contiguous along experts,
# which is the coalesced axis of the input, and transposed into the output,
# where the contiguous axis is tokens. 128x16 keeps the block resident without
# starving the grid: a 4K-token map still gets 32 tiles.
_TOKEN_TILE = 128
_EXPERT_TILE = 16
_TRANSPOSE_WARPS = 2

# Past a few million elements a map is bandwidth-bound rather than
# launch-bound, and the wider strip moves it measurably faster: the grid still
# has programs to spare, the 256-token stores are longer runs, and the halved
# tile count halves the partial counts the flip pass sums. Below that size the
# wider tile only costs occupancy.
_LARGE_MAP_ELEMENTS = 3_000_000
_TOKEN_TILE_LARGE = 256
_EXPERT_TILE_LARGE = 32
_TRANSPOSE_WARPS_LARGE = 4

# The flip pass walks a row in 512-token chunks. It has at most
# ``pad_multiple - 1`` zeros to find, and a routing map has far more zeros than
# that, so the first chunk usually holds all of them; the loop keeps going only
# when a row is unusually dense in ones, or saturates.
_FLIP_BLOCK = 512
_FLIP_WARPS = 4

# One program sums all of an expert's partial counts in a single block while
# there are at most this many tiles; longer rows sum the block at a time.
_MAX_PARTIAL_BLOCK = 1024

# Vendors whose ``gdc_wait``/``launch_pdl`` mean the CUDA ones, and whose
# capability is therefore reported in the sm_XX scale PDL's floor is written in.
_PDL_VENDORS = {"nvidia"}

_PDL_SUPPORTED = {}
_RUN_CONFIGS = {}


def _pdl_supported(device_index):
    """Whether the flip pass may use a programmatic dependent launch."""
    supported = _PDL_SUPPORTED.get(device_index)
    if supported is None:
        supported = False
        if _gdc_wait is not None and runtime_device.vendor_name in _PDL_VENDORS:
            try:
                capability = torch_device_fn.get_device_capability(device_index)
            except Exception:
                capability = None
            supported = capability is not None and capability[0] >= 9
        _PDL_SUPPORTED[device_index] = supported
    return supported


def _run_config(num_tokens, num_experts, device_index):
    """Tile geometry, warp counts and the PDL decision, cached per map shape."""
    key = (num_tokens, num_experts, device_index)
    config = _RUN_CONFIGS.get(key)
    if config is None:
        large = num_tokens * num_experts >= _LARGE_MAP_ELEMENTS
        block_t = _TOKEN_TILE_LARGE if large else _TOKEN_TILE
        # Never widen the expert axis past the map: the extra lanes would only
        # be masked out, and for a map with fewer experts than a tile the
        # expert axis is no longer the coalesced one.
        block_e = min(
            _EXPERT_TILE_LARGE if large else _EXPERT_TILE,
            triton.next_power_of_2(num_experts),
        )
        num_tiles = triton.cdiv(num_tokens, block_t)
        expert_tiles = triton.cdiv(num_experts, block_e)
        pdl = _pdl_supported(device_index)
        config = {
            "block_t": block_t,
            "block_e": block_e,
            "num_tiles": num_tiles,
            "transpose_grid": (num_tiles, expert_tiles),
            "transpose_warps": _TRANSPOSE_WARPS_LARGE if large else _TRANSPOSE_WARPS,
            "partial_block": min(triton.next_power_of_2(num_tiles), _MAX_PARTIAL_BLOCK),
            "flip_grid": (num_experts,),
            "flip_options": {"launch_pdl": True} if pdl else {},
            "pdl": pdl,
        }
        _RUN_CONFIGS[key] = config
    return config


@triton.jit
def _transpose_count_kernel(
    routing_map_ptr,
    output_ptr,
    partials_ptr,
    num_tokens,
    num_experts,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Transpose one tile into the [experts, tokens] buffer and count its ones.

    One program per (token tile, expert tile). The counts it leaves in
    ``partials`` are what the flip pass needs before it can decide how many
    zeros its row must flip -- it cannot derive them from the row it is about to
    rewrite without reading all of it first.
    """
    token_tile = tle.program_id(0)
    expert_tile = tle.program_id(1)

    tokens = token_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    experts = expert_tile * BLOCK_E + tl.arange(0, BLOCK_E)
    token_mask = tokens < num_tokens
    expert_mask = experts < num_experts

    values = tl.load(
        routing_map_ptr + tokens[:, None] * num_experts + experts[None, :],
        mask=token_mask[:, None] & expert_mask[None, :],
        other=0,
    ).to(tl.int32)

    partial = tl.sum(values, axis=0)
    tl.store(
        partials_ptr + token_tile * num_experts + experts, partial, mask=expert_mask
    )

    # ``tl.trans`` moves the tile through shared memory so each lane holds
    # tokens -- the axis that is contiguous in the output -- rather than experts,
    # the axis that is contiguous in the input.
    tl.store(
        output_ptr + experts[:, None] * num_tokens + tokens[None, :],
        tl.trans(values),
        mask=expert_mask[:, None] & token_mask[None, :],
    )


@triton.jit(do_not_specialize=["pad_multiple"])
def _flip_prefix_kernel(
    output_ptr,
    partials_ptr,
    num_tokens,
    num_experts,
    num_tiles,
    pad_multiple,
    PARTIAL_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    WAIT_GDC: tl.constexpr,
):
    """Pad one expert's row in place, one program per expert.

    The row arrives already transposed and counted; this pass only has to sum
    the partial counts, then flip the first ``num_to_pad`` zeros in token order.
    Reading and writing only the chunks that hold those zeros is what makes it
    a prefix pass rather than a second full sweep.

    The row's first chunk and the partial counts are independent reads -- only
    the flip *mask* depends on the counts -- so both are issued before either
    is consumed and the kernel pays one memory round trip instead of two. At
    these sizes that round trip is most of the pass.
    """
    if WAIT_GDC:
        # Programmatic dependent launch: the transpose pass may still be
        # running when this grid is dispatched, and its writes are the row we
        # are about to read.
        _gdc_wait()

    expert = tle.program_id(0)
    row_ptr = output_ptr + expert * num_tokens

    first_tokens = tl.arange(0, BLOCK)
    first_mask = first_tokens < num_tokens
    # Lanes past the row load 1: they rank as neither zeros nor flips, so they
    # cannot be written and cannot shift a real token's rank.
    first_values = tl.load(row_ptr + first_tokens, mask=first_mask, other=1)

    num_ones = 0
    base = 0
    while base < num_tiles:
        tiles = base + tl.arange(0, PARTIAL_BLOCK)
        counts = tl.load(
            partials_ptr + tiles * num_experts + expert,
            mask=tiles < num_tiles,
            other=0,
        )
        num_ones += tl.sum(counts, axis=0)
        base += PARTIAL_BLOCK

    remainder = num_ones % pad_multiple
    num_to_pad = tl.where(remainder != 0, pad_multiple - remainder, 0)

    flipped = 0
    start = 0
    if num_to_pad > 0:
        is_zero = first_values == 0
        zero_ranks = tl.cumsum(is_zero.to(tl.int32), axis=0)
        flip_here = is_zero & (zero_ranks <= num_to_pad)
        tl.store(
            row_ptr + first_tokens,
            tl.where(flip_here, 1, first_values),
            mask=first_mask,
        )
        flipped = tl.sum(is_zero.to(tl.int32), axis=0)
        start = BLOCK

    while (flipped < num_to_pad) & (start < num_tokens):
        tokens = start + tl.arange(0, BLOCK)
        token_mask = tokens < num_tokens
        row_values = tl.load(row_ptr + tokens, mask=token_mask, other=1)
        is_zero = row_values == 0
        zero_ranks = flipped + tl.cumsum(is_zero.to(tl.int32), axis=0)
        flip_here = is_zero & (zero_ranks <= num_to_pad)
        tl.store(
            row_ptr + tokens,
            tl.where(flip_here, 1, row_values),
            mask=token_mask,
        )
        flipped += tl.sum(is_zero.to(tl.int32), axis=0)
        start += BLOCK


def fused_pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Pad every expert column of a routing map to a multiple of ``pad_multiple``.

    Args:
        routing_map (Tensor): boolean or integer map of shape ``[num_tokens,
            num_experts]``; a non-contiguous view is accepted. It is not
            modified.
        pad_multiple (int): the alignment each expert's token count must reach.

    Returns:
        Tensor: the padded map as an ``int32`` transposed view of a contiguous
            ``[num_experts, num_tokens]`` buffer. For ``num_tokens == 0`` the
            argument itself is returned, unchanged.
    """
    logger.debug("TRAIN FUSED_PAD_ROUTING_MAP")
    assert routing_map.dim() == 2, (
        "fused_pad_routing_map expects a [num_tokens, num_experts] tensor, "
        f"got shape {tuple(routing_map.shape)}"
    )
    # Accept whatever the active backend calls its device -- 'cuda' on
    # nvidia/hygon, 'npu' on ascend -- rather than hard-coding CUDA.
    assert (
        routing_map.device.type == runtime_device.name
    ), f"fused_pad_routing_map only supports {runtime_device.name} tensors"
    assert (
        isinstance(pad_multiple, int) and pad_multiple > 0
    ), f"pad_multiple must be a positive int, got {pad_multiple!r}"

    num_tokens, num_experts = routing_map.shape
    if num_tokens == 0:
        return routing_map

    # Both kernels index the map as a dense [tokens, experts] array; a view
    # with another layout is densified once rather than read strided.
    if not routing_map.is_contiguous():
        routing_map = routing_map.contiguous()

    output_map = torch.empty(
        (num_experts, num_tokens), dtype=torch.int32, device=routing_map.device
    )
    if num_experts == 0:
        return output_map.transpose(0, 1)

    device_index = routing_map.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    config = _run_config(num_tokens, num_experts, device_index)

    # One count of ones per transpose tile, per expert. Written, not
    # accumulated, so no zeroing launch is needed to reset it.
    partials = torch.empty(
        (config["num_tiles"], num_experts),
        dtype=torch.int32,
        device=routing_map.device,
    )

    # A Triton launch runs on the *current* device, while the tensors name
    # their own; the guard is only needed when the two disagree (a process
    # driving more than one device), and it is not free on every vendor.
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == device_index
        else torch_device_fn.device(routing_map.device)
    )

    with guard:
        _transpose_count_kernel[config["transpose_grid"]](
            routing_map,
            output_map,
            partials,
            num_tokens,
            num_experts,
            BLOCK_T=config["block_t"],
            BLOCK_E=config["block_e"],
            num_warps=config["transpose_warps"],
        )
        _flip_prefix_kernel[config["flip_grid"]](
            output_map,
            partials,
            num_tokens,
            num_experts,
            config["num_tiles"],
            pad_multiple,
            PARTIAL_BLOCK=config["partial_block"],
            BLOCK=_FLIP_BLOCK,
            WAIT_GDC=config["pdl"],
            num_warps=_FLIP_WARPS,
            # PDL is a launch attribute, and passing it to a launcher that does
            # not know it is an error there rather than a no-op, so the option
            # is only present when it is used.
            **config["flip_options"],
        )

    return output_map.transpose(0, 1)
