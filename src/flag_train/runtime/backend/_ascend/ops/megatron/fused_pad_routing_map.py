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
"""Ascend-correct fused padding of a token-to-expert routing map.

The arithmetic and the contract are ``flag_train.megatron.fused_pad_routing_map``'s,
unchanged: return int32 ``[num_tokens, num_experts]`` as a transposed view of a
contiguous ``[num_experts, num_tokens]`` buffer, flip each expert column's
earliest zeros until its set-bit count is a multiple of ``pad_multiple``, and
saturate when the zeros run out. The generic module is not retiled here but
replaced: its kernels do not compile on a 910B -- ``bishengir-compile`` aborts
in ``ConvertLinalgIRToBinary`` -- so the two passes are re-derived from the same
two-kernel shape:

* a transpose-and-count pass that materialises the ``[experts, tokens]`` int32
  map and leaves one partial count per tile per expert;
* one program per expert that sums those partials, then flips the row's first
  ``num_to_pad`` zeros in token order.

Everything else is geometry, and every choice below is measured on a 910B4:

* ``tl.sum(axis=0)`` of a ``[tokens, experts]`` tile costs about 13x what
  transposing first and reducing ``axis=1`` does (32768x8 tile: 310 us against
  4262 us for the whole pass), so both count kernels transpose.
* A map with at most 32 experts fits one tile, and its ``[tokens, experts]``
  tile is then one *contiguous* run of memory: loading it flat, reshaping and
  transposing beats the same tile loaded through computed indices by ~5x
  (32768x8: 10.4 us against 49.5 us), because the strided form pays a gather
  per lane. Wider maps have rows longer than a tile, so no flat run exists and
  they load a 2-D tile.
* ``tl.cumsum`` costs 16-20 ns per lane, sequential and independent of the tile
  shape -- a 512-lane scan puts 55 us on 256 experts' critical path, a 32-lane
  one a fifth of that -- so no scan here is wider than 32 lanes. A 512-lane
  first chunk is therefore found hierarchically: count zeros per 32-lane group,
  scan the 16 group counts, then scan only the one group that crosses
  ``num_to_pad``. For alignments of 32 or less the flip usually stops in the
  first few tokens, and a plain 64-lane first chunk is cheaper than the
  hierarchy (17.1 us against 22.2 us on 4096x64/pad=32); past that the hierarchy
  wins and the gap grows with the alignment (37.2 us against 51.0 us on
  8192x128/pad=256).
* A boolean map's counts are summed in fp32 and cast back: the sum is exact
  (0/1 lanes, at most 32768 of them, far under 2**24) and fp32 lanes are
  cheaper than int32 ones on this core -- 67.1 us against 93.9 us for the
  count pass alone on 32768x256. Integer maps keep the int32 sum, which is the
  only exact reduction for them.

Measured and rejected (kept here so they are not retried):

* A flat contiguous load for *wide* maps (``BLOCK_E == num_experts``, 16-128
  token tiles): 162-536 us against 67 us on 32768x256. The transpose survives,
  so the store runs shrink to the token tile's width while the strided form
  keeps a 512-lane run per expert.
* Dropping the count kernels' token/expert masks behind ``EVEN_T``/``EVEN_E``
  constexprs: neutral. The masks are not what the pass costs.
* Several experts per program in the flip -- a static unroll of 2 or 4 rows,
  launched on ``num_experts / 4`` programs: 1.2-1.6x slower. Rows do not
  pipeline inside a program; each keeps its own ~5 us of chain latency.
* Replacing the flip's scalar ``start`` with a broadcast group mask: neutral.
  The chunk store is not on the critical chain (removing it entirely frees
  2 us of 34 on 32768x256); the scans are.
* Locating the crossing tile from the per-tile partials and flipping the prefix
  whole: 11 us *worse* on 32768x256. The 64-lane scan over tile counts costs
  more than the 512-lane walk it removes, and the walk almost never runs --
  padded columns are ~94% zeros, so the ``num_to_pad``-th zero sits in the
  first chunk.
* A wider first chunk in the staged kernel (256 or 512 lanes) to serve large
  alignments: 13-30% worse; a single 512-lane ``tl.cumsum`` is dearer than the
  hierarchy's two short scans. The staged/hierarchical crossover is a cliff at
  64 (``_FLIP_FIRST_BLOCK``), so a threshold of 32 covers every realistic pad.

Two shapes crash the compiler rather than failing to be fast, and are branched
around: a partial-count vector one lane wide (``PB == 1``) and a row of a single
token both abort ``bishengir``. The first is avoided by never launching with
fewer than two partial lanes, the second by a dedicated single-chunk kernel.
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# A tile is capped by UB -- 192 KB on a 910B4, i.e. 16384 int32 lanes. One lane
# more is a compile-time UB-overflow, not a slow kernel.
_MAX_TILE_LANES = 16384

# At or below this many experts a tile spans the whole map, so the map's
# ``[tokens, experts]`` tile is a single contiguous run and the flat load
# applies. ``tl.reshape`` to ``(tokens, experts)`` also needs the expert count to
# be a power of two, which is checked separately when the plan is built.
_FLAT_EXPERT_MAX = 32

# Token tiles. A flat program owns ``BLOCK_T * num_experts`` lanes, so its token
# tile shrinks as the map widens; it is never widened past 2048 because that is
# all the parallelism a small map has to give.
_FLAT_TOKEN_TILE_MAX = 2048
_FLAT_TOKEN_TILE_MIN = 128

# A 2-D program owns BLOCK_T x BLOCK_E lanes with a 32-expert strip: 128 bytes of
# every row, which is one memory sector, and 16384 lanes at the 512-token tile.
_EXPERT_TILE_2D = 32

# Candidate token tiles for a 2-D program, widest first, each with the smallest
# grid it is worth using on. A wider tile halves the program count and doubles
# the store's run length (512 tokens x 4 B), but at 16384 lanes it also fills UB
# -- 64 KB of tile, a 64 KB transpose and a 64 KB fp32 copy -- so it only pays
# once the grid is several waves deep anyway: 32768x64 counts in 29.8 us on the
# 512-token tile against 33.2 us on the 256-token one, while 16384x64 counts in
# 19.1 us on 256 against 20.0 us on 512, because 64 programs is 1.6 waves. The
# narrow tile is for maps too small to give the medium one a grid.
_TILE_CANDIDATES_2D = ((512, 96), (256, 32), (128, 0))

# The flip walk's chunk past the first one. Chunks are read, scanned and written
# whole, so this bounds the scan cost of a row that is dense in ones.
_FLIP_BLOCK = 512

# Alignments at or below this stop inside one short chunk, where the plain scan
# beats the hierarchical select.
_SMALL_ALIGN_MAX = 32
_FLIP_FIRST_BLOCK = 64

# The hierarchical select's leaf: the widest scan this core does cheaply.
_HIER_SCAN_LANES = 32
_HIER_CHUNK = 512

# One program sums all of an expert's partial counts in a single block while
# there are at most this many tiles; longer rows sum the block at a time. The
# launch never drops below two lanes: a one-lane partial block miscompiles.
_MAX_PARTIAL_BLOCK = 1024
_MIN_PARTIAL_BLOCK = 2

# ``num_warps`` is a no-op on this backend (the vector core's sub-block count is
# chosen by the compiler), so one value is used everywhere.
_KERNEL_WARPS = 2

_RUN_CONFIGS = {}


def _run_config(num_tokens, num_experts):
    """Tile geometry, grid shapes and the flip kernel, cached per map shape."""
    key = (num_tokens, num_experts)
    config = _RUN_CONFIGS.get(key)
    if config is not None:
        return config

    flat = (
        num_experts <= _FLAT_EXPERT_MAX
        and num_experts & (num_experts - 1) == 0
        and num_experts > 0
    )
    # Never tile past the map: masked lanes cost a load slot for nothing.
    widest_tokens = triton.next_power_of_2(num_tokens)
    if flat:
        block_e = num_experts
        block_t = min(
            _FLAT_TOKEN_TILE_MAX,
            max(_FLAT_TOKEN_TILE_MIN, _MAX_TILE_LANES // num_experts),
            widest_tokens,
        )
    else:
        block_e = min(_EXPERT_TILE_2D, triton.next_power_of_2(num_experts))
        expert_tiles = triton.cdiv(num_experts, block_e)
        block_t = widest_tokens
        for candidate, min_programs in _TILE_CANDIDATES_2D:
            if triton.cdiv(num_tokens, candidate) * expert_tiles >= min_programs:
                block_t = min(candidate, widest_tokens)
                break

    num_tiles = triton.cdiv(num_tokens, block_t)
    config = {
        "flat": flat,
        "block_t": block_t,
        "block_e": block_e,
        "num_tiles": num_tiles,
        "count_grid": (
            (num_tiles,) if flat else (num_tiles, triton.cdiv(num_experts, block_e))
        ),
        "partial_block": max(
            _MIN_PARTIAL_BLOCK,
            min(triton.next_power_of_2(num_tiles), _MAX_PARTIAL_BLOCK),
        ),
    }
    _RUN_CONFIGS[key] = config
    return config


@triton.jit
def _transpose_count_flat_kernel(
    routing_map_ptr,
    output_ptr,
    partials_ptr,
    num_tokens,
    num_experts,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    FLOAT_COUNT: tl.constexpr,
):
    """Transpose and count one whole-map tile, loaded as a contiguous run.

    ``BLOCK_E == num_experts``, so a token tile is one flat run of the map and
    the reshape below is free; a computed-index load of the same tile costs a
    gather per lane instead.
    """
    token_tile = tl.program_id(0)
    lanes: tl.constexpr = BLOCK_T * BLOCK_E
    flat = token_tile * lanes + tl.arange(0, lanes)
    values = tl.load(
        routing_map_ptr + flat, mask=flat < num_tokens * num_experts, other=0
    ).to(tl.int32)
    # ``tl.trans`` moves the tile through shared memory so each lane holds
    # tokens -- the axis contiguous in the output -- rather than experts, the
    # axis contiguous in the input.
    transposed = tl.trans(tl.reshape(values, (BLOCK_T, BLOCK_E)))

    experts = tl.arange(0, BLOCK_E)
    tokens = token_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    if FLOAT_COUNT:
        partial = tl.sum(transposed.to(tl.float32), 1).to(tl.int32)
    else:
        partial = tl.sum(transposed, 1)
    tl.store(partials_ptr + experts * tl.num_programs(0) + token_tile, partial)
    tl.store(
        output_ptr + experts[:, None] * num_tokens + tokens[None, :],
        transposed,
        # Rank-1 masks do not broadcast here, so the expert side is spelled as
        # a comparison that is always true rather than left out.
        mask=(tokens < num_tokens)[None, :] & (experts >= 0)[:, None],
    )


@triton.jit
def _transpose_count_2d_kernel(
    routing_map_ptr,
    output_ptr,
    partials_ptr,
    num_tokens,
    num_experts,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    FLOAT_COUNT: tl.constexpr,
):
    """Transpose and count one ``[BLOCK_T, BLOCK_E]`` tile of a wide map."""
    token_tile = tl.program_id(0)
    expert_tile = tl.program_id(1)

    tokens = token_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    experts = expert_tile * BLOCK_E + tl.arange(0, BLOCK_E)
    token_mask = tokens < num_tokens
    expert_mask = experts < num_experts

    values = tl.load(
        routing_map_ptr + tokens[:, None] * num_experts + experts[None, :],
        mask=token_mask[:, None] & expert_mask[None, :],
        other=0,
    ).to(tl.int32)
    transposed = tl.trans(values)

    if FLOAT_COUNT:
        partial = tl.sum(transposed.to(tl.float32), 1).to(tl.int32)
    else:
        partial = tl.sum(transposed, 1)
    tl.store(
        partials_ptr + experts * tl.num_programs(0) + token_tile,
        partial,
        mask=expert_mask,
    )
    tl.store(
        output_ptr + experts[:, None] * num_tokens + tokens[None, :],
        transposed,
        mask=expert_mask[:, None] & token_mask[None, :],
    )


@triton.jit(do_not_specialize=["pad_multiple"])
def _flip_prefix_kernel(
    output_ptr,
    partials_ptr,
    num_tokens,
    num_tiles,
    pad_multiple,
    PARTIAL_BLOCK: tl.constexpr,
    FIRST_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pad one expert's row in place: a short first chunk, then 512 at a time.

    Used for alignments small enough that the flip stops inside the first chunk.
    The loop keeps going only when the row is unusually dense in ones, or
    saturates.
    """
    expert = tl.program_id(0)

    num_ones = 0
    base = 0
    while base < num_tiles:
        tiles = base + tl.arange(0, PARTIAL_BLOCK)
        counts = tl.load(
            partials_ptr + expert * num_tiles + tiles,
            mask=tiles < num_tiles,
            other=0,
        )
        num_ones += tl.sum(counts, 0)
        base += PARTIAL_BLOCK

    remainder = num_ones % pad_multiple
    num_to_pad = tl.where(remainder != 0, pad_multiple - remainder, 0)

    row_ptr = output_ptr + expert * num_tokens
    flipped = 0
    if flipped < num_to_pad:
        tokens = tl.arange(0, FIRST_BLOCK)
        token_mask = tokens < num_tokens
        row_values = tl.load(row_ptr + tokens, mask=token_mask, other=1)
        is_zero = row_values == 0
        zero_ranks = tl.cumsum(is_zero.to(tl.int32), 0)
        tl.store(
            row_ptr + tokens,
            tl.where(is_zero & (zero_ranks <= num_to_pad), 1, row_values),
            mask=token_mask,
        )
        flipped += tl.sum(is_zero.to(tl.int32), 0)

    start = FIRST_BLOCK
    while (flipped < num_to_pad) & (start < num_tokens):
        tokens = start + tl.arange(0, BLOCK)
        token_mask = tokens < num_tokens
        row_values = tl.load(row_ptr + tokens, mask=token_mask, other=1)
        is_zero = row_values == 0
        zero_ranks = flipped + tl.cumsum(is_zero.to(tl.int32), 0)
        tl.store(
            row_ptr + tokens,
            tl.where(is_zero & (zero_ranks <= num_to_pad), 1, row_values),
            mask=token_mask,
        )
        flipped += tl.sum(is_zero.to(tl.int32), 0)
        start += BLOCK


@triton.jit(do_not_specialize=["pad_multiple"])
def _flip_prefix_hierarchical_kernel(
    output_ptr,
    partials_ptr,
    num_tokens,
    num_tiles,
    pad_multiple,
    PARTIAL_BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    GROUPS: tl.constexpr,
    SCAN: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN_CHUNK: tl.constexpr,
):
    """Pad one expert's row in place, locating the stop hierarchically.

    One read/write of ``CHUNK`` tokens, one ``GROUPS``-wide scan of per-``SCAN``
    zero counts, and one ``SCAN``-wide cumsum inside the group that crosses
    ``num_to_pad``. That is the whole prefix for any alignment the chunk can
    satisfy -- 512 tokens per expert instead of one 512-lane scan per 512 chunks
    of a short alignment. ``EVEN_CHUNK`` drops the masks when the row is a whole
    number of chunks, which is the common shape.
    """
    expert = tl.program_id(0)

    num_ones = 0
    base = 0
    while base < num_tiles:
        tiles = base + tl.arange(0, PARTIAL_BLOCK)
        counts = tl.load(
            partials_ptr + expert * num_tiles + tiles,
            mask=tiles < num_tiles,
            other=0,
        )
        num_ones += tl.sum(counts, 0)
        base += PARTIAL_BLOCK

    remainder = num_ones % pad_multiple
    num_to_pad = tl.where(remainder != 0, pad_multiple - remainder, 0)

    row_ptr = output_ptr + expert * num_tokens
    if num_to_pad > 0:
        chunk = tl.arange(0, CHUNK)
        chunk_mask = chunk < num_tokens
        if EVEN_CHUNK:
            row_values = tl.load(row_ptr + chunk)
        else:
            # Lanes past the row read as ones: neither zeros nor flips.
            row_values = tl.load(row_ptr + chunk, mask=chunk_mask, other=1)
        is_zero = row_values == 0
        # The chunk total comes from the group counts rather than a second
        # CHUNK-wide reduction: it is the same number for the cost of a
        # GROUPS-lane sum, and the groups are needed on the common path anyway.
        group_zeros = tl.sum(tl.reshape(is_zero.to(tl.int32), (GROUPS, SCAN)), 1)
        zeros = tl.sum(group_zeros, 0)

        if zeros >= num_to_pad:
            # Which SCAN-lane group holds the boundary: groups strictly before
            # it hold fewer than num_to_pad zeros.
            groups = tl.arange(0, GROUPS)
            full_groups = tl.sum(
                (tl.cumsum(group_zeros, 0) < num_to_pad).to(tl.int32), 0
            )
            start = full_groups * SCAN
            remaining = num_to_pad - tl.sum(
                tl.where(groups < full_groups, group_zeros, 0), 0
            )

            # Everything before the boundary group is flipped whole; the
            # boundary group is scanned to find the exact lane.
            updated = tl.where(is_zero & (chunk < start), 1, row_values)
            if EVEN_CHUNK:
                tl.store(row_ptr + chunk, updated)
            else:
                tl.store(row_ptr + chunk, updated, mask=chunk_mask)

            if remaining > 0:
                inside = start + tl.arange(0, SCAN)
                inside_mask = inside < num_tokens
                inner = tl.load(row_ptr + inside, mask=inside_mask, other=1)
                inner_zero = inner == 0
                inner_ranks = tl.cumsum(inner_zero.to(tl.int32), 0)
                tl.store(
                    row_ptr + inside,
                    tl.where(inner_zero & (inner_ranks <= remaining), 1, inner),
                    mask=inside_mask,
                )
        else:
            # The chunk holds fewer zeros than the alignment needs: flip them
            # all and keep walking, exactly as the plain prefix would.
            updated = tl.where(is_zero, 1, row_values)
            if EVEN_CHUNK:
                tl.store(row_ptr + chunk, updated)
            else:
                tl.store(row_ptr + chunk, updated, mask=chunk_mask)
            flipped = zeros
            pos = CHUNK
            while (flipped < num_to_pad) & (pos < num_tokens):
                walk = pos + tl.arange(0, BLOCK)
                walk_mask = walk < num_tokens
                walk_values = tl.load(row_ptr + walk, mask=walk_mask, other=1)
                walk_zero = walk_values == 0
                walk_ranks = flipped + tl.cumsum(walk_zero.to(tl.int32), 0)
                tl.store(
                    row_ptr + walk,
                    tl.where(walk_zero & (walk_ranks <= num_to_pad), 1, walk_values),
                    mask=walk_mask,
                )
                flipped += tl.sum(walk_zero.to(tl.int32), 0)
                pos += BLOCK


@triton.jit(do_not_specialize=["pad_multiple"])
def _flip_prefix_single_chunk_kernel(
    output_ptr,
    partials_ptr,
    num_tokens,
    num_tiles,
    pad_multiple,
    PARTIAL_BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Pad a one-token-wide map: no loop, so no ``num_tokens == 1`` miscompile.

    A row of one token holds at most one zero, so the single chunk both finds
    and saturates the prefix; the walk the other kernels need cannot start.
    """
    expert = tl.program_id(0)

    num_ones = 0
    base = 0
    while base < num_tiles:
        tiles = base + tl.arange(0, PARTIAL_BLOCK)
        counts = tl.load(
            partials_ptr + expert * num_tiles + tiles,
            mask=tiles < num_tiles,
            other=0,
        )
        num_ones += tl.sum(counts, 0)
        base += PARTIAL_BLOCK

    remainder = num_ones % pad_multiple
    num_to_pad = tl.where(remainder != 0, pad_multiple - remainder, 0)

    tokens = tl.arange(0, CHUNK)
    token_mask = tokens < num_tokens
    row_ptr = output_ptr + expert * num_tokens
    row_values = tl.load(row_ptr + tokens, mask=token_mask, other=1)
    is_zero = row_values == 0
    zero_ranks = tl.cumsum(is_zero.to(tl.int32), 0)
    tl.store(
        row_ptr + tokens,
        tl.where(is_zero & (zero_ranks <= num_to_pad), 1, row_values),
        mask=token_mask,
    )


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
    logger.debug("TRAIN_ASCEND FUSED_PAD_ROUTING_MAP")
    assert routing_map.dim() == 2, (
        "fused_pad_routing_map expects a [num_tokens, num_experts] tensor, "
        f"got shape {tuple(routing_map.shape)}"
    )
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

    config = _run_config(num_tokens, num_experts)
    num_tiles = config["num_tiles"]
    partial_block = config["partial_block"]

    # One count of ones per tile, per expert. Written, not accumulated, so no
    # zeroing launch is needed to reset it.
    partials = torch.empty(
        (num_experts, num_tiles), dtype=torch.int32, device=routing_map.device
    )

    device_index = routing_map.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    # A Triton launch runs on the *current* device, while the tensors name
    # their own; the guard is only needed when the two disagree.
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == device_index
        else torch_device_fn.device(routing_map.device)
    )
    # fp32 lanes are cheaper than int32 ones here, and a boolean map's counts
    # are exact in fp32; an integer map's need not be, so it stays int32.
    float_count = routing_map.dtype == torch.bool

    with guard:
        if config["flat"]:
            _transpose_count_flat_kernel[config["count_grid"]](
                routing_map,
                output_map,
                partials,
                num_tokens,
                num_experts,
                BLOCK_T=config["block_t"],
                BLOCK_E=config["block_e"],
                FLOAT_COUNT=float_count,
                num_warps=_KERNEL_WARPS,
            )
        else:
            _transpose_count_2d_kernel[config["count_grid"]](
                routing_map,
                output_map,
                partials,
                num_tokens,
                num_experts,
                BLOCK_T=config["block_t"],
                BLOCK_E=config["block_e"],
                FLOAT_COUNT=float_count,
                num_warps=_KERNEL_WARPS,
            )

        if num_tokens == 1:
            _flip_prefix_single_chunk_kernel[(num_experts,)](
                output_map,
                partials,
                num_tokens,
                num_tiles,
                pad_multiple,
                PARTIAL_BLOCK=partial_block,
                CHUNK=_FLIP_FIRST_BLOCK,
                num_warps=_KERNEL_WARPS,
            )
        elif pad_multiple <= _SMALL_ALIGN_MAX:
            _flip_prefix_kernel[(num_experts,)](
                output_map,
                partials,
                num_tokens,
                num_tiles,
                pad_multiple,
                PARTIAL_BLOCK=partial_block,
                FIRST_BLOCK=_FLIP_FIRST_BLOCK,
                BLOCK=_FLIP_BLOCK,
                num_warps=_KERNEL_WARPS,
            )
        else:
            # A whole number of chunks, and at least one scan lane of it.
            chunk = max(
                _HIER_SCAN_LANES,
                min(_HIER_CHUNK, triton.next_power_of_2(num_tokens)),
            )
            _flip_prefix_hierarchical_kernel[(num_experts,)](
                output_map,
                partials,
                num_tokens,
                num_tiles,
                pad_multiple,
                PARTIAL_BLOCK=partial_block,
                CHUNK=chunk,
                GROUPS=chunk // _HIER_SCAN_LANES,
                SCAN=_HIER_SCAN_LANES,
                BLOCK=_FLIP_BLOCK,
                EVEN_CHUNK=num_tokens % chunk == 0,
                num_warps=_KERNEL_WARPS,
            )

    return output_map.transpose(0, 1)
