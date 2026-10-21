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
"""Iluvatar (IxUDA) tuned blocked flash attention.

The generic implementation is ``flag_train.deepspeed.blocked_flash`` and is what
every backend without an override runs. This one exists because three of that
launcher's assumptions about the device are false on IxUDA, and all three are
failures rather than preferences -- the tests, not the benchmark, are what
settled them. The launcher is not tuned differently here; it is rewritten against
a device the generic one rejects, sizes for, and miscompiles for:

* **The generic launcher rejects the device outright.** It mirrors the reference
  ``TORCH_CHECK`` for compute capability 8.0 or newer, which is how a CUDA-only
  launcher spells "this needs tensor cores". IxUDA presents itself as
  ``torch.cuda`` and answers that question in NVIDIA's terms -- a BI-V150 returns
  ``(7, 1)``, which on an NVIDIA part means Volta, no tensor cores -- so the
  check rejects a device its own kernels run on. The number describes an
  architecture this part is not, which is worse than no answer because it reads
  as one; the launcher here therefore hands the generic check a device that will
  not say, which is the case that check already provides for. See
  ``_capability_gate``, the one place in this module that reaches into the
  generic one.

* **The generic tiling does not fit in this device's shared memory.** It assumes
  the 163 KB an A100 has. A BI-V150 has 128 KB, and the generic tiling at
  ``head_size=256`` asks for 196608 bytes, so ``test_head_size[256]`` fails with
  ``OutOfResources`` from inside the compiler -- a message about shared memory
  that names neither the query tile nor the parameter that decides it. The tiling
  here is checked against the device's limit, with the compiler's own figure for
  it, before there is a launch to refuse, and the KV tile is narrowed until the
  tiling fits rather than the launch simply being declined.

* **The generic kernel's two-loop KV walk is miscompiled by this part's
  Triton.** The generic kernel peels the leading cache blocks that neither the
  causal mask nor the extent mask can touch and walks them unmasked. That is
  arithmetic-preserving, and here it is also worth 3-6% on the long shapes. It is
  given up anyway, because on this backend it does not compute: with the generic
  kernel unchanged except for forcing ``peeled`` to zero -- one line, the same
  arithmetic -- ``test_kv_block_size[256]`` goes from 1021 wrong elements and
  3.19 of absolute error to passing. The trigger is neither the unmasked loads
  nor the warp count nor ``num_stages``, all three of which were ruled out by
  measurement; it is two sequential loops over one accumulator. So the KV walk
  here is a single always-masked loop.

The rest is tuning, and it is this part's own. It was chosen by measurement on a
BI-V150, on the benchmark's own shapes, one sequence each:

* the KV tile is the widest that divides the cache block, and at most 64 -- see
  ``_block_n_candidates``. It is *not* the narrowest tile ``tl.dot`` accepts,
  which is what an earlier version of this file shipped, and the reason it did is
  the measurement that matters here: the two orderings are opposite, and which
  one a sweep sees depends on a parameter that is not in the kernel cache key.
  At 4096 tokens and head 128, one process per tiling, ``num_warps=4``:

      stages   BLOCK_N=16   BLOCK_N=32   BLOCK_N=64
      3             4652         8110        10046   us
      1             4026         3618         3354   us

  Three is Triton's default, and it is what the narrow tile was chosen under: a
  pipelined loop gives each stage its own copy of what the load feeds, so the
  wide tile's staging buffers scale past the device's 128 KB and the kernel
  serialises. At one stage the ordering reverses and the wide tile wins -- by
  22% at 4096 tokens and head 128, and on every other shape measured. See
  ``_NUM_STAGES``;
* the key is loaded transposed rather than loaded and transposed, worth 17% at
  4096 tokens and head 128 (3284 us against 3941) -- see the load in the kernel;
* both KV tiles are loaded before the first dot, and the value tile first --
  worth 8% at 4096 tokens and head 128 (2880 us against 3125) and 18% at 1024,
  and nothing or a percent the wrong way on the two 64-head long shapes (1441
  against 1425 at 4096, 398 against 389 at 2048). This part's Triton does not
  hoist a global load above the shared-memory traffic the dots need, so source
  order is the schedule and a tile loaded after a dot waits for it; at one
  stage there is nothing else in flight to wait behind. Issuing the value tile
  first puts its latency under the key dot and the softmax, which is the longest
  chain in the iteration -- see the loads in the kernel;
* a flat 1-D grid with the head index varying fastest and the atoms numbered
  widest-first. Programs are dispatched in linear-id order, so numbering decides
  what runs together, and on this device both halves of that are worth having:
  head-fastest beats atom-fastest by 9% across the eight shapes it was measured
  on (234 us against 368 at 1024 tokens and head 64), and widest-first beats
  narrowest-first by 1-17% wherever there is a tail to fill, costing about 6% at
  128 tokens where there is not. The flat grid is what lets both hold at once --
  a 2-D grid would have to put one of them on the slow axis;
* ``num_warps`` set by the query tile's MMA rows rather than by the accumulator's
  size -- see ``_num_warps_for``. Four warps is what an accumulator-based rule
  gives the two head sizes the benchmark uses, and four is what an earlier
  version of this file shipped; eight is faster on all eleven shapes, by 9% on
  their total (0.429 against 0.392 measured this way) and by 45% at 128 tokens
  (14.0 us against 20.3), and sixteen is much slower. One MMA row-tile per warp
  is where that lands.

What that buys against the benchmark's baseline is a speedup of 0.4 to 1.15
over its eleven shapes and two dtypes, and it is worth being plain about where
the rest of the gap is, because it is not in this file. The baseline on this
backend is the device's own ``flash_attn_varlen_func`` over a dense KV that the
benchmark builds for it outside the timed region -- a kernel written for this
part, running at 58 TFLOP/s on the 4096-token shape, against a vendor
``torch.matmul`` peak of 94.8 and a best-of-sweep Triton fp16 GEMM of 77.4.

The ceiling here is the chained dot itself, and it was measured by taking
everything else out. A probe holding this kernel's grid (512 programs), its
iteration count (16896, the same 1056 KV blocks per head), its two dots and the
softmax between them, and nothing else -- no atom bookkeeping, no masks, no
causal comparison, and the operand tiles resident in registers so that the loop
loads nothing at all -- still runs at 36 TFLOP/s (1961 us against the shipped
kernel's 2805 on this shape, for 70.9 GFLOP). Feeding those same tiles from a
fresh load each iteration, every one of them L1-resident, does not move it
(2016 us): so it is not the memory, and it is not the layout conversion of a
freshly loaded tile either. Dropping only the softmax from the probe gets 40.4
TFLOP/s. Nothing that reads a tile reaches the 58 an attention kernel written
for this part gets on the same shape, and the best any Triton form of this
kernel can do here is about 0.6 of the baseline, which is the short version of
why this file does not reach 1.

What is left is a lowering question about this part's Triton backend, and the
list of things that are not it is longer than the list of things that are.
Swept, all on this shape: ``num_stages`` 1 to 4 crossed with ``BLOCK_N`` 16, 32
and 64 (1 and 64 are both best and the rest are strictly worse -- the pipeliner
costs more than it hides at every value above 1); ``num_warps`` 4, 8 and 16
(16 collapses); ``maxnreg`` 64 to 255 (spilling below about 96, no gain above);
``TRITON_DISABLE_SME`` both ways; expressing the KV loop as a nested ``range``,
a ``tl.static_range`` and a single flat loop (identical to within noise, so the
one-trip inner loop is not the cost -- the flat one is what ships, but for a
compiler reason the loop's own comment gives, not a performance one);
dropping the load masks, which are
provably all-true here (0.3%); and halving ``BLOCK_M`` (much worse). Pointing
every iteration at a single cached cache block, which removes the DRAM traffic
and leaves the access pattern intact, recovers 3.4% -- that measurement is what
rules the memory system out, and the comparison above is what rules the
operand path out.

Everything shared is imported from the generic module so the two cannot drift:
the atom-field constants (``_ATOM_*``, which are the kernel's view of the
layout), ``_LOG2E``, and ``_check_launch_arguments`` -- the input checks are one
implementation used by both, so a backend cannot accept something another
rejects. Importing ``_check_launch_arguments`` rather than restating it is also
what keeps the reference-launcher tests working: they monkeypatch the capability
reader in the module it is defined in, and that has to be the generic module
whichever backend dispatched here -- which is also the module ``_capability_gate``
has to reach into, and the only such reach in this file.

Registration is by name. ``_iluvatar/ops/deepspeed/__init__.py`` re-exports this
module's ``blocked_flash`` straight from here, and ``SpecOpRegistrar`` writes it
over the same-named generic function in ``flag_train.deepspeed`` when that
package is imported -- the namespace the tests and the benchmark read the
operator from.
"""

import functools
import importlib
import logging

import triton
import triton.language as tl

from flag_train.deepspeed.blocked_flash import (
    _ATOM_BLOCK_OFFSET,
    _ATOM_GLOBAL_Q_IDX,
    _ATOM_KV_BLOCKS,
    _ATOM_Q_LEN,
    _ATOM_Q_START,
    _ATOM_STRIDE,
    _ATOM_TOTAL_EXTENT,
    _LOG2E,
    _check_launch_arguments,
)
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# The generic module as a *module*. Importing ``flag_train.deepspeed.blocked_flash``
# by name yields the function once registration has written this backend's
# operator over it, so the module object -- which is what ``_capability_gate``
# needs -- is reached through ``sys.modules`` like this.
_generic_module = importlib.import_module("flag_train.deepspeed.blocked_flash")

# The capability reader the generic module was imported with, which on this
# device is the one that answers in CUDA's terms. See ``_capability_gate``.
_READER_AS_IMPORTED = _generic_module._device_capability

# The widest KV tile the inner loop walks, in rows. Measured, and it is a
# ceiling rather than a preference: see ``_block_n_candidates``.
_BLOCK_N_MAX = 64

# ``tl.dot``'s floor for N. A cache block narrower than this cannot be tiled at
# all, and the launcher refuses it rather than launching into a compiler error.
_BLOCK_N_MIN = 16

# Software pipelining stages for the KV loop. One, and that is a measurement
# rather than a preference. More stages cost work rather than hiding it, at
# every value that Triton will take -- one process per value, the shipped
# kernel, 4096 tokens, a 64-row KV tile:
#
#     stages     head 128      head 64    (us, at 4096 tokens)
#     1              2807         1465
#     2              3318         1379
#     3              6076         2307
#     4              6117         1766
#
# Once each stage carries its own copy of what the load feeds, the staging is
# past the device's 128 KB of shared memory, so what should be an overlap
# becomes traffic that does not fit -- which is the three- and four-stage
# column. Two does fit and is still 18% worse on head 128, which is where the
# benchmark's weight is; it is 6% better on head 64, and one value has to serve
# both. Three is Triton's default, and it is what the KV tile in an earlier
# version of this file was chosen under -- see the module docstring for that
# measurement.
#
# One value for every tiling, and never overridden. IxUDA's Triton keys its
# kernel cache on the signature and the constexprs and *not* on ``num_stages`` or
# ``num_warps`` (``jit.py:732``), so the first compile of a given tiling fixes
# both options for the rest of the process: two tilings that disagreed would
# silently share one binary, and the loser is whichever compiled second. That is
# also why the tile sweep above is one process per tiling, and why the
# measurement the narrow tile was chosen under could not see the wide one.
_NUM_STAGES = 1

# Rows in one MMA instruction along M. This part lowers ``tl.dot`` to
# ``iluvatar_mma<instrShape = [16, 16]>``, so a query tile of ``block_m`` rows is
# ``block_m // 16`` instruction rows, and the CTA is sized to give each warp one
# of them. See ``_num_warps_for``, which has the measurement behind the floor and
# the cap.
_MMA_M = 16
_NUM_WARPS_MIN = 4
_NUM_WARPS_MAX = 8


@functools.lru_cache(maxsize=None)
def _device_properties(device_index):
    """The device's property dict, or ``None`` if it cannot be read.

    Cached, and it has to be: the driver call is host-side and this launcher
    makes it twice per invocation, against kernels that run in tens of
    microseconds -- uncached it does not size the tiling, it *is* the timing.
    Device properties do not change under a running process, so there is nothing
    to invalidate.

    ``None`` is not an error path to be avoided at all costs: the callers below
    each fall back to the behaviour that existed before the device could be
    asked at all.
    """
    try:
        from triton.runtime.driver import driver as triton_driver

        return triton_driver.active.utils.get_device_properties(device_index)
    except Exception:
        return None


def _max_shared_memory(device_index):
    """The device's shared-memory limit in bytes, or ``None`` if unreadable."""
    props = _device_properties(device_index)
    limit = props.get("max_shared_mem") if props else None
    return int(limit) if limit else None


def _num_warps_for(block_m):
    """Warps in the CTA: one MMA row-tile along M, floored at 4 and capped at 8.

    The floor is the generic launcher's, and it is a floor because below it the
    CTA has too few warps to fill the KV tile's staging. The cap is measured: at
    4096 tokens and head 128 with a 64-row KV tile, sixteen warps of 64 threads
    takes 4019 us against 2804 at eight, and it loses on every other shape too,
    including the 64-head ones (2307 against 1465 at 4096 tokens). Eight wins on
    all eleven of the benchmark's shapes, by 9% on their total, so this is one
    row-tile per warp and not a tuning per shape.

    Counting rows rather than threads is deliberate: an accumulator-based rule
    -- 64 fp32 elements per thread, the generic launcher's -- gives four warps to
    both head sizes here, and the warp is 64 threads on this part against
    NVIDIA's 32, so a thread count is a factor of two away from the CTA it
    describes. Rows do not have that problem, and ``BLOCK_M`` is the whole of
    what the measurement turned on.
    """
    return min(_NUM_WARPS_MAX, max(_NUM_WARPS_MIN, block_m // _MMA_M))


def _block_n_candidates(kv_block_size):
    """The KV tiles this cache's block size can be walked in, widest first.

    Every tile here divides the cache block, which is what keeps one tile inside
    one physical block. The inner loop's lane mask is ``kv_pos < total_extent``,
    which says nothing about the block boundary, so a tile that did *not* divide
    the block would load rows belonging to the next physical block and attend to
    KV the query was never given. Paged caches have power-of-two block sizes and
    so does every candidate, which is why this is a property of the layout
    rather than something to guard at the load.

    Which of them is *used* is decided by what fits -- see the launcher -- but
    the order is the measurement, and it is widest first. On a BI-V150, at 4096
    tokens and head 128: a 64-row tile is 3354 us, a 32-row one 3618, and a
    16-row one 4026. That is the reverse of the ordering the same sweep shows
    with the pipeliner at its default three stages (10046, 8110, 4652), because
    the pipeline is the room the narrow tile was buying -- see the table in the
    module docstring. Each figure is one process: ``num_stages`` is not in this
    backend's kernel cache key, so a single process cannot measure two of them
    (``_NUM_STAGES``). The narrow tile is not dead -- it is what a tiling that
    will not fit falls back to -- but it is no longer the answer of first
    resort.

    The ceiling is 64 rather than the block size itself. A 128-row tile does not
    fit a query tile this kernel accepts even at one stage, and where a scan of
    what does fit stops is not the same question as where the peak is: at 4096
    tokens and head 128 the wide tile is ahead of the narrow one by 22%, and the
    gap grows with the shape rather than shrinking, so the peak is at or past
    what fits rather than below it.
    """
    if kv_block_size < _BLOCK_N_MIN:
        raise ValueError(
            "blocked flash walks a cache block in KV tiles of at least %d rows, "
            "so the key cache's block size must be at least that; this one is %d"
            % (_BLOCK_N_MIN, kv_block_size)
        )
    widest = min(1 << (int(kv_block_size).bit_length() - 1), _BLOCK_N_MAX)
    candidates = []
    while widest >= _BLOCK_N_MIN:
        candidates.append(widest)
        widest //= 2
    return candidates


def _allocated_shared_memory(
    out,
    q,
    k,
    v,
    attention_atoms,
    kv_block_idx,
    softmax_scale,
    is_causal,
    kv_block_size,
    head_size,
    h_h_k_ratio,
    num_heads_q,
    num_warps,
    block_m,
    block_n,
    block_d,
):
    """Shared memory Triton allocates for one tiling, in bytes, without launching.

    ``warmup`` compiles and hands back the compiled kernel without running it,
    which is the whole point: a tiling past the device's limit still *compiles* --
    only the launch refuses it, inside ``CompiledKernel._init_handles``. So the
    allocator's own answer is readable for a tiling that could never be launched,
    and the check in front of the launch needs neither a failed launch nor a
    closed form.

    A closed form is what this is replacing, and it is not available here: the
    allocation is the allocator's answer about buffer reuse, which is a function
    of the layouts it picked for the two dots, and that is a function of the
    compiler. The expression the other backends fit to their parts -- a query
    tile against the KV loop's live set -- was checked against this one and
    under-states the wide-head tilings: it puts a 32-row tile at head 256 with
    ``BLOCK_N`` 64 at 69632 bytes, where this part's compiler allocates 196608.
    Wrong in the direction that lets a doomed launch through is the failure being
    replaced, so the figure is asked for rather than derived.

    The arguments are the launch's, because the kernel is compiled per signature
    and the strides are part of it; they are not read.

    ``LibEntry`` does not forward ``warmup``, so it is reached through the
    ``JITFunction`` it wraps -- the same one the launch compiles. That is also
    why ``num_stages`` and ``num_warps`` are passed here and not only at the
    launch: the two are the same compile, and the figure this returns is only
    the figure the launch will get if the options agree.
    """
    compiled = _blocked_flash_fwd_kernel.jit_function.warmup(
        out,
        q,
        k,
        v,
        attention_atoms,
        kv_block_idx,
        softmax_scale,
        q.stride(0),
        out.stride(0),
        k.stride(1),
        h_h_k_ratio,
        num_heads_q,
        grid=(1,),
        IS_CAUSAL=is_causal,
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    # ``warmup`` returns the same ``(kernel, hooks)`` pair a launch does.
    compiled = compiled[0] if isinstance(compiled, tuple) else compiled
    return compiled.metadata.shared


def _widest_tile_that_fits(
    needed_for_tile,
    limit,
    candidates,
    block_m,
    head_size,
):
    """The widest KV tile whose tiling the device can hold, or ``None``.

    Shared memory is the limit that actually bites here, and it is a property of
    the *device*: a BI-V150 has 128 KB against the 163 KB the generic launcher's
    tiling assumes, so the same tiling that launches elsewhere is refused here --
    deep inside the compiler, where the caller cannot act on it. The KV tile is
    the one dimension free to move: ``BLOCK_M`` has to span the widest atom or
    query rows are dropped silently, and ``BLOCK_D`` is the whole head, so a
    query tile that does not fit is not something this launcher can fix by
    trying again -- it raises instead, and says which two figures met.

    ``needed_for_tile`` is called with a candidate and returns the allocator's
    own answer for it, so nothing is derived here: the allocation is buffer
    reuse, which is a function of the layouts the compiler picked for the two
    dots.
    """
    last = None
    narrowest = None
    compile_error = None
    for candidate in candidates:
        try:
            size = needed_for_tile(candidate)
        except Exception as exc:
            # A tiling this compiler will not compile is not a tiling the device
            # can hold, so narrow and ask again rather than fail the launch.
            # Which widths compile is not monotone in the tile width -- a newer
            # FlagTree than the one this was tuned on failed in a TTGIR pass at
            # the 64-row tile of a 256-token cache block while every narrower
            # one compiled -- so the candidates below are worth trying. If none
            # of them compile, the first such error is the real explanation and
            # is what gets raised; a shared-memory figure would not be.
            if compile_error is None:
                compile_error = exc
            continue
        last = size
        narrowest = candidate
        if size <= limit:
            return candidate
    if last is None:
        raise compile_error
    raise RuntimeError(
        "blocked flash needs %d bytes of shared memory even at the narrowest KV "
        "tile this kernel compiles with (%d rows), for a %d-row query tile at "
        "head_size %d, and the device has %d. Build the atoms with a smaller "
        "q_block_size." % (last, narrowest, block_m, head_size, limit)
    )


@libentry()
@triton.jit
def _blocked_flash_fwd_kernel(
    out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    atoms_ptr,
    block_idx_ptr,
    qk_scale,
    q_row_stride,
    o_row_stride,
    k_row_stride,
    h_h_k_ratio,
    n_heads_q,
    IS_CAUSAL: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Which (atom, head) pair this program runs is a scheduling decision, and it
    # is worth about 9% of the kernel: programs are dispatched in linear-id
    # order, so the order they are numbered in decides which of them run at the
    # same time. Both halves are measured on the BI-V150, both are order-only --
    # permuting independent programs cannot change a result -- and both are
    # cheap enough to be worth taking unconditionally:
    #
    #   * heads vary fastest, worth 9% across the eight shapes measured and 36%
    #     at 1024 tokens (234 us against 368). The programs resident at any
    #     moment are then the same atom on several heads, which for
    #     grouped-query attention are reading the same cache blocks; inside a
    #     block the heads are contiguous, so memory sees one sequential stream
    #     instead of ``n_heads_q`` strided ones.
    #   * atoms run from the widest to the narrowest, worth 1-17% wherever there
    #     is a tail to fill, and 6% the wrong way at 128 tokens, where there is
    #     not. An atom's work grows with the query it carries, and under the
    #     causal layout the query grows with the index: in the benchmark's
    #     1024-token sequence each atom covers 128 queries and atom *i* walks
    #     2(i+1) cache blocks, so the last atom does eight times the work of the
    #     first. Numbering them backwards hands the long programs to the device
    #     first and leaves it the short ones to drain on -- the tail is where the
    #     imbalance would otherwise sit. For atoms built in any other order the
    #     reversal is an arbitrary permutation, which is why it costs nothing to
    #     apply unconditionally.
    #
    # The flat 1-D grid is what lets both hold at once. A 2-D ``(heads, atoms)``
    # grid would put the atoms on one of them whether or not that is the fast
    # axis, and would need ``num_atoms`` to fit in grid dimension 1, capping a
    # batch at 65535 atoms; flattening keeps the fast-varying axis under our
    # control while leaving the grid bound at 2**31 programs.
    pid = tl.program_id(0)
    hq = pid % n_heads_q
    atom = tl.num_programs(0) // n_heads_q - 1 - pid // n_heads_q

    # --- atom metadata -----------------------------------------------------
    atom_base = atoms_ptr + atom * _ATOM_STRIDE
    block_offset = tl.load(atom_base + _ATOM_BLOCK_OFFSET)
    q_start = tl.load(atom_base + _ATOM_Q_START)
    q_len = tl.load(atom_base + _ATOM_Q_LEN)
    kv_blocks = tl.load(atom_base + _ATOM_KV_BLOCKS)
    total_extent = tl.load(atom_base + _ATOM_TOTAL_EXTENT)
    global_q_idx = tl.load(atom_base + _ATOM_GLOBAL_Q_IDX)

    # Grouped-query attention: query heads are grouped in contiguous runs over one
    # KV head, so dividing maps 32 query heads and 8 KV heads to runs of four
    # (heads 0-3 read KV head 0, and so on). At h_h_k_ratio == 1 the division is
    # the identity and this is plain multi-head attention.
    hk = hq // h_h_k_ratio

    # --- queries -----------------------------------------------------------
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < q_len
    d_mask = offs_d < HEAD_SIZE

    q_rows = q_start + offs_m
    q_ptrs = q_ptr + q_rows[:, None] * q_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    # Kept in the input dtype so it matches ``k`` in the dot; the scale is applied
    # to the fp32 scores instead, which is where it belongs numerically.
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # A query token's position in the sequence, not in the tensor: a continuation
    # starts at global_q_idx, which is what the causal mask compares against.
    global_q_pos = global_q_idx + offs_m

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # One loop, every tile masked. The causal and extent comparisons below are
    # the same ones for every tile of every cache block, so a tile that lies
    # wholly before the program's first query row could skip them -- and the
    # generic kernel peels exactly that prefix, for a few percent. It is not
    # peeled here because on this part it does not compute: two sequential loops
    # over one accumulator return wrong numbers, and one always-masked loop is
    # the whole of the difference. The module docstring has the measurement.
    #
    # The blocks and the tiles inside a block are one flat loop, not a loop
    # nested in a loop. ``BLOCK_N`` is chosen to divide the cache block (see
    # ``_block_n_candidates``), so tile ``i`` belongs to block
    # ``i // TILES_PER_BLOCK`` at offset ``(i % TILES_PER_BLOCK) * BLOCK_N`` and
    # the two structures visit the same tiles in the same order. The nesting is
    # what has to go: the compiler in CI (a newer FlagTree than the one this was
    # tuned on) fails in its ``mma-reduce-thread-locality`` TTGIR pass -- a
    # PassManager error at compile time, not a miscompile -- for any cache block
    # holding more than one tile. Block sizes 32 and 64 pass there, where the
    # inner loop runs once; 256 at a 64-row tile is four tiles per block and
    # fails. Flattened, the loop that carries the accumulators is the one
    # dynamic loop that compiler already takes for the other cases. The cost is
    # one extra scalar load of the block index per tile instead of per block,
    # and a divide by a constant: the flat form was already one of the three
    # loop shapes the sweep in the module docstring found indistinguishable from
    # the nested one on this part.
    TILES_PER_BLOCK: tl.constexpr = (KV_BLOCK_SIZE + BLOCK_N - 1) // BLOCK_N
    for i in range(0, kv_blocks * TILES_PER_BLOCK):
        b = i // TILES_PER_BLOCK
        t0 = (i % TILES_PER_BLOCK) * BLOCK_N
        block = tl.load(block_idx_ptr + block_offset + b)
        block_row_base = block * k_row_stride * KV_BLOCK_SIZE

        offs_n = t0 + tl.arange(0, BLOCK_N)
        kv_pos = b * KV_BLOCK_SIZE + offs_n
        n_mask = kv_pos < total_extent

        # Both tiles are loaded here, before the first dot, and the value one
        # goes first. Source order is the schedule: this part's Triton does
        # not reorder global loads across the shared-memory traffic the dots
        # need, so a load issued after a dot waits for it, and at one
        # pipeline stage there is nothing else in flight to wait behind.
        # Issuing the value tile first puts its whole latency under the key
        # dot and the softmax, which is the longest dependency chain in the
        # iteration. Worth 8% at 4096 tokens and head 128 (2880 us against
        # 3125) and less or nothing on the short and 64-head shapes, where
        # there is less chain to hide it under; see the docstring.
        v_ptrs = (
            v_ptr
            + block_row_base
            + offs_n[:, None] * k_row_stride
            + hk * HEAD_SIZE
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # The key is loaded already transposed, as a (BLOCK_D, BLOCK_N) tile,
        # rather than loaded as (BLOCK_N, BLOCK_D) and transposed here. The
        # two are the same bytes; the difference is where the transpose
        # happens. ``tl.trans`` on this backend is a shared-memory round trip
        # with the layout the compiler picks for it (a store with
        # ``order=[0, 1]``), and the dot's operand is going through shared
        # anyway -- so the explicit load and the implicit one pay the same
        # traffic, and this one lets the load vectorise along the head
        # dimension, which is the contiguous one. Measured at 4096 tokens,
        # head 128, a 64-row tile: 3284 us against 3941.
        kt_ptrs = (
            k_ptr
            + block_row_base
            + hk * HEAD_SIZE
            + offs_d[:, None]
            + offs_n[None, :] * k_row_stride
        )
        kt = tl.load(kt_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0)

        qk = tl.dot(q, kt) * qk_scale
        if IS_CAUSAL:
            # A query attends to every KV position up to and including its own.
            qk = tl.where(kv_pos[None, :] <= global_q_pos[:, None], qk, float("-inf"))
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)
        # ``p`` is rounded into the compute dtype here. That is deliberate:
        # the bar for this operator is the reference implementation's own
        # tolerance, which this clears with room to spare, and carrying the
        # rounding residual in a second dot buys accuracy the requirement
        # does not ask for at a cost the benchmark would show.
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # A row whose keys were all masked out has no contribution to renormalize.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    o_ptrs = out_ptr + q_rows[:, None] * o_row_stride + hq * HEAD_SIZE + offs_d[None, :]
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )


def _no_capability_reported():
    """What ``_device_capability`` answers on this backend. See ``_capability_gate``."""
    return None


def _capability_gate():
    """Leave the generic launcher's capability gate to the device.

    ``_check_launch_arguments`` reads the capability from the module it is
    defined in, and that module's reader answers in CUDA's terms here: ``(7, 1)``,
    Volta in NVIDIA's numbering, on a part that has tensor cores and runs this
    kernel. A translated number is not evidence about this hardware, so the gate
    is given the "will not say" its own check already provides for -- by replacing
    the reader, which is the only way to reach a value it reads for itself. The
    registrar writes this backend's operator over the generic one the same way;
    this is that move applied to the one hook that decides whether the launch
    happens at all.

    Installed once, on the first launch from this backend, and left in place: a
    scope that restored the original would leave a window in which another
    thread's launch reads the number this device's API answers with, which is the
    number being rejected here.

    A reader that has already been replaced is not touched. A replacement is a
    deliberate statement about the device -- the tests that pin this rule replace
    it, one with a too-old number and one with none -- so it is honoured, and this
    cannot be used to hide from a limit someone else has stated.
    """
    if _generic_module._device_capability is _READER_AS_IMPORTED:
        _generic_module._device_capability = _no_capability_reported


def blocked_flash(
    out,
    q,
    k,
    v,
    attention_atoms,
    kv_block_idx,
    softmax_scale,
    is_causal=True,
    q_block_size=None,
):
    """Flash attention forward over the blocked KV-cache described by ``atoms``.

    Args:
        out (Tensor): output of shape ``[total_q, n_heads_q * head_size]``.
        q (Tensor): queries of shape ``[total_q, n_heads_q * head_size]``.
        k (Tensor): key cache of shape ``[n_blocks, block_size, n_heads_kv, head_size]``,
            contiguous in the last dimension.
        v (Tensor): value cache, same shape and layout as ``k``.
        attention_atoms (Tensor): ``[num_atoms, 8]`` int32; see the generic
            module's docstring for the layout.
        kv_block_idx (Tensor): int32 physical block indices, addressed by each
            atom's ``block_offset`` field.
        softmax_scale (float): scale applied to the query/key dot product.
        is_causal (bool): mask each query against its position in the sequence.
        q_block_size (int, optional): an upper bound on the atoms' ``q_len``,
            normally the ``q_block_size`` the atoms were built with. Supplying it
            avoids a device sync -- see below. ``None`` reads the maximum out of
            ``attention_atoms``, which is always correct and is the default.
            It is a **bound, not a hint**: an atom wider than it would have its
            extra rows masked away, silently.

    Returns:
        Tensor: ``out``, updated in place.
    """
    logger.debug("TRAIN_ILUVATAR BLOCKED_FLASH")

    # The query tile must span the widest atom. Taking that maximum off the
    # device costs a device sync -- about 70 us on this part, plus the copy --
    # which is several times the 41 us kernel it is sizing at 128 tokens. So a
    # caller who already knows the bound hands it over and the sync never
    # happens.
    head_size = k.size(-1)
    num_heads_q = q.size(-1) // head_size
    num_heads_kv = k.size(-2)
    num_atoms = attention_atoms.size(0)

    _capability_gate()
    _check_launch_arguments(q, k, v, head_size, num_heads_q, num_heads_kv)

    # k is [n_blocks, block_size, n_heads_kv, head_size] with contiguous heads, so
    # stride(0) / stride(1) is the cache's block size -- the same quantity the
    # DeepSpeed launcher leaves implicit.
    kv_block_size = k.stride(0) // k.stride(1)
    if kv_block_size * k.stride(1) != k.stride(0):
        raise ValueError("key cache must be contiguous across the block dimension")

    if q_block_size is None:
        # ``int()`` because the constants are constexpr for the kernel's benefit
        # and torch will not index a tensor with one; ``item()`` is the sync.
        q_len_max = (
            int(attention_atoms[:, int(_ATOM_Q_LEN)].max().item()) if num_atoms else 0
        )
    else:
        q_len_max = q_block_size

    # One program per (atom, query head), so the query tile has to span the widest
    # atom in the batch -- the rest mask off their padding. BLOCK_N is the KV tile
    # walked inside a single cache block. BLOCK_D is the whole head, padded up.
    BLOCK_M = max(triton.next_power_of_2(q_len_max), 16)
    BLOCK_D = triton.next_power_of_2(head_size)

    # Size the CTA to the query tile's MMA rows rather than to the accumulator's
    # size. See ``_num_warps_for``: the two disagree here, and the rows are what
    # the measurement turned on.
    device_index = q.device.index if q.device.index is not None else 0
    num_warps = _num_warps_for(BLOCK_M)

    # The KV tile, widest first, narrowed until the tiling fits -- see
    # ``_block_n_candidates`` for the ordering and ``_widest_tile_that_fits``
    # for what narrows it. A device that will not report its limit keeps the
    # widest tile: the check it would have failed is one nothing has stated, and
    # the launch itself is the thing that would refuse it.
    candidates = _block_n_candidates(kv_block_size)
    limit = _max_shared_memory(device_index)
    if limit is None:
        BLOCK_N = candidates[0]
    else:

        def needed_for_tile(candidate):
            return _allocated_shared_memory(
                out,
                q,
                k,
                v,
                attention_atoms,
                kv_block_idx,
                softmax_scale * _LOG2E,
                is_causal,
                kv_block_size,
                head_size,
                num_heads_q // num_heads_kv,
                num_heads_q,
                num_warps,
                BLOCK_M,
                candidate,
                BLOCK_D,
            )

        BLOCK_N = _widest_tile_that_fits(
            needed_for_tile, limit, candidates, BLOCK_M, head_size
        )

    _blocked_flash_fwd_kernel[(num_atoms * num_heads_q,)](
        out,
        q,
        k,
        v,
        attention_atoms,
        kv_block_idx,
        # The kernel's softmax runs on exp2, so log2(e) rides on the scale rather
        # than costing an exp() per score.
        softmax_scale * _LOG2E,
        q.stride(0),
        out.stride(0),
        k.stride(1),
        # Query heads that share one KV head; the kernel divides by it to find
        # this program's KV head.
        num_heads_q // num_heads_kv,
        num_heads_q,
        IS_CAUSAL=is_causal,
        # Both are host-side values that the kernel only ever uses as a loop
        # bound and a mask extent. As runtime scalars they force the inner KV
        # loop to stay dynamic and every load mask to carry a term that is
        # statically true; as constexpr they fold away.
        KV_BLOCK_SIZE=kv_block_size,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out
