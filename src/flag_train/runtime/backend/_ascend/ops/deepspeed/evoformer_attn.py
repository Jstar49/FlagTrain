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
"""Ascend-correct evoformer attention forward.

The arithmetic is ``flag_train.deepspeed.evoformer_attn``'s, unchanged. Two
things the generic kernel does are wrong here, both measured on a 910B.

An ``or`` of two *runtime* comparisons is miscompiled as a store mask: the
generic ``lse`` mask drops whole rows, a different set each run. Either side
alone, their ``and``, and an ``or`` with a foldable constant are all clean. Since
``seq_len <= lse_dim`` the left side is a subset of the right, so one comparison
stores exactly the same rows.

A kernel containing a ``tl.dot`` cannot launch more than 32768 programs; above
exactly 2**15, whatever the grid shape or ``num_warps``, it hangs rather than
failing. Vector-only kernels at 40000 programs are fine. So the grid is capped
and the ``(batch, pair)`` axis is walked with a stride instead. That walk is what
makes the *head* axis the launch's only other pressure point, and it is chunked
across several launches when heads x query tiles would not fit -- so the operator
serves any number of heads, and any number of slices, rather than refusing them.

Everything else -- tiles, ``_LOG2E``, bias loaders, checks, shape helpers -- is
imported from the generic module so the two cannot drift.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.deepspeed.evoformer_attn import (
    _BLOCK_M,
    _BLOCK_N,
    _LOG2E,
    _batch_of,
    _bias1_row,
    _bias2_tile,
    _check_launch_arguments,
    _lse_shape,
    _normalize_bias,
    _pairs_of,
    _stride_bn,
)
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

_MAX_PROGRAMS = 32768

# Widen the grid to here when it would otherwise be one program per slice: each
# program pays ~170 ns of startup, so a natural grid of thousands of one-slice
# programs is charged it thousands of times. Mean speedup peaks shallowly at 256.
# A cap on the width, never a floor under it.
_TARGET_PROGRAMS = 256


@libentry()
@triton.jit
def _evoformer_fwd_kernel(
    out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    bias1_ptr,
    bias2_ptr,
    lse_ptr,
    qk_scale,
    seq_len,
    lse_dim,
    num_slices,
    n_pairs,
    q_stride_bn,
    q_stride_l,
    q_stride_h,
    o_stride_bn,
    o_stride_l,
    o_stride_h,
    bias2_stride_b,
    bias2_stride_h,
    lse_stride_bn,
    lse_stride_h,
    HAS_BIAS1: tl.constexpr,
    HAS_BIAS2: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NITERS: tl.constexpr,
    NEEDS_GUARD: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    slots = tl.num_programs(2)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len
    d_mask = offs_d < HEAD_SIZE

    for it in range(NITERS):
        idx = pid_n + it * slots
        # The clamp guards an overshooting walk. When NEEDS_GUARD is false every
        # idx is already in range and it is the identity, but writing it anyway
        # makes every load address in the body unfoldable and costs 1.3%.
        if NEEDS_GUARD:
            bn = tl.minimum(idx, num_slices - 1)
        else:
            bn = idx

        q_base = q_ptr + bn * q_stride_bn + head * q_stride_h
        k_base = k_ptr + bn * q_stride_bn + head * q_stride_h
        v_base = v_ptr + bn * q_stride_bn + head * q_stride_h

        if EVEN:
            q = tl.load(
                q_base + offs_m[:, None] * q_stride_l + offs_d[None, :],
                mask=d_mask[None, :],
                other=0.0,
            )
        else:
            q = tl.load(
                q_base + offs_m[:, None] * q_stride_l + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for n0 in range(0, seq_len, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len

            if EVEN:
                k = tl.load(
                    k_base + offs_n[:, None] * q_stride_l + offs_d[None, :],
                    mask=d_mask[None, :],
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_base + offs_n[:, None] * q_stride_l + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
            # The scale has to stay on the scores: folding it into q or k is a
            # bishengir miscompile (NaN, 25-33% slower).
            s = tl.dot(q, tl.trans(k)) * qk_scale

            bias2_offset = (bn // n_pairs) * bias2_stride_b + head * bias2_stride_h
            if HAS_BIAS1:
                if EVEN:
                    s += tl.load(bias1_ptr + bn * seq_len + offs_n)[None, :]
                else:
                    s += _bias1_row(bias1_ptr, bn * seq_len, offs_n, n_mask)[None, :]
            if HAS_BIAS2:
                if EVEN:
                    s += tl.load(
                        bias2_ptr
                        + bias2_offset
                        + offs_m[:, None] * seq_len
                        + offs_n[None, :]
                    ).to(tl.float32)
                else:
                    s += _bias2_tile(
                        bias2_ptr,
                        bias2_offset,
                        offs_m,
                        offs_n,
                        m_mask,
                        n_mask,
                        seq_len,
                    )

            if not EVEN:
                s = tl.where(n_mask[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2((m_i - m_new) * _LOG2E)
            p = tl.exp2((s - m_new[:, None]) * _LOG2E)

            if EVEN:
                v = tl.load(
                    v_base + offs_n[:, None] * q_stride_l + offs_d[None, :],
                    mask=d_mask[None, :],
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_base + offs_n[:, None] * q_stride_l + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
            l_i = l_i * alpha + tl.sum(p, 1)
            # The rescale belongs in the dot's accumulator init; the equivalent
            # add afterwards is 40% slower, and hoisting the cast above the sum
            # returns a wrong answer.
            acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]

        # ``and`` chains, as in the generic kernel -- not the ``or`` this backend
        # drops stores on.
        if EVEN:
            o_mask = d_mask[None, :]
        else:
            o_mask = m_mask[:, None] & d_mask[None, :]
        # Rows from seq_len to lse_dim (the next multiple of 32) carry +inf, which
        # is what upstream's backward reads; rows at or past lse_dim are never
        # written. Same rows as the generic mask, minus the ``or``.
        lse_mask = offs_m < lse_dim

        if NEEDS_GUARD:
            tail = idx < num_slices
            o_mask = o_mask & tail
            lse_mask = lse_mask & tail

        tl.store(
            out_ptr
            + bn * o_stride_bn
            + head * o_stride_h
            + offs_m[:, None] * o_stride_l
            + offs_d[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=o_mask,
        )
        if EVEN:
            tl.store(
                lse_ptr + bn * lse_stride_bn + head * lse_stride_h + offs_m,
                m_i + tl.log(l_safe),
            )
        else:
            tl.store(
                lse_ptr + bn * lse_stride_bn + head * lse_stride_h + offs_m,
                tl.where(offs_m < seq_len, m_i + tl.log(l_safe), float("inf")),
                mask=lse_mask,
            )


# Largest area first: the Cube's MAC rate rises steeply with tile area (1.5 at
# 64x64, 3.95 at 128x128). 16384 elements is the UB ceiling while the scores
# carry the scale, and of the two pairs that fit, 128x128 beats 256x64.
_TILE_PAIRS = ((128, 128), (128, 64), (64, 64))


def _tiles_for(seq_len):
    """``(block_m, block_n, even)``: largest score tile dividing ``seq_len``.

    When a pair divides the sequence every index is in range, so the masks, the
    masked bias loads and the ``-inf`` fill all drop -- removing a BLOCK_M x
    BLOCK_N predicate from the loop. No pair dividing it falls back to the
    generic tiles with every mask in place.
    """
    for block_m, block_n in _TILE_PAIRS:
        if seq_len % block_m == 0 and seq_len % block_n == 0:
            return block_m, block_n, True
    return _BLOCK_M, _BLOCK_N, False


def _grid_for(m_tiles, head_axis, num_slices):
    """``(grid, n_iters, needs_guard)``: the launch shape and how far it walks.

    ``head_axis`` is the width of this launch's head axis -- the caller's head
    chunk, not the operand's head count; the two differ only when heads had to be
    split. The z axis is as wide as ``_MAX_PROGRAMS`` allows and each program
    walks the rest; where the natural grid fits, it is the generic kernel's and
    the walk is the identity. It is then narrowed to ``_TARGET_PROGRAMS``.
    ``needs_guard`` is whether the last pass runs past the slice count.
    """
    per_slice = m_tiles * head_axis
    slots = min(num_slices, max(1, min(_TARGET_PROGRAMS, _MAX_PROGRAMS) // per_slice))
    n_iters = triton.cdiv(num_slices, slots)
    return (m_tiles, head_axis, slots), n_iters, n_iters * slots != num_slices


def evoformer_attn(q, k, v, bias1=None, bias2=None):
    """Evoformer attention forward; see ``flag_train.deepspeed.evoformer_attn``.

    Same signature, arithmetic and returned ``(out, lse)``. Only the ``lse`` store
    mask and the grid differ -- see this module's docstring.
    """
    logger.debug("TRAIN_ASCEND EVOFORMER_ATTN")

    bias1 = _normalize_bias(bias1, q)
    bias2 = _normalize_bias(bias2, q)

    # ``max_slices=None``: the (batch, pair) dimension is walked here, not placed
    # in the grid, so the z axis width is not this kernel's ceiling.
    _check_launch_arguments(q, k, v, bias1, bias2, max_slices=None)

    seq_len = q.shape[-3]
    heads = q.shape[-2]
    head_size = q.shape[-1]
    batch = _batch_of(q)
    n_pairs = _pairs_of(q)

    block_m, block_n, even = _tiles_for(seq_len)

    m_tiles = triton.cdiv(seq_len, block_m)
    if m_tiles > _MAX_PROGRAMS:
        raise ValueError(
            f"{seq_len} queries is {m_tiles} query tiles at {block_m} rows each, "
            f"but a kernel using the Cube cannot launch more than {_MAX_PROGRAMS} "
            f"programs on this device; use a shorter sequence"
        )

    out = torch.empty_like(q)
    lse = torch.empty(
        _lse_shape(batch, heads, seq_len), dtype=torch.float32, device=q.device
    )

    # bias1 is [B, N, 1, 1, L]: one L-row per (batch, pair).
    # bias2 is [B, 1, H, L, L]: one L x L matrix per (batch, head).
    bias2_stride_b = heads * seq_len * seq_len
    bias2_stride_h = seq_len * seq_len

    has_bias1 = bias1.numel() != 0
    has_bias2 = bias2.numel() != 0
    block_d = max(16, triton.next_power_of_2(head_size))
    num_warps = max(4, (block_m * block_d) // 2048)

    # A launch cannot hold more than ``_MAX_PROGRAMS`` programs, so when the head
    # axis and the query tiles together would not fit, the head axis is split
    # across launches. The split is done by *slicing the operands*, not by handing
    # the kernel a head offset: a slice of a contiguous tensor keeps every stride
    # and moves only the base pointer, so the kernel is compiled from the same
    # source either way and the one-chunk case every shape takes today is
    # untouched. Passing an offset instead is worth -1.1% on (1, 8, 1024, 4, 32)
    # -- the same unfoldable-address cost the slice clamp below is branched away
    # from.
    head_chunk = max(1, min(heads, _MAX_PROGRAMS // max(1, m_tiles)))

    with torch_device_fn.device(q.device):
        for head0 in range(0, heads, head_chunk):
            n_heads_here = min(head_chunk, heads - head0)
            head_slice = slice(head0, head0 + n_heads_here)

            q_here = q[..., head_slice, :]
            k_here = k[..., head_slice, :]
            v_here = v[..., head_slice, :]
            out_here = out[..., head_slice, :]
            lse_here = lse[..., head_slice, :]
            # An absent bias borrows the query's pointer; never dereferenced.
            bias1_here = bias1 if has_bias1 else q_here
            bias2_here = bias2[..., head_slice, :, :] if has_bias2 else q_here

            grid, n_iters, needs_guard = _grid_for(m_tiles, n_heads_here, batch)
            _evoformer_fwd_kernel[grid](
                out_here,
                q_here,
                k_here,
                v_here,
                bias1_here,
                bias2_here,
                lse_here,
                head_size**-0.5,
                seq_len,
                lse_here.shape[-1],
                batch,
                n_pairs,
                _stride_bn(q_here),
                q_here.stride(-3),
                q_here.stride(-2),
                _stride_bn(out_here),
                out_here.stride(-3),
                out_here.stride(-2),
                bias2_stride_b,
                bias2_stride_h,
                lse_here.stride(-3),
                lse_here.stride(-2),
                HAS_BIAS1=has_bias1,
                HAS_BIAS2=has_bias2,
                HEAD_SIZE=head_size,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                NITERS=n_iters,
                NEEDS_GUARD=needs_guard,
                EVEN=even,
                num_warps=num_warps,
            )
    return out, lse
