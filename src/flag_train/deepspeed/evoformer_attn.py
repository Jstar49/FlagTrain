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
"""Evoformer attention forward: non-causal attention with two additive biases.

Port of upstream's ``_attention`` -- the forward kernel of DeepSpeed's
``DS4Sci_EvoformerAttention`` (``deepspeed/ops/deepspeed4science/evoformer_attn.py``,
kernel in ``csrc/deepspeed4science/evoformer_attn/attention_cu.cu``).

    ``_attention`` -> ``evoformer_attn``

**The backward is not ported.** Upstream's module holds four things in one file --
``_attention``, ``attention_bwd``, ``EvoformerFusedAttention`` and the public
``DS4Sci_EvoformerAttention`` -- and the last three exist to produce gradients.
This port has only the first, so ``evoformer_attn_bwd`` and the trainable entry
are *absent* rather than present-and-raising: a name that fails when called is a
worse answer than no name, because it still reads as a promise.

The operator is AlphaFold's attention, as OpenFold runs it: every query attends to
every key (there is no causal mask), and the attention matrix is biased by two
tensors that exist because Evoformer's attention is over *pairs*:

    q, k, v   ``[B, N, L, H, D]`` -- ``N`` is the pair dimension
    bias1     ``[B, N, 1, 1, L]`` -- one value per key position, shared by every
                                     query row and every head
    bias2     ``[B, 1, H, L, L]`` -- a full ``L x L`` bias per head, shared by every
                                     ``N``

    S = q k^T / sqrt(D) + bias1 + bias2
    P = softmax(S)
    out = P v

Both biases are optional; pass an empty tensor for the one that is absent, which is
what upstream does (and what makes its kernel dispatch to one of four template
instantiations -- here the same split is two ``tl.constexpr`` flags).

Why the bias shapes are what they are: ``bias1`` is where a key-padding mask lives
(the upstream test builds it as ``1e9 * (mask - 1)``), and ``bias2`` is where the
pair representation enters. Both are added to the *scores*, in fp32, before the
softmax -- not to the output.

**This port is not bit-identical to upstream**, and does not try to be: upstream
accumulates in CUTLASS tensor-core fragments and broadcasts biases through shared
memory, while this is a Triton online-softmax kernel. What it does match is the
arithmetic contract -- the same fp32 score accumulation, the same ``exp2``-based
softmax, the same ``lse`` definition -- so results agree to the tolerance upstream's
own test asserts (``1e-2`` fp16, ``5e-2`` bf16; see ``evoformer_attn.md``).

What upstream's ``EvoformerFusedAttention.forward`` does around the kernel is not
copied either: it normalises absent biases (``torch.tensor([])``) and calls
``.contiguous()`` on everything, and that part *is* here -- ``evoformer_attn``
accepts ``None`` for a bias and requires contiguous operands, because those are
what the kernel needs to be handed. What is not here is the ``biases`` list
handling of the public entry, which appends to the caller's list
(``biases.append`` at evoformer_attn.py:92-95); with the public entry unported,
nothing takes a list.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# log2(e): the softmax runs on exp2, so this converts the natural-log domain the
# scores and ``lse`` live in into the exponent the hardware has an instruction for.
#
# ``tl.constexpr`` rather than a plain float because the kernels read it too:
# Triton resolves a global inside ``@triton.jit`` only when it was instantiated as
# constexpr, and rejects a bare float at compile time.
_LOG2E = tl.constexpr(1.4426950408889634)

# The key tile. 64 is upstream's ``kKeysPerBlock`` -- and, sized against the
# ``L x L`` bias2 tile this kernel loads per iteration, it is also the point past
# which the extra rows only add shared memory: bias2 is read in full whatever the
# tile is, so a wider tile buys fewer iterations at the cost of the accumulators
# that have to stay live across them.
#
# Neither tile has been swept for this operator. They are upstream's numbers, and
# the kernel meets its baseline by 2.4-3.1x with them; a sweep is what would say
# whether it can do better (see evoformer_attn.md §5).
_BLOCK_N = 64

# The query tile, at the upstream kernel's ``kQueriesPerBlock`` -- which is also
# the alignment its ``lse`` padding is defined against (``kAlignLSE = 32``, and 64
# is a multiple of it, so the last query tile always covers the padding).
_BLOCK_M = 64


def _batch_of(q):
    """Number of (batch, pair) slices the kernel indexes as one flat dimension.

    Contiguous, the flat ``(batch, pair)`` dimension advances by exactly one
    ``L x H x D`` slice, which is what the kernels' ``bn`` indexes.
    """
    return q.numel() // (q.shape[-3] * q.shape[-2] * q.shape[-1])


def _pairs_of(q):
    """``N``, the pair dimension a bias2 is shared across.

    Only defined when a bias is present (``_check_bias_shapes`` pins the query to
    five dimensions then). Without one nothing divides by it: the kernel reads it
    inside the ``HAS_BIAS2`` branch, which is compiled out when there is no bias2.
    """
    return q.shape[1] if q.dim() >= 4 else 1


def _stride_bn(tensor):
    """Stride of the flat ``(batch, pair)`` dimension; 0 when the tensor has none.

    A 3-D operand is one ``(batch, pair)`` slice, so its ``bn`` is always 0 and the
    stride never multiplies anything.
    """
    return tensor.stride(-4) if tensor.dim() >= 4 else 0


@triton.jit
def _bias1_row(bias1_ptr, bias1_offset, offs_n, n_mask):
    """One ``bias1`` row, broadcast over the query axis by the caller.

    ``bias1`` is ``[B, N, 1, 1, L]``: contiguous, that is a row vector per
    ``(batch, pair)`` whose ``L`` values are added to *every* query row and every
    head. Upstream gets the same effect from a tile iterator with a row stride of
    zero (``BroadcastA::load``); here the row is simply broadcast.
    """
    return tl.load(bias1_ptr + bias1_offset + offs_n, mask=n_mask, other=0.0).to(
        tl.float32
    )


@triton.jit
def _bias2_tile(bias2_ptr, bias2_offset, offs_m, offs_n, m_mask, n_mask, seq_len):
    """The ``[BLOCK_M, BLOCK_N]`` corner of the ``bias2`` matrix for one head.

    ``bias2`` is ``[B, 1, H, L, L]``; ``bias2_offset`` has already folded in the
    batch and head, so what is left is a row-major ``L x L`` matrix.
    """
    ptrs = bias2_ptr + bias2_offset + offs_m[:, None] * seq_len + offs_n[None, :]
    return tl.load(ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(
        tl.float32
    )


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
):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    bn = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len
    d_mask = offs_d < HEAD_SIZE

    # q, k and v share a shape and a layout, so one set of strides serves all three.
    q_base = q_ptr + bn * q_stride_bn + head * q_stride_h
    k_base = k_ptr + bn * q_stride_bn + head * q_stride_h
    v_base = v_ptr + bn * q_stride_bn + head * q_stride_h

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

        k = tl.load(
            k_base + offs_n[:, None] * q_stride_l + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * qk_scale

        # Both biases land on the scores, in fp32. Upstream does the same in its
        # epilogue (`accum[idx] = accum[idx] * p.scale + bias`), which is also why
        # its softmax is told the scale is 1.0 when a bias is present.
        if HAS_BIAS1:
            s += _bias1_row(bias1_ptr, bn * seq_len, offs_n, n_mask)[None, :]
        if HAS_BIAS2:
            s += _bias2_tile(
                bias2_ptr,
                (bn // n_pairs) * bias2_stride_b + head * bias2_stride_h,
                offs_m,
                offs_n,
                m_mask,
                n_mask,
                seq_len,
            )

        # A key past the end of the sequence contributes nothing. Its score is
        # -inf, so it neither wins the row max nor adds to the denominator.
        s = tl.where(n_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2((m_i - m_new) * _LOG2E)
        p = tl.exp2((s - m_new[:, None]) * _LOG2E)

        v = tl.load(
            v_base + offs_n[:, None] * q_stride_l + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # A query row whose keys were all masked out has nothing to renormalise.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    o_base = out_ptr + bn * o_stride_bn + head * o_stride_h
    tl.store(
        o_base + offs_m[:, None] * o_stride_l + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )

    # lse is the log-sum-exp of the row's scores, in nats and in fp32, at the same
    # ``m + log(sum exp(s - m))`` that upstream computes from its running max and
    # running sum. Rows from ``seq_len`` up to ``lse_dim`` (the next multiple of 32)
    # are written as +inf: upstream pads the buffer that way, and the padding is
    # what its backward reads instead of bounding every load.
    # Rows at or past ``lse_dim`` are never written -- the buffer stops there.
    lse_ptrs = lse_ptr + bn * lse_stride_bn + head * lse_stride_h + offs_m
    tl.store(
        lse_ptrs,
        tl.where(offs_m < seq_len, m_i + tl.log(l_safe), float("inf")),
        mask=(offs_m < seq_len) | (offs_m < lse_dim),
    )


def _device_capability():
    """``(major, minor)`` of the current device, or ``None`` if it will not say.

    ``None`` means "cannot tell", and the caller then leaves the device alone
    rather than guessing a limit it did not state.
    """
    try:
        return torch_device_fn.get_device_capability(torch_device_fn.current_device())
    except Exception:
        return None


# The per-dimension grid limit. A launch's z axis cannot be wider than this on
# any backend's driver, which is what makes it the flattened (batch, pair)
# dimension's ceiling for a kernel that puts that dimension in the grid.
_GRID_Z_MAX = 65535


def _check_launch_arguments(q, k, v, bias1, bias2, max_slices=_GRID_Z_MAX):
    """Reject inputs the kernels cannot serve.

    Mirrors the assertions upstream makes around ``_attention`` -- in it and in
    the C++ ``check_supported`` it calls -- so a caller gets the same diagnosis
    upstream gives rather than a failure inside the Triton compiler.

    Three of them cannot be literal on a port that runs on several vendors:

    * upstream asserts tensors are CUDA; here the test is that they are not on
      the CPU, which is what "must be a device tensor" means anywhere else;
    * upstream asserts Ampere-or-newer through ``CheckArch``. That still holds,
      but it is only checked when the device actually reports a capability -- a
      backend that will not say keeps working, because inventing a limit it never
      stated is how a port rejects a device that was fine;
    * upstream predicates on pointer and stride alignment (``check_supported``)
      because its kernel loads 128-bit vectors. This port's Triton loads do not
      need that, but it *does* require contiguous operands (see the docstring of
      ``evoformer_attn``).

    ``max_slices`` is the one argument here that is a property of the *kernel*
    rather than of the inputs. A kernel that puts the flattened (batch, pair)
    dimension in its grid is capped at that axis's width; one that *walks* the
    dimension instead pays the cap on the walk's width and can serve any number
    of slices at all. A backend doing the latter passes ``None``.
    """
    if q.dim() < 3:
        raise ValueError(
            "query, key and value must be at least 3-D ([..., L, H, D]); got a "
            f"{q.dim()}-D query"
        )
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("query, key and value must have the same shape")

    for name, tensor in (("query", q), ("key", k), ("value", v)):
        if tensor.device.type == "cpu":
            raise ValueError(f"{name} must be a device tensor, not on the CPU")
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous; upstream's Python wrapper calls "
                f".contiguous() before reaching the kernel, so this port expects "
                f"the same thing it would have received"
            )

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("evoformer attention only supports fp16 and bf16")

    # The kernel's grid is (query tiles, heads, flattened batch and pairs) --
    # upstream's is the same three (kernel_forward.h `getBlocksGrid` returns
    # `dim3(ceil_div(num_queries, kQueriesPerBlock), num_heads, num_batches)`),
    # and the pair dimension therefore lands in the grid's z slot. CUDA caps that
    # at 65535; past it the launch fails with "[CUDA]: invalid argument", which
    # says nothing about which argument. So this is upstream's own ceiling, said
    # out loud. Chunking the batch is the way around it.
    num_slices = _batch_of(q)
    if max_slices is not None and num_slices > max_slices:
        raise ValueError(
            f"the flattened (batch, pair) dimension has {num_slices} slices, but "
            f"it is the kernel's grid z axis and allows only {max_slices} there "
            f"(upstream's grid is shaped the same way); split the batch"
        )

    # Upstream instantiates its forward kernel with ``kSingleValueIteration =
    # true``, which is its way of asserting that V fits in one tile -- so this is
    # its limit, not this port's. (Its backward spells the same one out: "Hidden
    # size is too large. Need to change kMax to a larger value".)
    if q.shape[-1] > 64:
        raise ValueError(
            "head_size must be <= 64; upstream's kernels fix kMaxK at 64 "
            f"(got {q.shape[-1]})"
        )

    capability = _device_capability()
    if capability is not None and capability[0] < 7:
        raise ValueError(
            "evoformer attention needs tensor cores (compute capability 7.0 or "
            "newer); this device reports %d.%d" % capability
        )

    _check_bias_shapes(q, bias1, bias2)


def _check_bias_shapes(q, bias1, bias2):
    """Hold the two biases to the shapes upstream's public entry asserts.

    The shapes are not decorative: they are what tells the kernel how the bias
    broadcasts, and they are indexed by ``B`` and ``N`` -- the first two leading
    dimensions -- so they also pin down how many leading dimensions the query may
    have. With no bias present nothing indexes them and any leading shape is fine.
    """
    has_bias1 = bias1 is not None and bias1.numel() != 0
    has_bias2 = bias2 is not None and bias2.numel() != 0
    if not (has_bias1 or has_bias2):
        return

    if q.dim() != 5:
        raise ValueError(
            "a bias indexes the query's first two dimensions (batch, pair), so a "
            f"biased query must be 5-D [B, N, L, H, D]; got {q.dim()} dimensions"
        )
    batch, n_pairs, seq_len, heads, _ = q.shape

    if has_bias1:
        expected = (batch, n_pairs, 1, 1, seq_len)
        if tuple(bias1.shape) != expected:
            raise ValueError(
                f"bias1 must have shape {expected} (one value per key position, "
                f"shared by every query row and head); got {tuple(bias1.shape)}"
            )
        if bias1.dtype != q.dtype:
            raise ValueError("bias1 must have the same dtype as the query")
        if not bias1.is_contiguous():
            raise ValueError(
                "bias1 must be contiguous; its layout is what tells the kernel "
                "where each (batch, pair) row starts"
            )

    if has_bias2:
        expected = (batch, 1, heads, seq_len, seq_len)
        if tuple(bias2.shape) != expected:
            raise ValueError(
                f"bias2 must have shape {expected} (one L x L matrix per head, "
                f"shared by every pair); got {tuple(bias2.shape)}"
            )
        if bias2.dtype != q.dtype:
            raise ValueError("bias2 must have the same dtype as the query")
        if not bias2.is_contiguous():
            raise ValueError(
                "bias2 must be contiguous; a transposed L x L matrix has the "
                "same shape and would be read as the wrong matrix"
            )


def _empty_bias_like(q):
    """The "no bias" sentinel upstream uses: an empty tensor of the input dtype."""
    return torch.empty((0,), dtype=q.dtype, device=q.device)


def _normalize_bias(bias, q):
    return _empty_bias_like(q) if bias is None else bias


def _lse_shape(batch, heads, seq_len):
    """``[B*N, H, ceil(L/32)*32]`` -- upstream pads lse to a multiple of 32.

    ``kAlignLSE = 32`` is the block size of its backward, which reads the padding
    instead of bounding every load; the padded rows carry +inf so that
    ``exp(s - lse)`` is 0 there.
    """
    return (batch, heads, triton.cdiv(seq_len, 32) * 32)


def evoformer_attn(q, k, v, bias1=None, bias2=None):
    """Evoformer attention forward.

    Args:
        q (Tensor): queries ``[..., L, H, D]``, contiguous.
        k (Tensor): keys, same shape and layout as ``q``.
        v (Tensor): values, same shape and layout as ``q``.
        bias1 (Tensor, optional): ``[B, N, 1, 1, L]`` added to every query row
            and head, or ``None``/an empty tensor for no such bias.
        bias2 (Tensor, optional): ``[B, 1, H, L, L]`` added per head, or
            ``None``/an empty tensor for no such bias.

    Returns:
        ``(out, lse)``: the attention output, shaped like ``q``, and the fp32
        log-sum-exp of each query row's scores, shaped ``[B*N, H, ceil(L/32)*32]``.
        This is exactly what upstream's ``_attention`` returns. ``lse`` is part
        of that contract rather than an extra: it is the second half of what the
        forward is defined to produce, and it is what a gradient would need.

    Requires ``D <= 64``, which is upstream's own limit: its forward is compiled
    for it. ``L`` is *not* bounded below here. Upstream asserts ``L > 16``, but
    that is an assertion about the kernel being ported, not about this one -- the
    tiles are masked and the arithmetic is the same at any ``L``, so this port
    serves ``L`` down to 1 where upstream refuses it. Accepting a superset of what
    upstream accepts keeps the port a drop-in replacement; the reverse would not.
    ``k`` and ``v`` must have the same ``L`` as ``q`` -- upstream sets
    ``num_queries = num_keys = seq_length`` from the query, so a cross-attention
    with different lengths is not something it supports either.

    This is the generic implementation, and the one every backend without an
    override runs. A backend with a specialised kernel replaces this *name*
    through ``runtime.backend.SpecOpRegistrar`` when ``flag_train.deepspeed`` is
    imported, so that is where to import the operator from -- importing it from
    this module gets the generic kernel on every backend.
    """
    logger.debug("TRAIN EVOFORMER_ATTN")

    bias1 = _normalize_bias(bias1, q)
    bias2 = _normalize_bias(bias2, q)

    _check_launch_arguments(q, k, v, bias1, bias2)

    seq_len = q.shape[-3]
    heads = q.shape[-2]
    head_size = q.shape[-1]
    batch = _batch_of(q)
    n_pairs = _pairs_of(q)

    out = torch.empty_like(q)
    lse = torch.empty(
        _lse_shape(batch, heads, seq_len), dtype=torch.float32, device=q.device
    )

    # bias1 is [B, N, 1, 1, L]: contiguous, that is one L-row per (batch, pair).
    # bias2 is [B, 1, H, L, L]: one L x L matrix per (batch, head).
    bias2_stride_b = heads * seq_len * seq_len
    bias2_stride_h = seq_len * seq_len

    has_bias1 = bias1.numel() != 0
    has_bias2 = bias2.numel() != 0
    block_d = max(16, triton.next_power_of_2(head_size))
    num_warps = max(4, (_BLOCK_M * block_d) // 2048)

    with torch_device_fn.device(q.device):
        _evoformer_fwd_kernel[(triton.cdiv(seq_len, _BLOCK_M), heads, batch)](
            out,
            q,
            k,
            v,
            # An empty tensor is not a usable pointer, so an absent bias borrows
            # the query's -- never dereferenced under HAS_BIAS* = False.
            bias1 if has_bias1 else q,
            bias2 if has_bias2 else q,
            lse,
            # The scores are scaled before the softmax, in fp32; upstream applies
            # the same 1/sqrt(head_dim) in its epilogue.
            head_size**-0.5,
            seq_len,
            lse.shape[-1],
            n_pairs,
            _stride_bn(q),
            q.stride(-3),
            q.stride(-2),
            _stride_bn(out),
            out.stride(-3),
            out.stride(-2),
            bias2_stride_b,
            bias2_stride_h,
            lse.stride(-3),
            lse.stride(-2),
            HAS_BIAS1=has_bias1,
            HAS_BIAS2=has_bias2,
            HEAD_SIZE=head_size,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
    return out, lse
