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
"""Fused n-stream weighted aggregation (forward).

This is the ``h_aggregate`` stage of Megatron-LM's
``megatron/core/fusions/fused_mhc_kernels.py`` as a standalone operator. For
``x`` of shape ``[s, b, n, C]`` and ``h_pre`` of shape ``[s, b, n]`` it computes

.. math::

    out[s, b, c] = \\sum_{i=0}^{n-1} h_{pre}[s, b, i] \\, x[s, b, i, c]

i.e. the ``n`` residual streams mHC keeps per token are collapsed back into one,
each scaled by its own per-token scalar. The Megatron reference is
``native_h_aggregate`` in ``megatron.core.transformer.hyper_connection`` --
``(x * h_pre.unsqueeze(-1)).sum(dim=2)`` -- which is the oracle the functional
suite compares against and the baseline the performance suite times.

Only the first three axes are index arithmetic: ``s`` and ``b`` never appear
separately, so ``x`` is read as ``[s*b, n, C]`` and one program owns a
``[BLOCK_M, BLOCK_C]`` tile of the ``[s*b, C]`` output. The stream axis is
unrolled rather than tiled. ``n`` is 2-8 in mHC and it is the *outer* axis of
``x``, so a stream's operand is one contiguous ``[BLOCK_M, BLOCK_C]`` tile a
fixed stride away from the previous one, and its weight is a single scalar per
row: the loop costs an add and a fused multiply per element and nothing that
needs a tile of its own. Tiling it would instead put ``n`` in the innermost
loop, where the same rows of the next stream land ``C`` elements apart.

``x`` and ``h_pre`` may have different floating dtypes -- the mHC mapping
parameters are held in fp32 while the activations are bf16/fp16 -- in which case
both are promoted for the computation and the output carries
``torch.promote_types(x.dtype, h_pre.dtype)``. That is what Megatron's reference
produces (the Python multiply broadcasts and promotes), and the promotion is
done inside the kernel: promoting ``x`` on the host would copy every element of
the largest operand for no arithmetic benefit.

Every element of the output is an independent short reduction, so the operator
is bound by how fast ``x`` streams in, not by arithmetic.
"""

import functools
import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Elements one program streams per output tile. The launch is trivial: the tile
# only has to be wide enough in ``C`` for each stream's row segment to coalesce
# and tall enough in rows to amortise the per-program setup.
_TILE_ELEMENTS = 2048
_MAX_BLOCK_C = 512
_MIN_BLOCK_C = 32
# The tile is [BLOCK_M, BLOCK_C] of the *output*, but a program reads n of them.
# Past this many streams the row block shrinks so a program does not outgrow the
# registers a [BLOCK_M, BLOCK_C] fp32 accumulator needs.
_MAX_BLOCK_M = 16


def _check_input(tensor, name, rank):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dim() != rank:
        raise ValueError(f"{name} must be {rank}-D, got shape {tuple(tensor.shape)}")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise ValueError(f"{name} must have dtype {supported}, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} must be a device tensor, not a CPU tensor")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} must be a {runtime_device.name} tensor, got {tensor.device.type}"
        )


@functools.lru_cache(maxsize=128)
def _config(C, sb):
    """Tile geometry for one launch.

    ``BLOCK_C`` takes the whole row when it is narrow, and the row block then
    grows to keep the tile near ``_TILE_ELEMENTS``; a wide row gets the widest
    column tile and a short row block. The row block never exceeds the row count:
    a tile taller than ``s * b`` is masked lanes a program still pays to carry,
    and at the mHC shapes a single row is a real case. Cached because it is pure,
    and because at these shapes one step's worth of arithmetic is smaller than
    the lookup.

    A wider tile was measured too -- one row per program, the whole row in a
    ``[1, 4096]`` tile -- and it is 3.6-4.0% ahead in fp16 but 2-4% *behind* in
    bf16, which is the dtype the mHC stages actually run in. The margin does not
    survive that split, so the geometry stays one rule for every dtype.
    """
    block_c = min(_MAX_BLOCK_C, max(_MIN_BLOCK_C, triton.next_power_of_2(max(C, 1))))
    # ``tl.arange`` takes a power of two, so the cap is the largest one below sb.
    rows = 1 << (max(sb, 1).bit_length() - 1)
    block_m = max(1, min(_MAX_BLOCK_M, _TILE_ELEMENTS // block_c, rows))
    num_warps = 4 if block_m * block_c <= 4096 else 8
    return block_m, block_c, num_warps


@triton.jit
def _h_aggregate_forward_kernel(
    x_ptr,
    h_ptr,
    out_ptr,
    sb,
    C,
    stride_x_s,
    stride_x_n,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_C]`` tile of ``out``, reducing over the streams.

    ``stride_x_s`` is the distance between rows of ``x`` (``n * C`` for the
    contiguous input the host guarantees) and ``stride_x_n`` the distance
    between streams (``C``); the column stride is 1 and is not passed.
    """
    pid_m = tle.program_id(0)
    pid_c = tle.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_m = offs_m < sb
    mask_c = offs_c < C
    mask = mask_m[:, None] & mask_c[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    for i in tl.static_range(N):
        x_i = tl.load(
            x_ptr + offs_m[:, None] * stride_x_s + i * stride_x_n + offs_c[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        h_i = tl.load(h_ptr + offs_m * N + i, mask=mask_m, other=0.0).to(tl.float32)
        acc += h_i[:, None] * x_i

    tl.store(
        out_ptr + offs_m[:, None] * C + offs_c[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_h_aggregate_forward(
    x: torch.Tensor,
    h_pre: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of ``n`` streams.

    Args:
        x (Tensor): ``[s, b, n, C]`` fp16, bf16 or fp32 stream activations on
            the active device. A non-contiguous input is copied.
        h_pre (Tensor): ``[s, b, n]`` per-token stream weights. Its dtype may
            differ from ``x``'s; both are promoted to
            ``torch.promote_types(x.dtype, h_pre.dtype)``.

    Returns:
        Tensor: ``[s, b, C]`` in the promoted input dtype.
    """
    logger.debug("TRAIN FUSED_H_AGGREGATE_FORWARD")
    _check_input(x, "x", 4)
    _check_input(h_pre, "h_pre", 3)
    if x.shape[:3] != h_pre.shape:
        raise ValueError(
            "x and h_pre must agree on the (s, b, n) axes, got "
            f"{tuple(x.shape)} and {tuple(h_pre.shape)}"
        )

    s, b, n, C = x.shape
    out_dtype = torch.promote_types(x.dtype, h_pre.dtype)
    out = x.new_empty((s, b, C), dtype=out_dtype)
    if s == 0 or b == 0 or C == 0:
        return out
    if n == 0:
        # The empty sum: Megatron's reference returns zeros, not uninitialised
        # memory.
        out.zero_()
        return out

    if not x.is_contiguous():
        x = x.contiguous()
    if not h_pre.is_contiguous():
        h_pre = h_pre.contiguous()

    sb = s * b
    block_m, block_c, num_warps = _config(C, sb)
    grid = (triton.cdiv(sb, block_m), triton.cdiv(C, block_c))

    device_index = x.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == device_index
        else torch_device_fn.device(x.device)
    )

    with guard:
        _h_aggregate_forward_kernel[grid](
            x,
            h_pre,
            out,
            sb,
            C,
            # ``s`` and ``b`` are flattened away, so the row stride is the one
            # between adjacent ``b``, not between adjacent ``s``.
            x.stride(1),
            x.stride(2),
            N=n,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    return out
