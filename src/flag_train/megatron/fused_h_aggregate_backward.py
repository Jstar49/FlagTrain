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
"""Fused n-stream weighted aggregation (backward).

The derivative of ``fused_h_aggregate_forward``, and the backward of the
``h_aggregate`` stage of ``megatron/core/fusions/fused_mhc_kernels.py``. For
``grad_output`` of shape ``[s, b, C]``, ``x`` of ``[s, b, n, C]`` and ``h_pre``
of ``[s, b, n]`` it produces

.. math::

    grad_x[s, b, i, c] = grad_{out}[s, b, c] \\, h_{pre}[s, b, i]

    grad_h[s, b, i] = \\sum_c grad_{out}[s, b, c] \\, x[s, b, i, c]

which is Megatron's ``_torch_h_aggregate_bwd`` -- the function its Triton
``FusedHAggregate`` autograd node calls -- evaluated in one pass. The functional
suite differentiates ``native_h_aggregate`` and holds both outputs to that; the
performance suite times the same autograd derivative.

The two gradients have opposite shapes of work and cannot share a launch
naively: ``grad_x`` is a broadcast multiply with no reduction, while ``grad_h``
is a reduction over ``C``. One program therefore owns a block of rows for the
*whole* of ``C``, streaming column tiles: every tile writes its slice of
``grad_x`` and folds ``grad_output * x`` into a ``[BLOCK_M, n]`` accumulator
that is written once at the end. That keeps ``grad_h`` exact and ordered --
splitting ``C`` across programs would need atomics on an ``[s*b, n]`` buffer,
which is neither deterministic nor faster, since the reduction is over the same
bytes ``grad_x`` already reads.

``x`` and ``h_pre`` are read once each and ``grad_output`` once per column tile,
so the pass costs what the forward does plus the ``n`` times larger ``grad_x``
write. ``grad_output`` is re-read for each tile of ``C`` -- there is one tile per
program -- but consecutive programs read the same few rows of it, which is what
the L2 holds.

The outputs are cast the way Megatron casts them: ``grad_x`` to ``x``'s dtype
and ``grad_h`` to ``h_pre``'s, with the arithmetic itself in fp32. With both
inputs in a narrow dtype that is one rounding better than the reference's
elementwise product, and the tolerance in the functional suite is set from the
measured difference.
"""

import functools
import logging
from contextlib import nullcontext
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_MAX_BLOCK_C = 2048
_MIN_BLOCK_C = 32
# Elements of the ``[BLOCK_M, n, BLOCK_C]`` operand tile a program holds at once.
# The tile is the whole register budget here: it is read, multiplied by the
# ``[BLOCK_M, BLOCK_C]`` gradient tile and reduced, all in fp32. The budget is
# spent on columns rather than rows -- see ``_config``.
_TILE_ELEMENTS = 8192
_MAX_BLOCK_M = 8


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
def _config(C, n_pad, sb):
    """Tile geometry for one launch.

    The column tile takes the whole row when the row fits in ``_MAX_BLOCK_C``,
    and the row block gets whatever the tile budget leaves -- which at the wide
    shapes is a single row. Wide columns are what this pass wants: the stream
    loop over ``C`` shortens to two iterations at ``C = 4096`` and one at
    ``C = 2048``, ``grad_output`` arrives in wide coalesced runs, and
    ``grad_h`` reduces one wide contiguous axis per iteration instead of a
    narrow one many times. Measured over the sweep: 1.05-1.20x over the
    ``BLOCK_C = 512`` this started with at the shapes a step runs, 1.4x where
    the row count is small enough that the loop was the cost, and no more than
    1.6% behind at the one mid-size shape (``(32, 32, 4, 4096)``) where the
    narrow tile edges it out. The widest shapes moved from ~3.3 TB/s to
    ~4.0 TB/s, against the ~3.7 TB/s ``copy_`` reaches on this device.

    The row block also decides the program count -- every program walks all of
    ``C`` -- so it is capped at ``_MAX_BLOCK_M`` rather than growing to fill the
    tile budget, and never exceeds the row count: a tile taller than ``s * b``
    is masked lanes the program still carries.

    ``num_stages`` is 2 (the default is 3). The loop is a pure stream -- each
    iteration loads a fresh column tile, there is no operand reuse to prefetch
    across iterations -- so the third stage only buys shared memory and
    bookkeeping; measured at 0-15% across the sweep, never worse.
    """
    block_c = min(_MAX_BLOCK_C, max(_MIN_BLOCK_C, triton.next_power_of_2(max(C, 1))))
    rows = 1 << (max(sb, 1).bit_length() - 1)
    block_m = max(1, min(_MAX_BLOCK_M, _TILE_ELEMENTS // (block_c * n_pad), rows))
    num_warps = 4 if block_m * block_c <= 4096 else 8
    num_stages = 2
    return block_m, block_c, num_warps, num_stages


@triton.jit
def _h_aggregate_backward_kernel(
    grad_out_ptr,
    x_ptr,
    h_ptr,
    grad_x_ptr,
    grad_h_ptr,
    sb,
    C,
    stride_x_s,
    stride_x_n,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One row block for all of ``C``: writes ``grad_x`` and ``grad_h``.

    ``N_PAD`` is ``N`` rounded up to a power of two, which is what ``tl.arange``
    requires; the lanes above ``N`` are masked out of every load and store, so
    the padding costs nothing but the registers it occupies. ``stride_x_s`` is
    the row stride of the flattened ``[s*b, n, C]`` view and ``stride_x_n`` the
    distance between streams.

    ``grad_h`` is reduced per iteration and folded into a ``[BLOCK_M, N_PAD]``
    accumulator, so the loop carries no ``[BLOCK_M, N_PAD, BLOCK_C]`` fp32 tile.
    Holding one and reducing once at the end instead removes the cross-lane
    reduction from the loop's critical path -- and is what a small shape wants,
    where the serial chain *is* the kernel's cost -- but it costs 16 KB of
    registers per program and measured 2-9% slower at every shape large enough
    to be bandwidth-bound. Those are the shapes a training step runs.
    """
    pid_m = tle.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < sb
    offs_n = tl.arange(0, N_PAD)
    mask_n = offs_n < N

    h = tl.load(
        h_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)

    grad_h = tl.zeros((BLOCK_M, N_PAD), dtype=tl.float32)
    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask_mc = mask_m[:, None] & mask_c[None, :]
        grad_out = tl.load(
            grad_out_ptr + offs_m[:, None] * C + offs_c[None, :],
            mask=mask_mc,
            other=0.0,
        ).to(tl.float32)

        mask_x = mask_m[:, None, None] & mask_n[None, :, None] & mask_c[None, None, :]
        x = tl.load(
            x_ptr
            + offs_m[:, None, None] * stride_x_s
            + offs_n[None, :, None] * stride_x_n
            + offs_c[None, None, :],
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)

        grad_x = grad_out[:, None, :] * h[:, :, None]
        tl.store(
            grad_x_ptr
            + offs_m[:, None, None] * stride_x_s
            + offs_n[None, :, None] * stride_x_n
            + offs_c[None, None, :],
            grad_x.to(grad_x_ptr.dtype.element_ty),
            mask=mask_x,
        )
        grad_h += tl.sum(grad_out[:, None, :] * x, axis=2)

    tl.store(
        grad_h_ptr + offs_m[:, None] * N + offs_n[None, :],
        grad_h.to(grad_h_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def fused_h_aggregate_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    h_pre: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradients of ``fused_h_aggregate_forward`` with respect to ``x`` and ``h_pre``.

    Args:
        grad_output (Tensor): ``[s, b, C]`` gradient of the aggregated output.
        x (Tensor): ``[s, b, n, C]`` stream activations the forward read.
        h_pre (Tensor): ``[s, b, n]`` stream weights the forward read.

    Returns:
        Tuple[Tensor, Tensor]: ``grad_x`` of shape ``[s, b, n, C]`` in ``x``'s
        dtype and ``grad_h`` of shape ``[s, b, n]`` in ``h_pre``'s dtype.
    """
    logger.debug("TRAIN FUSED_H_AGGREGATE_BACKWARD")
    _check_input(grad_output, "grad_output", 3)
    _check_input(x, "x", 4)
    _check_input(h_pre, "h_pre", 3)
    if x.shape[:3] != h_pre.shape:
        raise ValueError(
            "x and h_pre must agree on the (s, b, n) axes, got "
            f"{tuple(x.shape)} and {tuple(h_pre.shape)}"
        )

    s, b, n, C = x.shape
    if grad_output.shape != (s, b, C):
        raise ValueError(
            f"grad_output must have shape {(s, b, C)}, got {tuple(grad_output.shape)}"
        )

    grad_x = x.new_empty((s, b, n, C))
    grad_h = h_pre.new_empty((s, b, n))
    if s == 0 or b == 0 or n == 0:
        return grad_x, grad_h
    if C == 0:
        # The reduction is over an empty axis: zero, not whatever the allocator
        # handed back.
        grad_h.zero_()
        return grad_x, grad_h

    if not x.is_contiguous():
        x = x.contiguous()
    if not h_pre.is_contiguous():
        h_pre = h_pre.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    sb = s * b
    n_pad = triton.next_power_of_2(n)
    block_m, block_c, num_warps, num_stages = _config(C, n_pad, sb)
    grid = (triton.cdiv(sb, block_m),)

    device_index = x.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == device_index
        else torch_device_fn.device(x.device)
    )

    with guard:
        _h_aggregate_backward_kernel[grid](
            grad_output,
            x,
            h_pre,
            grad_x,
            grad_h,
            sb,
            C,
            # ``s`` and ``b`` are flattened away, so the row stride is the one
            # between adjacent ``b``, not between adjacent ``s``.
            x.stride(1),
            x.stride(2),
            N=n,
            N_PAD=n_pad,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return grad_x, grad_h
