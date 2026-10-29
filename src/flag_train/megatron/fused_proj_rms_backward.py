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
"""Fused projection + RMS normalization (backward).

The counterpart of :func:`flag_train.megatron.fused_proj_rms_forward`. Given
``x`` ``[M, K]``, ``weight`` ``[N, K]``, the gradients ``grad_proj`` ``[M, N]``
and ``grad_r`` ``[M, 1]`` of the forward's outputs, and the forward's saved
``r``, it returns ``grad_x`` ``[M, K]`` and ``grad_weight`` ``[N, K]``.

With ``proj = x @ weight^T`` and ``r = 1 / (||x|| / sqrt(K) + eps)``:

.. math::

    grad_{weight} = grad_{proj}^T @ x

    grad_x = grad_{proj} @ weight
             - \\frac{grad_r \\cdot r^2}{\\sqrt{K} \\lVert x \\rVert} \\, x

The oracle is Megatron-LM's ``native_proj_rms`` differentiated by autograd, so
the formulas above are the reference's own gradient rather than a hand-written
approximation of it. As in PyTorch's norm backward, a row whose norm is zero
contributes no ``grad_r`` term -- the norm is not differentiable there -- while
its ``grad_proj`` term is unaffected.

``grad_x`` comes back in ``x``'s dtype and ``grad_weight`` in ``weight``'s,
matching what autograd produces through the promotion the forward applies.
Inputs are promoted to ``torch.promote_types(x.dtype, weight.dtype)`` for the
computation, exactly as in the forward.

Two or three launches, because the work has three different shapes of
parallelism: the row coefficient (a reduction over K), ``grad_proj @ weight``
(tiles over M and K), and ``grad_proj^T @ x`` (tiles over N and K, reducing over
M). Folding the coefficient into either matrix product would make one long-lived
program walk the whole row while the rest of the grid only sees a K tile, so it
keeps programs of its own; at mHC's narrow N with a short M, where the operator
is dispatch bound and its programs are few, they ride in the ``grad_weight``
launch instead (see ``_coef_grad_weight_kernel``). Merging the two products that
way -- one launch, ``grad_weight`` programs interleaved with ``grad_x`` programs
so their reads of ``x`` share L2 -- measured 1.5-1.7x worse: a launch's
shared-memory request is the maximum over its programs, so the narrower
product's programs inherit the deeper one's pipeline footprint and lose the
occupancy they stream with. The coefficient pays the same tax, which is why its
merge is limited to the shapes where it is too small to.

Each launch therefore carries its own geometry (see the ``_*_config``
functions). They differ more than the two products' tiles do in the forward,
because the three kernels are limited by different things: the coefficient and
``grad_x`` by traffic, ``grad_weight`` by how much of the device its
``(N tiles) x (K tiles)`` grid covers.

Every operand is contiguous -- the caller makes ``x``, ``weight`` and
``grad_proj`` so and allocates both outputs so -- which is why the kernels take
their strides from the row lengths instead of as arguments. Each argument
dropped from a launch is a per-call cost the host pays; at the mHC shapes a
training step waits on the host, not on the card. For the same reason the
kernels are plain ``@triton.jit`` rather than ``flag_train.utils.libentry``:
that wrapper exists to give the tuner somewhere to hang, and this operator picks
its geometry itself, so all it would add here is its per-call key computation
and cache lookup -- measured at 9-10 us per launch, three launches per call, on
a dispatch that already dominates the mHC shapes (at the four narrowest shapes it
is worth 1.6-1.7x end to end). Where the device, not the host, sets the pace the
wrapper is free, which is why the forward keeps it.
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

# Row block of a coefficient program. The reduction streams ``x`` and nothing
# else, so this is only about how many programs the read is spread over: four
# rows put M/4 of them on the device, which is within 2% of the best geometry
# tested and well ahead of the two rows this used to take -- at (M=8192, K=8192)
# the narrower tile reached 1.5 TB/s where four rows reach 3.1, and the whole
# operator is 14% quicker. Wider blocks leave too few programs (64 rows is 1.7x
# this one's time at M=4096). K is the other half: 128 keeps the per-program
# loop long enough to amortise its prologue without making the single program
# that a short M launches take a long serial chain.
_COEF_BLOCK_M = 4
_COEF_BLOCK_K = 128
# ``grad_x`` tiles: 128 deep in K at three stages, which is the deepest
# pipeline that still leaves two program blocks resident per SM at the N=1024
# width. Measured within 1% of the best per-shape choice at every shape in the
# suite except the 24-wide ones, where it is the best.
_GRAD_X_BLOCK_K = 128
_GRAD_X_STAGES = 3
# Shared memory a pipelined tile pair may claim, in bytes. Both operands are
# buffered ``stages`` deep, so the request scales with the element size and an
# fp32 launch takes the next K tile down. What does not fit is rejected at
# launch, not at the first wrong answer.
_SMEM_BUDGET = 200 * 1024
_BLOCK_K_CHOICES = (128, 64, 32, 16)
# Narrowest N that gets a 64-wide ``grad_x`` tile instead of the 32-wide one.
_GRAD_X_WIDE_N = 512
# Row blocks from this many rows up get BLOCK_M=64. Below it a 64-row tile
# leaves the device with too few ``grad_x`` programs, and the parallelism of a
# 16-row tile is worth more than the traffic it repeats.
_SMALL_M = 1024
# ``grad_weight`` is the one launch whose grid does not grow with M, so its
# tiles are chosen to cover the device instead: mHC's N=24 leaves one or two n
# tiles, and the grid is then only as wide as the K sweep.
_NARROW_N = 32
_MID_N = 128
_WIDE_N = 2048
# At or below this many rows, and at mHC's narrow N, the coefficient and
# ``grad_weight`` share one launch: the operator is dispatch bound there, and at
# that size the coefficient's programs are a small addition to the product's grid
# rather than competition for it. See ``_coef_grad_weight_kernel``. Wider N is
# left alone -- the product's tile is then too large to hand to the coefficient
# programs, and merging measured 0.83x at (M=1024, N=80).
_MERGE_M = 1024
_MERGED_BLOCK_M = 128
# Past this K the reduction is long enough that a single 32-wide n tile with a
# 128-deep K pipeline beats splitting ``x`` over two 16-wide tiles, which is the
# ordering at K=4096.
_LONG_K = 4096


def _check_dtype_and_device(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise ValueError(f"{name} must have dtype {supported}, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} must be a device tensor, not a CPU tensor")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} must be a {runtime_device.name} tensor, got {tensor.device.type}"
        )


def _check_tensor(tensor, name, ndim):
    _check_dtype_and_device(tensor, name)
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(tensor.shape)}")


def _check_vector(tensor, name, length, device):
    """``r`` and ``grad_r`` are row vectors: ``[M, 1]`` or ``[M]``."""
    _check_dtype_and_device(tensor, name)
    if tensor.dim() not in (1, 2) or tensor.numel() != length:
        raise ValueError(
            f"{name} must have {length} elements, got shape {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dim() == 1:
        return tensor if tensor.is_contiguous() else tensor.contiguous()
    return tensor.reshape(-1).contiguous()


def _block_k(cap, block_m, block_n, K, dtype, stages):
    """Widest K tile up to ``cap`` whose tile pair fits the budget.

    Never wider than the reduction itself, so a short K is a single masked
    iteration rather than a tile of padding.
    """
    element = torch.finfo(dtype).bits // 8
    width = triton.next_power_of_2(max(K, 1))
    for candidate in _BLOCK_K_CHOICES:
        if candidate <= cap and (
            (block_m + block_n) * candidate * element * stages <= _SMEM_BUDGET
        ):
            return min(candidate, width)
    return _BLOCK_K_CHOICES[-1]


@functools.lru_cache(maxsize=128)
def _coef_config(K):
    """Row block, K tile and launch parameters for the coefficient launch.

    Cached because it is pure, and because at the mHC shapes a training step's
    worth of work is smaller than what deciding the tiles costs: the lookup is
    the whole per-call cost once the shape has been seen.
    """
    return _COEF_BLOCK_M, min(_COEF_BLOCK_K, triton.next_power_of_2(max(K, 1))), 4, 2


@functools.lru_cache(maxsize=128)
def _grad_x_config(M, N, K, dtype):
    """Tile geometry for ``grad_proj @ weight``.

    N decides the tile width: 32 while N is narrow enough that a row of ``x``
    is the traffic and the ``weight`` tile is a rounding error, 64 from
    ``_GRAD_X_WIDE_N`` up, where re-reading ``weight`` per row block is what the
    wider tile is there to halve. Capped at 64 rather than at the forward's 128:
    at N=4096 the 64-wide tile was 8% ahead, the same occupancy effect that made
    the mHC shapes want a narrow tile. Measured within about 1% of the best
    per-shape choice everywhere in the suite, and the best at the 24-wide ones.

    Four warps on the half-precision tiles, not eight: this launch re-reads
    ``x`` for the ``coef * x`` term rather than feeding the tensor core off a
    wide operand tile, so the extra warps buy nothing -- eight measured 7%
    behind at N=80 and within noise everywhere else, up to N=4096. fp32 keeps
    them, because there each tile carries twice the bytes per thread and eight
    warps are worth 3-4%.
    """
    block_m = 64 if M >= _SMALL_M else 16
    block_n = max(16, min(32, triton.next_power_of_2(max(N, 1))))
    if N > _GRAD_X_WIDE_N:
        block_n = 64
    block_k = _block_k(_GRAD_X_BLOCK_K, block_m, block_n, K, dtype, _GRAD_X_STAGES)
    num_warps = 8 if dtype == torch.float32 else 4
    num_stages = _GRAD_X_STAGES if K >= block_k else 1
    return block_m, block_n, block_k, num_warps, num_stages


@functools.lru_cache(maxsize=128)
def _grad_weight_config(M, N, K, dtype):
    """Tile geometry for ``grad_proj^T @ x``.

    Three bands, because this grid is ``cdiv(N, BLOCK_N) * cdiv(K, BLOCK_K)``
    and does not grow with M:

    How deep the K pipeline may go is set by how often the tile re-reads ``x``,
    which is once per n tile, i.e. ``cdiv(N, BLOCK_N)``:

    * ``N <= _NARROW_N`` (mHC): one or two n tiles, so the row block is as deep
      as the reduction allows (256 rows) and the K tile sets the pipeline. At
      K=4096 that is 64 deep over two 16-wide tiles, which is what fills the
      device; at K=8192 the reduction is long enough to keep one 16-wide tile in
      flight at a 128-deep, three-stage pipeline, and that is 9% ahead of the
      wider two-tile grid.
    * ``_NARROW_N < N <= _MID_N``: at most two 64-wide n tiles, so ``x`` is read
      at most twice and the deeper pipeline is still nearly free -- a 128-deep
      K tile at three stages is 13% ahead of a 64-deep two-stage one at N=80.
    * wider: more n tiles means more re-reads of ``x``, and the memory system
      stops paying for the deeper pipeline; the tile drops to 64 deep. At N=288
      that is 7% ahead of the 128-deep one, at N=1024 within 1%.
    """
    if N <= _NARROW_N:
        if K > _LONG_K:
            return 128, 16, _block_k(128, 128, 16, K, dtype, 3), 8, 3
        return 256, 16, _block_k(64, 256, 16, K, dtype, 2), 8, 2
    if N <= _MID_N:
        return 64, 64, _block_k(128, 64, 64, K, dtype, 3), 4, 3
    if N > _WIDE_N:
        return 64, 64, _block_k(128, 64, 64, K, dtype, 3), 8, 3
    return 64, 64, _block_k(64, 64, 64, K, dtype, 2), 4, 2


@functools.lru_cache(maxsize=128)
def _merged_config(M, N, K, dtype):
    """Geometry for the merged coefficient + ``grad_weight`` launch.

    Narrow N only, so the n tile is the 16 that covers mHC in one or two
    programs. The row block is the widest the shared-memory budget allows rather
    than the 256 the split launch uses: it is the product's tile that decides
    what every program in the merged launch reserves, and the coefficient's
    programs would pay for a deeper one with their occupancy. A 64-deep K tile
    keeps the request near 37 KB, so they still run several to an SM.
    """
    block_k = _block_k(64, _MERGED_BLOCK_M, 16, K, dtype, 2)
    return _MERGED_BLOCK_M, 16, block_k, 4, 2


@triton.jit
def _coef_tile(
    x_ptr,
    r_ptr,
    grad_r_ptr,
    coef_ptr,
    M,
    K,
    pid,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``-grad_r * r^2 / (sqrt(K) * ||x||)`` for one block of rows.

    Shared by ``_coef_kernel`` and the merged launch, which is why the program
    index is handed in rather than read.
    """
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=1)

    norm = tl.sqrt(sum_sq)
    r = tl.load(r_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    grad_r = tl.load(grad_r_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    # A zero row has no norm direction: PyTorch's norm backward yields zero
    # there, so this where() does too, and it keeps 0 / 0 from becoming NaN.
    coef = tl.where(norm > 0.0, -grad_r * r * r / (norm * tl.sqrt(K * 1.0)), 0.0)
    tl.store(coef_ptr + offs_m, coef, mask=mask_m)


@triton.jit
def _coef_kernel(
    x_ptr,
    r_ptr,
    grad_r_ptr,
    coef_ptr,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _coef_tile(
        x_ptr, r_ptr, grad_r_ptr, coef_ptr, M, K, tle.program_id(0), BLOCK_M, BLOCK_K
    )


@triton.jit
def _grad_x_kernel(
    grad_proj_ptr,
    weight_ptr,
    x_ptr,
    coef_ptr,
    grad_x_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
):
    """``grad_proj @ weight`` for one ``[BLOCK_M, BLOCK_K]`` tile, plus ``coef * x``."""
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        grad_proj = tl.load(
            grad_proj_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        w = tl.load(
            weight_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        if IEEE:
            acc += tl.dot(grad_proj, w, input_precision="ieee")
        else:
            acc += tl.dot(grad_proj, w)

    x = tl.load(
        x_ptr + offs_m[:, None] * K + offs_k[None, :],
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    coef = tl.load(coef_ptr + offs_m, mask=mask_m, other=0.0)
    acc += coef[:, None] * x

    tl.store(
        grad_x_ptr + offs_m[:, None] * K + offs_k[None, :],
        acc.to(grad_x_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_k[None, :],
    )


@triton.jit
def _grad_weight_tile(
    grad_proj_ptr,
    x_ptr,
    grad_weight_ptr,
    M,
    N,
    K,
    pid,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
):
    """``grad_proj^T @ x`` for one ``[BLOCK_N, BLOCK_K]`` tile of grad_weight.

    Programs are ordered so that consecutive ones share a k tile and differ in
    the n tile (``pid_n`` varies fastest), which is what lets the ``x`` tile a
    program has just read serve its neighbours out of L2.
    """
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_n = pid % num_pid_n
    pid_k = pid // num_pid_n

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        grad_proj = tl.load(
            grad_proj_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        if IEEE:
            acc += tl.dot(tl.trans(grad_proj), x, input_precision="ieee")
        else:
            acc += tl.dot(tl.trans(grad_proj), x)

    tl.store(
        grad_weight_ptr + offs_n[:, None] * K + offs_k[None, :],
        acc.to(grad_weight_ptr.dtype.element_ty),
        mask=mask_n[:, None] & mask_k[None, :],
    )


@triton.jit
def _grad_weight_kernel(
    grad_proj_ptr,
    x_ptr,
    grad_weight_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
):
    _grad_weight_tile(
        grad_proj_ptr,
        x_ptr,
        grad_weight_ptr,
        M,
        N,
        K,
        tle.program_id(0),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        IEEE,
    )


@triton.jit
def _coef_grad_weight_kernel(
    grad_proj_ptr,
    x_ptr,
    grad_weight_ptr,
    r_ptr,
    grad_r_ptr,
    coef_ptr,
    M,
    N,
    K,
    num_gw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
    COEF_BLOCK_M: tl.constexpr,
    COEF_BLOCK_K: tl.constexpr,
):
    """``grad_weight`` and the row coefficient out of one launch.

    The two have no dependence on each other -- only ``grad_x`` consumes the
    coefficient -- so shipping them together costs a launch less, which is what
    the narrow shapes are short of. Only used while M is small enough that the
    coefficient's programs are a small addition to the product's grid: a launch
    reserves shared memory as the maximum over its programs, so past that point
    they would each reserve the product's pipeline footprint and stream ``x`` at
    a fraction of the occupancy they get on their own.
    """
    pid = tle.program_id(0)
    if pid < num_gw:
        _grad_weight_tile(
            grad_proj_ptr,
            x_ptr,
            grad_weight_ptr,
            M,
            N,
            K,
            pid,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            IEEE,
        )
    else:
        _coef_tile(
            x_ptr,
            r_ptr,
            grad_r_ptr,
            coef_ptr,
            M,
            K,
            pid - num_gw,
            COEF_BLOCK_M,
            COEF_BLOCK_K,
        )


@triton.jit
def _scale_rows_kernel(
    coef_ptr,
    x_ptr,
    grad_x_ptr,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``grad_x = coef * x``, for the degenerate ``N == 0`` case."""
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    coef = tl.load(coef_ptr + offs_m, mask=mask_m, other=0.0)
    x = tl.load(
        x_ptr + offs_m[:, None] * K + offs_k[None, :],
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_x_ptr + offs_m[:, None] * K + offs_k[None, :],
        (coef[:, None] * x).to(grad_x_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_k[None, :],
    )


def fused_proj_rms_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_proj: torch.Tensor,
    grad_r: torch.Tensor,
    r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradients of :func:`fused_proj_rms_forward` with respect to ``x`` and
    ``weight``.

    Args:
        x (Tensor): ``[M, K]`` forward input. A non-contiguous input is copied.
        weight (Tensor): ``[N, K]`` forward weight. Its dtype may differ from
            ``x``'s; both are promoted to
            ``torch.promote_types(x.dtype, weight.dtype)`` for the computation.
        grad_proj (Tensor): ``[M, N]`` gradient of the forward's ``proj``. A
            non-contiguous input is copied.
        grad_r (Tensor): ``[M, 1]`` or ``[M]`` gradient of the forward's ``r``.
        r (Tensor): the ``[M, 1]`` or ``[M]`` RMS scale the forward returned.

    Returns:
        Tuple[Tensor, Tensor]: ``grad_x`` ``[M, K]`` in ``x``'s dtype and
        ``grad_weight`` ``[N, K]`` in ``weight``'s dtype.
    """
    logger.debug("TRAIN FUSED_PROJ_RMS_BACKWARD")
    _check_tensor(x, "x", 2)
    _check_tensor(weight, "weight", 2)
    _check_tensor(grad_proj, "grad_proj", 2)
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            "x and weight must agree on the reduction dimension, got "
            f"{tuple(x.shape)} and {tuple(weight.shape)}"
        )
    M, K = x.shape
    N = weight.shape[0]
    if tuple(grad_proj.shape) != (M, N):
        raise ValueError(
            f"grad_proj must have shape {(M, N)}, got {tuple(grad_proj.shape)}"
        )
    if grad_proj.device != x.device:
        raise ValueError(f"grad_proj must be on {x.device}, got {grad_proj.device}")
    grad_r = _check_vector(grad_r, "grad_r", M, x.device)
    r = _check_vector(r, "r", M, x.device)

    grad_x = x.new_empty((M, K))
    grad_weight = weight.new_empty((N, K))
    if M == 0 or K == 0:
        # Neither product contributes anything: with no rows there is nothing to
        # reduce over, and with no reduction dimension every product is an empty
        # sum. ``K == 0`` leaves two empty tensors, but ``M == 0`` still returns
        # an ``[N, K]`` grad_weight -- and an empty allocation would hand back
        # uninitialised memory, not the zeros autograd produces.
        grad_x.zero_()
        grad_weight.zero_()
        return grad_x, grad_weight

    # One dtype for the whole computation, as the forward does. Each conversion
    # is guarded rather than requested outright: at the mHC shapes the caller
    # hands over the promoted dtype already, and the kernels take their strides
    # from M, N and K, so every operand has to be contiguous by the time one
    # runs.
    compute = torch.promote_types(x.dtype, weight.dtype)
    if x.dtype != compute:
        x = x.to(compute)
    if not x.is_contiguous():
        x = x.contiguous()
    if weight.dtype != compute:
        weight = weight.to(compute)
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if grad_proj.dtype != compute:
        grad_proj = grad_proj.to(compute)
    if not grad_proj.is_contiguous():
        grad_proj = grad_proj.contiguous()
    if r.dtype != compute:
        r = r.to(compute)
    if grad_r.dtype != compute:
        grad_r = grad_r.to(compute)

    device_index = x.device.index
    if device_index is None:
        device_index = torch_device_fn.current_device()
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == device_index
        else torch_device_fn.device(x.device)
    )

    ieee = compute == torch.float32
    # The coefficient is fp32 regardless of the operand dtype: it multiplies
    # x and is added to an fp32 accumulator.
    coef = x.new_empty((M,), dtype=torch.float32)
    coef_block_m, coef_block_k, coef_warps, coef_stages = _coef_config(K)
    grad_x_block_m, grad_x_block_n, grad_x_block_k, warps, stages = _grad_x_config(
        M, N, K, compute
    )

    with guard:
        # The two launches the coefficient is not on the critical path of --
        # ``grad_weight`` -- go first, together, when M is small; ``grad_x``
        # needs the coefficient and always comes after. See
        # ``_coef_grad_weight_kernel`` for why the merge stops at small M.
        merged = 0 < N <= _NARROW_N and M <= _MERGE_M
        if merged:
            (
                weight_block_m,
                weight_block_n,
                weight_block_k,
                weight_warps,
                weight_stages,
            ) = _merged_config(M, N, K, compute)
            num_gw = triton.cdiv(N, weight_block_n) * triton.cdiv(K, weight_block_k)
            _coef_grad_weight_kernel[(num_gw + triton.cdiv(M, coef_block_m),)](
                grad_proj,
                x,
                grad_weight,
                r,
                grad_r,
                coef,
                M,
                N,
                K,
                num_gw,
                BLOCK_M=weight_block_m,
                BLOCK_N=weight_block_n,
                BLOCK_K=weight_block_k,
                IEEE=ieee,
                COEF_BLOCK_M=coef_block_m,
                COEF_BLOCK_K=coef_block_k,
                num_warps=weight_warps,
                num_stages=weight_stages,
            )
        else:
            _coef_kernel[(triton.cdiv(M, coef_block_m),)](
                x,
                r,
                grad_r,
                coef,
                M,
                K,
                BLOCK_M=coef_block_m,
                BLOCK_K=coef_block_k,
                num_warps=coef_warps,
                num_stages=coef_stages,
            )

        if N == 0:
            _scale_rows_kernel[
                (triton.cdiv(M, grad_x_block_m), triton.cdiv(K, grad_x_block_k))
            ](
                coef,
                x,
                grad_x,
                M,
                K,
                BLOCK_M=grad_x_block_m,
                BLOCK_K=grad_x_block_k,
                num_warps=4,
            )
            return grad_x, grad_weight

        _grad_x_kernel[
            (triton.cdiv(M, grad_x_block_m), triton.cdiv(K, grad_x_block_k))
        ](
            grad_proj,
            weight,
            x,
            coef,
            grad_x,
            M,
            N,
            K,
            BLOCK_M=grad_x_block_m,
            BLOCK_N=grad_x_block_n,
            BLOCK_K=grad_x_block_k,
            IEEE=ieee,
            num_warps=warps,
            num_stages=stages,
        )

        if not merged:
            (
                weight_block_m,
                weight_block_n,
                weight_block_k,
                weight_warps,
                weight_stages,
            ) = _grad_weight_config(M, N, K, compute)
            _grad_weight_kernel[
                (triton.cdiv(N, weight_block_n) * triton.cdiv(K, weight_block_k),)
            ](
                grad_proj,
                x,
                grad_weight,
                M,
                N,
                K,
                BLOCK_M=weight_block_m,
                BLOCK_N=weight_block_n,
                BLOCK_K=weight_block_k,
                IEEE=ieee,
                num_warps=weight_warps,
                num_stages=weight_stages,
            )

    return grad_x, grad_weight
