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
"""Forward of Megatron's fused projection + RMS + compute_h mHC mapping.

The reference is ``megatron.core.fusions.fused_mhc_kernels`` (the native
``_torch_proj_rms_compute_h`` and the cuTile ``_cutile_proj_rms_compute_h_fwd``).
For ``x`` of shape ``[M, K]``, ``weight`` of shape ``[N, K]`` with
``N = n * n + 2 * n`` and scalar ``alpha_pre`` / ``alpha_post`` / ``alpha_res``::

    proj  = x @ weight.T                                # [M, N]
    r     = ||x||_2 / sqrt(K)                           # [M, 1], the RMS
    alpha = [alpha_pre] * n + [alpha_post] * n + [alpha_res] * (N - 2n)
    h     = proj * alpha / (r + eps) + bias             # [M, N]
    h_pre   = sigmoid(h[:, :n]) + compute_h_eps         # [M, n]
    h_post  = 2 * sigmoid(h[:, n:2n])                   # [M, n]
    h_res   = h[:, 2n:]                                 # [M, n*n], pre-Sinkhorn

``eps`` is added to the RMS before taking its reciprocal, matching RMSNorm's
denominator. ``h_res`` is returned as raw logits: the caller applies the
Sinkhorn-Knopp projection afterwards.

The work is split into two streaming kernels. ``_rms_norm_kernel`` reduces the
square sum over ``K`` and writes ``r``; ``_proj_compute_h_kernel`` runs the
pipelined matmul and, with the whole projection tile in registers, scales,
biases, activates and scatters it to the three segment outputs. That reads
``x`` twice, but both loops run with software pipelining, which the fused
single-kernel form cannot: Triton 3.6 miscompiles a pipelined loop whose tile
feeds both ``tl.dot`` and a reduction (wrong, non-deterministic accumulator
rows), and the unpipelined single-kernel loop measured slower than these two
pipelined passes together on the H20 mHC shapes.

The norm kernel keeps the plainest possible loop body -- load, fp32 square,
row ``tl.sum`` -- with no ``tl.dot`` in the loop, so the dot/reduction
interaction above does not apply and no tile-sized accumulator is needed.
A ``K`` that fits in one tile also turns pipelining off (``STAGES=1``): the
loop cannot pipeline across a single iteration, and a vendor compiler that
mis-handles the prologue/epilogue of such a loop produces a wrong ``r``
(observed on Iluvatar's Triton 3.2 fork for M=1, K=128).

Precision: with fp32 operands ``tl.dot`` uses the backend's default input
precision (tf32 on NVIDIA), which is what Megatron's cuTile forward uses
(``tfloat32`` MMA); low-precision activations keep their own dtype for the dot
and accumulate in fp32. The square sum is always fp32. ``r`` is written in the
promoted matmul dtype (the reference's dtype) and read back by the epilogue
kernel, so a low-precision path uses the same rounded ``r`` as the reference.

The reduced projection is returned as well (as Megatron's cuTile forward does)
because the backward consumes it.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

_NUM_WARPS = 4
_NUM_STAGES = 3

# Vendor workaround: Iluvatar's Triton 3.2 fork miscompiles the pipelined norm
# loop. A [BLOCK_M, BLOCK_K] square accumulator on a single-iteration loop
# returned r ~ 0.75x, and the next revision still returned a near-zero r for
# K=1000 (h_pre saturates to compute_h_eps / 1.0). The norm kernel is
# memory-latency bound without pipelining, so on that vendor correctness wins
# over the pipeline; every other vendor keeps _NUM_STAGES.
_NORM_STAGES = 1 if runtime_device.vendor_name == "iluvatar" else _NUM_STAGES

# RMS kernel: one program covers a small row block and the full K axis.
_NORM_BLOCK_M = 32
_NORM_BLOCK_K = 256

# Projection kernel: at least 4096 rows fill the GPU well enough for 64-row
# tiles (better weight reuse); below that the extra parallelism of 32-row tiles
# wins. The tile sizes were measured on the H20 mHC shapes.
_PROJ_LARGE_M = 4096
_PROJ_BLOCK_M_LARGE = 64
_PROJ_BLOCK_K_LARGE = 128
_PROJ_BLOCK_M_SMALL = 32
_PROJ_BLOCK_K_SMALL = 256

# Triton allocates one shared-memory buffer per pipelined tile per stage, so
# the projection kernel needs _NUM_STAGES * (x_tile + weight_tile) bytes. That
# overruns devices with a 128 KiB limit (Iluvatar reports "Required: 147456,
# Hardware limit: 131072" for 32x256 tiles with a bf16 x and fp32 weight), so
# _proj_config halves BLOCK_K until the estimate fits. 128 KiB is the tightest
# limit among the supported platforms; devices with more keep the larger tile.
_SHARED_MEMORY_BUDGET = 128 * 1024

_MIN_BLOCK_N = 16

_FORWARD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_input(tensor, name, dtypes):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} needs a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype not in dtypes:
        wanted = " / ".join(str(dtype) for dtype in dtypes)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


def _check_matrix(tensor, name, dtypes):
    _check_input(tensor, name, dtypes)
    if tensor.dim() != 2:
        raise ValueError(f"{name} needs a 2-D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} needs a contiguous tensor; call .contiguous() on the argument"
        )


def _check_scalar(tensor, name, dtypes):
    _check_input(tensor, name, dtypes)
    if tensor.numel() != 1:
        raise ValueError(f"{name} needs one element, got shape {tuple(tensor.shape)}")


@libentry()
@triton.jit
def _rms_norm_kernel(
    x_ptr,
    r_ptr,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
):
    """``r = ||x|| / sqrt(K)`` for BLOCK_M rows, pipelined over K.

    The loop body is only a load, an fp32 square and a row reduction: this
    kernel has no ``tl.dot``, so the dot/reduction pipelining miscompile does
    not apply, and no tile-sized square accumulator is needed. The simpler
    form also compiles correctly on vendor Triton forks (a ``[BLOCK_M,
    BLOCK_K]`` accumulator with a single-iteration pipelined loop produced a
    wrong ``r`` on Iluvatar's 3.2 fork).
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    ks = tl.arange(0, BLOCK_K)
    # Row offsets scale by K (up to M*K elements), which can exceed int32.
    x_rows = x_ptr + rows.to(tl.int64)[:, None] * K

    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=STAGES):
        ks_mask = k0 + ks < K
        x_tile = tl.load(
            x_rows + (k0 + ks)[None, :],
            mask=row_mask[:, None] & ks_mask[None, :],
            other=0.0,
        )
        # Squaring in bf16/fp16 loses most of the RMS scale's precision.
        x_fp32 = x_tile.to(tl.float32)
        sum_sq += tl.sum(x_fp32 * x_fp32, axis=1)

    r = tl.sqrt(sum_sq / K.to(tl.float32))
    tl.store(r_ptr + rows, r, mask=row_mask)


@libentry()
@triton.jit
def _proj_compute_h_kernel(
    x_ptr,
    weight_ptr,
    alpha_pre_ptr,
    alpha_post_ptr,
    alpha_res_ptr,
    bias_ptr,
    r_ptr,
    h_pre_ptr,
    h_post_ptr,
    h_res_ptr,
    proj_ptr,
    M,
    K,
    eps,
    compute_h_eps,
    N: tl.constexpr,
    N_RES: tl.constexpr,
    n: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
):
    """Projection, scale by ``r``, and the three compute_h segment outputs.

    ``BLOCK_N`` is ``N`` rounded up to a power of two, so one tile covers the
    whole weight row count and the segment layout (pre / post / res) is a
    column mask rather than a loop. The projection tile stays in registers
    through the epilogue.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    ks = tl.arange(0, BLOCK_K)
    rows64 = rows.to(tl.int64)

    # Scalar gating factors; the segment each one applies to is a column mask.
    alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)
    alpha = tl.where(cols < n, alpha_pre, tl.where(cols < 2 * n, alpha_post, alpha_res))
    bias_tile = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_rows = x_ptr + rows64[:, None] * K
    weight_rows = weight_ptr + cols[None, :] * K
    for k0 in tl.range(0, K, BLOCK_K, num_stages=STAGES):
        ks_mask = k0 + ks < K
        x_tile = tl.load(
            x_rows + (k0 + ks)[None, :],
            mask=row_mask[:, None] & ks_mask[None, :],
            other=0.0,
        )
        weight_tile = tl.load(
            weight_rows + (k0 + ks)[:, None],
            mask=ks_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        if FP32_INPUT:
            # Backend-default precision (tf32 on NVIDIA), matching the tf32 MMA
            # of Megatron's cuTile forward; fp32 IEEE dot is ~10x slower here.
            acc = tl.dot(x_tile.to(tl.float32), weight_tile.to(tl.float32), acc=acc)
        else:
            acc = tl.dot(x_tile, weight_tile, acc=acc)

    r = tl.load(r_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    inv_r_eps = 1.0 / (r + eps)
    h = acc * (alpha[None, :] * inv_r_eps[:, None]) + bias_tile[None, :]
    sigmoid = tl.sigmoid(h)

    pre_mask = row_mask[:, None] & (cols < n)[None, :]
    tl.store(
        h_pre_ptr + rows64[:, None] * n + cols[None, :],
        sigmoid + compute_h_eps,
        mask=pre_mask,
    )

    post_mask = row_mask[:, None] & ((cols >= n) & (cols < 2 * n))[None, :]
    tl.store(
        h_post_ptr + rows64[:, None] * n + (cols - n)[None, :],
        sigmoid * 2.0,
        mask=post_mask,
    )

    res_mask = row_mask[:, None] & ((cols >= 2 * n) & col_mask)[None, :]
    tl.store(
        h_res_ptr + rows64[:, None] * N_RES + (cols - 2 * n)[None, :],
        h,
        mask=res_mask,
    )

    tl.store(
        proj_ptr + rows64[:, None] * N + cols[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _pipeline_stages(K, block_k, max_stages=_NUM_STAGES):
    """One K tile means no cross-iteration overlap, so do not pipeline it."""
    return 1 if K <= block_k else max_stages


def _norm_config(K, x_itemsize):
    """Tile sizes and pipeline depth for the RMS kernel."""
    block_k = max(16, min(_NORM_BLOCK_K, triton.next_power_of_2(max(K, 1))))
    stages = _pipeline_stages(K, block_k, _NORM_STAGES)
    while block_k > 16 and (
        stages * _NORM_BLOCK_M * block_k * x_itemsize > _SHARED_MEMORY_BUDGET
    ):
        block_k //= 2
        stages = _pipeline_stages(K, block_k, _NORM_STAGES)
    return _NORM_BLOCK_M, block_k, stages


def _proj_shared_bytes(block_m, block_n, block_k, x_itemsize, weight_itemsize, stages):
    """Shared memory Triton allocates for the projection kernel's pipeline."""
    return stages * (
        block_m * block_k * x_itemsize + block_k * block_n * weight_itemsize
    )


def _proj_config(M, N, K, x_itemsize, weight_itemsize):
    """Tile sizes for the projection kernel; tl.dot needs axes of at least 16.

    ``BLOCK_K`` also has to keep the pipelined tile buffers within
    ``_SHARED_MEMORY_BUDGET``; the tile bytes are what Triton reports as
    ``metadata.shared``, so this is the same quantity the launch checks.
    """
    block_n = max(_MIN_BLOCK_N, triton.next_power_of_2(N))
    if M >= _PROJ_LARGE_M:
        block_m, block_k = _PROJ_BLOCK_M_LARGE, _PROJ_BLOCK_K_LARGE
    else:
        block_m, block_k = _PROJ_BLOCK_M_SMALL, _PROJ_BLOCK_K_SMALL
    block_k = max(16, min(block_k, triton.next_power_of_2(max(K, 1))))
    stages = _pipeline_stages(K, block_k)
    while (
        block_k > 16
        and _proj_shared_bytes(
            block_m, block_n, block_k, x_itemsize, weight_itemsize, stages
        )
        > _SHARED_MEMORY_BUDGET
    ):
        block_k //= 2
        stages = _pipeline_stages(K, block_k)
    return block_m, block_n, block_k, stages


def fused_proj_rms_compute_h_forward(
    x,
    weight,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias,
    n,
    eps=1e-6,
    compute_h_eps=1e-6,
):
    """Fused projection, RMS scale, and mHC ``compute_h`` forward.

    Args:
        x: ``[M, K]`` contiguous activations; ``K = n * hidden_size``.
        weight: ``[N, K]`` contiguous projection weight, ``N = n * n + 2 * n``.
        alpha_pre: single-element tensor scaling the first ``n`` columns.
        alpha_post: single-element tensor scaling the next ``n`` columns.
        alpha_res: single-element tensor scaling the remaining ``n * n`` columns.
        bias: ``[N]`` bias added before the activations.
        n: number of residual streams; a Python int, constant for the launch.
        eps: added to the RMS before its reciprocal (default ``1e-6``).
        compute_h_eps: added to ``sigmoid(h_pre)`` (default ``1e-6``).

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            ``h_pre`` ``[M, n]``, ``h_post`` ``[M, n]``, ``h_res`` ``[M, n*n]``,
            ``r`` ``[M, 1]`` and the reduced ``proj`` ``[M, N]`` (kept for the
            backward, as Megatron's cuTile forward does).

    This is the forward only: it builds no autograd graph, matching Megatron's
    ``_cutile_proj_rms_compute_h_fwd``, which its ``torch.autograd.Function``
    calls.
    """
    logger.debug("TRAIN FUSED_PROJ_RMS_COMPUTE_H_FORWARD")

    _check_matrix(x, "x", _FORWARD_DTYPES)
    _check_matrix(weight, "weight", _FORWARD_DTYPES)
    _check_input(bias, "bias", _FORWARD_DTYPES)
    for name, tensor in (
        ("alpha_pre", alpha_pre),
        ("alpha_post", alpha_post),
        ("alpha_res", alpha_res),
    ):
        _check_scalar(tensor, name, _FORWARD_DTYPES)
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n needs an int, got {type(n).__name__}")
    if n <= 0:
        raise ValueError(f"n needs to be positive, got {n}")

    M, K = x.shape
    N = weight.shape[0]
    if weight.shape[1] != K:
        raise ValueError(f"weight's K ({weight.shape[1]}) must match x's K ({K})")
    if N != n * n + 2 * n:
        raise ValueError(
            f"weight's N ({N}) must be n * n + 2 * n ({n * n + 2 * n}) for n={n}"
        )
    if K == 0:
        raise ValueError("x needs at least one column (K > 0)")
    if bias.dim() != 1 or bias.shape[0] != N:
        raise ValueError(f"bias needs shape ({N},), got {tuple(bias.shape)}")
    if not bias.is_contiguous():
        raise ValueError(
            "bias needs a contiguous tensor; call .contiguous() on the argument"
        )

    matmul_dtype = torch.promote_types(x.dtype, weight.dtype)
    # Matches the reference's promotion: proj is compute dtype, but the alpha
    # and bias operands widen the epilogue (they are fp32 in the mHC module).
    h_dtype = torch.promote_types(
        torch.promote_types(matmul_dtype, alpha_pre.dtype),
        torch.promote_types(
            torch.promote_types(alpha_post.dtype, alpha_res.dtype), bias.dtype
        ),
    )

    device = x.device
    h_pre = torch.empty(M, n, dtype=h_dtype, device=device)
    h_post = torch.empty(M, n, dtype=h_dtype, device=device)
    h_res = torch.empty(M, N - 2 * n, dtype=h_dtype, device=device)
    r = torch.empty(M, 1, dtype=matmul_dtype, device=device)
    proj = torch.empty(M, N, dtype=matmul_dtype, device=device)

    if M == 0:
        # No launch when there are no rows to map.
        return h_pre, h_post, h_res, r, proj

    norm_m, norm_k, norm_stages = _norm_config(K, x.element_size())
    block_m, block_n, block_k, proj_stages = _proj_config(
        M, N, K, x.element_size(), weight.element_size()
    )

    with torch_device_fn.device(device):
        _rms_norm_kernel[(triton.cdiv(M, norm_m),)](
            x,
            r,
            M,
            K,
            BLOCK_M=norm_m,
            BLOCK_K=norm_k,
            STAGES=norm_stages,
            num_warps=_NUM_WARPS,
            num_stages=norm_stages,
        )
        _proj_compute_h_kernel[(triton.cdiv(M, block_m),)](
            x,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            r,
            h_pre,
            h_post,
            h_res,
            proj,
            M,
            K,
            float(eps),
            float(compute_h_eps),
            N=N,
            N_RES=N - 2 * n,
            n=n,
            FP32_INPUT=matmul_dtype == torch.float32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            STAGES=proj_stages,
            num_warps=_NUM_WARPS,
            num_stages=proj_stages,
        )

    return h_pre, h_post, h_res, r, proj
