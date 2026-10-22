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

The matmul is the only reduction over ``K``, so the kernel maps one program per
``BLOCK_M`` rows of ``x`` and loops over ``K``. The projection tile stays in
registers: it is scaled, biased, activated and scattered to the three segment
outputs without a second pass over memory. The RMS square sum is accumulated in
fp32 in the same loop, so ``x`` is read once for both outputs.

Precision: with fp32 operands ``tl.dot`` uses the backend's default input
precision (tf32 on NVIDIA), which is what Megatron's cuTile forward uses
(``tfloat32`` MMA); low-precision activations keep their own dtype for the dot
and accumulate in fp32. The square sum is always fp32. The ``K`` loop opts out
of Triton's software pipelining: with a pipelined loop the same ``x`` tile
feeds both ``tl.dot`` and the reduction, and Triton 3.6 miscompiles that
combination (wrong and non-deterministic accumulator rows), while the
unpipelined loop is verified against the reference; reading ``x`` once still
beats a separate RMS kernel that reads it twice.

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

_BLOCK_M = 32
_BLOCK_K = 256
_MIN_BLOCK_N = 16

_FWD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
def _fused_proj_rms_compute_h_fwd_kernel(
    x_ptr,
    weight_ptr,
    alpha_pre_ptr,
    alpha_post_ptr,
    alpha_res_ptr,
    bias_ptr,
    h_pre_ptr,
    h_post_ptr,
    h_res_ptr,
    r_ptr,
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
):
    """One program covers BLOCK_M rows and reduces the full K axis.

    ``BLOCK_N`` is ``N`` rounded up to a power of two, so one tile covers the
    whole weight row count and the segment layout (pre / post / res) is a
    column mask rather than a loop.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    ks = tl.arange(0, BLOCK_K)
    # Row offsets scale by K (up to M*K elements), which can exceed int32.
    rows64 = rows.to(tl.int64)

    # Scalar gating factors; the segment each one applies to is a column mask.
    alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)
    alpha = tl.where(cols < n, alpha_pre, tl.where(cols < 2 * n, alpha_post, alpha_res))
    bias_tile = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    x_rows = x_ptr + rows64[:, None] * K
    weight_rows = weight_ptr + cols[None, :] * K
    # num_stages=1 is deliberate: a pipelined loop that uses one x tile for both
    # tl.dot and the square-sum reduction is miscompiled by Triton 3.6 (wrong,
    # non-deterministic accumulator rows). Correctness first; the single read of
    # x still beats splitting the RMS into a second kernel.
    for k0 in tl.range(0, K, BLOCK_K, num_stages=1):
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
        # The square sum is fp32 even for a low-precision activation dtype:
        # squaring in bf16/fp16 loses most of the RMS scale's precision.
        x_fp32 = x_tile.to(tl.float32)
        sum_sq += tl.sum(x_fp32 * x_fp32, axis=1)
        if FP32_INPUT:
            # Backend-default precision (tf32 on NVIDIA), matching the tf32 MMA
            # of Megatron's cuTile forward; fp32 IEEE dot is ~10x slower here.
            acc = tl.dot(x_fp32, weight_tile.to(tl.float32), acc=acc)
        else:
            acc = tl.dot(x_tile, weight_tile, acc=acc)

    r = tl.sqrt(sum_sq / K.to(tl.float32))
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

    tl.store(r_ptr + rows, r, mask=row_mask)

    tl.store(
        proj_ptr + rows64[:, None] * N + cols[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _fwd_config(M, N, K):
    """Tile sizes for one launch; tl.dot needs every axis to be at least 16."""
    block_n = max(_MIN_BLOCK_N, triton.next_power_of_2(N))
    block_m = min(_BLOCK_M, max(16, triton.next_power_of_2(max(M, 1))))
    block_k = max(16, min(_BLOCK_K, triton.next_power_of_2(max(K, 1))))
    return block_m, block_n, block_k


def fused_proj_rms_compute_h_fwd(
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
    logger.debug("TRAIN FUSED_PROJ_RMS_COMPUTE_H_FWD")

    _check_matrix(x, "x", _FWD_DTYPES)
    _check_matrix(weight, "weight", _FWD_DTYPES)
    _check_input(bias, "bias", _FWD_DTYPES)
    for name, tensor in (
        ("alpha_pre", alpha_pre),
        ("alpha_post", alpha_post),
        ("alpha_res", alpha_res),
    ):
        _check_scalar(tensor, name, _FWD_DTYPES)
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

    block_m, block_n, block_k = _fwd_config(M, N, K)
    grid = (triton.cdiv(M, block_m),)

    with torch_device_fn.device(device):
        _fused_proj_rms_compute_h_fwd_kernel[grid](
            x,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            h_pre,
            h_post,
            h_res,
            r,
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
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )

    return h_pre, h_post, h_res, r, proj
