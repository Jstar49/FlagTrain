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
"""Backward of Megatron's fused projection + RMS + compute_h mHC mapping.

Forward (``fused_proj_rms_compute_h_fwd``) returns ``h_pre``, ``h_post``,
``h_res``, ``r`` and the reduced ``proj``. With upstream gradients
``grad_h_pre``, ``grad_h_post``, ``grad_h_res`` and optionally ``grad_r``::

    s_pre    = sigmoid(h_linear[:, :n])          = h_pre - compute_h_eps
    s_post   = h_linear[:, n:2n].sigmoid()       = h_post / 2
    grad_h   = [dL/dh_pre * s_pre * (1 - s_pre),
                dL/dh_post * 2 * s_post * (1 - s_post),
                dL/dh_res]                       # [M, N]
    alpha    = [alpha_pre] * n + [alpha_post] * n + [alpha_res] * (N - 2n)
    inv      = 1 / (r + eps)
    grad_proj = grad_h * alpha * inv                       # [M, N]
    grad_r    = sum_seg(grad_h * proj * alpha) * (-inv**2) + grad_r_ext
    grad_x    = grad_proj @ weight + grad_r * x / (r * K)  # [M, K]
    grad_W    = grad_proj.T @ x                            # [N, K]
    grad_alpha_seg = sum_{m, seg}(grad_h * proj * inv)     # [1] each
    grad_bias      = sum_m(grad_h)                         # [N]

This mirrors Megatron's ``_cutile_fused_compute_h_proj_rms_bwd`` and the
autograd of ``_torch_proj_rms_compute_h``. ``grad_h_*`` and ``grad_r_ext`` may
be ``None`` (an unused forward output); their contributions are then zero.

Precision follows the forward: fp32 accumulation everywhere; the two matmuls
use the promoted dtype of ``x`` and ``weight`` with the backend's default dot
precision (tf32 for fp32 operands on NVIDIA). Results are written in the input
dtypes (``grad_x`` like ``x``, ``grad_weight`` like ``weight``, the scalars like
their parameters).

The launch sequence is: one pass over rows producing ``grad_proj``,
``grad_r_total`` and the scalar-gradient partials; one pass over the
``(M, K)`` grid for ``grad_x``; one pass over ``K`` (split over M) for
``grad_weight``; then the two cheap reductions.
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
_GRAD_STAGES = 2

_GRAD_BLOCK_M = 32
_GRAD_BLOCK_K = 128

# Triton allocates one shared buffer per pipelined tile per stage; keep the
# matmul kernels within the smallest limit among the supported platforms.
_SHARED_MEMORY_BUDGET = 128 * 1024
_TARGET_GRAD_WEIGHT_PROGRAMS = 128

_MIN_BLOCK_N = 16
_BWD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_input(tensor, name, dtypes=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} needs a torch.Tensor, got {type(tensor).__name__}")
    if dtypes is not None and tensor.dtype not in dtypes:
        wanted = " / ".join(str(dtype) for dtype in dtypes)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} needs a floating-point tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


def _check_matrix(tensor, name, shape=None):
    _check_input(tensor, name)
    if tensor.dim() != 2:
        raise ValueError(f"{name} needs a 2-D tensor, got shape {tuple(tensor.shape)}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} needs shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} needs a contiguous tensor; call .contiguous() on the argument"
        )


def _check_scalar(tensor, name):
    _check_input(tensor, name, _BWD_DTYPES)
    if tensor.numel() != 1:
        raise ValueError(f"{name} needs one element, got shape {tuple(tensor.shape)}")


def _check_grad(tensor, name, shape):
    if tensor is None:
        return
    _check_matrix(tensor, name, shape)


@libentry()
@triton.jit
def _grad_h_proj_kernel(
    grad_h_pre_ptr,
    grad_h_post_ptr,
    grad_h_res_ptr,
    h_pre_ptr,
    h_post_ptr,
    proj_ptr,
    r_ptr,
    grad_r_ext_ptr,
    alpha_pre_ptr,
    alpha_post_ptr,
    alpha_res_ptr,
    grad_proj_ptr,
    grad_r_total_ptr,
    alpha_pre_partials_ptr,
    alpha_post_partials_ptr,
    alpha_res_partials_ptr,
    bias_partials_ptr,
    M,
    eps,
    compute_h_eps,
    N: tl.constexpr,
    N_RES: tl.constexpr,
    n: tl.constexpr,
    HAS_GRAD_H_PRE: tl.constexpr,
    HAS_GRAD_H_POST: tl.constexpr,
    HAS_GRAD_H_RES: tl.constexpr,
    HAS_GRAD_R_EXT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-row grad_h, grad_proj, grad_r_total and scalar-gradient partials.

    One program covers BLOCK_M rows and the whole N axis; the pre / post / res
    segments are column masks. Grid: (ceil(M / BLOCK_M),).
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    rows64 = rows.to(tl.int64)

    alpha_pre = tl.load(alpha_pre_ptr).to(tl.float32)
    alpha_post = tl.load(alpha_post_ptr).to(tl.float32)
    alpha_res = tl.load(alpha_res_ptr).to(tl.float32)
    alpha = tl.where(cols < n, alpha_pre, tl.where(cols < 2 * n, alpha_post, alpha_res))

    r = tl.load(r_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    inv_r_eps = 1.0 / (r + eps)

    proj_tile = tl.load(
        proj_ptr + rows64[:, None] * N + cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    grad_h = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    pre_mask = row_mask[:, None] & (cols < n)[None, :]
    if HAS_GRAD_H_PRE:
        gy_pre = tl.load(
            grad_h_pre_ptr + rows64[:, None] * n + cols[None, :],
            mask=pre_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        gy_pre = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    h_pre = tl.load(
        h_pre_ptr + rows64[:, None] * n + cols[None, :], mask=pre_mask, other=0.0
    ).to(tl.float32)
    sigmoid_pre = h_pre - compute_h_eps
    grad_h += gy_pre * sigmoid_pre * (1.0 - sigmoid_pre)

    post_mask = row_mask[:, None] & ((cols >= n) & (cols < 2 * n))[None, :]
    if HAS_GRAD_H_POST:
        gy_post = tl.load(
            grad_h_post_ptr + rows64[:, None] * n + (cols - n)[None, :],
            mask=post_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        gy_post = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    h_post = tl.load(
        h_post_ptr + rows64[:, None] * n + (cols - n)[None, :],
        mask=post_mask,
        other=0.0,
    ).to(tl.float32)
    sigmoid_post = h_post * 0.5
    grad_h += gy_post * sigmoid_post * (1.0 - sigmoid_post) * 2.0

    res_mask = row_mask[:, None] & ((cols >= 2 * n) & col_mask)[None, :]
    if HAS_GRAD_H_RES:
        grad_h += tl.load(
            grad_h_res_ptr + rows64[:, None] * N_RES + (cols - 2 * n)[None, :],
            mask=res_mask,
            other=0.0,
        ).to(tl.float32)

    grad_proj = grad_h * alpha[None, :] * inv_r_eps[:, None]
    tl.store(
        grad_proj_ptr + rows64[:, None] * N + cols[None, :],
        grad_proj,
        mask=row_mask[:, None] & col_mask[None, :],
    )

    grad_r = tl.sum(grad_h * proj_tile * alpha[None, :], axis=1) * (
        -inv_r_eps * inv_r_eps
    )
    if HAS_GRAD_R_EXT:
        grad_r += tl.load(grad_r_ext_ptr + rows, mask=row_mask, other=0.0).to(
            tl.float32
        )
    tl.store(grad_r_total_ptr + rows, grad_r, mask=row_mask)

    # Scalar gradients: per-block partials, reduced by the next kernel.
    ga = grad_h * proj_tile * inv_r_eps[:, None]
    mask_pre = (cols < n).to(tl.float32)
    mask_post = ((cols >= n) & (cols < 2 * n)).to(tl.float32)
    mask_res = ((cols >= 2 * n) & col_mask).to(tl.float32)
    tl.store(
        alpha_pre_partials_ptr + pid,
        tl.sum(tl.sum(ga * mask_pre[None, :], axis=1), axis=0),
    )
    tl.store(
        alpha_post_partials_ptr + pid,
        tl.sum(tl.sum(ga * mask_post[None, :], axis=1), axis=0),
    )
    tl.store(
        alpha_res_partials_ptr + pid,
        tl.sum(tl.sum(ga * mask_res[None, :], axis=1), axis=0),
    )
    tl.store(
        bias_partials_ptr + pid * BLOCK_N + cols,
        tl.sum(grad_h, axis=0),
    )


@libentry()
@triton.jit
def _grad_x_kernel(
    x_ptr,
    weight_ptr,
    grad_proj_ptr,
    grad_r_total_ptr,
    r_ptr,
    grad_x_ptr,
    M,
    K,
    N: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
):
    """grad_x = grad_proj @ weight + grad_r_total * x / (r * K).

    Grid: (ceil(M / BLOCK_M), ceil(K / BLOCK_K)); each tile is independent.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = ks < K
    rows64 = rows.to(tl.int64)

    grad_proj = tl.load(
        grad_proj_ptr + rows64[:, None] * N + cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weight_tile = tl.load(
        weight_ptr + cols[:, None] * K + ks[None, :],
        mask=col_mask[:, None] & k_mask[None, :],
        other=0.0,
    )
    if FP32_INPUT:
        acc = tl.dot(grad_proj, weight_tile.to(tl.float32))
    else:
        acc = tl.dot(grad_proj.to(weight_tile.dtype), weight_tile)

    grad_r_total = tl.load(grad_r_total_ptr + rows, mask=row_mask, other=0.0).to(
        tl.float32
    )
    r = tl.load(r_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    coef = grad_r_total / (r * K.to(tl.float32))
    x_tile = tl.load(
        x_ptr + rows64[:, None] * K + ks[None, :],
        mask=row_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc += coef[:, None] * x_tile
    tl.store(
        grad_x_ptr + rows64[:, None] * K + ks[None, :],
        acc,
        mask=row_mask[:, None] & k_mask[None, :],
    )


@libentry()
@triton.jit
def _grad_weight_kernel(
    x_ptr,
    grad_proj_ptr,
    partial_ptr,
    M,
    K,
    N: tl.constexpr,
    FP32_INPUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
):
    """Split-M partials of grad_weight = grad_proj.T @ x.

    Grid: (ceil(K / BLOCK_K), SPLIT_M). Row range for split ``s`` is
    ``[s * chunk, min(M, (s + 1) * chunk))``; partials are fp32.
    """
    pid_k = tl.program_id(0)
    split = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = ks < K

    chunk = tl.cdiv(M, tl.num_programs(1))
    m_start = split * chunk
    m_end = tl.minimum(M, m_start + chunk)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for m0 in tl.range(m_start, m_end, BLOCK_M, num_stages=STAGES):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_mask = rows < m_end
        rows64 = rows.to(tl.int64)
        grad_proj = tl.load(
            grad_proj_ptr + rows64[:, None] * N + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x_tile = tl.load(
            x_ptr + rows64[:, None] * K + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        if FP32_INPUT:
            acc = tl.dot(tl.trans(x_tile.to(tl.float32)), grad_proj, acc=acc)
        else:
            acc = tl.dot(tl.trans(x_tile), grad_proj.to(x_tile.dtype), acc=acc)

    tl.store(
        partial_ptr + split * N * K + cols[None, :] * K + ks[:, None],
        acc,
        mask=col_mask[None, :] & k_mask[:, None],
    )


@libentry()
@triton.jit
def _grad_weight_reduce_kernel(
    partial_ptr,
    grad_weight_ptr,
    N: tl.constexpr,
    K,
    SPLIT_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum the split-M partials and write grad_weight in the weight dtype."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N * K
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in tl.static_range(SPLIT_M):
        acc += tl.load(partial_ptr + split * N * K + offs, mask=mask, other=0.0)
    tl.store(grad_weight_ptr + offs, acc, mask=mask)


@libentry()
@triton.jit
def _scalar_grads_reduce_kernel(
    alpha_pre_partials_ptr,
    alpha_post_partials_ptr,
    alpha_res_partials_ptr,
    bias_partials_ptr,
    grad_alpha_pre_ptr,
    grad_alpha_post_ptr,
    grad_alpha_res_ptr,
    grad_bias_ptr,
    NUM_BLOCKS,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce the per-row-block scalar-gradient partials."""
    cols = tl.arange(0, BLOCK_N)
    acc_pre = 0.0
    acc_post = 0.0
    acc_res = 0.0
    acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for block in range(NUM_BLOCKS):
        acc_pre += tl.load(alpha_pre_partials_ptr + block)
        acc_post += tl.load(alpha_post_partials_ptr + block)
        acc_res += tl.load(alpha_res_partials_ptr + block)
        acc_bias += tl.load(bias_partials_ptr + block * BLOCK_N + cols)

    tl.store(grad_alpha_pre_ptr, acc_pre)
    tl.store(grad_alpha_post_ptr, acc_post)
    tl.store(grad_alpha_res_ptr, acc_res)
    tl.store(grad_bias_ptr + cols, acc_bias, mask=cols < N)


def _dot_config(N, K, x_itemsize, weight_itemsize):
    """Tile sizes for the two matmul kernels; tl.dot needs axes of at least 16."""
    block_n = max(_MIN_BLOCK_N, triton.next_power_of_2(N))
    block_m = _GRAD_BLOCK_M
    block_k = max(16, min(_GRAD_BLOCK_K, triton.next_power_of_2(max(K, 1))))
    stages = _GRAD_STAGES

    def shared(block_k_value, stages_value):
        # x tile + weight tile + fp32 grad_proj tile, per pipeline stage.
        return stages_value * (
            block_m * block_k_value * x_itemsize
            + block_k_value * block_n * weight_itemsize
            + block_m * block_n * 4
        )

    while block_k > 16 and shared(block_k, stages) > _SHARED_MEMORY_BUDGET:
        block_k //= 2
    while stages > 1 and shared(block_k, stages) > _SHARED_MEMORY_BUDGET:
        stages -= 1
    return block_m, block_n, block_k, stages


def _grad_weight_split(K, block_k):
    """Enough M-splits to fill the device when K alone gives few programs."""
    k_tiles = triton.cdiv(max(K, 1), block_k)
    return max(1, min(8, triton.cdiv(_TARGET_GRAD_WEIGHT_PROGRAMS, k_tiles)))


def fused_proj_rms_compute_h_bwd(
    x,
    weight,
    grad_h_pre,
    grad_h_post,
    grad_h_res,
    h_pre,
    h_post,
    h_res,
    proj,
    r,
    grad_r_ext,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias,
    n,
    eps=1e-6,
    compute_h_eps=1e-6,
):
    """Backward of the fused projection + RMS + compute_h forward.

    Args:
        x: ``[M, K]`` contiguous activations saved by the forward.
        weight: ``[N, K]`` contiguous projection weight, ``N = n * n + 2 * n``.
        grad_h_pre: ``[M, n]`` upstream gradient, or ``None``.
        grad_h_post: ``[M, n]`` upstream gradient, or ``None``.
        grad_h_res: ``[M, n * n]`` upstream gradient, or ``None``.
        h_pre: ``[M, n]`` forward output.
        h_post: ``[M, n]`` forward output.
        h_res: ``[M, n * n]`` forward output.
        proj: ``[M, N]`` reduced projection returned by the forward.
        r: ``[M, 1]`` RMS scale returned by the forward.
        grad_r_ext: ``[M, 1]`` gradient of ``r``, or ``None``.
        alpha_pre: single-element tensor scaling the first ``n`` columns.
        alpha_post: single-element tensor scaling the next ``n`` columns.
        alpha_res: single-element tensor scaling the last ``n * n`` columns.
        bias: ``[N]`` bias added before the activations.
        n: number of residual streams; a Python int, constant for the launch.
        eps: added to the RMS before its reciprocal (default ``1e-6``).
        compute_h_eps: added to ``sigmoid(h_pre)`` (default ``1e-6``).

    Returns:
        Tuple of ``grad_x`` ``[M, K]``, ``grad_weight`` ``[N, K]``,
        ``grad_alpha_pre`` / ``grad_alpha_post`` / ``grad_alpha_res`` (one
        element each) and ``grad_bias`` ``[N]``, in the dtypes of the inputs.
    """
    logger.debug("TRAIN FUSED_PROJ_RMS_COMPUTE_H_BWD")

    _check_matrix(x, "x")
    _check_matrix(weight, "weight")
    for name, tensor in (
        ("alpha_pre", alpha_pre),
        ("alpha_post", alpha_post),
        ("alpha_res", alpha_res),
    ):
        _check_scalar(tensor, name)
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n needs an int, got {type(n).__name__}")
    if n <= 0:
        raise ValueError(f"n needs to be positive, got {n}")

    M, K = x.shape
    N = weight.shape[0]
    N_RES = N - 2 * n
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

    _check_matrix(h_pre, "h_pre", (M, n))
    _check_matrix(h_post, "h_post", (M, n))
    _check_matrix(h_res, "h_res", (M, N_RES))
    _check_matrix(proj, "proj", (M, N))
    _check_matrix(r, "r", (M, 1))
    _check_grad(grad_h_pre, "grad_h_pre", (M, n))
    _check_grad(grad_h_post, "grad_h_post", (M, n))
    _check_grad(grad_h_res, "grad_h_res", (M, N_RES))
    if grad_r_ext is not None:
        if grad_r_ext.numel() != M:
            raise ValueError(
                f"grad_r_ext needs {M} elements, got shape {tuple(grad_r_ext.shape)}"
            )
        _check_input(grad_r_ext, "grad_r_ext", _BWD_DTYPES)
        if not grad_r_ext.is_contiguous():
            raise ValueError(
                "grad_r_ext needs a contiguous tensor; call .contiguous() on the argument"
            )

    device = x.device
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    grad_alpha_pre = torch.empty(1, dtype=alpha_pre.dtype, device=device)
    grad_alpha_post = torch.empty(1, dtype=alpha_post.dtype, device=device)
    grad_alpha_res = torch.empty(1, dtype=alpha_res.dtype, device=device)
    grad_bias = torch.empty(N, dtype=bias.dtype, device=device)

    if M == 0:
        grad_weight.zero_()
        grad_alpha_pre.zero_()
        grad_alpha_post.zero_()
        grad_alpha_res.zero_()
        grad_bias.zero_()
        return (
            grad_x,
            grad_weight,
            grad_alpha_pre.view_as(alpha_pre),
            grad_alpha_post.view_as(alpha_post),
            grad_alpha_res.view_as(alpha_res),
            grad_bias,
        )

    matmul_dtype = torch.promote_types(x.dtype, weight.dtype)
    fp32_input = matmul_dtype == torch.float32
    block_m, block_n, block_k, stages = _dot_config(
        N, K, x.element_size(), weight.element_size()
    )
    split_m = _grad_weight_split(K, block_k)

    num_blocks = triton.cdiv(M, block_m)
    grad_proj = torch.empty(M, N, dtype=torch.float32, device=device)
    grad_r_total = torch.empty(M, 1, dtype=torch.float32, device=device)
    alpha_pre_partials = torch.empty(num_blocks, dtype=torch.float32, device=device)
    alpha_post_partials = torch.empty(num_blocks, dtype=torch.float32, device=device)
    alpha_res_partials = torch.empty(num_blocks, dtype=torch.float32, device=device)
    bias_partials = torch.empty(num_blocks, block_n, dtype=torch.float32, device=device)
    weight_partials = torch.empty(split_m, N, K, dtype=torch.float32, device=device)

    with torch_device_fn.device(device):
        _grad_h_proj_kernel[(num_blocks,)](
            grad_h_pre if grad_h_pre is not None else h_pre,
            grad_h_post if grad_h_post is not None else h_post,
            grad_h_res if grad_h_res is not None else h_res,
            h_pre,
            h_post,
            proj,
            r,
            grad_r_ext if grad_r_ext is not None else r,
            alpha_pre,
            alpha_post,
            alpha_res,
            grad_proj,
            grad_r_total,
            alpha_pre_partials,
            alpha_post_partials,
            alpha_res_partials,
            bias_partials,
            M,
            float(eps),
            float(compute_h_eps),
            N=N,
            N_RES=N_RES,
            n=n,
            HAS_GRAD_H_PRE=grad_h_pre is not None,
            HAS_GRAD_H_POST=grad_h_post is not None,
            HAS_GRAD_H_RES=grad_h_res is not None,
            HAS_GRAD_R_EXT=grad_r_ext is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=_NUM_WARPS,
        )
        _grad_x_kernel[(triton.cdiv(M, block_m), triton.cdiv(K, block_k))](
            x,
            weight,
            grad_proj,
            grad_r_total,
            r,
            grad_x,
            M,
            K,
            N=N,
            FP32_INPUT=fp32_input,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            STAGES=stages,
            num_warps=_NUM_WARPS,
            num_stages=stages,
        )
        _grad_weight_kernel[(triton.cdiv(K, block_k), split_m)](
            x,
            grad_proj,
            weight_partials,
            M,
            K,
            N=N,
            FP32_INPUT=fp32_input,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            STAGES=stages,
            num_warps=_NUM_WARPS,
            num_stages=stages,
        )
        _grad_weight_reduce_kernel[(triton.cdiv(N * K, 1024),)](
            weight_partials,
            grad_weight,
            N=N,
            K=K,
            SPLIT_M=split_m,
            BLOCK=1024,
            num_warps=_NUM_WARPS,
        )
        _scalar_grads_reduce_kernel[(1,)](
            alpha_pre_partials,
            alpha_post_partials,
            alpha_res_partials,
            bias_partials,
            grad_alpha_pre,
            grad_alpha_post,
            grad_alpha_res,
            grad_bias,
            num_blocks,
            N=N,
            BLOCK_N=block_n,
            num_warps=_NUM_WARPS,
        )

    return (
        grad_x,
        grad_weight,
        grad_alpha_pre.view_as(alpha_pre),
        grad_alpha_post.view_as(alpha_post),
        grad_alpha_res.view_as(alpha_res),
        grad_bias,
    )
