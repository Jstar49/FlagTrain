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

The oracle is autograd through Megatron's ``fused_proj_rms_compute_h`` torch
path (a verbatim copy of ``_torch_proj_rms_compute_h`` from
``megatron/core/fusions/fused_mhc_kernels.py``, including the internal reduced
projection). The test feeds the backward the oracle's own saved forward
tensors, so it compares the backward kernels rather than compounding two
forward implementations. Under ``--ref cpu`` the oracle runs on the CPU with
float64 inputs.

Run with::

    pytest -m fused_proj_rms_compute_h_backward tests/megatron -vs
"""

import math

import pytest
import torch

import flag_train
from flag_train.megatron import fused_proj_rms_compute_h_backward

from .. import accuracy_utils as utils

# M=0 exercises the no-launch path, M=1 the row masks, n=3 gives N=15 (padded
# to 16), K=1000 the K tail, and (128, 4, 4096) the mHC shape.
_CASES = [
    (0, 4, 128),
    (1, 2, 128),
    (17, 3, 1000),
    (128, 4, 4096),
    (256, 2, 512),
]

# Absolute tolerances against the autograd oracle. fp32 matmuls use the
# backend's default precision (tf32 on NVIDIA) while the oracle is IEEE; the
# scalar gradients additionally keep their per-step rounding in fp16/bf16.
_ATOL = {torch.float32: 5e-3, torch.float16: 2e-3, torch.bfloat16: 1e-2}


def _reference_forward(
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
    """Megatron's ``_torch_proj_rms_compute_h`` plus its internal projection.

    Copied from ``megatron/core/fusions/fused_mhc_kernels.py`` at Megatron-LM
    ``e3f528380``. The reduced projection is returned as well so the test can
    feed the backward the same saved tensors the oracle differentiates.
    """
    x = x.to(torch.promote_types(x.dtype, weight.dtype))
    proj = torch.matmul(x, weight.t())
    r = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
    alpha = torch.cat(
        [
            alpha_pre.expand(n),
            alpha_post.expand(n),
            alpha_res.expand(weight.shape[0] - 2 * n),
        ],
        dim=-1,
    )
    h = proj * alpha.unsqueeze(0) / (r + eps) + bias.unsqueeze(0)
    h_pre = h[..., :n].sigmoid() + compute_h_eps
    h_post = h[..., n : 2 * n].sigmoid() * 2
    h_res = h[..., 2 * n :]
    return h_pre, h_post, h_res, r, proj


def _to_reference(tensor):
    """Move a tensor to the oracle's device/dtype (CPU float64 under --ref cpu)."""
    if tensor is None:
        return None
    return tensor.cpu().double() if utils.TO_CPU else tensor


def _inputs(M, n, K, x_dtype, weight_dtype, aux_dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    device = flag_train.device
    N = n * n + 2 * n
    x = torch.empty(M, K, dtype=x_dtype, device=device).uniform_(-0.1, 0.1)
    weight = torch.empty(N, K, dtype=weight_dtype, device=device).uniform_(-0.1, 0.1)
    alpha_pre = torch.empty(1, dtype=aux_dtype, device=device).uniform_(0.005, 0.02)
    alpha_post = torch.empty(1, dtype=aux_dtype, device=device).uniform_(0.005, 0.02)
    alpha_res = torch.empty(1, dtype=aux_dtype, device=device).uniform_(0.005, 0.02)
    bias = torch.empty(N, dtype=aux_dtype, device=device).uniform_(-0.05, 0.05)
    return x, weight, alpha_pre, alpha_post, alpha_res, bias


def _upstream(M, n, N, dtype, seed=0):
    torch.manual_seed(seed + 1)
    device = flag_train.device
    return (
        torch.empty(M, n, dtype=dtype, device=device).uniform_(-0.1, 0.1),
        torch.empty(M, n, dtype=dtype, device=device).uniform_(-0.1, 0.1),
        torch.empty(M, N - 2 * n, dtype=dtype, device=device).uniform_(-0.1, 0.1),
        torch.empty(M, 1, dtype=dtype, device=device).uniform_(-0.1, 0.1),
    )


def _reference_grads(
    x,
    weight,
    grad_h_pre,
    grad_h_post,
    grad_h_res,
    grad_r_ext,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias,
    n,
    eps=1e-6,
    compute_h_eps=1e-6,
):
    """Autograd through the oracle forward on the same upstream gradients."""
    x, weight, alpha_pre, alpha_post, alpha_res, bias = [
        _to_reference(tensor).clone().requires_grad_(True)
        for tensor in (x, weight, alpha_pre, alpha_post, alpha_res, bias)
    ]
    grad_h_pre = _to_reference(grad_h_pre)
    grad_h_post = _to_reference(grad_h_post)
    grad_h_res = _to_reference(grad_h_res)
    grad_r_ext = _to_reference(grad_r_ext)

    h_pre, h_post, h_res, r, _proj = _reference_forward(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps, compute_h_eps
    )
    loss = (
        (
            h_pre * (grad_h_pre if grad_h_pre is not None else torch.zeros_like(h_pre))
        ).sum()
        + (
            h_post
            * (grad_h_post if grad_h_post is not None else torch.zeros_like(h_post))
        ).sum()
        + (
            h_res * (grad_h_res if grad_h_res is not None else torch.zeros_like(h_res))
        ).sum()
    )
    if grad_r_ext is not None:
        loss = loss + (r * grad_r_ext).sum()
    return torch.autograd.grad(
        loss, [x, weight, alpha_pre, alpha_post, alpha_res, bias]
    )


def _saved_forward_outputs(x, weight, alpha_pre, alpha_post, alpha_res, bias, n):
    """Oracle forward outputs, moved to the device and dtypes the backward uses."""
    with torch.no_grad():
        outputs = _reference_forward(
            _to_reference(x),
            _to_reference(weight),
            _to_reference(alpha_pre),
            _to_reference(alpha_post),
            _to_reference(alpha_res),
            _to_reference(bias),
            n,
        )
    matmul_dtype = torch.promote_types(x.dtype, weight.dtype)
    h_dtype = torch.promote_types(
        torch.promote_types(matmul_dtype, alpha_pre.dtype),
        torch.promote_types(
            torch.promote_types(alpha_post.dtype, alpha_res.dtype), bias.dtype
        ),
    )
    if utils.TO_CPU:
        # The oracle differentiates in float64; feed the backward float32 saved
        # tensors so their rounding does not dominate the comparison.
        matmul_dtype = torch.float32
        h_dtype = torch.float32
    h_pre, h_post, h_res, r, proj = outputs
    device = x.device
    # h_res is a slice of the [M, N] intermediate; the backward takes the
    # contiguous tensors the forward actually returns.
    return (
        h_pre.contiguous().to(device=device, dtype=h_dtype),
        h_post.contiguous().to(device=device, dtype=h_dtype),
        h_res.contiguous().to(device=device, dtype=h_dtype),
        r.contiguous().to(device=device, dtype=matmul_dtype),
        proj.contiguous().to(device=device, dtype=matmul_dtype),
        h_dtype,
    )


def _check(got, expected, dtype):
    expected = expected.to(dtype)
    if not utils.TO_CPU:
        expected = expected.to(got.device)
    utils.train_assert_close(got, expected, dtype, atol=_ATOL[dtype])


def _run(M, n, K, x_dtype, weight_dtype, aux_dtype, grads, with_r, seed=0):
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        M, n, K, x_dtype, weight_dtype, aux_dtype, seed
    )
    N = n * n + 2 * n
    h_pre, h_post, h_res, r, proj, h_dtype = _saved_forward_outputs(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n
    )
    use_pre, use_post, use_res = grads
    gy_pre, gy_post, gy_res, gy_r = _upstream(M, n, N, h_dtype, seed)
    if not use_pre:
        gy_pre = None
    if not use_post:
        gy_post = None
    if not use_res:
        gy_res = None
    if not with_r:
        gy_r = None

    got = fused_proj_rms_compute_h_backward(
        x,
        weight,
        gy_pre,
        gy_post,
        gy_res,
        h_pre,
        h_post,
        h_res,
        proj,
        r,
        gy_r,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
    )
    expected = _reference_grads(
        x,
        weight,
        gy_pre,
        gy_post,
        gy_res,
        gy_r,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
    )
    dtypes = [x_dtype, weight_dtype, aux_dtype, aux_dtype, aux_dtype, aux_dtype]
    for result, ref, dtype in zip(got, expected, dtypes):
        _check(result, ref, dtype)


@pytest.mark.parametrize("M,n,K", _CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_backward_vs_reference(M, n, K, dtype):
    """Catch wrong activation derivatives, segment layout, RMS term, or reductions."""
    if dtype == torch.bfloat16 and not utils.bf16_is_supported:
        pytest.skip("bfloat16 is not supported on this device")
    _run(M, n, K, dtype, dtype, dtype, (True, True, True), True)


@pytest.mark.parametrize("M,n,K", [(33, 4, 2048), (64, 2, 512)])
def test_backward_activation_dtype_with_fp32_weight(M, n, K):
    """mHC's real case: bf16 activations with keep_in_fp32 mapping parameters."""
    x_dtype = torch.bfloat16 if utils.bf16_is_supported else torch.float16
    _run(M, n, K, x_dtype, torch.float32, torch.float32, (True, True, True), True)


@pytest.mark.parametrize(
    "grads,with_r",
    [
        ((True, False, True), False),
        ((False, False, False), True),
        ((False, True, False), False),
    ],
)
def test_backward_partial_grads(grads, with_r):
    """Catch treating an unused forward output as a zero gradient."""
    _run(64, 4, 512, torch.float32, torch.float32, torch.float32, grads, with_r)


def test_backward_rejects_bad_arguments():
    """Catch silent wrong layouts and mis-sized saved tensors before launching."""
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        8, 4, 256, torch.float32, torch.float32
    )
    N = 4 * 4 + 2 * 4
    h_pre = torch.zeros(8, 4, device=flag_train.device)
    h_post = torch.zeros(8, 4, device=flag_train.device)
    h_res = torch.zeros(8, N - 8, device=flag_train.device)
    proj = torch.zeros(8, N, device=flag_train.device)
    r = torch.zeros(8, 1, device=flag_train.device)
    gy_pre = torch.zeros_like(h_pre)
    gy_post = torch.zeros_like(h_post)
    gy_res = torch.zeros_like(h_res)

    def call(**overrides):
        kwargs = {
            "x": x,
            "weight": weight,
            "grad_h_pre": gy_pre,
            "grad_h_post": gy_post,
            "grad_h_res": gy_res,
            "h_pre": h_pre,
            "h_post": h_post,
            "h_res": h_res,
            "proj": proj,
            "r": r,
            "grad_r_ext": None,
            "alpha_pre": alpha_pre,
            "alpha_post": alpha_post,
            "alpha_res": alpha_res,
            "bias": bias,
            "n": 4,
        }
        kwargs.update(overrides)
        return fused_proj_rms_compute_h_backward(**kwargs)

    with pytest.raises(ValueError, match="contiguous"):
        call(x=x.t())
    with pytest.raises(ValueError, match="n \\* n \\+ 2 \\* n"):
        call(n=3)
    with pytest.raises(ValueError, match="grad_h_pre"):
        call(grad_h_pre=gy_pre[:, :-1])
    with pytest.raises(ValueError, match="grad_r_ext"):
        call(grad_r_ext=r[:-1])
    with pytest.raises(ValueError, match="positive"):
        call(n=0)
