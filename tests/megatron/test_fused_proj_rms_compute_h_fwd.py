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

The oracle is Megatron's ``_torch_proj_rms_compute_h`` evaluated on the CPU in
float64, independent of the Triton kernel. Inputs use Megatron's small uniform
range so bf16/fp16 magnitudes stay well-conditioned.

Run with::

    pytest -m fused_proj_rms_compute_h tests/megatron -vs
"""

import math

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

pytestmark = pytest.mark.fused_proj_rms_compute_h

# M=0 has no launch, M=1 exercises row masking, n=3 gives N=15 (padded to 16),
# K=1000 is not a multiple of the K tile, and (128, 4, 4096) is the mHC shape.
_CASES = [
    (0, 4, 128),
    (1, 2, 128),
    (17, 3, 1000),
    (128, 4, 4096),
    (256, 2, 512),
]

# fp32 inputs use the backend's default dot precision (tf32 on NVIDIA, as
# Megatron's cuTile forward does), so they agree with an IEEE-fp32/fp64
# reference only to tf32 accuracy: ~1e-3 relative, three orders above the shared
# fp32 rtol (1.3e-6). fp16/bf16 need no override -- their shared tolerances
# (1e-3 / 0.016) already cover the dot's own rounding.
_FP32_ATOL = 2e-3


def _reference(x, weight, alpha_pre, alpha_post, alpha_res, bias, n, eps, cheps):
    """Megatron ``_torch_proj_rms_compute_h`` on CPU in float64."""
    x = x.cpu().double()
    weight = weight.cpu().double()
    proj = torch.matmul(x, weight.t())
    r = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
    N = weight.shape[0]
    alpha = torch.cat(
        [
            alpha_pre.cpu().double().expand(n),
            alpha_post.cpu().double().expand(n),
            alpha_res.cpu().double().expand(N - 2 * n),
        ],
        dim=-1,
    )
    h = proj * alpha.unsqueeze(0) / (r + eps) + bias.cpu().double().unsqueeze(0)
    return (
        h[..., :n].sigmoid() + cheps,
        h[..., n : 2 * n].sigmoid() * 2,
        h[..., 2 * n :],
        r,
        proj,
    )


def base_op():
    # Prefer the installed Megatron; fall back to the CPU fp64 oracle when the
    # package (or this operator) is unavailable, so the test still runs.
    try:
        import megatron.core.fusions.fused_mhc_kernels as fused_mhc_kernels

        return fused_mhc_kernels.fused_proj_rms_compute_h
    except (ImportError, AttributeError):
        return _reference


reference_fused_proj_rms_compute = base_op()


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


def _check(got, expected, dtype):
    # A CPU reference needs to be moved to the device; train_assert_close
    # expects both operands there unless the comparison is CPU-side (--ref cpu).
    expected = expected.to(dtype)
    if not utils.TO_CPU:
        expected = expected.to(got.device)
    if dtype == torch.float32:
        utils.train_assert_close(got, expected, dtype, atol=_FP32_ATOL)
    else:
        utils.train_assert_close(got, expected, dtype)


@pytest.mark.fused_proj_rms_compute_h_fwd
@pytest.mark.parametrize("M,n,K", _CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fwd_vs_reference(M, n, K, dtype):
    """Catch wrong segment layout, RMS scale, eps placement, bias, or masking."""
    if dtype == torch.bfloat16 and not utils.bf16_is_supported:
        pytest.skip("bfloat16 is not supported on this device")
    # Auxiliary parameters share the activation dtype here so the outputs stay
    # in `dtype`; the fp32 auxiliaries of the mHC module are covered below.
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        M, n, K, dtype, dtype, aux_dtype=dtype
    )
    expected = reference_fused_proj_rms_compute(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, 1e-6, 1e-6
    )
    got = flag_train.megatron.fused_proj_rms_compute_h_fwd(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n
    )
    for result, ref in zip(got, expected):
        _check(result, ref, dtype)


@pytest.mark.fused_proj_rms_compute_h_fwd
@pytest.mark.parametrize("M,n,K", [(33, 4, 2048), (64, 2, 512)])
def test_fwd_activation_dtype_with_fp32_weight(M, n, K):
    """mHC's real case: bf16 activations with keep_in_fp32 mapping parameters."""
    x_dtype = torch.bfloat16 if utils.bf16_is_supported else torch.float16
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        M, n, K, x_dtype, torch.float32
    )
    expected = reference_fused_proj_rms_compute(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, 1e-6, 1e-6
    )
    got = flag_train.megatron.fused_proj_rms_compute_h_fwd(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n
    )
    # promote_types(activation, fp32) makes every output fp32.
    for result, ref in zip(got, expected):
        _check(result, ref, torch.float32)


@pytest.mark.fused_proj_rms_compute_h_fwd
def test_fwd_eps_arguments():
    """Distinct eps values must reach both the RMS denominator and h_pre."""
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        64, 4, 512, torch.float32, torch.float32
    )
    eps, compute_h_eps = 1e-3, 1e-2
    expected = reference_fused_proj_rms_compute(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, 4, eps, compute_h_eps
    )
    got = flag_train.megatron.fused_proj_rms_compute_h_fwd(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, 4, eps, compute_h_eps
    )
    for result, ref in zip(got, expected):
        _check(result, ref, torch.float32)


@pytest.mark.fused_proj_rms_compute_h_fwd
def test_fwd_rejects_bad_arguments():
    """Catch silent wrong layouts and mis-sized parameters before launching."""
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        8, 4, 256, torch.float32, torch.float32
    )
    with pytest.raises(ValueError, match="contiguous"):
        flag_train.megatron.fused_proj_rms_compute_h_fwd(
            x.t(), weight, alpha_pre, alpha_post, alpha_res, bias, 4
        )
    with pytest.raises(ValueError, match="n \\* n \\+ 2 \\* n"):
        flag_train.megatron.fused_proj_rms_compute_h_fwd(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, 3
        )
    with pytest.raises(ValueError, match="bias"):
        flag_train.megatron.fused_proj_rms_compute_h_fwd(
            x, weight, alpha_pre, alpha_post, alpha_res, bias[:-1], 4
        )
    with pytest.raises(ValueError, match="positive"):
        flag_train.megatron.fused_proj_rms_compute_h_fwd(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, 0
        )
