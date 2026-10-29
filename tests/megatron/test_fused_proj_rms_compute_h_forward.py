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

The reference is Megatron's own ``fused_proj_rms_compute_h`` forward from
``megatron/core/fusions/fused_mhc_kernels.py``: the installed module when it
ships the mHC kernels, otherwise a verbatim copy of its torch path. It returns
four outputs; the operator's extra ``proj`` (kept for the backward) is not part
of the upstream contract and is not compared. Under ``--ref cpu`` the same
reference runs on the CPU with float64 inputs, so that pass compares against a
higher-precision oracle. Inputs use Megatron's small uniform range so bf16/fp16
magnitudes stay well-conditioned.

Run with::

    pytest -m fused_proj_rms_compute_h_forward tests/megatron -vs
"""

import math

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# M=0 has no launch, M=1 exercises row masking, n=3 gives N=15 (padded to 16),
# K=1000 is not a multiple of the K tile, and (128, 4, 4096) is the mHC shape.
_CASES = [
    (0, 4, 128),
    (1, 2, 128),
    (17, 3, 1000),
    (128, 4, 4096),
    (256, 2, 512),
]

# Explicit absolute tolerances for two measured effects:
#   fp32: the dot uses the backend's default precision (tf32 on NVIDIA, as
#   Megatron's cuTile forward does), so it agrees with the baseline's IEEE fp32
#   torch path only to tf32 accuracy (~1e-3 relative, three orders above the
#   shared fp32 rtol of 1.3e-6).
#   fp16/bf16: the baseline rounds *every* elementwise step of compute_h to the
#   activation dtype, while the kernel keeps the epilogue in fp32 and rounds
#   once at the store. Near-zero outputs (cancellation between the scaled
#   projection and the bias) therefore differ by more than the shared 1e-4
#   atol; the dtype rtol (1e-3 / 0.016) still covers the large values.
_ATOL = {torch.float32: 2e-3, torch.float16: 1e-3, torch.bfloat16: 2e-3}


def _megatron_torch_forward(
    x,
    weight,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias,
    n,
    eps=1e-6,
    compute_h_eps=1e-6,
    *,
    backend=None,
):
    """Megatron's ``fused_proj_rms_compute_h`` torch path, copied verbatim.

    Source: ``megatron/core/fusions/fused_mhc_kernels.py`` at Megatron-LM
    ``6c8f14a3ecbd8ac4095c5d795f2661732381d702``, function
    ``_torch_proj_rms_compute_h`` (the path the public function dispatches to
    when cuTile is unavailable). ``backend`` mirrors the public signature; this
    copy is the torch path, so it is accepted and ignored.

    It returns the public four outputs: the reduced projection is internal to
    the implementation.
    """
    # compute_mappings() hands us activations in the activation dtype while the
    # mapping parameters are keep_in_fp32, so matmul would reject the pair.
    # Compute in the wider of the two, matching the unfused path's fp32 upcast
    # without letting a lower-precision parameter downcast the activations.
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
    return h_pre, h_post, h_res, r


def base_op():
    return _megatron_torch_forward


reference_fused_proj_rms_compute = base_op()


def _reference(*args):
    """Call the Megatron reference; ``--ref cpu`` uses CPU float64 inputs."""
    # Megatron 0.21 wraps its torch path in @torch.compile. Run the reference
    # eagerly: the test is about the operator's semantics, and depending on
    # Inductor's async compile subprocess makes it flaky under suite load.
    import torch._dynamo

    with torch._dynamo.config.patch(disable=True):
        if utils.TO_CPU:
            args = tuple(
                arg.cpu().double() if isinstance(arg, torch.Tensor) else arg
                for arg in args
            )
            # The CPU oracle must take Megatron's torch path, not a device kernel.
            return reference_fused_proj_rms_compute(*args, backend="native")
        return reference_fused_proj_rms_compute(*args)


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
    utils.train_assert_close(got, expected, dtype, atol=_ATOL[dtype])


@pytest.mark.fused_proj_rms_compute_h_forward
@pytest.mark.parametrize("M,n,K", _CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_vs_reference(M, n, K, dtype):
    """Catch wrong segment layout, RMS scale, eps placement, bias, or masking."""
    if dtype == torch.bfloat16 and not utils.bf16_is_supported:
        pytest.skip("bfloat16 is not supported on this device")
    # Auxiliary parameters share the activation dtype here so the outputs stay
    # in `dtype`; the fp32 auxiliaries of the mHC module are covered below.
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        M, n, K, dtype, dtype, aux_dtype=dtype
    )
    expected = _reference(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, 1e-6, 1e-6
    )
    got = flag_train.megatron.fused_proj_rms_compute_h_forward(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n
    )
    # Megatron's forward returns four tensors; `proj` is our backward extra.
    for result, ref in zip(got[:4], expected):
        _check(result, ref, dtype)


@pytest.mark.fused_proj_rms_compute_h_forward
@pytest.mark.parametrize("M,n,K", [(33, 4, 2048), (64, 2, 512)])
def test_forward_activation_dtype_with_fp32_weight(M, n, K):
    """mHC's real case: bf16 activations with keep_in_fp32 mapping parameters."""
    x_dtype = torch.bfloat16 if utils.bf16_is_supported else torch.float16
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        M, n, K, x_dtype, torch.float32
    )
    expected = _reference(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n, 1e-6, 1e-6
    )
    got = flag_train.megatron.fused_proj_rms_compute_h_forward(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, n
    )
    # promote_types(activation, fp32) makes every output fp32.
    # Megatron's forward returns four tensors; `proj` is our backward extra.
    for result, ref in zip(got[:4], expected):
        _check(result, ref, torch.float32)


@pytest.mark.fused_proj_rms_compute_h_forward
def test_forward_eps_arguments():
    """Distinct eps values must reach both the RMS denominator and h_pre."""
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        64, 4, 512, torch.float32, torch.float32
    )
    eps, compute_h_eps = 1e-3, 1e-2
    expected = _reference(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, 4, eps, compute_h_eps
    )
    got = flag_train.megatron.fused_proj_rms_compute_h_forward(
        x, weight, alpha_pre, alpha_post, alpha_res, bias, 4, eps, compute_h_eps
    )
    # Megatron's forward returns four tensors; `proj` is our backward extra.
    for result, ref in zip(got[:4], expected):
        _check(result, ref, torch.float32)


@pytest.mark.fused_proj_rms_compute_h_forward
def test_forward_rejects_bad_arguments():
    """Catch silent wrong layouts and mis-sized parameters before launching."""
    x, weight, alpha_pre, alpha_post, alpha_res, bias = _inputs(
        8, 4, 256, torch.float32, torch.float32
    )
    with pytest.raises(ValueError, match="contiguous"):
        flag_train.megatron.fused_proj_rms_compute_h_forward(
            x.t(), weight, alpha_pre, alpha_post, alpha_res, bias, 4
        )
    with pytest.raises(ValueError, match="n \\* n \\+ 2 \\* n"):
        flag_train.megatron.fused_proj_rms_compute_h_forward(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, 3
        )
    with pytest.raises(ValueError, match="bias"):
        flag_train.megatron.fused_proj_rms_compute_h_forward(
            x, weight, alpha_pre, alpha_post, alpha_res, bias[:-1], 4
        )
    with pytest.raises(ValueError, match="positive"):
        flag_train.megatron.fused_proj_rms_compute_h_forward(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, 0
        )
