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
"""Performance of the fused mHC projection + RMS + compute_h backward.

The baseline is the same backward math in eager torch, fed the same saved
forward tensors and upstream gradients, so the measurement isolates the
backward kernels. Its expressions mirror the autograd of Megatron's
``_torch_proj_rms_compute_h`` (and Megatron's cuTile backward).

Shapes, dtypes and the input construction match the forward benchmark; the
saved ``h_pre`` / ``h_post`` / ``h_res`` / ``proj`` / ``r`` come from the
FlagTrain forward once per shape, outside the timed region.

Precision caveat: the operator's fp32 matmuls use the backend's default
precision (tf32 on NVIDIA) while the eager baseline is IEEE fp32, so the
speedup includes that difference (the same caveat as the forward benchmark).

Run::

    pytest benchmark/megatron -m fused_proj_rms_compute_h_backward --level core
"""

import pytest
import torch

import flag_train
from flag_train.megatron import fused_proj_rms_compute_h_forward

from .. import base, consts

# (M, n, K): M is the number of tokens (s * b) and K = n * hidden_size.
_SHAPES = [
    (1024, 4, 16384),
    (4096, 4, 16384),
    (4096, 4, 28672),
    (8192, 4, 8192),
    (4096, 2, 8192),
    (16384, 4, 4096),
]

_ACTIVATION_DTYPES = [torch.float16, torch.bfloat16]

# The mHC module keeps the mapping parameters in fp32 (keep_in_fp32).
_WEIGHT_DTYPE = torch.float32

_TOLERANCE = {torch.float16: 5e-3, torch.bfloat16: 2e-2}


def torch_op(
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
    """Eager backward of Megatron's torch forward from the saved tensors."""
    matmul_dtype = torch.promote_types(x.dtype, weight.dtype)
    inv = 1.0 / (r.float() + eps)
    sigmoid_pre = h_pre.float() - compute_h_eps
    grad_pre = grad_h_pre.float() * sigmoid_pre * (1.0 - sigmoid_pre)
    sigmoid_post = h_post.float() * 0.5
    grad_post = grad_h_post.float() * sigmoid_post * (1.0 - sigmoid_post) * 2.0
    grad_h = torch.cat([grad_pre, grad_post, grad_h_res.float()], dim=-1)

    alpha = torch.cat(
        [
            alpha_pre.float().expand(n),
            alpha_post.float().expand(n),
            alpha_res.float().expand(n * n),
        ],
        dim=-1,
    )
    proj_f32 = proj.float()
    grad_proj = grad_h * alpha * inv
    grad_r = (grad_h * proj_f32 * alpha).sum(-1, keepdim=True) * (-inv * inv)
    if grad_r_ext is not None:
        grad_r = grad_r + grad_r_ext.float()

    x_cast = x.to(matmul_dtype)
    weight_cast = weight.to(matmul_dtype)
    grad_proj_cast = grad_proj.to(matmul_dtype)
    grad_x = torch.matmul(grad_proj_cast, weight_cast)
    grad_x = grad_x + grad_r * x_cast / (r.to(matmul_dtype) * x.shape[-1])
    grad_weight = torch.matmul(grad_proj_cast.t(), x_cast)

    ga = grad_h * proj_f32 * inv
    grad_alpha_pre = ga[:, :n].sum().reshape_as(alpha_pre)
    grad_alpha_post = ga[:, n : 2 * n].sum().reshape_as(alpha_post)
    grad_alpha_res = ga[:, 2 * n :].sum().reshape_as(alpha_res)
    grad_bias = grad_h.sum(dim=0).to(bias.dtype)
    return (
        grad_x.to(x.dtype),
        grad_weight.to(weight.dtype),
        grad_alpha_pre.to(alpha_pre.dtype),
        grad_alpha_post.to(alpha_post.dtype),
        grad_alpha_res.to(alpha_res.dtype),
        grad_bias,
    )


def _input_fn(shape, dtype, device):
    M, n, K = shape
    N = n * n + 2 * n
    torch.manual_seed(0)
    x = torch.empty(M, K, dtype=dtype, device=device).uniform_(-0.1, 0.1)
    weight = torch.empty(N, K, dtype=_WEIGHT_DTYPE, device=device).uniform_(-0.1, 0.1)
    alpha = torch.full((1,), 0.01, dtype=torch.float32, device=device)
    bias = torch.zeros(N, dtype=torch.float32, device=device)
    h_pre, h_post, h_res, r, proj = fused_proj_rms_compute_h_forward(
        x, weight, alpha, alpha, alpha, bias, n
    )

    torch.manual_seed(1)
    grad_h_pre = torch.empty(M, n, dtype=h_pre.dtype, device=device).uniform_(-0.1, 0.1)
    grad_h_post = torch.empty(M, n, dtype=h_post.dtype, device=device).uniform_(
        -0.1, 0.1
    )
    grad_h_res = torch.empty(M, N - 2 * n, dtype=h_res.dtype, device=device).uniform_(
        -0.1, 0.1
    )
    grad_r_ext = torch.empty(M, 1, dtype=r.dtype, device=device).uniform_(-0.1, 0.1)

    args = (
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
        alpha,
        alpha,
        alpha,
        bias,
        n,
    )

    # One untimed correctness check per shape so both sides answer the same
    # question (the backward matmuls use tf32 for fp32 operands).
    expected = torch_op(*args)
    got = flag_train.megatron.fused_proj_rms_compute_h_backward(*args)
    tolerance = _TOLERANCE[dtype]
    for name, result, ref in zip(
        (
            "grad_x",
            "grad_weight",
            "grad_alpha_pre",
            "grad_alpha_post",
            "grad_alpha_res",
            "grad_bias",
        ),
        got,
        expected,
    ):
        torch.testing.assert_close(
            result,
            ref,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg, name=name: f"{name}: {msg}",
        )
    yield args


class ProjRmsComputeHBwdBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "M, n, K"
    DEFAULT_METRICS = consts.DEFAULT_METRICS + ["gbps", "tflops"]

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        return []

    def get_gbps(self, args, latency=None):
        """Logical bytes: saved tensors and upstream grads in, grads out."""
        if latency is None:
            return None
        x, weight = args[0], args[1]
        M, K = x.shape
        N = weight.shape[0]
        grad_h_pre, grad_h_post, grad_h_res = args[2], args[3], args[4]
        h_pre, _, _ = args[5], args[6], args[7]
        proj, r, grad_r_ext, alpha, bias = (
            args[8],
            args[9],
            args[10],
            args[11],
            args[14],
        )

        read_bytes = M * K * x.element_size() + N * K * weight.element_size()
        read_bytes += (
            2
            * (grad_h_pre.numel() + grad_h_post.numel() + grad_h_res.numel())
            * h_pre.element_size()
        )
        read_bytes += proj.numel() * proj.element_size()
        read_bytes += (r.numel() + grad_r_ext.numel()) * r.element_size()
        read_bytes += bias.numel() * bias.element_size()

        write_bytes = M * K * x.element_size() + N * K * weight.element_size()
        write_bytes += 3 * alpha.element_size() + bias.numel() * bias.element_size()
        return (read_bytes + write_bytes) * 1e-9 / (latency * 1e-3)


@pytest.mark.fused_proj_rms_compute_h_backward(
    recommended_shapes=_SHAPES, shape_desc="M, n, K"
)
def test_fused_proj_rms_compute_h_backward_perf():
    """mHC's real case: bf16 activations with a keep_in_fp32 weight."""
    bench = ProjRmsComputeHBwdBenchmark(
        input_fn=_input_fn,
        op_name="fused_proj_rms_compute_h_backward",
        torch_op=torch_op,
        dtypes=_ACTIVATION_DTYPES,
    )
    bench.set_train(flag_train.megatron.fused_proj_rms_compute_h_backward)
    bench.run()
