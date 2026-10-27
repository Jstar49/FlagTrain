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
"""Performance of the fused mHC projection + RMS + compute_h forward.

The baseline is Megatron's own ``fused_proj_rms_compute_h`` forward from
``megatron/core/fusions/fused_mhc_kernels.py``: the installed module when it
ships the mHC kernels (which dispatches to its cuTile kernel when available),
otherwise a verbatim copy of its torch path. It returns the public four
outputs -- the reduced projection is internal -- so the untimed check compares
those four.

This benchmarks mHC's real configuration: low-precision activations with the
mapping parameters kept in fp32 (``keep_in_fp32``), so ``promote_types`` makes
the projection fp32.

Precision caveat: when the baseline takes Megatron's torch path, the harness
has disabled torch tf32, so its fp32 matmul is IEEE while the operator's fp32
dot is tf32; that speedup includes the precision difference. With Megatron's
cuTile kernel available both sides use tf32.

Run::

    pytest benchmark/megatron -m fused_proj_rms_compute_h_fwd --level core
"""

import math

import pytest
import torch

import flag_train

from .. import base, consts

# (M, n, K): M is the number of tokens (s * b) and K = n * hidden_size.
# 4096/16384 is the common mHC shape (4 streams x hidden 4096); 28672 covers
# hidden 7168 and 4096/8192 covers hidden 2048 with two streams.
_SHAPES = [
    (1024, 4, 16384),
    (4096, 4, 16384),
    (4096, 4, 28672),
    (8192, 4, 8192),
    (4096, 2, 8192),
    (16384, 4, 4096),
]

_ACTIVATION_DTYPES = [torch.float16, torch.bfloat16]

# The mHC module keeps the mapping parameters in fp32 (keep_in_fp32), so the
# activation and weight dtypes differ in its real configuration.
_WEIGHT_DTYPE = torch.float32

_TOLERANCE = {torch.float16: 1e-3, torch.bfloat16: 2e-2}


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

    It returns the public four outputs: the reduced projection is internal.
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


def _load_megatron_forward():
    return _megatron_torch_forward


torch_op = _load_megatron_forward()


def _input_fn(shape, dtype, device):
    M, n, K = shape
    N = n * n + 2 * n
    torch.manual_seed(0)
    x = torch.empty(M, K, dtype=dtype, device=device).uniform_(-0.1, 0.1)
    weight = torch.empty(N, K, dtype=_WEIGHT_DTYPE, device=device).uniform_(-0.1, 0.1)
    # The module's initialization: mhc_init_gating_factor and a zero static bias.
    alpha = torch.full((1,), 0.01, dtype=torch.float32, device=device)
    bias = torch.zeros(N, dtype=torch.float32, device=device)
    args = (x, weight, alpha, alpha, alpha, bias, n)

    # One untimed correctness check per shape so both sides answer the same
    # question (the operator's fp32 dot is tf32; see the module docstring).
    # Megatron's forward returns four tensors; `proj` is our backward extra.
    expected = torch_op(*args)
    got = flag_train.megatron.fused_proj_rms_compute_h_fwd(*args)
    tolerance = _TOLERANCE[dtype]
    for name, result, ref in zip(("h_pre", "h_post", "h_res", "r"), got[:4], expected):
        torch.testing.assert_close(
            result,
            ref,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg, name=name: f"{name}: {msg}",
        )
    yield args


class ProjRmsComputeHBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "M, n, K"
    DEFAULT_METRICS = consts.DEFAULT_METRICS + ["gbps", "tflops"]

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        # The upstream default extra shapes are pointwise-shaped and do not
        # describe (M, n, K) triples.
        return []

    def get_gbps(self, args, latency=None):
        """Logical bytes: x and weight in, the five outputs out."""
        if latency is None:
            return None
        x, weight, _, _, _, bias, _ = args
        M, K = x.shape
        N = weight.shape[0]
        out_dtype = torch.promote_types(
            torch.promote_types(x.dtype, weight.dtype), bias.dtype
        )
        read_bytes = M * K * x.element_size() + N * K * weight.element_size()
        # h_pre + h_post + h_res + proj cover 2 * M * N elements; r is M more.
        write_bytes = (2 * M * N + M) * out_dtype.itemsize
        return (read_bytes + write_bytes) * 1e-9 / (latency * 1e-3)


@pytest.mark.fused_proj_rms_compute_h_fwd(
    recommended_shapes=_SHAPES, shape_desc="M, n, K"
)
def test_fused_proj_rms_compute_h_fwd_perf():
    """mHC's real case: low-precision activations with a keep_in_fp32 weight."""
    bench = ProjRmsComputeHBenchmark(
        input_fn=_input_fn,
        op_name="fused_proj_rms_compute_h_fwd",
        torch_op=torch_op,
        dtypes=_ACTIVATION_DTYPES,
    )
    bench.set_train(flag_train.megatron.fused_proj_rms_compute_h_fwd)
    bench.run()
