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

"""NVIDIA baseline for fused_proj_rms_compute_h_forward (forward).

Self-contained module; depends only on torch and megatron. ``native()`` returns
Megatron-LM's ``megatron.core.fusions.fused_mhc_kernels.fused_proj_rms_compute_h``
(the public forward: it dispatches to the cuTile kernel when available and
otherwise runs ``_torch_proj_rms_compute_h``); when Megatron is unavailable --
or the installed release predates the mHC kernels, which megatron-core 0.19.2
does not ship -- it raises instead of collecting a baseline or falling back to
a torch composition.

The upstream forward returns four tensors ``(h_pre, h_post, h_res, r)``;
FlagTrain's operator also returns the reduced projection for its backward, but
the collector records latency and NCU metrics rather than comparing values.

Shapes, dtypes and the input construction are copied from
``FlagTrain/benchmark/megatron/test_fused_proj_rms_compute_h_forward.py``: mHC's
real configuration, low-precision activations with the mapping parameters
(weight, alphas, bias) kept in fp32.
"""

import torch

OP_NAME = "fused_proj_rms_compute_h_forward"
# benchmark/megatron/test_fused_proj_rms_compute_h_forward.py::_ACTIVATION_DTYPES
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fused_proj_rms_compute_h_forward.py::_SHAPES
_SHAPES = [
    (1024, 4, 16384),
    (4096, 4, 16384),
    (4096, 4, 28672),
    (8192, 4, 8192),
    (4096, 2, 8192),
    (16384, 4, 4096),
]

# The mHC module keeps the mapping parameters in fp32 (``keep_in_fp32``), so
# the activation and weight dtypes differ in its real configuration.
_WEIGHT_DTYPE = torch.float32

# mhc_init_gating_factor; the static bias is initialized to zero.
_GATING_FACTOR = 0.01


def base_op():
    """Megatron-LM's fused_proj_rms_compute_h; raises when it is unavailable."""
    try:
        import megatron.core.fusions.fused_mhc_kernels as fused_mhc_kernels

        kernel = fused_mhc_kernels.fused_proj_rms_compute_h
    except Exception as exc:
        raise RuntimeError(
            "fused_proj_rms_compute_h_forward baseline requires Megatron-LM's "
            "megatron.core.fusions.fused_mhc_kernels, which could not be "
            f"imported: {exc!r}. Install a Megatron-LM build that ships this "
            "module (megatron-core 0.19.2 does not)."
        ) from exc

    def eager_forward(*args, **kwargs):
        # Megatron 0.21 wraps its torch path in @torch.compile. The collector
        # profiles ATen kernels, not Inductor output (same rule as the
        # quick_gelu baseline's disable_jit_fuser), so run it eagerly.
        import torch._dynamo

        with torch._dynamo.config.patch(disable=True):
            return kernel(*args, **kwargs)

    return eager_forward


def native():
    return base_op()


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    """Mirror FlagTrain/benchmark/megatron/test_fused_proj_rms_compute_h_forward.py."""
    M, n, K = binding["shape"]
    N = n * n + 2 * n
    torch.manual_seed(0)
    x = torch.empty(M, K, dtype=dtype, device=device).uniform_(-0.1, 0.1)
    weight = torch.empty(N, K, dtype=_WEIGHT_DTYPE, device=device).uniform_(-0.1, 0.1)
    alpha = torch.full((1,), _GATING_FACTOR, dtype=torch.float32, device=device)
    bias = torch.zeros(N, dtype=torch.float32, device=device)
    # The operator defaults eps and compute_h_eps to 1e-6; the baseline calls
    # the upstream forward with the same defaults.
    args = (x, weight, alpha, alpha, alpha, bias, n)
    return args, {}


def key_shape(binding):
    return list(binding["shape"])
