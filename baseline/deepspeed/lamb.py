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

"""NVIDIA baseline for lamb: DeepSpeed fused_lamb.

This collects the baseline implementation the FlagTrain benchmark uses on
NVIDIA, not FlagTrain's own Triton kernel:

    FlagTrain/benchmark/deepspeed/test_lamb.py
        _DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}
        _load_deepspeed_lamb() -> FusedLambBuilder().load().lamb
        _call(lamb, p, p_copy, m, v, g, lr, beta1, beta2, max_coeff,
              min_coeff, eps, grad_scale, step, mode, bias_correction, decay)

    DeepSpeed source paths (these move between versions; check the source):
        deepspeed/ops/op_builder/fused_lamb.py
        deepspeed/ops/fused_lamb.py

One fused_lamb call launches three CUDA kernels internally (part1/part2/part3),
so the NCU per_kernel list should contain three entries.

When DeepSpeed is unavailable native() raises, so the collector does **not**
collect this operator's baseline and does not fall back to FlagTrain's torch
reference lamb_ref.

The input construction mirrors FlagTrain's lamb_input_fn:
    p = randn(N); p_copy = empty(0); m = zeros(N); v = zeros(N); g = randn(N)
Hyperparameters match _LR/_BETA1/... in test_lamb.py.
Shapes come from test_lamb.py::_LAMB_SHAPES (seven 1-D lengths).

Requirements:
    - an NVIDIA GPU with NCU performance-counter permission (otherwise units /
      bottleneck are empty);
    - DeepSpeed installed (the first load JIT-compiles for several minutes,
      then uses the cache);
    - the collection machine matches --reference-chip (H800 by default).

Usage:
    python tools/collect_baseline_nvidia.py --check --repo flagtrain --ops lamb
    python tools/collect_baseline_nvidia.py --repo flagtrain --ops lamb \
        --output op_flagtrain_lamb_h800_baseline.json --device 0
"""

import torch

OP_NAME = "lamb"
# fused_lamb only supports float32 parameters/state (see test_lamb.py).
DTYPES = [torch.float32]
IS_INPLACE = True  # p / m / v / p_copy are updated in place

# FlagTrain/benchmark/deepspeed/test_lamb.py::_LAMB_SHAPES
_SHAPES = [
    (1024,),
    (4096,),
    (16384,),
    (65536,),
    (262144,),
    (1048576,),
    (11048576,),
]

# Same hyper-parameters as test_lamb.py, shared by both implementations.
_LR = 1e-3
_BETA1 = 0.9
_BETA2 = 0.999
_MAX_COEFF = 10.0
_MIN_COEFF = 0.01
_EPS = 1e-8
_GRAD_SCALE = 1.0
_STEP = 1
_MODE = 1  # eps outside the sqrt (DeepSpeed default)
_BIAS_CORRECTION = 1
_DECAY = 0.01


def native():
    """Resolve DeepSpeed's fused_lamb; raise when it cannot be loaded.

    The JIT build is cached under TORCH_EXTENSIONS_DIR, so the first call is
    slow and later worker/NCU subprocesses reuse the build.
    """
    try:
        from deepspeed.ops.op_builder import FusedLambBuilder

        return FusedLambBuilder().load().lamb
    except Exception as exc:
        raise RuntimeError(
            "lamb baseline requires DeepSpeed's fused_lamb, which could not be "
            f"loaded: {exc!r}. Install deepspeed and ensure its CUDA extension "
            "builds on this machine."
        ) from exc


def grid():
    return [{"N": n} for (n,) in _SHAPES]


def build_inputs(binding, dtype, device):
    """Mirror FlagTrain/benchmark/deepspeed/test_lamb.py::lamb_input_fn."""
    n = binding["N"]
    p = torch.randn(n, dtype=dtype, device=device)
    p_copy = torch.empty(0, dtype=dtype, device=device)  # skip the copy
    m = torch.zeros(n, dtype=dtype, device=device)
    v = torch.zeros(n, dtype=dtype, device=device)
    g = torch.randn(n, dtype=dtype, device=device)
    args = (
        p,
        p_copy,
        m,
        v,
        g,
        _LR,
        _BETA1,
        _BETA2,
        _MAX_COEFF,
        _MIN_COEFF,
        _EPS,
        _GRAD_SCALE,
        _STEP,
        _MODE,
        _BIAS_CORRECTION,
        _DECAY,
    )
    return args, {}


def key_shape(binding):
    # LAMB operates on a flattened parameter array; N is the only dimension.
    return [binding["N"]]


def config(binding, dtype):
    """Describe the real inputs/outputs for baseline matching."""
    n = binding["N"]
    dtype_str = str(dtype)
    return {
        "inputs": {
            "p": {"shape": [n], "dtype": dtype_str},
            "p_copy": {"shape": [0], "dtype": dtype_str},
            "m": {"shape": [n], "dtype": dtype_str},
            "v": {"shape": [n], "dtype": dtype_str},
            "g": {"shape": [n], "dtype": dtype_str},
            "lr": {"scalar": _LR},
            "beta1": {"scalar": _BETA1},
            "beta2": {"scalar": _BETA2},
            "max_coeff": {"scalar": _MAX_COEFF},
            "min_coeff": {"scalar": _MIN_COEFF},
            "eps": {"scalar": _EPS},
            "grad_scale": {"scalar": _GRAD_SCALE},
            "step": {"scalar": _STEP},
            "mode": {"scalar": _MODE},
            "bias_correction": {"scalar": _BIAS_CORRECTION},
            "decay": {"scalar": _DECAY},
        },
        "outputs": {
            "p": {"shape": [n], "dtype": dtype_str},
            "m": {"shape": [n], "dtype": dtype_str},
            "v": {"shape": [n], "dtype": dtype_str},
        },
        "dims": {"N": n},
    }
