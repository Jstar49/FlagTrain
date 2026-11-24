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

"""NVIDIA baseline for bias_gelu_impl.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns Megatron-LM's
``megatron.core.fusions.fused_bias_gelu.bias_gelu_impl``
(``GeLUFunction.apply``) and raises when Megatron is unavailable instead of
falling back to a torch composition.

The shapes and dtypes are ``FlagTrain/benchmark/megatron/test_bias_gelu_impl.py``
(``_CORE_SHAPES`` and ``_dtypes``); ``bias_gelu_impl`` is the non-gated sibling
of the fused-MLP benchmark, so the same token counts and hidden widths keep the
two baselines comparable. The input is ``[tokens, hidden]``. The bias is one
``[hidden]`` vector broadcast across rows by default -- matching Megatron's own
use (``bias_gelu_impl(x, bias)`` in ``transformer/mlp.py`` and
``training/kernel_warmup.py``) -- and, at the shapes in ``_FULL_BIAS_SHAPES``,
a full ``[tokens, hidden]`` tensor: the kernel's elementwise path, which reads
a bias as large as the input. The two are separate bindings with distinct
``key_shape`` values so their baseline records cannot collide.

Megatron decorates the helper behind ``bias_gelu_impl`` with ``@jit_fuser``,
which is ``torch.compile`` on supported PyTorch versions. This module disables
the fuser so the collector profiles the ATen kernels rather than Inductor
output. Because ``import megatron.core`` may already have imported and wrapped
the helper, the module is reloaded once after the fuser is disabled. Megatron is
imported lazily on the first ``native()`` call, so ops discovery does not
require it.
"""

import math

import torch

OP_NAME = "bias_gelu_impl"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_bias_gelu_impl.py::_CORE_SHAPES
# (tokens, hidden); the activation keeps the full width (no gate/linear split).
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]

# benchmark/megatron/test_bias_gelu_impl.py::_FULL_BIAS_SHAPES
_FULL_BIAS_SHAPES = {
    (4096, 4096),
    (16, 128, 8192),
    (2048, 11008),
    (256, 32768),
}

_NATIVE = None


def _megatron_bias_gelu_impl():
    """Megatron-LM's bias_gelu_impl (eager); raise when it cannot be imported."""
    global _NATIVE
    if _NATIVE is None:
        try:
            import importlib

            import megatron.core.jit

            # The collector profiles native kernels, not Inductor output.
            megatron.core.jit.disable_jit_fuser()
            import megatron.core.fusions.fused_bias_gelu as fused_bias_gelu

            # megatron.core imports the fusions package at import time, so the
            # helper may already be wrapped by torch.compile. Reloading after
            # disable_jit_fuser() rebinds @jit_fuser to the no-op decorator.
            if hasattr(fused_bias_gelu.bias_gelu, "_torchdynamo_orig_callable"):
                importlib.reload(fused_bias_gelu)
        except Exception as exc:
            raise RuntimeError(
                "bias_gelu_impl baseline requires Megatron-LM's "
                "megatron.core.fusions.fused_bias_gelu.bias_gelu_impl, which "
                f"could not be imported: {exc!r}. Install megatron-core on "
                "this machine."
            ) from exc
        _NATIVE = fused_bias_gelu.bias_gelu_impl
    return _NATIVE


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    return _megatron_bias_gelu_impl()


def grid():
    bindings = [{"shape": shape, "bias_mode": "vector"} for shape in _SHAPES]
    bindings += [
        {"shape": shape, "bias_mode": "full"}
        for shape in _SHAPES
        if tuple(shape) in _FULL_BIAS_SHAPES
    ]
    return bindings


def build_inputs(binding, dtype, device):
    shape = binding["shape"]
    input = torch.randn(shape, dtype=dtype, device=device)
    if binding["bias_mode"] == "full":
        bias = torch.randn(shape, dtype=dtype, device=device)
    else:
        bias = torch.randn((shape[-1],), dtype=dtype, device=device)
    return (input, bias), {}


def key_shape(binding):
    # Both bias modes share the input's shape, so the key names the mode too:
    # a vector bias and a full bias of one shape are different measurements.
    dims = "x".join(str(dim) for dim in binding["shape"])
    return f"{dims}_{binding['bias_mode']}_bias"


def config(binding, dtype):
    shape = list(binding["shape"])
    hidden = shape[-1]
    bias_shape = shape if binding["bias_mode"] == "full" else [hidden]
    return {
        "inputs": {
            "input": {"shape": shape, "dtype": str(dtype)},
            "bias": {"shape": bias_shape, "dtype": str(dtype)},
        },
        "outputs": {
            "out": {"shape": shape, "dtype": str(dtype)},
        },
        "dims": {
            "num_tokens": math.prod(shape[:-1]),
            "hidden": hidden,
            "shape": shape,
            "bias_mode": binding["bias_mode"],
        },
    }
