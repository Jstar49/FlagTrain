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

FlagTrain does not benchmark ``bias_gelu_impl`` yet. The shapes and dtypes are
the fused-MLP activation cases of
``FlagTrain/benchmark/megatron/test_bias_geglu_impl.py`` (``_CORE_SHAPES`` and
``_dtypes``); ``bias_gelu_impl`` is the non-gated sibling measured there, so the
same token counts and hidden widths keep the two baselines comparable. The
input is ``[tokens, hidden]`` and the bias is one ``[hidden]`` vector broadcast
across rows, matching Megatron's own use (``bias_gelu_impl(x, bias)`` in
``transformer/mlp.py`` and ``training/kernel_warmup.py``).

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

# benchmark/megatron/test_bias_geglu_impl.py::_CORE_SHAPES
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
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    hidden = binding["shape"][-1]
    input = torch.randn(binding["shape"], dtype=dtype, device=device)
    bias = torch.randn((hidden,), dtype=dtype, device=device)
    return (input, bias), {}


def key_shape(binding):
    # First tensor: the [tokens, hidden] input; unique per deployment.
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    hidden = shape[-1]
    return {
        "inputs": {
            "input": {"shape": shape, "dtype": str(dtype)},
            # One [hidden] bias vector is broadcast across all rows.
            "bias": {"shape": [hidden], "dtype": str(dtype)},
        },
        "outputs": {
            "out": {"shape": shape, "dtype": str(dtype)},
        },
        "dims": {
            "num_tokens": math.prod(shape[:-1]),
            "hidden": hidden,
            "shape": shape,
        },
    }
