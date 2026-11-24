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

"""NVIDIA baseline for bias_swiglu_impl.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns Megatron-LM's
``megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl`` and raises when
Megatron is unavailable instead of falling back to a torch composition.

Shapes and dtypes are copied from
``FlagTrain/benchmark/megatron/test_bias_swiglu_impl.py`` (``_CORE_SHAPES`` and
``_dtypes``; fp64 is included where the benchmark's device advertises it). The
last input dimension is the concatenated ``[gate | linear]``,
so the output keeps half of it. The bias is the ``skip_bias_add`` ``[hidden]``
tensor added to both halves. The plain, clamped and SiTU-GLU variants selected
by the benchmark are collected. The clamp/SiTU arguments require megatron-core
0.20 or newer, the version the benchmark requires for the same variants.

Megatron decorates the inner helpers of that module with ``@jit_fuser``, which
is ``torch.compile`` on torch >= 2.2. Like ``quick_gelu.py`` this module
disables the fuser so the collector profiles the ATen kernels rather than
Inductor output. Disabling alone is not enough here: ``import megatron.core``
already imports the fusions package, so the helpers are wrapped before this
module gets a chance to switch the fuser off, and the wrapper sits on the inner
helpers rather than on the public function. The module is therefore reloaded
once after the fuser is disabled, which rebinds the decorators to the no-op
fuser. Megatron is imported lazily on the first ``native()`` call, so ops
discovery does not trigger it.
"""

import math

import torch

OP_NAME = "bias_swiglu_impl"
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]
IS_INPLACE = False

# benchmark/megatron/test_bias_swiglu_impl.py::_CORE_SHAPES
# (tokens, 2 * hidden); the output shape is [..., hidden].
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]

_MEGATRON_BIAS_SWIGLU_IMPL = None


def _megatron_bias_swiglu_impl():
    """Megatron-LM's bias_swiglu_impl (eager); raise when it cannot be imported."""
    global _MEGATRON_BIAS_SWIGLU_IMPL
    if _MEGATRON_BIAS_SWIGLU_IMPL is None:
        try:
            import importlib

            import megatron.core.jit

            # The collector profiles native kernels, not Inductor output.
            megatron.core.jit.disable_jit_fuser()
            import megatron.core.fusions.fused_bias_swiglu

            # megatron.core imports the fusions package at import time, so the
            # helpers may already be wrapped by torch.compile. Reloading after
            # disable_jit_fuser() rebinds @jit_fuser to the no-op decorator.
            if hasattr(
                megatron.core.fusions.fused_bias_swiglu.swiglu,
                "_torchdynamo_orig_callable",
            ):
                importlib.reload(megatron.core.fusions.fused_bias_swiglu)
        except Exception as exc:
            raise RuntimeError(
                "bias_swiglu_impl baseline requires Megatron-LM's "
                "megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl, "
                f"which could not be imported: {exc!r}. Install megatron-core "
                "on this machine."
            ) from exc
        _MEGATRON_BIAS_SWIGLU_IMPL = (
            megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl
        )
    return _MEGATRON_BIAS_SWIGLU_IMPL


# benchmark/megatron/test_bias_swiglu_impl.py::_VARIANT_KWARGS
_VARIANT_KWARGS = {
    "plain": {},
    "clamped": {"clamp_value": 2.0},
    "situ": {"gate_clamp_scale": 4.0, "linear_clamp_scale": 2.5},
}


def torch_op(input, bias, variant):
    """Megatron's bias_swiglu_impl for one benchmark variant."""
    return _megatron_bias_swiglu_impl()(input, bias, **_VARIANT_KWARGS[variant])


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_bias_swiglu_impl()
    return torch_op


def grid():
    return [
        {"shape": shape, "variant": variant}
        for shape in _SHAPES
        for variant in _VARIANT_KWARGS
    ]


def build_inputs(binding, dtype, device):
    hidden = binding["shape"][-1]
    input = torch.randn(binding["shape"], dtype=dtype, device=device)
    bias = torch.randn((hidden,), dtype=dtype, device=device)
    return (input, bias, binding["variant"]), {}


def key_shape(binding):
    # The plain case follows the first-tensor convention; the variants share
    # that shape, so they carry a suffix to keep their records distinct.
    shape = binding["shape"]
    variant = binding["variant"]
    if variant == "plain":
        return list(shape)
    return "x".join(str(dim) for dim in shape) + "_" + variant


def config(binding, dtype):
    """Describe the real inputs/outputs for baseline matching."""
    shape = list(binding["shape"])
    width = shape[-1]
    variant = binding["variant"]
    dtype_str = str(dtype)
    inputs = {
        "input": {"shape": shape, "dtype": dtype_str},
        # One [2 * d] bias vector is shared by all rows.
        "bias": {"shape": [width], "dtype": dtype_str},
    }
    inputs.update(
        {key: {"scalar": value} for key, value in _VARIANT_KWARGS[variant].items()}
    )
    return {
        "inputs": inputs,
        "outputs": {
            # The forward splits the last dimension in half: 2 * d -> d.
            "out": {
                "shape": shape[:-1] + [width // 2],
                "dtype": dtype_str,
            },
        },
        "dims": {
            "num_tokens": math.prod(shape[:-1]),
            "hidden": width,
            "d": width // 2,
            "shape": shape,
            "variant": variant,
        },
    }
