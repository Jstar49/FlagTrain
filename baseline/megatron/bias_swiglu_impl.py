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
``_dtypes``). The last input dimension is the concatenated ``[gate | linear]``,
so the output keeps half of it. The bias is the ``skip_bias_add`` ``[hidden]``
tensor added to both halves. Only the plain SwiGLU activation is collected: the
clamped and SiTU variants run the same kernel with two extra min/max ops, and
``clamp_value`` / ``gate_clamp_scale`` only exist from megatron-core 0.20 on,
while this file must also work with the version installed here.

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

import torch

OP_NAME = "bias_swiglu_impl"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
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


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    return _megatron_bias_swiglu_impl()


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    hidden = binding["shape"][-1]
    input = torch.randn(binding["shape"], dtype=dtype, device=device)
    bias = torch.randn((hidden,), dtype=dtype, device=device)
    return (input, bias), {}


def key_shape(binding):
    # First tensor: the [tokens, 2 * hidden] input; unique per deployment.
    return list(binding["shape"])
