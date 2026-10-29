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

"""NVIDIA baseline for fast_gelu (forward).

Self-contained module; depends only on torch and megatron. ``native()`` returns
Megatron-LM's eager ``megatron.core.activations.fast_gelu`` (Megatron's
``jit_fuser``, i.e. ``torch.compile``, is disabled so the collector profiles
ATen kernels rather than Inductor output); when Megatron is unavailable it
raises instead of collecting a baseline or falling back to a torch composition.
Shapes and dtypes are copied from
``FlagTrain/benchmark/megatron/test_fast_gelu.py``.
"""

import torch

OP_NAME = "fast_gelu"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fast_gelu.py::_CORE_SHAPES
_SHAPES = [
    (1,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (4, 8192, 4096),
    (8, 4096, 11008),
]


def base_op():
    """Megatron-LM fast_gelu (eager); raises when Megatron is unavailable."""
    try:
        import megatron.core.jit as megatron_jit

        # Megatron's @jit_fuser is torch.compile by default. The collector must
        # profile Megatron's eager kernels rather than Inductor output, so
        # disable the fuser before activations binds the decorator.
        megatron_jit.disable_jit_fuser()
        import megatron.core.activations

        kernel = megatron.core.activations.fast_gelu
        # activations may have been imported before the fuser was disabled; in
        # that case unwrap the torch.compile wrapper instead of compiling here.
        return getattr(kernel, "_torchdynamo_orig_callable", kernel)
    except Exception as exc:
        raise RuntimeError(
            "fast_gelu baseline requires Megatron-LM's "
            "megatron.core.activations.fast_gelu, which could not be "
            f"imported: {exc!r}. Install megatron-core on this machine."
        ) from exc


def native():
    return base_op()


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    return (torch.randn(binding["shape"], dtype=dtype, device=device),), {}


def key_shape(binding):
    return list(binding["shape"])
