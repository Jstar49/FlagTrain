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

"""NVIDIA baseline for tanh_soft_clamp (forward).

Self-contained module; depends only on torch and megatron. ``native()`` returns
Megatron-LM's ``megatron.core.activations.tanh_soft_clamp`` as Megatron runs
it by default (``jit_fuser`` is ``torch.compile`` on supported PyTorch
versions). The operator was added in
megatron-core 0.20.0; when Megatron is unavailable or the installed version
predates it, this module raises instead of collecting a baseline or falling back
to a torch composition. Shapes, dtypes and the scale are copied from
``FlagTrain/benchmark/megatron/test_tanh_soft_clamp.py``.
"""

import torch
from torch._dynamo import config as dynamo_config

OP_NAME = "tanh_soft_clamp"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_tanh_soft_clamp.py::_CORE_SHAPES
_SHAPES = [
    (1,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (4, 8192, 4096),
    (8, 4096, 11008),
]

# benchmark/megatron/test_tanh_soft_clamp.py::_SCALE
_SCALE = 0.5


def base_op():
    """Megatron-LM tanh_soft_clamp; raises when unavailable."""
    try:
        import megatron.core.activations
    except Exception as exc:
        raise RuntimeError(
            "tanh_soft_clamp baseline requires Megatron-LM's "
            "megatron.core.activations.tanh_soft_clamp, which could not be "
            f"imported: {exc!r}. Install megatron-core >= 0.20.0 on this machine."
        ) from exc

    kernel = getattr(megatron.core.activations, "tanh_soft_clamp", None)
    if kernel is None:
        raise RuntimeError(
            "tanh_soft_clamp baseline requires Megatron-LM's "
            "megatron.core.activations.tanh_soft_clamp, which was added in "
            "megatron-core 0.20.0; the installed megatron-core predates it. "
            "This module does not fall back to the torch composition."
        )
    return kernel


def native():
    return base_op()


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    args = (torch.randn(binding["shape"], dtype=dtype, device=device), _SCALE)
    # Compile this shape/dtype before the collector starts timing.
    with dynamo_config.patch(
        recompile_limit=len(grid()) * len(DTYPES) + 8,
        fail_on_recompile_limit_hit=True,
    ):
        native()(*args)
    return args, {}


def key_shape(binding):
    # The scale is fixed at _SCALE, so the tensor shape is the full case key.
    return list(binding["shape"])


def config(binding, dtype):
    return {
        "inputs": {
            "x": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "scale": {"scalar": _SCALE},
        },
        "dims": {},
    }
