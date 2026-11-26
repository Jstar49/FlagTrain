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
Megatron-LM's ``megatron.core.activations.fast_gelu`` as Megatron runs it by
default (``jit_fuser`` is ``torch.compile`` on supported PyTorch versions);
when Megatron is unavailable it raises instead of collecting a baseline or
falling back to a torch composition.
Shapes and dtypes are copied from
``FlagTrain/benchmark/megatron/test_fast_gelu.py``.
"""

import math

import torch
from torch._dynamo import config as dynamo_config

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
    """Megatron-LM fast_gelu; raises when Megatron is unavailable."""
    try:
        import megatron.core.activations

        return megatron.core.activations.fast_gelu
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
    args = (torch.randn(binding["shape"], dtype=dtype, device=device),)
    # Compile this shape/dtype before the collector starts timing.
    with dynamo_config.patch(
        recompile_limit=len(grid()) * len(DTYPES) + 8,
        fail_on_recompile_limit_hit=True,
    ):
        native()(*args)
    return args, {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    """Describe the real inputs/outputs for baseline matching."""
    shape = list(binding["shape"])
    return {
        "inputs": {"x": {"shape": shape, "dtype": str(dtype)}},
        "outputs": {"out": {"shape": shape, "dtype": str(dtype)}},
        "dims": {"numel": math.prod(shape), "shape": shape},
    }
