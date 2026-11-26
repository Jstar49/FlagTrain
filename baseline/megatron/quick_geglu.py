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

"""NVIDIA baseline for quick_geglu (forward).

Self-contained module; depends only on torch and megatron. ``native()`` returns
Megatron-LM's ``megatron.core.fusions.fused_bias_geglu.quick_geglu`` as
Megatron runs it by default (``jit_fuser`` is ``torch.compile`` on supported
PyTorch versions); when Megatron is unavailable it raises instead of collecting
a baseline or falling back to a torch composition. Shapes and dtypes are copied from
``FlagTrain/benchmark/megatron/test_quick_geglu.py``.
"""

import math

import torch
from torch._dynamo import config as dynamo_config

OP_NAME = "quick_geglu"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_quick_geglu.py::_CORE_SHAPES
_SHAPES = [
    (2,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (1024, 1024, 1024),
]


def base_op():
    """Megatron-LM quick_geglu; raises when Megatron is unavailable."""
    try:
        import megatron.core.fusions.fused_bias_geglu as fused_bias_geglu

        return fused_bias_geglu.quick_geglu
    except Exception as exc:
        raise RuntimeError(
            "quick_geglu baseline requires Megatron-LM's "
            "megatron.core.fusions.fused_bias_geglu.quick_geglu, which could "
            f"not be imported: {exc!r}. Install megatron-core on this machine."
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
    hidden = shape[-1]
    dtype_str = str(dtype)
    return {
        "inputs": {
            "y": {"shape": shape, "dtype": dtype_str},
            # The benchmark calls quick_geglu(y); linear_offset keeps its default.
            "linear_offset": {"scalar": 0.0},
        },
        "outputs": {
            # The forward splits the last dimension in half: 2 * d -> d.
            "out": {"shape": shape[:-1] + [hidden // 2], "dtype": dtype_str},
        },
        "dims": {
            "num_tokens": math.prod(shape[:-1]),
            "hidden": hidden,
            "d": hidden // 2,
            "shape": shape,
        },
    }
