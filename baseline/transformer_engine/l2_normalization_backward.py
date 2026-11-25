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

"""NVIDIA backward baseline using Transformer Engine's L2 JIT helpers."""

from functools import lru_cache

import torch
from torch._dynamo import config as dynamo_config

OP_NAME = "l2_normalization_backward"
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = (
    (8, 64),
    (32, 128),
    (2, 16, 80),
    (1, 1),
    (31, 127),
    (128, 768),
    (512, 1024),
    (8, 128, 1536),
    (2048, 4096),
)


@lru_cache(maxsize=1)
def _upstream():
    from transformer_engine.pytorch.jit import (
        l2normalization_backward_fused,
        l2normalization_fwd_fused,
    )

    return l2normalization_fwd_fused, l2normalization_backward_fused


def native():
    return _upstream()[1]


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = binding["shape"]
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(shape, generator=rng, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    grad = torch.randn(shape, generator=rng, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    forward, backward = _upstream()
    # Prepare the saved norm and compile the backward before timing.
    with dynamo_config.patch(
        recompile_limit=len(_SHAPES) * len(DTYPES), fail_on_recompile_limit_hit=True
    ):
        _, inverse_norm = forward(x, _EPS)
        args = (grad, x, inverse_norm, _EPS)
        backward(*args)
    return args, {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = binding["shape"]
    return {
        "inputs": {
            "grad_output": {"shape": list(shape), "dtype": str(dtype)},
            "x": {"shape": list(shape), "dtype": str(dtype)},
            "rsqrt_norm": {"shape": [*shape[:-1], 1], "dtype": str(torch.float32)},
            "eps": {"scalar": _EPS},
        },
        "stage": "backward",
    }
