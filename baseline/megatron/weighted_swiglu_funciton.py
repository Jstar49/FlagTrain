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

"""NVIDIA forward baseline for the default Megatron-LM weighted SwiGLU."""

from functools import lru_cache

import torch
from torch._dynamo import config as dynamo_config

OP_NAME = "weighted_swiglu_funciton"
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
IS_INPLACE = False

_SHAPES = (
    (16, 32),
    (2, 5, 64),
    (1, 2),
    (33, 258),
    (128, 2048),
    (2, 128, 4096),
    (256, 8192),
)


@lru_cache(maxsize=1)
def _upstream():
    from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu

    return weighted_swiglu


def native():
    return _upstream()


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = binding["shape"]
    rng = torch.Generator().manual_seed(6466)
    y = torch.randn(shape, generator=rng).to(device=device, dtype=dtype)
    weights = torch.rand((*shape[:-1], 1), generator=rng).to(device=device)
    # Compile this input signature before the collector starts timing.
    with dynamo_config.patch(
        recompile_limit=len(_SHAPES) * len(DTYPES), fail_on_recompile_limit_hit=True
    ):
        _upstream()(y, weights)
    return (y, weights), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = binding["shape"]
    return {
        "inputs": {
            "y": {"shape": list(shape), "dtype": str(dtype)},
            "weights": {"shape": [*shape[:-1], 1], "dtype": str(torch.float32)},
        }
    }
