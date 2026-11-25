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

"""NVIDIA forward baseline using Megatron-LM's compiled native_sinkhorn."""

from functools import lru_cache

import torch
from torch._dynamo import config as dynamo_config

OP_NAME = "fused_sinkhorn"
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
IS_INPLACE = False

_CASES = (
    ((8, 4, 4), 1, 1e-6),
    ((2, 16, 8, 8), 3, 1e-4),
    ((1, 1, 1), 1, 1e-6),
    ((1, 4, 4), 1, 1e-6),
    ((3, 5, 5), 2, 1e-5),
    ((257, 4, 4), 3, 1e-6),
    ((8, 128, 4, 4), 20, 1e-6),
    ((2, 256, 8, 8), 20, 1e-4),
)


@lru_cache(maxsize=1)
def _upstream():
    from megatron.core.transformer.hyper_connection import native_sinkhorn

    return native_sinkhorn


def native():
    return _upstream()


def grid():
    return [
        {"shape": shape, "num_iterations": num_iterations, "eps": eps}
        for shape, num_iterations, eps in _CASES
    ]


def build_inputs(binding, dtype, device):
    rng = torch.Generator().manual_seed(6472)
    logits = torch.randn(binding["shape"], generator=rng).to(device=device, dtype=dtype)
    args = (logits, binding["num_iterations"], binding["eps"])
    # Compile this input signature before the collector starts timing.
    with dynamo_config.patch(
        recompile_limit=len(_CASES) * len(DTYPES), fail_on_recompile_limit_hit=True
    ):
        _upstream()(*args)
    return args, {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    return {
        "inputs": {
            "input_logits": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "num_iterations": {"scalar": binding["num_iterations"]},
            "eps": {"scalar": binding["eps"]},
        }
    }
