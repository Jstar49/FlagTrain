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

"""NVIDIA router forward baseline using Megatron-LM's TE GEMM path."""

from functools import lru_cache

import torch

OP_NAME = "router_gating_linear_function"
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
IS_INPLACE = False

_INPUT_DTYPE = torch.bfloat16
_WEIGHT_DTYPE = torch.float16
_BIAS_DTYPE = torch.float32
_SHAPES = (
    ((16, 32), 13),
    ((2, 5, 32), 13),
    ((1, 64), 1),
    ((33, 127), 17),
    ((128, 768), 32),
    ((4, 128, 2048), 64),
    ((1024, 4096), 128),
)


@lru_cache(maxsize=1)
def _upstream():
    from megatron.core.transformer.moe import moe_utils

    if moe_utils.te_general_gemm is None:
        raise RuntimeError(
            "The Megatron router baseline requires NVIDIA Transformer Engine's general_gemm"
        )
    return moe_utils.router_gating_linear


def _forward(inp, weight, bias, router_dtype):
    # Match FlagTrain's input conversion before dispatch to TE's same-dtype GEMM.
    # Keep these casts in the timed forward because the public inputs are mixed.
    return _upstream()(
        inp.to(router_dtype), weight.to(router_dtype), bias, router_dtype
    )


def native():
    _upstream()
    return _forward


def grid():
    return [
        {"shape": shape, "num_experts": num_experts, "with_bias": with_bias}
        for shape, num_experts in _SHAPES
        for with_bias in (False, True)
    ]


def build_inputs(binding, dtype, device):
    shape, num_experts = binding["shape"], binding["num_experts"]
    rng = torch.Generator().manual_seed(421)
    inp = torch.randn(shape, generator=rng)
    weight = torch.randn((num_experts, shape[-1]), generator=rng)
    bias = None
    if binding["with_bias"]:
        bias = torch.randn(num_experts, generator=rng)
        first_row = inp.reshape(-1, shape[-1])[0]
        first_row.zero_()
        first_row[0] = 1 + 2**-7
        weight[0].zero_()
        weight[0, 0] = 1 + 2**-7
        bias[0] = -(1 + 2**-6)
        bias = bias.to(device=device, dtype=_BIAS_DTYPE)
    return (
        inp.to(device=device, dtype=_INPUT_DTYPE),
        weight.to(device=device, dtype=_WEIGHT_DTYPE),
        bias,
        dtype,
    ), {}


def key_shape(binding):
    # A first-tensor key would select the same record for both bias variants.
    dims = "x".join(str(dim) for dim in binding["shape"])
    variant = "bias" if binding["with_bias"] else "no_bias"
    return f'{dims}x{binding["num_experts"]}:{variant}'


def config(binding, dtype):
    shape, num_experts = binding["shape"], binding["num_experts"]
    inputs = {
        "inp": {"shape": list(shape), "dtype": str(_INPUT_DTYPE)},
        "weight": {"shape": [num_experts, shape[-1]], "dtype": str(_WEIGHT_DTYPE)},
    }
    if binding["with_bias"]:
        inputs["bias"] = {"shape": [num_experts], "dtype": str(_BIAS_DTYPE)}
    return {"inputs": inputs, "router_dtype": str(dtype)}
