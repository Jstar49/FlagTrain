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

"""NVIDIA baseline for moe_gemm.

The baseline is DeepSpeed's CUTLASS 16-bit MoE GEMM:

    deepspeed.ops.op_builder.InferenceCutlassBuilder().load().moe_gemm
    (output, hidden_states, weight, bias,
     total_rows_before_expert, activation)

Source: ``deepspeed/inference/v2/kernels/cutlass_ops/moe_gemm/moe_gemm.cu``.
The binding writes the preallocated ``output`` and returns nothing, so it is
``IS_INPLACE``. ``activation`` uses DeepSpeed's ``ActivationType`` ids: 6 is
identity and 2 is SiLU, the two variants the mixed-precision benchmark measures.
``weights`` is ``[experts, K, N]`` in the activation dtype and ``bias`` is
``[experts, N]``; ``total_rows_before_expert`` gives each expert's exclusive row
boundary in the stacked ``[experts * tokens, K]`` input.

When DeepSpeed is unavailable, ``native()`` raises; the collector does not
collect this operator's baseline and does not fall back to a torch composition.

FlagTrain does not benchmark the 16-bit ``moe_gemm`` yet, only
``mixed_moe_gemm``. The shapes, the expert row layout and the activation
variants mirror ``FlagTrain/benchmark/deepspeed/test_mixed_moe_gemm.py`` so the
two baselines stay comparable; the weights and biases are 16-bit here instead of
quantized. The shape key carries the activation so variants of one deployment
do not overwrite each other.
"""

import torch

OP_NAME = "moe_gemm"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True  # the operator writes the preallocated `output`

# benchmark/deepspeed/test_mixed_moe_gemm.py::_SHAPES
# (experts, tokens_per_expert, K, N)
_SHAPES = [
    (1, 128, 2048, 2048),
    (8, 16, 1024, 2048),
    (64, 1, 1024, 1024),
    (8, 128, 1024, 2048),
]

# DeepSpeed ActivationType ids: IDENTITY = 6, SILU = 2.
_ACTIVATIONS = (6, 2)


def native():
    """DeepSpeed's moe_gemm binding; raise when it cannot be loaded."""
    try:
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        kernel = InferenceCutlassBuilder().load().moe_gemm
    except Exception as exc:
        raise RuntimeError(
            "moe_gemm baseline requires DeepSpeed's CUTLASS 16-bit MoE GEMM "
            "(InferenceCutlassBuilder). Install deepspeed and ensure its CUDA "
            f"extensions build on this machine: {exc!r}"
        ) from exc
    return kernel


def grid():
    return [
        {"shape": shape, "activation": activation}
        for shape in _SHAPES
        for activation in _ACTIVATIONS
    ]


def build_inputs(binding, dtype, device):
    torch.manual_seed(0)
    experts, tokens, k, n = binding["shape"]
    activation = binding["activation"]
    weights = torch.randn((experts, k, n), dtype=dtype, device=device)
    biases = torch.randn((experts, n), dtype=dtype, device=device)
    rows = torch.arange(1, experts + 1, dtype=torch.int64, device=device) * tokens
    x = torch.randn((experts * tokens, k), dtype=dtype, device=device)
    output = torch.empty((experts * tokens, n), dtype=dtype, device=device)
    args = (output, x, weights, biases, rows, activation)
    return args, {}


def key_shape(binding):
    experts, tokens, k, n = binding["shape"]
    return f"{experts}x{tokens}x{k}x{n}_act{binding['activation']}"


def config(binding, dtype):
    experts, tokens, k, n = binding["shape"]
    return {
        "inputs": {
            "output": {"shape": [experts * tokens, n], "dtype": str(dtype)},
            "hidden_states": {"shape": [experts * tokens, k], "dtype": str(dtype)},
            "weight": {"shape": [experts, k, n], "dtype": str(dtype)},
            "bias": {"shape": [experts, n], "dtype": str(dtype)},
            "total_rows_before_expert": {
                "shape": [experts],
                "dtype": "torch.int64",
            },
            "activation": {"scalar": binding["activation"]},
        },
        "dims": {
            "experts": experts,
            "tokens_per_expert": tokens,
            "K": k,
            "N": n,
        },
    }
