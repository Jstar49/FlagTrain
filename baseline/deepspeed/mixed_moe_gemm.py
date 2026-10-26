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

"""NVIDIA baseline for mixed_moe_gemm.

The baseline is DeepSpeed's CUTLASS mixed-precision MoE GEMM:

    deepspeed.ops.op_builder.InferenceCutlassBuilder().load().mixed_moe_gemm
    (output, hidden_states, weight, scales, bias,
     total_rows_before_expert, num_bits, activation)

Source: ``deepspeed/inference/v2/kernels/cutlass_ops/moe_gemm/moe_gemm.cu``.
DeepSpeed 0.19.8's ``MixedMoEGEMM`` wrapper calls the six-argument ``moe_gemm``
with eight arguments, so this module calls the pybind binding directly instead
of that wrapper.

When DeepSpeed is unavailable, ``native()`` raises; the collector does not
collect this operator's baseline and does not fall back to a torch composition.

The input construction mirrors the shapes, dtypes, weight values, scales, biases
and row boundaries in
``FlagTrain/benchmark/deepspeed/test_mixed_moe_gemm.py``. Weights are prepared
the way DeepSpeed expects:

* 8 bit: uint8 ``[E, K, N]`` in 0..255 (the kernel dequantizes with
  ``(q - 128) * scale``);
* 4 bit: packed uint8 with ``E*K*N/2`` backing bytes, exposed to the binding as
  an ``[E, K, N]`` ``as_strided`` view (the binding validates ``size(2) == N``;
  the kernel only takes ``data_ptr`` and addresses 4-bit data itself).

Note: the mixed kernel's weight byte order is a CUTLASS-internal interleave.
DeepSpeed 0.19.8 exposes no preprocessing entry point for it (and its wrapper is
broken), and the layout changes with the CUTLASS version / kernel configuration.
The collector measures latency and NCU metrics only and never compares values,
so this module guarantees the **logical shape and real packed footprint** but is
not a numerical oracle; numerical correctness is covered by the torch
composition in the FlagTrain benchmark. The baseline is still DeepSpeed's real
kernel with real FLOPs and real weight bandwidth.

The benchmark parameterizes ``num_bits`` (4/8) and ``activation`` (6=identity,
2=silu); both go into the binding, and the shape key uses a semantic string so
different quantization/activation configurations of the same (experts, tokens,
K, N) do not overwrite each other.
"""

import math

import torch

OP_NAME = "mixed_moe_gemm"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True  # the operator writes the preallocated `output`

# benchmark/deepspeed/test_mixed_moe_gemm.py::_SHAPES
_SHAPES = [
    (1, 128, 2048, 2048),
    (8, 16, 1024, 2048),
    (64, 1, 1024, 1024),
    (8, 128, 1024, 2048),
]
_NUM_BITS = (4, 8)
_ACTIVATIONS = (6, 2)  # ids: identity, silu


def native():
    """DeepSpeed's mixed_moe_gemm binding; raise when it cannot be loaded."""
    try:
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        kernel = InferenceCutlassBuilder().load().mixed_moe_gemm
    except Exception as exc:
        raise RuntimeError(
            "mixed_moe_gemm baseline requires DeepSpeed's CUTLASS mixed MoE GEMM "
            "(InferenceCutlassBuilder). Install deepspeed and ensure its CUDA "
            f"extensions build on this machine: {exc!r}"
        ) from exc
    return kernel


def grid():
    return [
        {"shape": shape, "num_bits": bits, "activation": activation}
        for shape in _SHAPES
        for bits in _NUM_BITS
        for activation in _ACTIVATIONS
    ]


def _weight_buffer(signed, num_bits):
    """The weight tensor the binding accepts.

    8 bit: one uint8 per logical weight, ``[E, K, N]``.
    4 bit: two nibbles per byte along N, stored in a ``E*K*N/2``-byte buffer and
    exposed through an ``[E, K, N]`` view (the binding validates that shape; the
    kernel reads the packed bytes by pointer).
    """
    if num_bits == 8:
        return (signed.to(torch.int16) + 128).to(torch.uint8)

    experts, k, n = signed.shape
    if n % 2:
        raise RuntimeError(
            "mixed_moe_gemm 4-bit weights need an even N; the binding validates N "
            f"before reading the packed E*K*N/2 bytes, got N={n}."
        )
    nibbles = (signed.to(torch.int16) + 8).to(torch.uint8)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    # The kernel reads E*K*N/2 bytes per expert; the [E, K, N] shape is only
    # what the binding validates. Zero strides keep the view inside the buffer.
    # The trailing slack absorbs any over-read from the kernel's internal tiling.
    size = experts * k * n // 2
    buffer = torch.zeros(size + 128, dtype=torch.uint8, device=signed.device)
    buffer[:size] = packed.reshape(-1)
    return torch.as_strided(buffer, (experts, k, n), (0, 0, 1))


def build_inputs(binding, dtype, device):
    torch.manual_seed(0)
    experts, tokens, k, n = binding["shape"]
    num_bits = binding["num_bits"]
    activation = binding["activation"]
    limit = 2 ** (num_bits - 1)
    signed = torch.randint(
        -limit, limit, (experts, k, n), dtype=torch.int8, device=device
    )
    weights = _weight_buffer(signed, num_bits)
    scales = (torch.rand((experts, n), device=device) / (limit * math.sqrt(k))).to(
        dtype
    )
    biases = torch.randn((experts, n), dtype=dtype, device=device)
    rows = torch.arange(1, experts + 1, dtype=torch.int64, device=device) * tokens
    x = torch.randn((experts * tokens, k), dtype=dtype, device=device)
    output = torch.empty((experts * tokens, n), dtype=dtype, device=device)
    args = (output, x, weights, scales, biases, rows, num_bits, activation)
    return args, {}


def key_shape(binding):
    experts, tokens, k, n = binding["shape"]
    return (
        f"{experts}x{tokens}x{k}x{n}"
        f"_bits{binding['num_bits']}_act{binding['activation']}"
    )


def config(binding, dtype):
    experts, tokens, k, n = binding["shape"]
    bits = binding["num_bits"]
    return {
        "inputs": {
            "output": {"shape": [experts * tokens, n], "dtype": str(dtype)},
            "x": {"shape": [experts * tokens, k], "dtype": str(dtype)},
            # 4 bit: the binding needs an [E, K, N] view over the same
            # E*K*N/2 packed bytes; report the logical packed tensor the
            # benchmark builds.
            "weights": {
                "shape": [experts, k, n if bits == 8 else (n + 1) // 2],
                "dtype": "torch.uint8" if bits == 4 else "torch.int8",
            },
            "scales": {"shape": [experts, n], "dtype": str(dtype)},
            "biases": {"shape": [experts, n], "dtype": str(dtype)},
            "rows": {"shape": [experts], "dtype": "torch.int64"},
            "num_bits": {"scalar": bits},
            "activation": {"scalar": binding["activation"]},
        },
        "dims": {
            "experts": experts,
            "tokens_per_expert": tokens,
            "K": k,
            "N": n,
        },
    }
