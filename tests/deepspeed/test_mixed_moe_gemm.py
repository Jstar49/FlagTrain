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
"""The oracle uses the original signed weights, never the kernel's decoder."""

import math

import pytest
import torch
import torch.nn.functional as F

import flag_train
from flag_train.runtime import torch_device_fn

from .. import accuracy_utils as utils

pytestmark = [
    pytest.mark.mixed_moe_gemm,
    pytest.mark.skipif(
        flag_train.vendor_name
        not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend"),
        reason="Unsupported accelerator",
    ),
]

_ACTIVATIONS = {0: F.gelu, 1: F.relu, 2: F.silu, 6: lambda x: x}
_CASES = [
    ([37], 128, 64),
    ([0, 3, 0, 19, 1, 0, 12, 0], 129, 65),
    ([i % 5 for i in range(64)], 33, 31),
    ([7, 35, 129], 127, 129),
]


def _pack_weights(signed, num_bits):
    if num_bits == 8:
        return signed.to(torch.int8).contiguous()
    padded = F.pad(signed.to(torch.int16), (0, signed.shape[-1] % 2))
    low = padded[..., 0::2] & 15
    high = (padded[..., 1::2] & 15) << 4
    return (low | high).to(torch.uint8).contiguous()


def _inputs(counts, k, n, num_bits, dtype, strided=False):
    torch.manual_seed(0)
    m, experts = sum(counts), len(counts)
    qmax = 2 ** (num_bits - 1)
    signed = torch.randint(-qmax, qmax, (experts, k, n), dtype=torch.int8)
    scales = torch.rand((experts, n), dtype=torch.float32) / (
        qmax * math.sqrt(max(k, 1))
    )
    scales = scales.to(dtype).to(flag_train.device)
    weights = _pack_weights(signed, num_bits).to(flag_train.device)
    rows = torch.tensor(counts, dtype=torch.int64, device=flag_train.device).cumsum(0)
    biases = torch.randn((experts, n), dtype=dtype, device=flag_train.device)
    if strided:
        x = torch.randn((2 * m, 2 * k), dtype=dtype, device=flag_train.device)[::2, ::2]
        output = torch.full(
            (2 * m, 2 * n), float("nan"), dtype=dtype, device=flag_train.device
        )[::2, ::2]
    else:
        x = torch.randn((m, k), dtype=dtype, device=flag_train.device)
        output = torch.full((m, n), float("nan"), dtype=dtype, device=flag_train.device)
    return output, x, weights, scales, biases, rows, signed


def _reference(x, signed, scales, biases, counts, activation):
    # CPU FP64 matmul is an independent oracle; rounding dequantized weights
    # to the activation dtype is part of the weight-only quantization contract.
    dequantized = signed.float() * scales.cpu().float()[:, None, :]
    dequantized = dequantized.to(x.dtype).double()
    x_cpu = x.cpu().double()
    result = torch.empty((x.shape[0], signed.shape[2]), dtype=torch.float64)
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        values = torch.matmul(x_cpu[start:end], dequantized[expert])
        if biases is not None:
            values = values + biases[expert].cpu().double()
        result[start:end] = _ACTIVATIONS[activation](values)
        start = end
    return utils.to_reference(result.to(x.dtype).to(x.device))


def _assert_close(output, expected):
    # Tiled FP32 reduction and activation approximations need not round exactly
    # like the CPU FP64 oracle. Keep the shared floating-point defaults.
    utils.train_assert_close(output, expected, output.dtype)


@pytest.mark.parametrize("counts,k,n", _CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("activation", [0, 1, 2, 6])
@pytest.mark.parametrize("has_bias", [False, True])
def test_mixed_moe_gemm(counts, k, n, dtype, num_bits, activation, has_bias):
    """Catch wrong expert boundaries, skipped tails, scales, bias, or activation."""
    output, x, weights, scales, biases, rows, signed = _inputs(
        counts, k, n, num_bits, dtype
    )
    if not has_bias:
        biases = None
    expected = _reference(x, signed, scales, biases, counts, activation)
    result = flag_train.deepspeed.mixed_moe_gemm(
        output, x, weights, scales, biases, rows, num_bits, activation
    )
    assert result is output
    _assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_bits", [4, 8])
def test_mixed_moe_gemm_strides(dtype, num_bits):
    """Catch assuming contiguous input/output or overwriting view padding."""
    counts = [0, 1, 17, 0, 5]
    output, x, weights, scales, biases, rows, signed = _inputs(
        counts, 71, 35, num_bits, dtype, strided=True
    )
    expected = _reference(x, signed, scales, biases, counts, 6)
    storage = torch.full(
        (2 * sum(counts), 70), float("nan"), dtype=dtype, device=x.device
    )
    output = storage[::2, ::2]
    flag_train.deepspeed.mixed_moe_gemm(
        output, x, weights, scales, biases, rows, num_bits
    )
    _assert_close(output, expected)
    assert torch.isnan(storage[1::2]).all()
    assert torch.isnan(storage[::2, 1::2]).all()


@pytest.mark.parametrize(
    "num_bits,weight_dtype", [(4, torch.int8), (4, torch.uint8), (8, torch.int8)]
)
def test_mixed_moe_gemm_signed_extrema(num_bits, weight_dtype):
    """Catch unsigned decoding and swapped low/high INT4 nibbles."""
    limit = 2 ** (num_bits - 1)
    signed = torch.tensor([[[-limit, -1, 0, limit - 1]]], dtype=torch.int8)
    # The bytes are independently specified, so the test cannot hide a decoder
    # bug by sharing the same erroneous packing convention with the kernel.
    raw = [0xF8, 0x70] if num_bits == 4 else [-128, -1, 0, 127]
    storage_dtype = torch.uint8 if num_bits == 4 else torch.int8
    weights = torch.tensor([[raw]], dtype=storage_dtype, device=flag_train.device).view(
        weight_dtype
    )
    x = torch.ones((1, 1), dtype=torch.float16, device=flag_train.device)
    output = torch.empty((1, 4), dtype=x.dtype, device=x.device)
    scales = torch.tensor([[1.0, 0.5, 2.0, 0.25]], dtype=x.dtype, device=x.device)
    rows = torch.tensor([1], dtype=torch.int64, device=x.device)
    flag_train.deepspeed.mixed_moe_gemm(
        output, x, weights, scales, None, rows, num_bits
    )
    expected = signed[0].to(x.device).to(x.dtype) * scales
    utils.train_assert_equal(output, utils.to_reference(expected))


@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize(
    "counts,k,n", [([0, 0], 17, 7), ([3, 0, 2], 0, 7), ([3, 0, 2], 17, 0)]
)
def test_mixed_moe_gemm_empty_dimensions(num_bits, counts, k, n):
    """Catch invalid zero-size launches or losing the bias when K is zero."""
    output, x, weights, scales, biases, rows, signed = _inputs(
        counts, k, n, num_bits, torch.float16
    )
    expected = _reference(x, signed, scales, biases, counts, 2)
    result = flag_train.deepspeed.mixed_moe_gemm(
        output, x, weights, scales, biases, rows, num_bits, 2
    )
    assert result is output
    _assert_close(output, expected)


@pytest.mark.parametrize("num_bits", [4, 8])
def test_mixed_moe_gemm_graph_capture(num_bits):
    """Catch hidden host synchronization; changed routing must work on replay."""
    counts = [1, 3, 0, 2]
    output, x, weights, scales, biases, rows, signed = _inputs(
        counts, 33, 17, num_bits, torch.float16
    )
    graph_type = getattr(torch_device_fn, "CUDAGraph", None)
    if graph_type is None:
        graph_type = getattr(torch_device_fn, "NPUGraph", None)
    if graph_type is None:
        pytest.skip("The backend has no graph capture API")
    stream = torch_device_fn.Stream()
    stream.wait_stream(torch_device_fn.current_stream())
    with torch_device_fn.stream(stream):
        for _ in range(3):
            flag_train.deepspeed.mixed_moe_gemm(
                output, x, weights, scales, biases, rows, num_bits
            )
    torch_device_fn.current_stream().wait_stream(stream)
    graph = graph_type()
    with torch_device_fn.graph(graph):
        flag_train.deepspeed.mixed_moe_gemm(
            output, x, weights, scales, biases, rows, num_bits
        )
    new_counts = [0, 0, 5, 1]
    rows.copy_(torch.tensor(new_counts, dtype=torch.int64, device=x.device).cumsum(0))
    graph.replay()
    expected = _reference(x, signed, scales, biases, new_counts, 6)
    _assert_close(output, expected)


@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_moe_gemm_large_k(num_bits, dtype):
    """Catch a kernel that compiles for small K but hangs or miscomputes at K=2048."""
    counts = [128]
    output, x, weights, scales, biases, rows, signed = _inputs(
        counts, 2048, 2048, num_bits, dtype
    )
    expected = _reference(x, signed, scales, biases, counts, 2)
    flag_train.deepspeed.mixed_moe_gemm(
        output, x, weights, scales, biases, rows, num_bits, 2
    )
    _assert_close(output, expected)


@pytest.mark.parametrize("num_bits", [4, 8])
def test_mixed_moe_gemm_two_layers(num_bits):
    """Catch incompatible output reuse across two expert linear layers."""
    counts = [0, 2, 17, 1]
    hidden, x, weights, scales, biases, rows, signed = _inputs(
        counts, 33, 47, num_bits, torch.float16
    )
    output, _, weights2, scales2, biases2, _, signed2 = _inputs(
        counts, 47, 19, num_bits, torch.float16
    )
    flag_train.deepspeed.mixed_moe_gemm(
        hidden, x, weights, scales, biases, rows, num_bits, 2
    )
    flag_train.deepspeed.mixed_moe_gemm(
        output, hidden, weights2, scales2, biases2, rows, num_bits
    )
    hidden_ref = _reference(x, signed, scales, biases, counts, 2)
    expected = _reference(hidden_ref, signed2, scales2, biases2, counts, 6)
    _assert_close(output, expected)


@pytest.mark.parametrize(
    "invalid",
    [
        "bits",
        "activation",
        "weight_dtype",
        "scales",
        "rows",
        "bias",
        "weight_shape",
        "contiguous",
        "device",
    ],
)
def test_mixed_moe_gemm_rejects_invalid_inputs(invalid):
    """Catch unsupported arguments reaching a launch with invalid memory access."""
    output, x, weights, scales, biases, rows, _ = _inputs(
        [3, 1], 17, 9, 4, torch.float16
    )
    num_bits, activation = 4, 6
    if invalid == "bits":
        num_bits = 3
    elif invalid == "activation":
        activation = 3
    elif invalid == "weight_dtype":
        weights = weights.float()
    elif invalid == "scales":
        scales = scales[:, :1]
    elif invalid == "rows":
        rows = rows.int()
    elif invalid == "bias":
        biases = biases.float()
    elif invalid == "weight_shape":
        weights = weights[:, :, :1]
    elif invalid == "contiguous":
        scales = torch.empty((2, 18), dtype=x.dtype, device=x.device)[:, ::2]
    elif invalid == "device":
        scales = scales.cpu()
    with pytest.raises((ValueError, TypeError)):
        flag_train.deepspeed.mixed_moe_gemm(
            output, x, weights, scales, biases, rows, num_bits, activation
        )
