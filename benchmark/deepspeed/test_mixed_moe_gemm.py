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
"""Performance benchmark for the mixed_moe_gemm operator.

The baseline is DeepSpeed's CUTLASS ``mixed_moe_gemm`` binding on the backends
that ship it -- the kernel this port is a port *of*. When it cannot be loaded,
there is no on-machine baseline: a dequantization plus per-expert PyTorch GEMM
is a correct implementation, not a competitor, so it is not timed;
``torch_op`` is ``None`` and ``--base-data`` is the only comparison. The same
composition is still used as an untimed correctness check in the input
builder.
"""

import math
import warnings

import pytest
import torch
import torch.nn.functional as F

import flag_train
from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn

from .. import base

_SHAPES = [
    (1, 128, 2048, 2048),
    (8, 16, 1024, 2048),
    (64, 1, 1024, 1024),
    (8, 128, 1024, 2048),
]


class MixedMoEGEMMBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "experts, tokens_per_expert, K, N"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        return []


# Backends on which DeepSpeed's CUTLASS MoE GEMM is expected to be usable.
# It ships as a CUDA extension, so only a backend that can compile and execute
# one can host it.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}

_DEEPSPEED_UNAVAILABLE_MSG = (
    "Its kernel ships prebuilt in DeepSpeed's CUTLASS extension "
    "(InferenceCutlassBuilder), which has to carry this GPU's compute "
    "capability. Install a build that does, or drop the backend from "
    "_DEEPSPEED_BASELINE_VENDORS."
)


def _deepspeed_weight_buffer(weights, num_bits, n):
    """Convert the benchmark's weights to the layout DeepSpeed's binding takes.

    ``weights`` is what the FlagTrain operator reads: int8 ``[E, K, N]`` for
    8-bit, packed uint8 ``[E, K, N/2]`` for 4-bit. DeepSpeed's CUTLASS binding
    takes unsigned offset-binary data: ``(q + 128)`` per byte for 8-bit, and
    ``(q + 8)`` nibbles packed over an ``E*K*N/2``-byte buffer exposed through
    an ``[E, K, N]`` ``as_strided`` view. The binding validates ``size(2)``
    but addresses the packed bytes by pointer, so the view has zero strides
    along E and K.

    This is a timing baseline, not a numerical oracle: DeepSpeed's weight byte
    order is a CUTLASS-internal interleave it exposes no preprocessing for, so
    the kernel reads the right footprint but the values are not compared.
    """
    if num_bits == 8:
        return (weights.to(torch.int16) + 128).to(torch.uint8)

    if n % 2:
        raise RuntimeError(f"mixed_moe_gemm 4-bit weights need an even N, got N={n}")
    unsigned = weights.to(torch.int16)
    low = unsigned & 15
    high = (unsigned >> 4) & 15
    values = torch.stack((low, high), dim=-1).flatten(-2)[..., :n]
    signed = torch.where(values >= 8, values - 16, values)
    nibbles = (signed + 8).to(torch.uint8)
    experts, k, _ = signed.shape
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    size = experts * k * n // 2
    buffer = torch.zeros(size + 128, dtype=torch.uint8, device=weights.device)
    buffer[:size] = packed.reshape(-1)
    return torch.as_strided(buffer, (experts, k, n), (0, 0, 1))


class _DeepSpeedBaseline:
    """DeepSpeed's binding, with the benchmark's weights converted once per input.

    ``base`` hands the same argument tuple to both ops, so the conversion has
    to happen here. It is cached per input tensor: the harness reuses one
    input for every timed iteration and the first call is warmup, so the
    conversion never lands inside the measured window.
    """

    def __init__(self, op):
        self._op = op
        self._source = None
        self._num_bits = None
        self._converted = None

    def __call__(self, output, x, weights, scales, biases, rows, num_bits, activation):
        if weights is not self._source or num_bits != self._num_bits:
            self._source = weights
            self._num_bits = num_bits
            self._converted = _deepspeed_weight_buffer(
                weights, num_bits, scales.shape[1]
            )
        return self._op(
            output,
            x,
            self._converted,
            scales,
            biases,
            rows,
            num_bits,
            activation,
        )


def _probe_deepspeed_mixed_moe_gemm(op):
    """Call the binding once per bit width, so a load that cannot run fails here."""
    experts, tokens, k, n = 2, 2, 1024, 1024
    device = runtime_device.name
    dtype = torch.float16
    scales = torch.rand((experts, n), dtype=dtype, device=device)
    biases = torch.randn((experts, n), dtype=dtype, device=device)
    rows = torch.arange(1, experts + 1, dtype=torch.int64, device=device) * tokens
    x = torch.randn((experts * tokens, k), dtype=dtype, device=device)
    output = torch.empty((experts * tokens, n), dtype=dtype, device=device)

    for num_bits, limit in ((8, 128), (4, 8)):
        signed = torch.randint(
            -limit, limit, (experts, k, n), dtype=torch.int8, device=device
        )
        if num_bits == 8:
            flag_weights = signed
        else:
            values = signed.to(torch.int16)
            flag_weights = (
                (values[..., 0::2] & 15) | ((values[..., 1::2] & 15) << 4)
            ).to(torch.uint8)
        weight = _deepspeed_weight_buffer(flag_weights, num_bits, n)
        op(output, x, weight, scales, biases, rows, num_bits, 6)

    torch_device_fn.synchronize()


def _load_deepspeed_mixed_moe_gemm():
    """``(op, reason)`` -- ``op`` is ``None`` when there is not a usable one."""
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        op = InferenceCutlassBuilder().load().mixed_moe_gemm
    except Exception as exc:  # noqa: BLE001
        return None, f"could not be loaded: {exc!r}"

    try:
        _probe_deepspeed_mixed_moe_gemm(op)
    except Exception as exc:  # noqa: BLE001
        return None, f"loaded but does not run here: {exc!r}"

    return op, None


# Resolved once, at module import time.
_DEEPSPEED_ORACLE, _DEEPSPEED_REASON = _load_deepspeed_mixed_moe_gemm()

_WHY = (
    f"{flag_train.vendor_name!r} is expected to use DeepSpeed's mixed_moe_gemm "
    f"as its baseline, which {_DEEPSPEED_REASON}. {_DEEPSPEED_UNAVAILABLE_MSG}"
)

if _DEEPSPEED_ORACLE is not None:
    _BASELINE = "deepspeed mixed_moe_gemm"
    torch_op = _DeepSpeedBaseline(_DEEPSPEED_ORACLE)
else:
    _BASELINE = "none (no on-machine baseline; --base-data is the only comparison)"
    if flag_train.vendor_name in _DEEPSPEED_BASELINE_VENDORS:
        warnings.warn(
            f"{_WHY} This run has no on-machine baseline; a torch composition "
            f"is not timed, so --base-data is the only comparison.",
            stacklevel=2,
        )
    torch_op = None


def _reference(output, x, weights, scales, biases, rows, num_bits, activation):
    """Untimed oracle: unpacking, dequantization, bias, and activation.

    Benchmark workloads allocate equally many tokens to every expert, so row
    boundaries are known from shapes; no device-to-host copies are timed.
    """
    experts, n = scales.shape
    if num_bits == 4:
        unsigned = weights.to(torch.int16)
        low = unsigned & 15
        high = (unsigned >> 4) & 15
        values = torch.stack((low, high), dim=-1).flatten(-2)[..., :n]
        signed = torch.where(values >= 8, values - 16, values)
    else:
        signed = weights
    dequantized = (signed.float() * scales.float()[:, None, :]).to(x.dtype)
    tokens_per_expert = x.shape[0] // experts
    for expert in range(experts):
        start = expert * tokens_per_expert
        end = start + tokens_per_expert
        result = torch.matmul(x[start:end], dequantized[expert]).float()
        result = result + biases[expert].float()
        if activation == 0:
            result = F.gelu(result)
        elif activation == 1:
            result = F.relu(result)
        elif activation == 2:
            result = F.silu(result)
        output[start:end].copy_(result)
    return output


def input_fn(shape, dtype, device, num_bits, activation):
    torch.manual_seed(0)
    experts, tokens, k, n = shape
    limit = 2 ** (num_bits - 1)
    signed = torch.randint(
        -limit, limit, (experts, k, n), dtype=torch.int8, device=device
    )
    if num_bits == 4:
        values = signed.to(torch.int16)
        weights = ((values[..., 0::2] & 15) | ((values[..., 1::2] & 15) << 4)).to(
            torch.uint8
        )
    else:
        weights = signed
    scales = (torch.rand((experts, n), device=device) / (limit * math.sqrt(k))).to(
        dtype
    )
    biases = torch.randn((experts, n), dtype=dtype, device=device)
    rows = torch.arange(1, experts + 1, dtype=torch.int64, device=device) * tokens
    x = torch.randn((experts * tokens, k), dtype=dtype, device=device)
    output = torch.empty((experts * tokens, n), dtype=dtype, device=device)
    args = (output, x, weights, scales, biases, rows, num_bits, activation)
    expected = _reference(*args).clone()
    flag_train.deepspeed.mixed_moe_gemm(*args)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
    yield args


@pytest.mark.mixed_moe_gemm
@pytest.mark.skipif(
    flag_train.vendor_name
    not in ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend", "mthreads"),
    reason="Unsupported accelerator",
)
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("activation", [6, 2], ids=["identity", "silu"])
def test_mixed_moe_gemm_perf(num_bits, activation):
    def make_inputs(shape, dtype, device):
        yield from input_fn(shape, dtype, device, num_bits, activation)

    print(f"\nBaseline: {_BASELINE}")
    bench = MixedMoEGEMMBenchmark(
        input_fn=make_inputs,
        op_name="mixed_moe_gemm",
        torch_op=torch_op,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_train(flag_train.deepspeed.mixed_moe_gemm)
    bench.run()
