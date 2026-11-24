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
"""Performance benchmark for the fused n-stream weighted aggregation (forward).

The baseline is Megatron-LM's ``_triton_h_aggregate_fwd`` when the installed
megatron-core provides a Triton kernel that can run here -- the kernel this port
is a port *of*. When that kernel cannot run, the baseline is Megatron's own
``native_h_aggregate``, the ``@torch.compile``d torch composition the vendor
defines this stage as; with an older megatron-core that predates it, the
transcription below is compiled the way Megatron compiles it. ``_BASELINE``
records which case was measured.

Both sides take the same ``(x, h_pre)`` tuple and return the aggregated
``[s, b, C]``, so there is nothing to pre-compute outside the timed region. The
stage is a pure streaming read of ``x`` plus a write of ``out`` -- every output
element is an independent sum of ``n`` products -- so latency is reported with
GB/s (the traffic the shape implies) and TFLOP/s (the ``2 * s * b * n * C``
multiply-adds), and the two together say whether a latency that moved was the
memory or the arithmetic.
"""

import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import base

# ``(s, b, n, C)``: ``x`` is ``[s, b, n, C]``, ``h_pre`` is ``[s, b, n]`` and the
# output is ``[s, b, C]`` -- the same sweep the correctness suite checks every
# entry of.
#
# ``n`` is the mHC stream count (2-8) and ``C`` the hidden size; together they
# set the traffic, which is what this stage is. The row count walks a single row
# (where the launch, not the kernel, is the measurement), the tiles the
# correctness suite masks, and a full micro-batch. 2 and 8 bracket the stream
# axis, 7168 is a real hidden size that is not a power of two, and the
# ``(256, 32, ...)`` rows are the largest that fit comfortably beside the
# reference's own allocations.
# !! KEEP IN SYNC with tests/megatron/test_fused_h_aggregate_forward.py !!
_BENCHMARK_SHAPES = [
    (1, 1, 2, 256),
    (1, 1, 4, 4096),
    (8, 16, 4, 4096),
    (64, 8, 4, 4096),
    (32, 32, 4, 4096),
    (128, 32, 4, 4096),
    (256, 32, 4, 4096),
    (256, 32, 2, 4096),
    (256, 32, 8, 4096),
    (256, 32, 4, 7168),
]

# The upstream stage runs on FP16/BF16 activations; FP32 is measured by the
# correctness suite (it is the exact-arithmetic path) but is not what a training
# step pays for here.
_DTYPES = [torch.float16]
if flag_train.runtime.device.support_bf16:
    _DTYPES.append(torch.bfloat16)


def native_h_aggregate_ref(x, h_pre):
    """Megatron-LM's ``native_h_aggregate`` body, transcribed."""
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


def _load_native_h_aggregate():
    """Megatron's ``native_h_aggregate``, or its transcribed body compiled."""
    try:
        from megatron.core.transformer.hyper_connection import native_h_aggregate
    except Exception:
        return torch.compile(native_h_aggregate_ref), "transcribed"
    return native_h_aggregate, "megatron-core"


# The compiled baseline needs one graph per (shape, dtype); with dynamo's
# default budget of 8 recompiles the tail of the sweep would be served eager.
_COMPILE_BUDGET = 256


@pytest.fixture(scope="module", autouse=True)
def _full_compile_budget():
    """Keep ``@torch.compile`` compiling across the whole sweep."""
    config = torch._dynamo.config
    names = [
        name
        for name in ("recompile_limit", "cache_size_limit")
        if hasattr(config, name)
    ]
    saved = {name: getattr(config, name) for name in names}
    for name in names:
        setattr(config, name, max(saved[name], _COMPILE_BUDGET))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


def _load_baseline():
    """Megatron's Triton forward kernel, else its ``native_h_aggregate``."""
    try:
        from megatron.core.fusions.fused_mhc_kernels import _triton_h_aggregate_fwd

        x = torch.randn((1, 1, 2, 8), dtype=torch.bfloat16, device=flag_train.device)
        h_pre = torch.randn((1, 1, 2), dtype=torch.bfloat16, device=flag_train.device)
        with base.torch_device_fn.device(x.device):
            _triton_h_aggregate_fwd(x, h_pre)
            base.torch_device_fn.synchronize()
        return _triton_h_aggregate_fwd, "megatron _triton_h_aggregate_fwd"
    except Exception as exc:
        warnings.warn(
            "Megatron-LM's Triton fused_h_aggregate_forward kernel is "
            f"unavailable ({exc!r}); timing its native_h_aggregate torch "
            "composition instead.",
            stacklevel=2,
        )

    op, source = _load_native_h_aggregate()
    return op, f"megatron native_h_aggregate ({source}, @torch.compile)"


torch_op, _BASELINE = _load_baseline()


class FusedHAggregateForwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _BENCHMARK_SHAPES
    DEFAULT_SHAPE_DESC = "s, b, n, C"
    # Memory-bound at every shape -- every output element is an independent sum
    # of ``n`` products and there is no reuse -- so GB/s is the reading that
    # moves, and TFLOP/s is reported beside it to say so.
    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["gbps", "tflops"]

    def set_shapes(self, shape_file=None):
        # Each row is an ``(s, b, n, C)`` tuple with a fixed meaning, so the
        # shared core_shapes.yaml -- and the generic 1D/2D/3D extra shapes -- do
        # not apply.
        self.shapes = list(_BENCHMARK_SHAPES)

    def set_more_shapes(self):
        return []

    def get_gbps(self, args, latency):
        """Boundary traffic of one call: read ``x`` and ``h_pre``, write ``out``."""
        x, h_pre = args[0], args[1]
        s, b, n, C = x.shape
        # The aggregate is cast back to x.dtype, matching Megatron main.
        element = x.element_size()
        io_bytes = x.numel() * element
        io_bytes += h_pre.numel() * h_pre.element_size()
        io_bytes += s * b * C * element
        return io_bytes * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        """``2 * s * b * n * C`` -- one multiply-add per input element.

        The harness divides this count by the *operator's* latency, so the
        reading describes the operator rather than the baseline.
        """
        s, b, n, C = args[0].shape
        return 2 * s * b * n * C


def fused_h_aggregate_forward_input_fn(shape, dtype, device):
    """``(x, h_pre)`` for one ``(s, b, n, C)`` case."""
    s, b, n, C = shape
    x = torch.randn((s, b, n, C), dtype=dtype, device=device)
    # The mHC mapping parameters are held in fp32 upstream; the benchmark keeps
    # them in the activation dtype so both sides see one dtype, which is the
    # case the kernel is tuned for. The mixed pair is a correctness contract,
    # checked in the functional suite.
    h_pre = torch.randn((s, b, n), dtype=dtype, device=device)
    yield x, h_pre


def train_op(x, h_pre):
    """The operator under test."""
    return flag_train.megatron.fused_h_aggregate_forward(x, h_pre)


@pytest.mark.fused_h_aggregate_forward
def test_fused_h_aggregate_forward_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = FusedHAggregateForwardBenchmark(
        input_fn=fused_h_aggregate_forward_input_fn,
        op_name="fused_h_aggregate_forward",
        # Megatron's Triton kernel, or its native_h_aggregate composition when
        # that kernel cannot run here.
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
