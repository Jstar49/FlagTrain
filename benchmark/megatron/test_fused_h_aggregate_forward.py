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

The baseline is Megatron-LM's ``native_h_aggregate``: the reference this
operator is a port of, so it is the only baseline that answers "what did the
port cost". It is taken from the installed ``megatron-core`` when that provides
the function, and otherwise from ``native_h_aggregate_ref`` -- the transcription
of ``hyper_connection.py`` this file carries -- wrapped in ``@torch.compile``
exactly as Megatron wraps it. megatron-core 0.19 and earlier have no
``hyper_connection`` module, so without that fallback an older container could
only skip this benchmark; with it, the comparison is still against Megatron's
implementation compiled the way Megatron compiles it. The two paths are import
success and import failure; the report line says which of the two was timed.

Unlike the correctness suite, the baseline here keeps the compile: that wrapper
is what Megatron pays for this stage. (``tests/megatron/
test_fused_h_aggregate_forward.py`` compares against
``_torchdynamo_orig_callable``, the same code without a dynamo compile in front
of every shape in the sweep.) The functional suite holds that function to the
operator, so the timed baseline is not an unchecked one; this file only times
it.

Keeping that compile across a twenty-row sweep takes one adjustment, made by
``_full_compile_budget``: dynamo stops compiling after 8 recompiles per code
object and serves the eager function from then on, so the tail of the sweep
would be timed unfused. The fixture raises the budget for this module's run.

Both sides take the same ``(x, h_pre)`` tuple and return the aggregated ``[s, b,
C]``, so there is nothing to pre-compute outside the timed region. The stage is
a pure streaming read of ``x`` plus a write of ``out`` -- every output element
is an independent sum of ``n`` products -- so latency is reported with GB/s (the
traffic the shape implies) and TFLOP/s (the ``2 * s * b * n * C`` multiply-adds),
and the two together say whether a latency that moved was the memory or the
arithmetic.
"""

import importlib.metadata
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

# ---------------------------------------------------------------------------
# Megatron baseline
#
# ``native_h_aggregate`` arrived with the mHC stages in megatron-core 0.20; an
# older install has no ``hyper_connection`` module. So the baseline is the
# installed function when it can be imported, and otherwise a transcription of
# Megatron-LM's ``hyper_connection.py`` compiled the way Megatron compiles it.
# The rule is the same on every supported backend: import success picks
# megatron-core, import failure the transcription.
# ---------------------------------------------------------------------------


def _mcore_version():
    try:
        return importlib.metadata.version("megatron-core")
    except Exception:
        return "not installed"


_MCORE_VERSION = _mcore_version()


def native_h_aggregate_ref(x, h_pre):
    """Megatron-LM's ``native_h_aggregate``, transcribed.

    From ``megatron/core/transformer/hyper_connection.py`` at Megatron-LM 0.20,
    kept operation for operation, because this is what the operator is measured
    against. Upstream the function carries a ``@torch.compile`` decorator; the
    transcription is compiled by ``_load_native_h_aggregate`` below so both
    baselines pay the same wrapper.
    """
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


def _load_native_h_aggregate():
    """The compiled baseline, and a label for where it came from.

    The compiled wrapper is kept: it is what Megatron pays for this stage, so it
    is what the operator has to be measured against. The import is done once at
    module import because ``megatron.core``'s ``__init__`` pulls in the whole
    core tree.
    """
    try:
        from megatron.core.transformer.hyper_connection import native_h_aggregate
    except Exception as exc:
        warnings.warn(
            f"megatron-core {_MCORE_VERSION} does not provide "
            f"native_h_aggregate ({exc!r}); fused_h_aggregate_forward is "
            f"benchmarked against native_h_aggregate_ref, this file's "
            f"transcription of Megatron-LM's implementation, compiled the way "
            f"Megatron compiles it",
            stacklevel=2,
        )
        return torch.compile(native_h_aggregate_ref), (
            f"vendored from Megatron-LM (megatron-core {_MCORE_VERSION})"
        )

    # The label reports the version of the source actually imported, which a
    # PYTHONPATH checkout can differ from the distribution metadata.
    try:
        from megatron.core.package_info import __version__ as source_version
    except Exception:
        source_version = _MCORE_VERSION
    return native_h_aggregate, f"megatron-core {source_version}"


_native_h_aggregate, _BASELINE_SOURCE = _load_native_h_aggregate()

# The baseline's source and version are part of the label: a speedup means little
# without the implementation it was measured against.
_BASELINE = f"megatron native_h_aggregate ({_BASELINE_SOURCE}, @torch.compile)"

# Dynamo counts recompiles per code object and stops compiling after 8 by
# default, serving the eager function from then on. The sweep below needs one
# graph per (shape, dtype), so its tail would otherwise be timed against an
# *unfused* baseline -- neither what a training step pays (it compiles once per
# shape and caches the graph) nor a stable number: which rows came out eager
# would depend on what ran earlier in the process. The budget is raised for this
# module's run instead; see ``_full_compile_budget``.
_COMPILE_BUDGET = 256


@pytest.fixture(scope="module", autouse=True)
def _full_compile_budget():
    """Keep ``@torch.compile`` compiling across the whole sweep.

    Only the recompile budget is raised -- the compiled code is untouched. The
    graphs dynamo builds under it stay cached after the fixture restores the
    setting, which is the point: every row is measured against the compiled
    implementation Megatron runs.
    """
    config = torch._dynamo.config
    # Older torch spells the knob ``cache_size_limit``; where both exist they are
    # aliases, so setting both is redundant but harmless.
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
        out_dtype = torch.promote_types(x.dtype, h_pre.dtype)
        element = x.element_size()
        io_bytes = x.numel() * element
        io_bytes += h_pre.numel() * h_pre.element_size()
        io_bytes += s * b * C * out_dtype.itemsize
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


def torch_op(x, h_pre):
    """Baseline: Megatron-LM's ``native_h_aggregate``. See the module docstring."""
    return _native_h_aggregate(x, h_pre)


def train_op(x, h_pre):
    """The operator under test."""
    return flag_train.megatron.fused_h_aggregate_forward(x, h_pre)


@pytest.mark.fused_h_aggregate_forward
def test_fused_h_aggregate_forward_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = FusedHAggregateForwardBenchmark(
        input_fn=fused_h_aggregate_forward_input_fn,
        op_name="fused_h_aggregate_forward",
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
