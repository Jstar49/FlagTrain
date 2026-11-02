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
"""Performance benchmark for the fused n-stream weighted aggregation (backward).

The baseline is ``torch.autograd.grad`` through Megatron-LM's
``native_h_aggregate`` -- the reference this operator is the derivative of --
taken from the installed ``megatron-core`` when that provides it, and otherwise
from ``native_h_aggregate_ref``: the transcription of ``hyper_connection.py``
this file carries, ``@torch.compile``-wrapped exactly as Megatron ships it.
megatron-core 0.19 and earlier have no ``hyper_connection`` module, so the
fallback is what lets an older container time a Megatron baseline instead of
skipping. The two paths are import success and import failure. The compiled
wrapper is kept either way, because it is what Megatron pays for.
The correctness suite differentiates the same function, so
the timed baseline is not an unchecked one; this file only times it.

Keeping that compile across a twenty-row sweep takes one adjustment, made by
``_full_compile_budget``: dynamo stops compiling after 8 recompiles per code
object and serves the eager function from then on, and the eager derivative of
this stage is unfused (it materializes the ``[s, b, n, C]`` product before
reducing) -- 2-3x slower than the compiled one. The fixture raises the budget
for this module's run, so every row is timed against the compiled derivative.

Timing range: only the backward. The operator under test takes ``grad_output``,
``x`` and ``h_pre`` and does no forward work; charging the baseline for the
forward that builds its graph would compare two different things.
``_MegatronBackwardBaseline`` therefore builds the graph outside the timed
region -- once per input set -- and every repetition runs ``torch.autograd.grad``
over it with ``retain_graph=True``. The saved state this leaves live is, like
``x`` and ``h_pre`` on the operator's side, what the stage is given rather than
what it produces.

That requires one adjustment to the baseline, made by the module's
``_no_buffer_donation`` fixture. A backward that inductor compiled with *donated
buffers* refuses to run with ``retain_graph=True``: it is allowed to write into
the forward's saved tensors, so it may run at most once. Disabling donation for
this benchmark's run lets the same compiled derivative be re-run over the same
graph. It is the conservative direction: the donating backward allocates less, so
the measured baseline is the slower of the two.

The setting is process-wide because the donation decision is baked into the
compiled artifact: a graph compiled while it was off keeps that shape, the
scoping is restored by the fixture. It only affects functions whose backward
goes through AOTAutograd, and this file's baseline is the only one in the
benchmark suite. ``base.py`` pins TF32 the same way, for the same reason.

The harness's own ``is_backward`` path is not used: it calls ``randn_like`` on
the op's output, and both this operator and the baseline return a tuple.

The gradient is worth its own benchmark because it is a *different* shape of
work from the forward: ``grad_x`` is a broadcast product with no reduction, while
``grad_h`` reduces over ``C``. Latency is reported with GB/s (the traffic the
three inputs and two outputs imply) and TFLOP/s (one multiply per element for
``grad_x`` and a multiply-add for ``grad_h``, i.e. ``3 * s * b * n * C``).
"""

import importlib.metadata
import warnings

import pytest
import torch
import torch._functorch.config

import flag_train
import flag_train.megatron

from .. import base

# ``(s, b, n, C)``: ``x`` is ``[s, b, n, C]``, ``h_pre`` is ``[s, b, n]``,
# ``grad_output`` is ``[s, b, C]``, and the outputs are ``grad_x`` of
# ``[s, b, n, C]`` and ``grad_h`` of ``[s, b, n]`` -- the same sweep the
# correctness suite checks every entry of.
#
# ``n`` is the mHC stream count (2-8) and ``C`` the hidden size; together they
# set the traffic, which is what this stage is. The row count walks a single row
# (where the launch, not the kernel, is the measurement), the tiles the
# correctness suite masks, and a full micro-batch. 2 and 8 bracket the stream
# axis, 7168 is a real hidden size that is not a power of two, and the
# ``(256, 32, ...)`` rows are the largest that fit comfortably beside the
# reference's own allocations.
# !! KEEP IN SYNC with tests/megatron/test_fused_h_aggregate_backward.py !!
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
    kept operation for operation, because this is the function whose derivative
    the operator is measured against. Upstream it carries a ``@torch.compile``
    decorator; the transcription is compiled by ``_load_native_h_aggregate``
    below so both baselines pay the same wrapper.
    """
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


def _load_native_h_aggregate():
    """The compiled baseline, and a label for where it came from.

    The compiled wrapper is kept: its derivative is what Megatron's training step
    actually runs, so it is what the operator has to be measured against. The
    import is done once at module import because ``megatron.core``'s ``__init__``
    pulls in the whole core tree.
    """
    try:
        from megatron.core.transformer.hyper_connection import native_h_aggregate
    except Exception as exc:
        warnings.warn(
            f"megatron-core {_MCORE_VERSION} does not provide "
            f"native_h_aggregate ({exc!r}); fused_h_aggregate_backward is "
            f"benchmarked against the autograd derivative of "
            f"native_h_aggregate_ref, this file's transcription of "
            f"Megatron-LM's implementation, compiled the way Megatron compiles "
            f"it",
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
_BASELINE = (
    "megatron native_h_aggregate autograd backward "
    f"({_BASELINE_SOURCE}, @torch.compile, donated buffers off)"
)


# Dynamo counts recompiles per code object and stops compiling after 8 by
# default, serving the eager function from then on. The sweep below needs one
# graph per (shape, dtype), so its tail would otherwise be timed against an
# *unfused* baseline -- neither what a training step pays (it compiles once per
# shape and caches the graph) nor a stable number: which rows came out eager
# would depend on what ran earlier in the process. Measured: with the default
# budget the bf16 half of the sweep, when a forward benchmark had already spent
# the budget in the same process, ran 2-3x slower than the fp16 half at the same
# shapes. The budget is raised for this module's run; see
# ``_full_compile_budget``.
_COMPILE_BUDGET = 256


@pytest.fixture(scope="module", autouse=True)
def _full_compile_budget():
    """Keep ``@torch.compile`` compiling across the whole sweep.

    Only the recompile budget is raised -- the compiled code is untouched. The
    graphs dynamo builds under it stay cached after the fixture restores the
    setting, which is the point: every row is measured against the compiled
    derivative Megatron runs.
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


@pytest.fixture(scope="module", autouse=True)
def _no_buffer_donation():
    """Compile the baseline's backward without buffer donation.

    A donating backward may write into the forward's saved tensors, so PyTorch
    only lets it run once (``retain_graph=False``); the graph here has to be
    re-run for every timed repetition. Module-scoped rather than wrapped around
    the graph build: the decision is baked into the compiled artifact, and
    inductor compiles lazily per shape and dtype, so the setting has to hold for
    every compile this benchmark triggers. Restored afterwards.
    """
    config = torch._functorch.config
    saved = config.donated_buffer
    config.donated_buffer = False
    try:
        yield
    finally:
        config.donated_buffer = saved


class _MegatronBackwardBaseline:
    """Megatron's backward for one input set, with the graph built up front.

    See the module docstring for why the forward is outside the timed region.
    The graph is keyed on the identity of the two differentiable inputs and the
    originals are held for exactly that reason -- a freed tensor's address can be
    handed to the next case's allocation, and an ``id()``-only cache would then
    reuse a graph built from different data.
    """

    def __init__(self):
        self._source_x = None
        self._source_h = None
        self._x = None
        self._h = None
        self._out = None

    def _build(self, x, h_pre):
        # Leaves on a copy: the harness's tensors are reused by the operator
        # under test, and ``requires_grad_`` on them would make that call build a
        # graph it never needs.
        self._x = x.detach().clone().requires_grad_(True)
        self._h = h_pre.detach().clone().requires_grad_(True)
        self._out = _native_h_aggregate(self._x, self._h)
        self._source_x = x
        self._source_h = h_pre

    def __call__(self, grad_output, x, h_pre):
        if self._source_x is not x or self._source_h is not h_pre:
            self._build(x, h_pre)
        return torch.autograd.grad(
            self._out, (self._x, self._h), grad_output, retain_graph=True
        )


_MEGATRON_BACKWARD = _MegatronBackwardBaseline()


class FusedHAggregateBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _BENCHMARK_SHAPES
    DEFAULT_SHAPE_DESC = "s, b, n, C"
    # The pass reads ``x`` once, ``grad_output`` once and writes ``grad_x`` --
    # three times the forward's traffic -- so GB/s is the reading that moves.
    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["gbps", "tflops"]

    def set_shapes(self, shape_file=None):
        # Each row is an ``(s, b, n, C)`` tuple with a fixed meaning, so the
        # shared core_shapes.yaml -- and the generic 1D/2D/3D extra shapes -- do
        # not apply.
        self.shapes = list(_BENCHMARK_SHAPES)

    def set_more_shapes(self):
        return []

    def get_gbps(self, args, latency):
        """Boundary traffic of one call: the three inputs in, both gradients out."""
        grad_output, x, h_pre = args
        s, b, n, C = x.shape
        element = x.element_size()
        io_bytes = x.numel() * element
        io_bytes += h_pre.numel() * element
        io_bytes += s * b * C * grad_output.element_size()
        io_bytes += x.numel() * element  # grad_x
        io_bytes += h_pre.numel() * element  # grad_h
        return io_bytes * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        """``3 * s * b * n * C``: a multiply for ``grad_x``, a multiply-add for ``grad_h``.

        The ``grad_h`` accumulation is counted as two FLOPs and the ``grad_x``
        product as one, which is what the kernel issues. The harness divides the
        count by the *operator's* latency, so the reading describes the operator
        rather than the baseline.
        """
        s, b, n, C = args[1].shape
        return 3 * s * b * n * C


def fused_h_aggregate_backward_input_fn(shape, dtype, device):
    """``(grad_output, x, h_pre)`` for one ``(s, b, n, C)`` case."""
    s, b, n, C = shape
    # ``x`` is what the forward computed from and ``grad_output`` what came back
    # through it; the operator reads both, so both are generated here rather than
    # derived from one another.
    grad_output = torch.randn((s, b, C), dtype=dtype, device=device)
    x = torch.randn((s, b, n, C), dtype=dtype, device=device)
    h_pre = torch.randn((s, b, n), dtype=dtype, device=device)
    yield grad_output, x, h_pre


def torch_op(grad_output, x, h_pre):
    """Baseline: Megatron's autograd backward. See the module docstring."""
    return _MEGATRON_BACKWARD(grad_output, x, h_pre)


def train_op(grad_output, x, h_pre):
    """The operator under test."""
    return flag_train.megatron.fused_h_aggregate_backward(grad_output, x, h_pre)


@pytest.mark.fused_h_aggregate_backward
def test_fused_h_aggregate_backward_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = FusedHAggregateBackwardBenchmark(
        input_fn=fused_h_aggregate_backward_input_fn,
        op_name="fused_h_aggregate_backward",
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
