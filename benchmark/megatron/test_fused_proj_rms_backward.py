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
"""Performance benchmark for the fused projection + RMS normalization (backward).

The baseline is ``torch.autograd.grad`` through Megatron-LM's ``native_proj_rms``
-- the reference implementation this operator differentiates -- taken from the
installed ``megatron-core`` when that provides it, and otherwise from
``native_proj_rms_ref``: the transcription of ``hyper_connection.py`` this file
carries, ``@torch.compile``-wrapped exactly as Megatron ships it. megatron-core
0.19 and earlier have no ``hyper_connection`` module, so the fallback is what
lets an older container time a Megatron baseline instead of skipping. Where
megatron-core is new enough to have the function but cannot import it, the
loader raises rather than timing the transcription under the package's name.
The compiled wrapper is kept either way, because it is what Megatron pays for.
The correctness suite differentiates the same function, so the timed baseline is
not an unchecked one; this file only times it.

Timing range: only the backward. The operator under test takes ``x``,
``weight``, ``grad_proj``, ``grad_r`` and the forward's saved ``r``, and does no
forward work; charging the baseline for the forward that builds its graph would
compare two different things. ``_MegatronBackwardBaseline`` therefore builds the
graph outside the timed region -- once per input set -- and every repetition runs
``torch.autograd.grad`` over it with ``retain_graph=True``. The saved state this
leaves live (the graph's own buffers) is, like ``r`` on the operator's side,
what the stage is given rather than what it produces.

That requires one adjustment to the baseline, made by the module's
``_no_buffer_donation`` fixture. A backward that inductor compiled with *donated
buffers* refuses to run with ``retain_graph=True``: it is allowed to write into
the forward's saved tensors, so it may run at most once. Disabling donation for
this benchmark's run lets the same compiled derivative be re-run over the same
graph. It is the conservative direction: the donating backward allocates less, so
the measured baseline is the slower of the two. The alternative, Megatron's
*eager* derivative, is slower still -- measured 1.3-3.3x at the shapes here --
and would flatter the operator rather than hold it to what Megatron pays.

The setting is process-wide because the donation decision is baked into the
compiled artifact: a graph compiled while it was off keeps that shape, the
scoping is restored by the fixture. It only affects functions whose backward
goes through AOTAutograd, and this file's baseline is the only one in the
benchmark suite. ``base.py`` pins TF32 the same way, for the same reason.

The harness's own ``is_backward`` path is not used: it calls ``randn_like`` on
the op's output, and both this operator and the baseline return a tuple.

Latency is reported with GB/s (the traffic the five inputs and two outputs
imply) and TFLOP/s (the two ``2 * M * N * K`` products).
"""

import importlib.metadata
import math
import re
import warnings

import pytest
import torch
import torch._functorch.config

import flag_train
import flag_train.megatron

from .. import base

_EPS = 1e-6

# ``(M, N, K)``: ``x`` is ``[M, K]``, ``weight`` is ``[N, K]``, ``grad_proj`` is
# ``[M, N]``, ``grad_r`` is ``[M, 1]``; the outputs are ``grad_x`` ``[M, K]`` and
# ``grad_weight`` ``[N, K]`` -- the same sweep the correctness suite checks every
# entry of.
#
# The two products parallelise differently: ``grad_weight`` reduces over M while
# ``grad_x`` reduces over N, so the sweep walks both reduction lengths -- M from
# a single row to a full micro-batch, N at mHC's 24 up to a tile-filling 1024,
# and K out to 8192.
# !! KEEP IN SYNC with tests/megatron/test_fused_proj_rms_backward.py !!
_BENCHMARK_SHAPES = [
    (1, 24, 4096),
    (128, 24, 4096),
    (256, 24, 4096),
    (1024, 24, 4096),
    (4096, 24, 4096),
    (8192, 24, 8192),
    (4096, 80, 4096),
    (4096, 288, 4096),
    (4096, 1024, 4096),
    (8192, 4096, 8192),
]

# The upstream stage runs on FP16/BF16 activations; FP32 is the correctness
# suite's exact-arithmetic path, not what a training step pays for here.
_DTYPES = [torch.float16]
if flag_train.runtime.device.support_bf16:
    _DTYPES.append(torch.bfloat16)

# ---------------------------------------------------------------------------
# Megatron baseline
#
# ``native_proj_rms`` arrived with the mHC proj_rms stage in megatron-core 0.20;
# 0.19 and earlier have no ``hyper_connection`` module. So the baseline is the
# installed function when it can be imported, and otherwise a transcription of
# Megatron-LM's ``hyper_connection.py`` compiled the way Megatron compiles it.
# The rule is the same on every supported backend: megatron-core when it has
# the function, the transcription otherwise. A megatron-core that is new enough
# to have the function but cannot import it is a broken installation, and that
# is raised rather than differentiated as if it were the package.
# ---------------------------------------------------------------------------
_MCORE_MIN_VERSION = (0, 20)


def _mcore_version():
    try:
        return importlib.metadata.version("megatron-core")
    except Exception:
        return "not installed"


_MCORE_VERSION = _mcore_version()


def _version_tuple(version):
    """``(major, minor)`` of a distribution version, or ``None`` without one."""
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", version)
    return None if match is None else (int(match.group(1)), int(match.group(2) or 0))


def native_proj_rms_ref(x, weight, eps=1e-6):
    """Megatron-LM's ``native_proj_rms``, transcribed.

    From ``megatron/core/transformer/hyper_connection.py`` at Megatron-LM 0.20,
    kept operation for operation, because its derivative is the baseline here.
    Upstream the function carries a ``@torch.compile`` decorator; the
    transcription is compiled by ``_load_native_proj_rms`` below so both
    baselines pay the same wrapper.
    """
    proj = torch.matmul(x, weight.t())
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    v = norm / math.sqrt(K) + eps
    r = 1.0 / v
    return proj, r


def _load_native_proj_rms():
    """The compiled baseline, and a label for where it came from.

    The compiled wrapper is kept: its derivative is what Megatron's training step
    actually runs, so it is what the operator has to be measured against. The
    import is done once at module import because ``megatron.core``'s ``__init__``
    pulls in the whole core tree.
    """
    try:
        from megatron.core.transformer.hyper_connection import native_proj_rms
    except Exception as exc:
        version = _version_tuple(_MCORE_VERSION)
        if version is not None and version >= _MCORE_MIN_VERSION:
            raise RuntimeError(
                f"megatron-core {_MCORE_VERSION} should provide "
                f"native_proj_rms, but importing it failed: {exc!r}. This "
                f"operator is the proj_rms stage of "
                f"megatron/core/fusions/fused_mhc_kernels.py, so it must be "
                f"measured against Megatron's own derivative; the vendored "
                f"transcription in this file only stands in for megatron-core "
                f"< 0.20."
            ) from exc
        warnings.warn(
            f"megatron-core {_MCORE_VERSION} does not provide native_proj_rms "
            f"({exc!r}); fused_proj_rms_backward is benchmarked against the "
            f"autograd derivative of native_proj_rms_ref, this file's "
            f"transcription of Megatron-LM's implementation, compiled the way "
            f"Megatron compiles it",
            stacklevel=2,
        )
        return torch.compile(native_proj_rms_ref), (
            f"vendored from Megatron-LM (megatron-core {_MCORE_VERSION})"
        )

    # The distribution metadata is what the gate above reads; the label reports
    # the version of the source actually imported, which a PYTHONPATH checkout
    # can differ from.
    try:
        from megatron.core.package_info import __version__ as source_version
    except Exception:
        source_version = _MCORE_VERSION
    return native_proj_rms, f"megatron-core {source_version}"


_native_proj_rms, _BASELINE_SOURCE = _load_native_proj_rms()

# The baseline's source and version are part of the label: a speedup means little
# without the implementation it was measured against.
_BASELINE = (
    "megatron native_proj_rms autograd backward "
    f"({_BASELINE_SOURCE}, @torch.compile, donated buffers off)"
)


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
        self._source_weight = None
        self._x = None
        self._weight = None
        self._proj = None
        self._r = None

    def _build(self, x, weight):
        # Leaves on a copy: the harness's tensors are reused by the operator
        # under test, and ``requires_grad_`` on them would make that call build a
        # graph it never needs.
        self._x = x.detach().clone().requires_grad_(True)
        self._weight = weight.detach().clone().requires_grad_(True)
        self._proj, self._r = _native_proj_rms(self._x, self._weight, _EPS)
        self._source_x = x
        self._source_weight = weight

    def __call__(self, x, weight, grad_proj, grad_r, r):
        """``r`` is the operator's saved scale; this graph recomputes its own."""
        if self._source_x is not x or self._source_weight is not weight:
            self._build(x, weight)
        return torch.autograd.grad(
            (self._proj, self._r),
            (self._x, self._weight),
            (grad_proj, grad_r),
            retain_graph=True,
        )


_baseline = _MegatronBackwardBaseline()


class FusedProjRmsBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _BENCHMARK_SHAPES
    DEFAULT_SHAPE_DESC = "M, N, K"
    # Compute-bound at the wide shapes and memory-bound at the narrow ones; both
    # readings are reported so a latency that moved says which side it moved on.
    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["gbps", "tflops"]

    def set_shapes(self, shape_file=None):
        # Each row is an ``(M, N, K)`` triple with a fixed meaning, so the shared
        # core_shapes.yaml -- and the generic 1D/2D/3D extra shapes -- do not
        # apply.
        self.shapes = list(_BENCHMARK_SHAPES)

    def set_more_shapes(self):
        return []

    def get_gbps(self, args, latency):
        """Boundary traffic: five inputs read, two gradients written."""
        x, weight, grad_proj, grad_r, r = args
        M, K = x.shape
        N = weight.shape[0]
        element = x.element_size()
        io_bytes = x.numel() * element  # x
        io_bytes += weight.numel() * weight.element_size()  # weight
        io_bytes += grad_proj.numel() * grad_proj.element_size()  # grad_proj
        io_bytes += 2 * M * element  # grad_r and r
        io_bytes += M * K * element  # grad_x
        io_bytes += N * K * element  # grad_weight
        return io_bytes * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        """``4 * M * N * K``: the two products, ``grad_proj @ weight`` and
        ``grad_proj^T @ x``.

        The coefficient term is one multiply per element of ``x``, three orders of
        magnitude below the products, and is not counted. The harness divides this
        count by the *operator's* latency, so the reading describes the operator
        rather than the baseline.
        """
        M, K = args[0].shape
        N = args[1].shape[0]
        return 4 * M * N * K


def fused_proj_rms_backward_input_fn(shape, dtype, device):
    """``(x, weight, grad_proj, grad_r, r)`` for one ``(M, N, K)`` case.

    ``r`` comes from *our* forward, because that is the tensor the operator is
    handed in training: the forward saves it and the backward consumes it.
    """
    M, N, K = shape
    x = torch.randn((M, K), dtype=dtype, device=device)
    # Megatron initialises the mapping with unit-variance rows scaled by
    # ``1/sqrt(K)``, which keeps the products O(1) at every K; a raw randn weight
    # would make them grow with K and the comparison dtype-dependent.
    weight = torch.randn((N, K), dtype=torch.float32, device=device)
    weight = (weight / math.sqrt(max(K, 1))).to(dtype)
    grad_proj = torch.randn((M, N), dtype=dtype, device=device)
    grad_r = torch.randn((M, 1), dtype=dtype, device=device)
    _, r = flag_train.megatron.fused_proj_rms_forward(x, weight, _EPS)
    yield x, weight, grad_proj, grad_r, r


def torch_op(x, weight, grad_proj, grad_r, r):
    """Baseline: Megatron's autograd backward. See the module docstring."""
    return _baseline(x, weight, grad_proj, grad_r, r)


def train_op(x, weight, grad_proj, grad_r, r):
    """The operator under test."""
    return flag_train.megatron.fused_proj_rms_backward(x, weight, grad_proj, grad_r, r)


@pytest.mark.fused_proj_rms_backward
def test_fused_proj_rms_backward_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = FusedProjRmsBackwardBenchmark(
        input_fn=fused_proj_rms_backward_input_fn,
        op_name="fused_proj_rms_backward",
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
