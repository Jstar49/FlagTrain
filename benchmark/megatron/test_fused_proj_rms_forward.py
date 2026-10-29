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
"""Performance benchmark for the fused projection + RMS normalization (forward).

The baseline is Megatron-LM's ``native_proj_rms``: the reference this operator
is a port of, so it is the only baseline that answers "what did the port cost".
It is taken from the installed ``megatron-core`` when that provides the
function, and otherwise from ``native_proj_rms_ref`` -- the transcription of
``hyper_connection.py`` this file carries -- wrapped in ``@torch.compile``
exactly as Megatron wraps it. megatron-core 0.19 and earlier have no
``hyper_connection`` module, so without that fallback an older container could
only skip this benchmark; with it, the comparison is still against Megatron's
implementation compiled the way Megatron compiles it. Where megatron-core is
new enough to have the function but cannot import it, the loader raises rather
than timing the transcription under the package's name. The report line says
which of the two was timed.

Unlike the correctness suite, the baseline here keeps the compile: that wrapper
is what Megatron pays for this stage.
(``tests/megatron/test_fused_proj_rms_forward.py`` compares against
``_torchdynamo_orig_callable``, the same code without a dynamo compile in front
of every shape in the sweep.) The functional suite holds that function to the
operator, so the timed baseline is not an unchecked one; this file only times
it.

Both sides take the same ``(x, weight, eps)`` tuple and return ``(proj, r)``, so
there is nothing to pre-compute outside the timed region. The operation is a
matmul plus a row reduction, so latency is reported with both GB/s (the traffic
the shape implies) and TFLOP/s (the ``2 * M * N * K`` product).
"""

import importlib.metadata
import math
import re
import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import base

_EPS = 1e-6

# ``(M, N, K)``: ``x`` is ``[M, K]``, ``weight`` is ``[N, K]``, ``proj`` is
# ``[M, N]`` and ``r`` is ``[M, 1]`` -- the same sweep the correctness suite
# checks every entry of.
#
# 24 is the mHC mapping height, i.e. the width the operator exists for; 80, 288
# and 1024 walk the N tile (one masked tile, several tiles, and a grid where the
# output tile is the work). M covers a single row, a masked row tile, the
# row-block switch at 1024, and a full micro-batch. 8192 x 8192 keeps the
# reduction long enough for split-K behaviour to show up.
# !! KEEP IN SYNC with tests/megatron/test_fused_proj_rms_forward.py !!
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

# The upstream stage runs on FP16/BF16 activations; FP32 is measured by the
# correctness suite (it is the exact-arithmetic path) but is not what a training
# step pays for here.
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
# is raised rather than timed as if it were the package.
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
    kept operation for operation, because this is what the operator is measured
    against. Upstream the function carries a ``@torch.compile`` decorator; the
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

    The compiled wrapper is kept: it is what Megatron pays for this stage, so it
    is what the operator has to be measured against. The import is done once at
    module import because ``megatron.core``'s ``__init__`` pulls in the whole
    core tree.
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
                f"measured against Megatron's own implementation; the vendored "
                f"transcription in this file only stands in for megatron-core "
                f"< 0.20."
            ) from exc
        warnings.warn(
            f"megatron-core {_MCORE_VERSION} does not provide native_proj_rms "
            f"({exc!r}); fused_proj_rms_forward is benchmarked against "
            f"native_proj_rms_ref, this file's transcription of Megatron-LM's "
            f"implementation, compiled the way Megatron compiles it",
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
_BASELINE = f"megatron native_proj_rms ({_BASELINE_SOURCE}, @torch.compile)"


class FusedProjRmsForwardBenchmark(base.GenericBenchmark):
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
        """Boundary traffic of one call: read ``x`` and ``weight``, write both outputs."""
        x, weight = args[0], args[1]
        M, K = x.shape
        N = weight.shape[0]
        element = x.element_size()
        io_bytes = x.numel() * element
        io_bytes += weight.numel() * weight.element_size()
        io_bytes += M * N * element  # proj
        io_bytes += M * element  # r
        return io_bytes * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        """``2 * M * N * K`` -- the projection's multiply-adds.

        The norm is a second pass over ``x`` with no multiplies to speak of, and
        is not counted. The harness divides this count by the *operator's*
        latency, so the reading describes the operator rather than the baseline.
        """
        M, K = args[0].shape
        N = args[1].shape[0]
        return 2 * M * N * K


def fused_proj_rms_forward_input_fn(shape, dtype, device):
    """``(x, weight, eps)`` for one ``(M, N, K)`` case."""
    M, N, K = shape
    x = torch.randn((M, K), dtype=dtype, device=device)
    # Megatron initialises the mapping with unit-variance rows scaled by
    # ``1/sqrt(K)``, which keeps ``proj`` O(1) at every K; a raw randn weight
    # would make the projection grow with K and the comparison dtype-dependent.
    weight = torch.randn((N, K), dtype=torch.float32, device=device)
    weight = (weight / math.sqrt(max(K, 1))).to(dtype)
    yield x, weight, _EPS


def torch_op(x, weight, eps):
    """Baseline: Megatron-LM's ``native_proj_rms``. See the module docstring."""
    return _native_proj_rms(x, weight, eps)


def train_op(x, weight, eps):
    """The operator under test."""
    return flag_train.megatron.fused_proj_rms_forward(x, weight, eps)


@pytest.mark.fused_proj_rms_forward
def test_fused_proj_rms_forward_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = FusedProjRmsForwardBenchmark(
        input_fn=fused_proj_rms_forward_input_fn,
        op_name="fused_proj_rms_forward",
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
