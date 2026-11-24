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
"""Performance benchmark for bias_swiglu_impl.

The baseline is Megatron-LM's ``bias_swiglu_impl`` when megatron-core >= 0.20
is installed. Older versions lack the clamp / SiTU arguments, so there is no
on-machine baseline for them: ``torch_op`` is ``None`` and ``--base-data`` is
the only comparison, rather than timing a vendored torch composition. The
``Baseline:`` line printed by the test records which one ran.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). ``--level core`` runs the shapes
below. One table covers the three activations (plain, clamped, SiTU-GLU);
``--dtypes`` chooses the dtypes. The operator is forward-only.
"""

import importlib.metadata
import re

import pytest
import torch

import flag_train

from .. import base, consts

# (tokens, 2 * hidden): the last dim is the concatenated [gate | linear].
_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


# Enough variants for 7 shapes x 3 activations x up to 4 dtypes of forward
# cases without Dynamo falling back to eager halfway through the suite.
_DYNAMO_RECOMPILE_LIMIT = 128

# megatron-core 0.20 is the first release with the clamp / SiTU arguments.
_MCORE_MIN_VERSION = (0, 20)


def _megatron_version():
    """Installed megatron-core version and ``(major, minor)``, or ``(None, None)``."""
    try:
        version = importlib.metadata.version("megatron-core")
    except Exception:
        return None, None
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", version)
    parsed = None if match is None else (int(match.group(1)), int(match.group(2) or 0))
    return version, parsed


_MCORE_VERSION, _MCORE_VERSION_TUPLE = _megatron_version()


@pytest.fixture(scope="module", autouse=True)
def _configure_torch_compile():
    """Keep the compiled baseline working across this module's case count."""
    import torch._dynamo
    import torch._inductor.config as inductor_config

    # One compile thread keeps Inductor's compilation in-process; vendor Triton
    # builds may fail to pickle the compiled kernel back from a worker.
    saved_threads = inductor_config.compile_threads
    inductor_config.compile_threads = 1
    saved_autotune = inductor_config.triton.autotune_pointwise
    # torch_musa's Inductor autotune timer imports
    # triton.backends.mtgpu.musa_testing, which the FlagTree Triton build does
    # not ship. Disable the pointwise config search on MUSA; the compiled
    # baseline itself still runs.
    if flag_train.vendor_name == "mthreads":
        inductor_config.triton.autotune_pointwise = False

    dynamo = torch._dynamo.config
    limit_name = next(
        (
            name
            for name in ("recompile_limit", "cache_size_limit")
            if hasattr(dynamo, name)
        ),
        None,
    )
    saved_limit = getattr(dynamo, limit_name) if limit_name else None
    if limit_name is not None:
        setattr(dynamo, limit_name, max(saved_limit, _DYNAMO_RECOMPILE_LIMIT))

    try:
        yield
    finally:
        inductor_config.compile_threads = saved_threads
        inductor_config.triton.autotune_pointwise = saved_autotune
        if limit_name is not None:
            setattr(dynamo, limit_name, saved_limit)


_CLAMPED_KWARGS = {"clamp_value": 2.0}
_SITU_KWARGS = {"gate_clamp_scale": 4.0, "linear_clamp_scale": 2.5}


def _dtypes():
    """``consts.FLOAT_DTYPES`` with bf16/fp64 dropped where the backend has none.

    Same rule as ``utils.FLOAT_DTYPES`` in the correctness suite, so both see
    the same dtypes on every backend.
    """
    dtypes = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    if flag_train.runtime.device.support_fp64:
        dtypes.append(torch.float64)
    return dtypes


class BiasSwigluImplBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


# Variant label -> operator kwargs. The label travels with each input row, so
# one benchmark table covers all three activations.
_VARIANT_KWARGS = {
    "plain": {},
    "clamped": _CLAMPED_KWARGS,
    "situ": _SITU_KWARGS,
}


def bias_swiglu_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    bias = torch.randn((shape[-1],), dtype=dtype, device=device)
    for variant in _VARIANT_KWARGS:
        yield (x, bias, variant)


def train_op(x, bias, variant):
    """The operator under test."""
    return flag_train.megatron.bias_swiglu_impl(x, bias, **_VARIANT_KWARGS[variant])


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_perf():
    """Plain, clamped, and SiTU-GLU rows for every shape in one table."""
    print(f"\nBaseline: {_BASELINE}")
    bench = BiasSwigluImplBenchmark(
        input_fn=bias_swiglu_input_fn,
        op_name="bias_swiglu_impl",
        torch_op=torch_op,
        dtypes=_dtypes(),
        is_backward=False,
    )
    bench.set_train(train_op)
    bench.run()


def _load_baseline():
    """A compiled Megatron baseline, or ``None`` when it cannot be provided."""
    if _MCORE_VERSION_TUPLE is None or _MCORE_VERSION_TUPLE < _MCORE_MIN_VERSION:
        return None

    try:
        from megatron.core.fusions.fused_bias_swiglu import (
            bias_swiglu_impl as installed,
        )
    except Exception:
        return None

    def torch_op(x, bias, variant):
        return installed(x, bias, **_VARIANT_KWARGS[variant])

    return torch_op


torch_op = _load_baseline()
_BASELINE = (
    f"megatron-core {_MCORE_VERSION}"
    if torch_op is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)
