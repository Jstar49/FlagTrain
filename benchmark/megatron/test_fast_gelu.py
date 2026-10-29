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
"""Performance benchmark for Megatron Fast GELU.

The preferred baseline is Megatron-LM's official
``megatron.core.activations.fast_gelu`` interface. If Megatron-LM is not
installed, the benchmark logs a warning and falls back to the equivalent
PyTorch composition. Timing, warmup and synchronization come from
``benchmark/base.py`` (``triton.testing.do_bench`` in kernel mode).
``--level core`` runs the small, medium and large shapes below;
``--level comprehensive`` adds the extra shapes.
"""

import warnings

import pytest
import torch

import flag_train

from .. import base

# Small, medium, and a Megatron-MLP-sized activation tensor (tokens, hidden).
_CORE_SHAPES = [
    (1,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (4, 8192, 4096),
    (8, 4096, 11008),
]


def fast_gelu_ref(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * x * (1 + 0.044715 * x**2)))


def base_op():
    try:
        import megatron.core.activations
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch Fast GELU "
            "composition as the benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return fast_gelu_ref

    compiled = megatron.core.activations.fast_gelu
    eager = getattr(compiled, "__wrapped__", compiled)
    chosen = compiled

    def reference(*args, **kwargs):
        nonlocal chosen
        if chosen is eager:
            return eager(*args, **kwargs)
        try:
            return chosen(*args, **kwargs)
        except Exception:
            warnings.warn(
                "Megatron fast_gelu failed to compile on this platform; "
                "falling back to its eager implementation",
                RuntimeWarning,
                stacklevel=2,
            )
            chosen = eager
            return eager(*args, **kwargs)

    return reference


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class FastGeluBenchmark(base.GenericBenchmark):
    """Shapes are fixed here because ``benchmark/core_shapes.yaml`` does not exist yet.

    ``Benchmark.set_shapes`` opens that file, so an override is required for an
    operator that is not listed in one. This matches the other benchmarks that
    keep their shapes in the test module.
    """

    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def fast_gelu_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device),)


def _run():
    bench = FastGeluBenchmark(
        input_fn=fast_gelu_input_fn,
        op_name="fast_gelu",
        torch_op=base_op(),
        dtypes=_dtypes(),
    )
    bench.set_train(flag_train.megatron.fast_gelu)
    bench.run()


@pytest.mark.fast_gelu
def test_fast_gelu_perf():
    _run()
