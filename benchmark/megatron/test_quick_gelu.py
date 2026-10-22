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
"""Performance benchmark for Megatron Quick GELU.

The preferred baseline is Megatron-LM's official
``megatron.core.activations.quick_gelu`` interface. If Megatron-LM is not
installed, the benchmark logs a warning and falls back to the equivalent
PyTorch composition. Timing, warmup and synchronization come from
``benchmark/base.py`` (``triton.testing.do_bench`` in kernel mode).
``--level core`` runs the small, medium and large shapes below;
``--level comprehensive`` adds the extra shapes.
"""

import os
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


def quick_gelu_ref(x):
    return x * torch.sigmoid(1.702 * x)


def base_op():
    try:
        import megatron.core.activations

        return megatron.core.activations.quick_gelu
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch Quick GELU "
            "composition as the benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return quick_gelu_ref


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class QuickGeluBenchmark(base.GenericBenchmark):
    """Shapes are fixed here because ``benchmark/core_shapes.yaml`` does not exist yet.

    ``Benchmark.set_shapes`` opens that file, so an override is required for an
    operator that is not listed in one. This matches ``LambBenchmark``.
    """

    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def quick_gelu_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device),)


def _run(is_backward):
    bench = QuickGeluBenchmark(
        input_fn=quick_gelu_input_fn,
        op_name="quick_gelu_backward" if is_backward else "quick_gelu",
        torch_op=base_op(),
        dtypes=_dtypes(),
        is_backward=is_backward,
    )
    bench.set_train(flag_train.megatron.quick_gelu)
    # On Ascend, enable parallel scheduling of all blocks to avoid excessive logical cores
    # and reduce scheduling overhead when the logical core count exceeds 65,535.
    # See https://github.com/flagos-ai/FlagGems/pull/7304
    if flag_train.vendor_name == "ascend":
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    bench.run()


@pytest.mark.quick_gelu
def test_quick_gelu_perf():
    _run(is_backward=False)


@pytest.mark.quick_gelu
def test_quick_gelu_backward_perf():
    _run(is_backward=True)
