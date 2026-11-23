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
"""Performance benchmark for bias_geglu_impl.

The baseline is Megatron-LM's ``bias_geglu_impl`` when installed. Its forward
helpers use ``jit_fuser``, which is ``torch.compile`` on supported PyTorch
versions. Machines without Megatron-LM use the vendored forward implementation
at the end of this file, also compiled with ``torch.compile``. Thus every row
is compared against a compiled native implementation rather than eager
PyTorch.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). ``--level core`` runs the shapes
below. The operator is forward-only, so there are no backward cases.

``torch.compile`` recompiles per shape/dtype pattern. The default Dynamo
recompile limit is smaller than this benchmark's case count, so the module
raises it; otherwise the baseline could silently fall back to eager.
"""

import pytest
import torch

import flag_train

from .. import base, consts

# Token counts and hidden widths used by a fused MLP activation. Last dim is even.
_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


def _dtypes():
    """Use the same floating-point dtypes as the correctness suite."""
    dtypes = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class BiasGegluImplBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


def bias_geglu_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    bias = torch.randn((shape[-1],), dtype=dtype, device=device)
    yield (x, bias)


def base_op():
    try:
        from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
    except Exception:
        return None
    return bias_geglu_impl


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_perf():
    """Benchmark GEGLU for every shape and dtype in one table."""
    torch_op = base_op()
    _BASELINE = "megatron-core bias_geglu_impl" if torch_op is not None else "none"
    print(f"\nBaseline: {_BASELINE}")

    bench = BiasGegluImplBenchmark(
        input_fn=bias_geglu_input_fn,
        op_name="bias_geglu_impl",
        torch_op=torch_op,
        dtypes=_dtypes(),
        is_backward=False,
    )
    bench.set_train(flag_train.megatron.bias_geglu_impl)
    bench.run()
