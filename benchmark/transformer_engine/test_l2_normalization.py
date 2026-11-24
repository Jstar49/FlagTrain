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

import pytest
import torch

import flag_train
from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from tests.transformer_engine.test_l2_normalization import OP_NAME, reference
from tests.transformer_engine.test_l2_normalization_backward import DTYPES, EPS, SHAPES


class L2NormalizationBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = list(SHAPES)

    def set_more_shapes(self):
        return []


def _cases(shape, dtype):
    yield BenchmarkCasePlan(
        shape={"x": list(shape)}, params={"eps": EPS}, builder_args=(shape,)
    )


def _build_inputs(plan, dtype, device):
    return torch.randn(plan.builder_args[0], device=device, dtype=dtype), EPS


@pytest.mark.l2_normalization
def test_l2_normalization_perf():
    bench = L2NormalizationBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=list(DTYPES),
        case_fn=_cases,
        build_inputs_fn=_build_inputs,
    )
    bench.set_train(flag_train.transformer_engine.l2_normalization)
    bench.run()
