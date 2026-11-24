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


"""Time the default weighted SwiGLU forward call."""

import pytest

import flag_train
from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from tests.megatron.test_weighted_swiglu_funciton import (
    DTYPES,
    OP_NAME,
    SHAPES,
    make_inputs,
    reference,
)


class WeightedSwiGLUBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = list(SHAPES)

    def set_more_shapes(self):
        return []


def _cases(shape, dtype):
    yield BenchmarkCasePlan(
        shape={"y": list(shape), "weights": [*shape[:-1], 1]},
        params={"weights_dtype": "torch.float32"},
        builder_args=(shape,),
    )


def _build_inputs(plan, dtype, device):
    return make_inputs(plan.builder_args[0], dtype, device)


@pytest.mark.weighted_swiglu_funciton
def test_weighted_swiglu_funciton_perf():
    bench = WeightedSwiGLUBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=list(DTYPES),
        case_fn=_cases,
        build_inputs_fn=_build_inputs,
    )
    bench.set_train(flag_train.megatron.weighted_swiglu_funciton)
    bench.run()
