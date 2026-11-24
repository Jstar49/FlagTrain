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


"""Time the forward projection under Megatron's compiled Torch Sinkhorn contract."""

import pytest

import flag_train
from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from tests.megatron.test_fused_sinkhorn import (
    CASES,
    DTYPES,
    OP_NAME,
    make_inputs,
    reference,
)


class FusedSinkhornBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = list(CASES)

    def set_more_shapes(self):
        return []


def _cases(case, dtype):
    shape, num_iterations, eps = case
    yield BenchmarkCasePlan(
        shape={"input_logits": list(shape)},
        params={"num_iterations": num_iterations, "eps": eps},
        builder_args=(shape, num_iterations, eps),
    )


def _build_inputs(plan, dtype, device):
    shape, num_iterations, eps = plan.builder_args
    return make_inputs(shape, dtype, device, num_iterations, eps)


@pytest.mark.fused_sinkhorn
def test_fused_sinkhorn_perf():
    bench = FusedSinkhornBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=list(DTYPES),
        case_fn=_cases,
        build_inputs_fn=_build_inputs,
    )
    bench.set_train(flag_train.megatron.fused_sinkhorn)
    bench.run()
