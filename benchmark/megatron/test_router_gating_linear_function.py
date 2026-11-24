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

"""Time the mixed-dtype router forward call with and without bias."""

import pytest

import flag_train
from benchmark.base import GenericBenchmark
from benchmark.cases import BenchmarkCasePlan
from tests.megatron.test_router_gating_linear_function import (
    BIAS_DTYPE,
    DTYPES,
    INPUT_DTYPE,
    OP_NAME,
    WEIGHT_DTYPE,
    WORKLOADS,
    make_inputs,
    reference,
)


class RouterGatingLinearBenchmark(GenericBenchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = list(WORKLOADS)

    def set_more_shapes(self):
        return []


def _cases(workload, dtype):
    shape, num_experts = workload
    for with_bias in (False, True):
        yield BenchmarkCasePlan(
            shape={
                "inp": list(shape),
                "weight": [num_experts, shape[-1]],
                "bias": [num_experts] if with_bias else None,
            },
            params={
                "input_dtype": str(INPUT_DTYPE),
                "weight_dtype": str(WEIGHT_DTYPE),
                "bias_dtype": str(BIAS_DTYPE) if with_bias else None,
                "router_dtype": str(dtype),
                "with_bias": with_bias,
            },
            builder_args=(shape, with_bias, num_experts),
        )


def _build_inputs(plan, dtype, device):
    shape, with_bias, num_experts = plan.builder_args
    return make_inputs(shape, dtype, device, with_bias, num_experts)


@pytest.mark.router_gating_linear_function
def test_router_gating_linear_function_perf():
    bench = RouterGatingLinearBenchmark(
        op_name=OP_NAME,
        torch_op=reference,
        dtypes=list(DTYPES),
        case_fn=_cases,
        build_inputs_fn=_build_inputs,
    )
    bench.set_train(flag_train.megatron.router_gating_linear_function)
    bench.run()
