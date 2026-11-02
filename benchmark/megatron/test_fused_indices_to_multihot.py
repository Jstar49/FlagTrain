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
import os

import pytest
import torch
from megatron.core.config import set_experimental_flag
from megatron.core.fusions import fused_indices_converter

import flag_train

from .. import base, consts

set_experimental_flag(True)

_SHAPES = [
    (4096, 2, 8),
    (4096, 6, 8),
    (4096, 4, 15),
    (2048, 8, 16),
    (16384, 8, 32),
    (8192, 1, 128),
    (65536, 8, 32),
    (131072, 8, 16),
    (65536, 2, 128),
    (256, 8, 32),
]

if flag_train.vendor_name == "ascend":
    # Auto-maps the logical blocks onto the physical cores, which allows a
    # grid above 65535.
    os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
    # Megatron reference kernel does not compile on Ascend when top-k
    # or the expert count is 1.
    _SHAPES = [shape for shape in _SHAPES if shape[1] > 1]


class _FusedIndicesToMultihotBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)


def fused_indices_to_multihot_input_fn(shape, dtype, device):
    num_tokens, topk, num_experts = shape
    scores = torch.rand(num_tokens, num_experts, dtype=torch.float32, device=device)
    indices = scores.topk(topk, dim=1).indices.to(torch.int64)
    probs = torch.rand(num_tokens, topk, dtype=dtype, device=device)
    yield indices, probs, num_experts


@pytest.mark.fused_indices_to_multihot
def test_fused_indices_to_multihot():
    bench = _FusedIndicesToMultihotBenchmark(
        input_fn=fused_indices_to_multihot_input_fn,
        op_name="fused_indices_to_multihot",
        torch_op=fused_indices_converter.fused_indices_to_multihot,
        dtypes=consts.FLOAT_DTYPES,
        # The module imports megatron-core unconditionally, so the baseline is
        # always the upstream kernel and --base-data's cross-hardware figure
        # does not apply.
        native_baseline=True,
    )
    bench.set_train(flag_train.megatron.fused_indices_to_multihot)
    bench.run()
