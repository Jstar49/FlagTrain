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
from megatron.core.config import set_experimental_flag
from megatron.core.fusions import fused_indices_converter

import flag_train

from .. import accuracy_utils as utils

set_experimental_flag(True)


_SHAPES = [
    (1, 1, 1),
    (1, 8, 128),
    (5, 3, 7),
    (17, 4, 64),
    (33, 16, 257),
    (1024, 2, 32),
    (1025, 6, 100),
    (1024, 8, 128),
    (4096, 6, 256),
    (8192, 1, 1),
    (16384, 8, 64),
]

_INDEX_DTYPES = [torch.int32, torch.int64]


def _make_inputs(shape, index_dtype, probs_dtype, masked=False):
    num_tokens, topk, num_experts = shape
    device = flag_train.device
    scores = torch.rand(num_tokens, num_experts, dtype=torch.float32, device=device)
    indices = scores.topk(topk, dim=1).indices.to(index_dtype)

    if masked:
        if topk > 1:
            indices[::3, 0] = -1
        indices[10::11] = -1
        if num_experts > 0:
            indices[12::13, -1] = num_experts

    probs = torch.rand(num_tokens, topk, dtype=probs_dtype, device=device)
    return indices, probs


@pytest.mark.fused_indices_to_multihot
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("probs_dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("masked", [False, True])
def test_fused_indices_to_multihot(shape, index_dtype, probs_dtype, masked):
    if flag_train.vendor_name == "ascend" and (shape[1] == 1 or shape[2] == 1):
        pytest.skip(
            "Megatron reference kernel does not compile on Ascend when top-k "
            "or the expert count is 1"
        )

    indices, probs = _make_inputs(shape, index_dtype, probs_dtype, masked)

    ref_map, ref_probs = fused_indices_converter.fused_indices_to_multihot(
        indices, probs, shape[2]
    )
    res_map, res_probs = flag_train.megatron.fused_indices_to_multihot(
        indices, probs, shape[2]
    )

    # Megatron needs device inputs and returns CUDA tensors; only the results
    # can follow ``--ref cpu``.
    if utils.TO_CPU:
        ref_map = utils.to_reference(ref_map)
        ref_probs = utils.to_reference(ref_probs)

    utils.train_assert_equal(res_map, ref_map)
    utils.train_assert_equal(res_probs, ref_probs)


@pytest.mark.fused_indices_to_multihot
@pytest.mark.parametrize("probs_dtype", utils.FLOAT_DTYPES)
def test_fused_indices_to_multihot_non_contiguous(probs_dtype):
    width = 8
    indices, probs = _make_inputs((64, width, 32), torch.int64, probs_dtype)
    indices = indices[:, ::2]
    probs = probs[:, ::2]

    assert not indices.is_contiguous()
    assert not probs.is_contiguous()

    ref_map, ref_probs = fused_indices_converter.fused_indices_to_multihot(
        indices.contiguous(), probs.contiguous(), 32
    )
    res_map, res_probs = flag_train.megatron.fused_indices_to_multihot(
        indices, probs, 32
    )

    if utils.TO_CPU:
        ref_map = utils.to_reference(ref_map)
        ref_probs = utils.to_reference(ref_probs)

    utils.train_assert_equal(res_map, ref_map)
    utils.train_assert_equal(res_probs, ref_probs)
