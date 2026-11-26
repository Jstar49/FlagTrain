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
from functools import lru_cache

import numpy as np
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

_INDEX_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]

# Upstream stores probabilities without arithmetic, so these dtypes pass
# through unchanged.
_PROBS_NONFLOAT_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.bool,
]

# Largest expert id each index dtype can hold; the masked case writes
# ``num_experts`` as an out-of-range sentinel, so wider expert counts cannot be
# exercised on narrow index tensors.
_INDEX_MAX = {
    torch.int8: torch.iinfo(torch.int8).max,
    torch.int16: torch.iinfo(torch.int16).max,
    torch.int32: torch.iinfo(torch.int32).max,
    torch.int64: torch.iinfo(torch.int64).max,
}


@lru_cache(maxsize=None)
def _device_compares(dtype):
    """Whether the device's ``torch.isclose`` accepts ``dtype``.

    torch_npu 2.8 rejects uint16/32/64; the probe is built from int64 because
    ``torch.zeros`` is missing for some of them too.
    """
    try:
        probe = torch.ones(1, dtype=torch.int64, device=flag_train.device).to(dtype)
        torch.isclose(probe, probe)
    except Exception:
        return False
    return True


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
    if shape[2] > _INDEX_MAX[index_dtype]:
        pytest.skip(f"num_experts={shape[2]} does not fit {index_dtype} indices")

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
def test_fused_indices_to_multihot_int_like_num_experts(probs_dtype):
    # Upstream accepts numpy integers for the width; the candidate must too.
    indices, probs = _make_inputs((17, 4, 64), torch.int64, probs_dtype)
    width = np.int64(64)

    ref_map, ref_probs = fused_indices_converter.fused_indices_to_multihot(
        indices, probs, width
    )
    res_map, res_probs = flag_train.megatron.fused_indices_to_multihot(
        indices, probs, width
    )

    if utils.TO_CPU:
        ref_map = utils.to_reference(ref_map)
        ref_probs = utils.to_reference(ref_probs)

    utils.train_assert_equal(res_map, ref_map)
    utils.train_assert_equal(res_probs, ref_probs)

    # Upstream's width is a Triton constexpr, so a tensor is rejected there
    # too (with a compile error); the candidate reports it as a ValueError.
    with pytest.raises(ValueError):
        flag_train.megatron.fused_indices_to_multihot(
            indices, probs, torch.tensor(64, device=flag_train.device)
        )


@pytest.mark.fused_indices_to_multihot
@pytest.mark.parametrize("probs_dtype", _PROBS_NONFLOAT_DTYPES)
def test_fused_indices_to_multihot_nonfloat_probs(probs_dtype):
    if not _device_compares(probs_dtype):
        pytest.skip(f"{probs_dtype} is not supported by the device's exact comparison")
    # Upstream copies the probabilities without arithmetic, so integer and
    # bool probs are valid; the candidate must accept them too.
    indices, _ = _make_inputs((17, 4, 64), torch.int64, torch.float32, masked=True)
    device = flag_train.device
    if probs_dtype is torch.bool:
        probs = torch.rand(17, 4, device=device) < 0.5
    else:
        probs = torch.randint(0, 100, (17, 4), dtype=torch.int64, device=device).to(
            probs_dtype
        )

    ref_map, ref_probs = fused_indices_converter.fused_indices_to_multihot(
        indices, probs, 64
    )
    res_map, res_probs = flag_train.megatron.fused_indices_to_multihot(
        indices, probs, 64
    )

    if utils.TO_CPU:
        ref_map = utils.to_reference(ref_map)
        ref_probs = utils.to_reference(ref_probs)

    utils.train_assert_equal(res_map, ref_map)
    utils.train_assert_equal(res_probs, ref_probs)


@pytest.mark.fused_indices_to_multihot
@pytest.mark.parametrize("probs_dtype", utils.FLOAT_DTYPES)
def test_fused_indices_to_multihot_bool_indices(probs_dtype):
    # Upstream accepts bool indices (False -> expert 0, True -> expert 1).
    # Keep the lanes distinct so the check does not depend on the undefined
    # scatter order for duplicates.
    num_tokens, num_experts = 32, 2
    device = flag_train.device
    indices = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device=device)
    indices[:, 1] = True
    probs = torch.rand(num_tokens, num_experts, dtype=probs_dtype, device=device)

    ref_map, ref_probs = fused_indices_converter.fused_indices_to_multihot(
        indices, probs, num_experts
    )
    res_map, res_probs = flag_train.megatron.fused_indices_to_multihot(
        indices, probs, num_experts
    )

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
