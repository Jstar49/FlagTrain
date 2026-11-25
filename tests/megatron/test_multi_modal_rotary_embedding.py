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


"""Check explicit Megatron non-interleaved mRoPE inputs and layout.

Non-interleaved Megatron mRoPE workloads for FlagOS requirement 6478.
"""

from typing import NamedTuple

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from flag_train.testing import assert_close, assert_equal

OP_NAME = "multi_modal_rotary_embedding"
OVERRIDE_KEY = f"megatron.{OP_NAME}"


class _MRoPEWorkload(NamedTuple):
    batch: int
    length: int
    sections: tuple[int, int, int]
    interpolation: float | None = None
    strided_positions: bool = False
    strided_frequency: bool = False
    repeated_positions: bool = False


WORKLOADS = (
    _MRoPEWorkload(2, 7, (1, 1, 2)),
    _MRoPEWorkload(1, 128, (3, 5, 8), 2.5),
    _MRoPEWorkload(4, 257, (6, 10, 16)),
    _MRoPEWorkload(1, 1, (1, 1, 1)),
    _MRoPEWorkload(2, 512, (8, 12, 12), 1.5),
    _MRoPEWorkload(8, 1025, (11, 11, 10)),
    _MRoPEWorkload(1, 4096, (16, 24, 24), 4.0),
)

TIMING_WORKLOADS = WORKLOADS + (
    _MRoPEWorkload(1, 1, (8, 12, 12)),  # Single-token decode with D=32
    _MRoPEWorkload(32, 1, (16, 24, 24)),  # Batched decode
    _MRoPEWorkload(16, 32, (8, 11, 14), 1.75),
    _MRoPEWorkload(2, 8192, (15, 24, 24)),
    _MRoPEWorkload(1, 16384, (16, 24, 25), 3.5),
)

CORRECTNESS_WORKLOADS = TIMING_WORKLOADS + (
    _MRoPEWorkload(3, 9, (1, 1, 29), strided_positions=True),
    _MRoPEWorkload(2, 17, (30, 1, 1), strided_frequency=True),
    _MRoPEWorkload(
        3, 17, (1, 31, 1), 2.5, strided_positions=True, strided_frequency=True
    ),
    _MRoPEWorkload(2, 13, (1, 1, 61), repeated_positions=True),
    _MRoPEWorkload(3, 7, (62, 1, 1), 1.2, repeated_positions=True),
    _MRoPEWorkload(
        2, 19, (1, 63, 1), 3.7, strided_positions=True, repeated_positions=True
    ),
    _MRoPEWorkload(
        4, 11, (9, 11, 11), 1.5, strided_frequency=True, repeated_positions=True
    ),
    _MRoPEWorkload(
        5,
        23,
        (4, 15, 13),
        2.2,
        strided_positions=True,
        strided_frequency=True,
        repeated_positions=True,
    ),
    _MRoPEWorkload(2, 31, (11, 11, 11)),
    _MRoPEWorkload(2, 31, (20, 21, 23), 2.0),
    _MRoPEWorkload(2, 31, (20, 21, 24), strided_positions=True, strided_frequency=True),
    _MRoPEWorkload(2, 5, (128, 256, 129)),  # D=513 crosses one output tile
)


def make_inputs(workload, device):
    batch, length = workload.batch, workload.length
    tokens = torch.arange(batch * length, device=device, dtype=torch.int64)
    tokens = tokens.reshape(batch, length)
    if workload.repeated_positions:
        order = tokens // 2
        positions = (order * 13 + 3, order * 17 + 11, order * 23 + 19)
        position_ids = torch.stack(tuple(axis.remainder(37) for axis in positions))
    else:
        position_ids = torch.stack((tokens + 1, tokens * 2 + 10, tokens * 3 + 100))
    width = sum(workload.sections)
    inv_freq = torch.arange(1, width + 1, device=device, dtype=torch.float32)
    if workload.strided_positions:
        storage = torch.empty(
            (3, length, 2 * batch + 1), device=device, dtype=torch.int64
        )
        view = storage[..., 1::2].transpose(1, 2)
        view.copy_(position_ids)
        position_ids = view
    if workload.strided_frequency:
        storage = torch.empty(2 * width + 1, device=device, dtype=torch.float32)
        view = storage[1::2]
        view.copy_(inv_freq)
        inv_freq = view
    return position_ids, inv_freq, workload.sections, workload.interpolation


def reference(position_ids, inv_freq, mrope_section, seq_len_interpolation_factor=None):
    seq = position_ids.to(device=inv_freq.device, dtype=inv_freq.dtype)
    if seq_len_interpolation_factor is not None:
        seq *= 1 / seq_len_interpolation_factor
    freqs = seq[..., None].float() * inv_freq
    doubled = torch.cat((freqs, freqs), dim=-1)
    chunks = doubled.split(mrope_section * 2, dim=-1)
    selected = [chunk[index % 3] for index, chunk in enumerate(chunks)]
    return torch.cat(selected, dim=-1).unsqueeze(-2).transpose(0, 1).contiguous()


@pytest.mark.multi_modal_rotary_embedding
@pytest.mark.parametrize("workload", CORRECTNESS_WORKLOADS)
def test_multi_modal_rotary_embedding(workload):
    candidate = get_candidate(OVERRIDE_KEY, reference=reference)

    position_ids, inv_freq, sections, interpolation = make_inputs(
        workload, flag_train.device
    )
    original_ids, original_freqs = position_ids.clone(), inv_freq.clone()
    expected = reference(original_ids, original_freqs, sections, interpolation)
    actual = candidate(position_ids, inv_freq, sections, interpolation)

    assert actual.shape == (workload.length, workload.batch, 1, 2 * inv_freq.numel())
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    assert_close(actual, expected, expected.dtype)
    assert_equal(position_ids, original_ids)
    assert_equal(inv_freq, original_freqs)
    actual_storage = actual.untyped_storage().data_ptr()
    assert actual_storage != position_ids.untyped_storage().data_ptr()
    assert actual_storage != inv_freq.untyped_storage().data_ptr()

    if workload == WORKLOADS[0]:
        first_position = actual.new_tensor([1, 20, 300, 400, 1, 20, 300, 400])
        assert_equal(actual[0, 0, 0], first_position)
