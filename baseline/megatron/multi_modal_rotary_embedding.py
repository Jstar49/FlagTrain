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

"""NVIDIA baseline using Megatron-LM's non-interleaved mRoPE forward."""

from functools import lru_cache

import torch

OP_NAME = "multi_modal_rotary_embedding"
DTYPES = (torch.float32,)
IS_INPLACE = False

_CASES = (
    (2, 7, (1, 1, 2), None),
    (1, 128, (3, 5, 8), 2.5),
    (4, 257, (6, 10, 16), None),
    (1, 1, (1, 1, 1), None),
    (2, 512, (8, 12, 12), 1.5),
    (8, 1025, (11, 11, 10), None),
    (1, 4096, (16, 24, 24), 4.0),
    (1, 1, (8, 12, 12), None),
    (32, 1, (16, 24, 24), None),
    (16, 32, (8, 11, 14), 1.75),
    (2, 8192, (15, 24, 24), None),
    (1, 16384, (16, 24, 25), 3.5),
)


@lru_cache(maxsize=1)
def _upstream():
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        MultimodalRotaryEmbedding,
    )

    return MultimodalRotaryEmbedding


def _forward(rotary, position_ids, sections):
    return rotary(position_ids, sections)


def native():
    _upstream()
    return _forward


def grid():
    return [
        {
            "batch": batch,
            "length": length,
            "sections": sections,
            "interpolation": factor,
        }
        for batch, length, sections, factor in _CASES
    ]


def build_inputs(binding, dtype, device):
    batch, length = binding["batch"], binding["length"]
    sections = binding["sections"]
    width = sum(sections)
    tokens = torch.arange(batch * length, device=device, dtype=torch.int64).reshape(
        batch, length
    )
    positions = torch.stack((tokens + 1, tokens * 2 + 10, tokens * 3 + 100))
    rotary = _upstream()(
        kv_channels=2 * width,
        rotary_percent=1.0,
        rotary_interleaved=False,
        seq_len_interpolation_factor=binding["interpolation"],
    )
    rotary.inv_freq = torch.arange(1, width + 1, device=device, dtype=dtype)
    return (rotary, positions, list(sections)), {}


def key_shape(binding):
    # Several workloads share the position_ids shape. Match both tensors.
    return f'{binding["batch"]}x{binding["length"]}x{sum(binding["sections"])}'


def config(binding, dtype):
    return {
        "inputs": {
            "position_ids": {
                "shape": [3, binding["batch"], binding["length"]],
                "dtype": str(torch.int64),
            },
            "inv_freq": {"shape": [sum(binding["sections"])], "dtype": str(dtype)},
            "seq_len_interpolation_factor": {"scalar": binding["interpolation"]},
        },
        "mrope_section": list(binding["sections"]),
    }
