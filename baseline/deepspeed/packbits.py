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

"""NVIDIA baseline for packbits.

Self-contained module; depends only on torch. There is no usable upstream
native implementation: DeepSpeed's pack_bits was XPU-only and has been removed,
and torch has no packbits, so the baseline is ``ref_packbits`` (a torch
composition), copied from
``FlagTrain/benchmark/deepspeed/test_packbits.py``.
"""

import torch

OP_NAME = "packbits"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/deepspeed/test_packbits.py::_PACKBITS_SHAPES
_SHAPES = [
    (4096,),
    (16384,),
    (65536,),
    (262144,),
    (1048576,),
    (4194304,),
    (16777216,),
    (1000003,),
    (1024, 1024),
    (4096, 4096),
    (11008, 1024),
    (128256, 128),
    (512, 4097),
    (32000, 8),
]

_WEIGHTS = (128, 64, 32, 16, 8, 4, 2, 1)


def ref_packbits(x, dim=-1):
    """PyTorch reference for packbits."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    if x.numel() == 0:
        packed_len = (n + 7) // 8
        empty = x.new_empty(*x.shape[:-1], packed_len, dtype=torch.uint8)
        return empty.movedim(-1, dim)

    bits = (~torch.signbit(x)).to(torch.uint8)
    if n % 8:
        pad = [0] * (2 * x.ndim)
        pad[1] = 8 - n % 8
        bits = torch.nn.functional.pad(bits, pad)

    weights = torch.tensor(_WEIGHTS, dtype=torch.uint8, device=x.device)
    packed = (bits.reshape(*x.shape[:-1], -1, 8) * weights).sum(-1, dtype=torch.uint8)
    return packed.movedim(-1, dim)


def native():
    """Torch primitive composition used as the baseline."""
    return ref_packbits


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    return (torch.randn(binding["shape"], dtype=dtype, device=device),), {}


def key_shape(binding):
    # First (and only) tensor is the input itself; unique per shape.
    return list(binding["shape"])
