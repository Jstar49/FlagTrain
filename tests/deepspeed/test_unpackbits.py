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

from .. import accuracy_utils as utils

_SHIFTS = (7, 6, 5, 4, 3, 2, 1, 0)

BYTE_LENGTHS = [
    1,
    7,
    8,
    9,
    63,
    64,
    65,
    1000,
    4096,
    16387,
    1000003,
    1048577,
]

BYTE_SHAPES = [
    (5, 16),
    (5, 13),
    (4, 7, 26),
    (5, 5000),
    (1024, 1024),
    (65536, 8),
    (70000, 8),
    (70000, 32),
    (2, 40000, 2),
]

BYTE_DIMS = [-1, 0, -2]

BYTE_SHAPE_COUNTS = [
    ((5, 200), 0),
    ((5, 200), 1),
    ((5, 200), 7),
    ((5, 200), 200),
    ((5, 200), 1599),
    ((5, 200), 1600),
    ((4, 5000), 5000),
    ((4, 5000), 39999),
    ((4, 5000), 40000),
    ((70000, 8), 1),
    ((70000, 8), 63),
    ((70000, 8), 64),
    ((70000, 32), 100),
    ((4097, 100), 800),
    ((16384, 4), 32),
    ((16383, 4), 32),
    ((3, 8), 64),
    ((300, 300), 2000),
    ((1, 100000), 800000),
]

# One byte per case, MSB first: 0xAA -> +1 -1 +1 -1 ..., 0x01 -> only the last is
# +1, and the two ends of the range pin the padding-free extremes.
_BYTE_VALUES = (0xAA, 0x01, 0x80, 0x00, 0xFF)

_BYTE_EXPECTED = (
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,  # 0xAA = 0b10101010
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,  # 0x01 = 0b00000001
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,  # 0x80 = 0b10000000
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,  # 0x00 = 0b00000000
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,  # 0xFF = 0b11111111
)


# DeepSpeed's unpackbits was XPU-only and deleted upstream (8ae0509ef), and torch
# has no unpackbits, so the baseline is assembled from primitives.
def _ref_unpackbits(packed, dim=-1, count=None, dtype=torch.float32):
    """PyTorch reference for unpackbits."""
    x = packed.movedim(dim, -1)
    n_bytes = x.shape[-1]
    n_out = n_bytes * 8 if count is None else count

    shifts = torch.tensor(_SHIFTS, dtype=torch.uint8, device=x.device)
    bits = (x.unsqueeze(-1).expand(*x.shape, len(_SHIFTS)) >> shifts) & 1
    values = bits.reshape(*x.shape[:-1], n_bytes * 8).to(dtype) * 2 - 1
    return values[..., :n_out].movedim(-1, dim)


@pytest.mark.unpackbits
@pytest.mark.parametrize("n", BYTE_LENGTHS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unpackbits_1d(n, dtype):
    packed = torch.randint(0, 256, (n,), dtype=torch.uint8, device=flag_train.device)

    ref_out = _ref_unpackbits(utils.to_reference(packed), dtype=dtype)
    res_out = flag_train.deepspeed.unpackbits(packed, dtype=dtype)

    utils.train_assert_equal(res_out, ref_out)


@pytest.mark.unpackbits
@pytest.mark.parametrize("shape", BYTE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unpackbits_nd(shape, dtype):
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device=flag_train.device)

    ref_out = _ref_unpackbits(utils.to_reference(packed), dtype=dtype)
    res_out = flag_train.deepspeed.unpackbits(packed, dtype=dtype)

    utils.train_assert_equal(res_out, ref_out)


@pytest.mark.unpackbits
def test_unpackbits_bit_order():
    packed = torch.tensor(_BYTE_VALUES, dtype=torch.uint8, device=flag_train.device)

    res_out = flag_train.deepspeed.unpackbits(packed)

    expected = torch.tensor(_BYTE_EXPECTED, dtype=torch.float32)
    assert torch.equal(res_out.cpu(), expected), f"{res_out.cpu()} != {expected}"


@pytest.mark.unpackbits
@pytest.mark.parametrize("dim", BYTE_DIMS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unpackbits_dim(dim, dtype):
    packed = torch.randint(
        0, 256, (4, 7, 26), dtype=torch.uint8, device=flag_train.device
    )

    ref_out = _ref_unpackbits(utils.to_reference(packed), dim, dtype=dtype)
    res_out = flag_train.deepspeed.unpackbits(packed, dim=dim, dtype=dtype)

    utils.train_assert_equal(res_out, ref_out)


@pytest.mark.unpackbits
@pytest.mark.parametrize("shape, count", BYTE_SHAPE_COUNTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unpackbits_count(shape, count, dtype):
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device=flag_train.device)

    ref_out = _ref_unpackbits(utils.to_reference(packed), count=count, dtype=dtype)
    res_out = flag_train.deepspeed.unpackbits(packed, count=count, dtype=dtype)

    assert res_out.shape == ref_out.shape, f"{res_out.shape} != {ref_out.shape}"
    utils.train_assert_equal(res_out, ref_out)


@pytest.mark.unpackbits
@pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 4)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unpackbits_empty(shape, dtype):
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device=flag_train.device)

    ref_out = _ref_unpackbits(utils.to_reference(packed), dtype=dtype)
    res_out = flag_train.deepspeed.unpackbits(packed, dtype=dtype)

    assert res_out.shape == (*shape[:-1], shape[-1] * 8), f"{res_out.shape}"
    utils.train_assert_equal(res_out, ref_out)
