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

_WEIGHTS = (128, 64, 32, 16, 8, 4, 2, 1)

PACK_LENGTHS = [
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

PACK_SHAPES = [
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

PACK_DIMS = [-1, 0, -2]

_SPECIAL_VALUES = (
    -0.0,
    0.0,
    float("inf"),
    float("-inf"),
    float("nan"),
)


# No executable oracle to call: DeepSpeed's pack_bits was XPU-only and deleted upstream
# (8ae0509ef), and torch has no packbits, so the baseline is assembled from primitives.
def _ref_packbits(x, dim=-1):
    """PyTorch reference for packbits."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    if x.numel() == 0:
        packed_len = (n + 7) // 8
        empty = x.new_empty(*x.shape[:-1], packed_len, dtype=torch.uint8)
        return empty.movedim(-1, dim)

    bits = (~torch.signbit(x)).to(torch.uint8)
    # On NPU ``torch.signbit`` calls -0.0 non-negative (CANN 9.0.0), where the CPU says
    # otherwise; its bit is back in reach as 1/-0.0 = -inf against 1/+0.0 = +inf.
    if flag_train.vendor_name == "ascend":
        negative_zero = (x == 0) & (torch.reciprocal(x) < 0)
        bits = torch.where(negative_zero, 0, bits)
    if n % 8:
        pad = [0] * (2 * x.ndim)
        pad[1] = 8 - n % 8
        bits = torch.nn.functional.pad(bits, pad)

    weights = torch.tensor(_WEIGHTS, dtype=torch.uint8, device=x.device)
    packed = (bits.reshape(*x.shape[:-1], -1, 8) * weights).sum(-1, dtype=torch.uint8)
    return packed.movedim(-1, dim)


@pytest.mark.packbits
@pytest.mark.parametrize("n", PACK_LENGTHS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_packbits_1d(n, dtype):
    x = torch.randn(n, dtype=dtype, device=flag_train.device)

    ref_out = _ref_packbits(utils.to_reference(x))
    res_out = flag_train.deepspeed.packbits(x)

    utils.train_assert_equal(res_out, ref_out)


@pytest.mark.packbits
@pytest.mark.parametrize("shape", PACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_packbits_nd(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)

    ref_out = _ref_packbits(utils.to_reference(x))
    res_out = flag_train.deepspeed.packbits(x)

    utils.train_assert_equal(res_out, ref_out)


@pytest.mark.packbits
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_packbits_special_values(dtype):
    x = torch.tensor(_SPECIAL_VALUES, dtype=dtype, device=flag_train.device)

    ref_out = _ref_packbits(utils.to_reference(x))
    res_out = flag_train.deepspeed.packbits(x)

    utils.train_assert_equal(res_out, ref_out)


@pytest.mark.packbits
@pytest.mark.parametrize("dim", PACK_DIMS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_packbits_dim(dim, dtype):
    x = torch.randn(4, 7, 26, dtype=dtype, device=flag_train.device)

    ref_out = _ref_packbits(utils.to_reference(x), dim)
    res_out = flag_train.deepspeed.packbits(x, dim=dim)

    utils.train_assert_equal(res_out, ref_out)


@pytest.mark.packbits
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape, packed_shape",
    [((0,), (0,)), ((3, 0), (3, 0)), ((0, 4), (0, 1))],
)
def test_packbits_empty(shape, dtype, packed_shape):
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    res = flag_train.deepspeed.packbits(x)

    assert res.shape == packed_shape, f"{res.shape} != {packed_shape}"
    assert res.dtype == torch.uint8
    utils.train_assert_equal(res, _ref_packbits(utils.to_reference(x)))
