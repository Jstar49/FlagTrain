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

from .. import base, consts

_PACKBITS_SHAPES = [
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


# No executable oracle to call: DeepSpeed's pack_bits was XPU-only and deleted upstream
# (8ae0509ef), and torch has no packbits, so the baseline is assembled from primitives.
def ref_packbits(x, dim=-1):
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


class _PackbitsBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _PACKBITS_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_PACKBITS_SHAPES)


def packbits_input_fn(shape, dtype, device):
    yield (torch.randn(shape, dtype=dtype, device=device),)


@pytest.mark.packbits
def test_packbits_perf():
    bench = _PackbitsBenchmark(
        input_fn=packbits_input_fn,
        op_name="packbits",
        torch_op=ref_packbits,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_train(flag_train.deepspeed.packbits)
    bench.run()
