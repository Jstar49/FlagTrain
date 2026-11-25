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


"""Check the explicit TransformerEngine L2 backward interface."""

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from flag_train.testing import assert_close

OP_NAME = "l2_normalization_backward"
SHAPES = (
    (8, 64),
    (32, 128),
    (2, 16, 80),
    (1, 1),
    (31, 127),
    (128, 768),
    (512, 1024),
    (8, 128, 1536),
    (2048, 4096),
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
EPS = 1e-6


def make_inputs(shape, dtype, device):
    rng = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(shape, generator=rng, dtype=torch.float32).to(device, dtype)
    grad = torch.randn(shape, generator=rng, dtype=torch.float32).to(device, dtype)
    inverse_norm = (x.float().square().sum(dim=-1, keepdim=True) + EPS).rsqrt()
    return grad, x, inverse_norm, EPS


def reference(grad_output, x, rsqrt_norm, eps):
    x32, grad32 = x.float(), grad_output.float()
    projection = (x32 * grad32).sum(dim=-1, keepdim=True)
    norm_squared = x32.square().sum(dim=-1, keepdim=True) + eps
    return (rsqrt_norm * (grad32 - x32 * projection / norm_squared)).to(x.dtype)


@pytest.mark.l2_normalization_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_l2_normalization_backward(shape, dtype):
    candidate = get_candidate(f"transformer_engine.{OP_NAME}", reference=reference)
    inputs = make_inputs(shape, dtype, flag_train.device)
    expected = reference(*inputs)
    actual = candidate(*inputs)
    assert actual.device == inputs[1].device
    assert_close(actual, expected, expected.dtype)
