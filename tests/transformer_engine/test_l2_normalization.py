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
from flag_train.reference_candidate import get_candidate
from flag_train.testing import assert_close

from .test_l2_normalization_backward import DTYPES, EPS, SHAPES

OP_NAME = "l2_normalization"


def reference(x, eps=EPS):
    x32 = x.float()
    inverse_norm = (x32.square().sum(dim=-1, keepdim=True) + eps).rsqrt()
    return (x32 * inverse_norm).to(x.dtype)


@pytest.mark.l2_normalization
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape, eps, transpose",
    [(shape, EPS, False) for shape in SHAPES]
    + [((3, 0), EPS, False), ((31, 7), 1e-3, True), ((3, 2051), EPS, False)],
)
def test_l2_normalization(shape, eps, transpose, dtype):
    candidate = get_candidate(f"transformer_engine.{OP_NAME}", reference=reference)
    generator = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        flag_train.device, dtype
    )
    if transpose:
        x = x.transpose(0, 1)
    grad = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x)

    x_ref = x.detach().requires_grad_()
    expected = reference(x_ref, eps)
    (expected_grad,) = torch.autograd.grad(expected, x_ref, grad)
    with torch.no_grad():
        assert_close(candidate(x, eps), expected, expected.dtype)

    x.requires_grad_()
    actual = candidate(x, eps)
    assert_close(actual, expected, expected.dtype)
    (actual_grad,) = torch.autograd.grad(actual, x, grad)
    assert_close(actual_grad, expected_grad, expected_grad.dtype)
