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


"""Check default WeightedSwiGLU forward and both first-order gradients.

Megatron default WeightedSwiGLUFunction at d2b0e35cc4f05bc6b70734d558617050fbbb51ed.
"""

import pytest
import torch
from torch.nn import functional as F

import flag_train
from flag_train.reference_candidate import get_candidate
from flag_train.testing import assert_close, assert_equal

OP_NAME = "weighted_swiglu_funciton"
OVERRIDE_KEY = f"megatron.{OP_NAME}"
SHAPES = (
    (16, 32),
    (2, 5, 64),
    (1, 2),
    (33, 258),
    (128, 2048),
    (2, 128, 4096),
    (256, 8192),
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def make_inputs(shape, dtype, device):
    generator = torch.Generator().manual_seed(6466)
    y = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    weights = torch.rand((*shape[:-1], 1), generator=generator).to(device=device)
    return y, weights


class _WeightedSwiGLUReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, weights):
        ctx.save_for_backward(y, weights)
        gate, linear = y.chunk(2, dim=-1)
        return (F.silu(gate) * linear * weights).to(y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, weights = ctx.saved_tensors
        gate, linear = y.chunk(2, dim=-1)
        sigmoid_gate = gate.sigmoid()
        scaled_grad = grad_output * weights
        grad_gate = (
            scaled_grad * sigmoid_gate * (1 + gate * (1 - sigmoid_gate)) * linear
        )
        silu_gate = F.silu(gate)
        grad_y = torch.cat((grad_gate, scaled_grad * silu_gate), dim=-1).to(y.dtype)
        grad_weights = (silu_gate * linear * grad_output.to(weights.dtype)).sum(
            dim=-1, keepdim=True
        )
        return grad_y, grad_weights.to(weights.dtype)


def reference(y, weights):
    return _WeightedSwiGLUReference.apply(y, weights)


@pytest.mark.weighted_swiglu_funciton
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_weighted_swiglu_funciton(shape, dtype):
    candidate = get_candidate(OVERRIDE_KEY, reference=reference)

    inputs = make_inputs(shape, dtype, flag_train.device)
    candidate_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    reference_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    actual = candidate(*candidate_inputs)
    expected = reference(*reference_inputs)
    assert_close(actual, expected, expected.dtype)
    for actual_input, original in zip(candidate_inputs, inputs):
        assert_equal(actual_input, original)
        assert (
            actual.untyped_storage().data_ptr()
            != actual_input.untyped_storage().data_ptr()
        )

    grad_output = (
        torch.linspace(-0.8, 0.9, expected.numel()).reshape(expected.shape).to(expected)
    )
    actual_grads = torch.autograd.grad(actual, candidate_inputs, grad_output.clone())
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert_close(actual_grad, expected_grad, expected_grad.dtype)
    for actual_input, original in zip(candidate_inputs, inputs):
        assert_equal(actual_input, original)


def test_weighted_swiglu_rejects_incompatible_shapes():
    call = flag_train.megatron.weighted_swiglu_funciton
    weights = torch.empty((2, 1), device=flag_train.device)
    with pytest.raises(ValueError, match="even final dimension"):
        call(torch.empty((2, 3), device=flag_train.device), weights)
    with pytest.raises(ValueError, match="one value per token"):
        call(torch.empty((2, 4), device=flag_train.device), weights[:1])
