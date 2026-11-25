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

"""Check Megatron's Torch router branch, including the mixed-dtype bias epilogue.

Megatron router Torch fallback at d2b0e35cc4f05bc6b70734d558617050fbbb51ed.
"""

import math

import pytest
import torch

import flag_train
from flag_train.runtime import torch_backend_device
from flag_train.testing import assert_close, assert_equal

OP_NAME = "router_gating_linear_function"
WORKLOADS = (
    ((16, 32), 13),
    ((2, 5, 32), 13),
    ((1, 64), 1),
    ((33, 127), 17),
    ((128, 768), 32),
    ((4, 128, 2048), 64),
    ((1024, 4096), 128),
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
INPUT_DTYPE = torch.bfloat16
WEIGHT_DTYPE = torch.float16
BIAS_DTYPE = torch.float32


def make_inputs(shape, router_dtype, device, with_bias, num_experts):
    generator = torch.Generator().manual_seed(421)
    inp = torch.randn(shape, generator=generator)
    weight = torch.randn((num_experts, shape[-1]), generator=generator)
    bias = None
    if with_bias:
        bias = torch.randn(num_experts, generator=generator)
        # This first logit loses the 2**-14 residual when GEMM rounds before adding bias.
        first_row = inp.reshape(-1, shape[-1])[0]
        first_row.zero_()
        first_row[0] = 1 + 2**-7
        weight[0].zero_()
        weight[0, 0] = 1 + 2**-7
        bias[0] = -(1 + 2**-6)
        bias = bias.to(device=device, dtype=BIAS_DTYPE)
    return (
        inp.to(device=device, dtype=INPUT_DTYPE),
        weight.to(device=device, dtype=WEIGHT_DTYPE),
        bias,
        router_dtype,
    )


class _RouterGatingLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, router_dtype):
        ctx.save_for_backward(inp, weight, bias)
        ctx.router_dtype = router_dtype
        rows = inp.flatten(end_dim=-2).to(router_dtype)
        columns = weight.t().to(router_dtype)
        if bias is None:
            logits = torch.mm(rows, columns)
        else:
            logits = torch.addmm(bias.to(router_dtype), rows, columns)
        return logits.reshape(*inp.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        router_dtype = ctx.router_dtype
        grad_rows = grad_output.flatten(end_dim=-2)
        input_rows = inp.flatten(end_dim=-2).to(router_dtype)
        grad_input = torch.mm(grad_rows, weight.to(router_dtype)).to(inp.dtype)
        grad_weight = torch.mm(grad_rows.t(), input_rows).to(weight.dtype)
        # Megatron rounds the bias gradient to weight.dtype before autograd
        # converts it to the original bias tensor's dtype.
        grad_bias = grad_rows.sum(dim=0).to(weight.dtype) if bias is not None else None
        return grad_input.reshape_as(inp), grad_weight, grad_bias, None


def reference(inp, weight, bias, router_dtype):
    return _RouterGatingLinear.apply(inp, weight, bias, router_dtype)


@pytest.fixture(autouse=True)
def _matmul_precision():
    # Some backends do not expose reduced-precision reduction controls.
    matmul = getattr(torch_backend_device, "matmul", None)
    fp32_precision = torch.get_float32_matmul_precision()
    saved = {}
    try:
        torch.set_float32_matmul_precision("highest")
        for setting in (
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        ):
            if not hasattr(matmul, setting):
                continue
            previous = getattr(matmul, setting)
            split_k = getattr(matmul, setting + "_split_k", None)
            saved[setting] = previous if split_k is None else (previous, split_k)
            setattr(matmul, setting, False)
        yield
    finally:
        for setting, previous in reversed(saved.items()):
            setattr(matmul, setting, previous)
        torch.set_float32_matmul_precision(fp32_precision)


def _assert_fp32_reduction_close(actual, expected, reduction_size):
    # These dot products have an output scale proportional to sqrt(reduction_size).
    # Account for FP32 accumulation order when cancellation brings a result near zero.
    assert_close(
        actual,
        expected,
        expected.dtype,
        atol=1e-5,
        reduce_dim=math.sqrt(reduction_size),
    )


@pytest.mark.router_gating_linear_function
@pytest.mark.parametrize(
    "shape,num_experts",
    WORKLOADS,
    ids=[f"shape{index}" for index in range(len(WORKLOADS))],
)
@pytest.mark.parametrize("router_dtype", DTYPES)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_router_gating_linear_function(shape, num_experts, router_dtype, with_bias):

    inp, weight, bias, _ = make_inputs(
        shape, router_dtype, flag_train.device, with_bias, num_experts
    )
    inputs = (inp, weight, bias) if with_bias else (inp, weight)
    candidate_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    reference_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    candidate_bias = candidate_inputs[2] if with_bias else None
    reference_bias = reference_inputs[2] if with_bias else None
    actual = flag_train.megatron.router_gating_linear_function(
        *candidate_inputs[:2], candidate_bias, router_dtype
    )
    expected = reference(*reference_inputs[:2], reference_bias, router_dtype)
    if router_dtype == torch.float32:
        _assert_fp32_reduction_close(actual, expected, inp.size(-1))
    else:
        assert_close(actual, expected, expected.dtype)
    if with_bias:
        assert_equal(actual.flatten()[0], expected.flatten()[0])
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
    if router_dtype == torch.float32:
        _assert_fp32_reduction_close(actual_grads[0], expected_grads[0], num_experts)
        _assert_fp32_reduction_close(
            actual_grads[1], expected_grads[1], grad_output.numel() // num_experts
        )
        if with_bias:
            assert_equal(actual_grads[-1], expected_grads[-1])
    else:
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            assert_close(actual_grad, expected_grad, expected_grad.dtype)
    for actual_input, original in zip(candidate_inputs, inputs):
        assert_equal(actual_input, original)


@pytest.mark.router_gating_linear_function
def test_router_gating_linear_noncontiguous():
    def strided(tensor):
        return tensor.repeat_interleave(2, dim=-1)[..., ::2]

    inp, weight, bias, dtype = make_inputs(
        (2, 5, 32), torch.float32, flag_train.device, True, 13
    )
    originals = (inp, weight, bias)
    candidate_inputs = tuple(strided(tensor).requires_grad_() for tensor in originals)
    reference_inputs = tuple(strided(tensor).requires_grad_() for tensor in originals)
    actual = flag_train.megatron.router_gating_linear_function(*candidate_inputs, dtype)
    expected = reference(*reference_inputs, dtype)
    assert_close(actual, expected, expected.dtype)
    grad = torch.randn_like(expected)
    candidate_grad = strided(grad)
    actual_grads = torch.autograd.grad(actual, candidate_inputs, candidate_grad)
    expected_grads = torch.autograd.grad(expected, reference_inputs, strided(grad))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert_close(actual_grad, expected_grad, expected_grad.dtype)
    for actual_value, original_value in zip(
        (*candidate_inputs, candidate_grad), (*originals, grad)
    ):
        assert_equal(actual_value, original_value)


@pytest.mark.router_gating_linear_function
def test_router_gating_linear_empty_reduction_with_bias():
    device = flag_train.device
    inp = torch.empty((2, 0), device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.empty((3, 0), device=device, dtype=torch.float16, requires_grad=True)
    bias = torch.tensor([0.1, 0.2, 0.4], device=device, requires_grad=True)
    actual = flag_train.megatron.router_gating_linear_function(
        inp, weight, bias, torch.float32
    )
    assert_close(actual, bias.expand(2, 3), bias.dtype)
    grad = torch.tensor([[0.2, 0.4, 0.7], [0.15, 0.25, 0.35]], device=device)
    actual_grads = torch.autograd.grad(actual, (inp, weight, bias), grad)
    expected_grads = (
        torch.zeros_like(inp),
        torch.zeros_like(weight),
        grad.sum(0).half().float(),
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert_close(actual_grad, expected_grad, expected_grad.dtype)
