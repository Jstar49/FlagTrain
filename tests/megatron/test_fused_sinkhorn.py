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


"""Check the compiled Megatron Sinkhorn projection and its recomputed first-order gradient.

Megatron compiled Torch Sinkhorn fallback at d2b0e35cc4f05bc6b70734d558617050fbbb51ed.
"""

from functools import lru_cache

import pytest
import torch
from torch._dynamo import config as dynamo_config

import flag_train
from flag_train.reference_candidate import get_candidate
from flag_train.testing import assert_close, assert_equal

OP_NAME = "fused_sinkhorn"
OVERRIDE_KEY = f"megatron.{OP_NAME}"
CASES = (
    ((8, 4, 4), 1, 1e-6),
    ((2, 16, 8, 8), 3, 1e-4),
    ((1, 1, 1), 1, 1e-6),
    ((1, 4, 4), 1, 1e-6),
    ((3, 5, 5), 2, 1e-5),
    ((257, 4, 4), 3, 1e-6),
    ((8, 128, 4, 4), 20, 1e-6),
    ((2, 256, 8, 8), 20, 1e-4),
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def make_inputs(shape, dtype, device, num_iterations, eps):
    generator = torch.Generator().manual_seed(6472)
    input_logits = (
        torch.randn(shape, generator=generator)
        .to(device=device, dtype=dtype)
        .contiguous()
    )
    return input_logits, num_iterations, eps


def _project(input_logits, num_iterations, eps):
    matrix = input_logits.softmax(dim=-1) + eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(1, num_iterations):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


_compiled_project = torch.compile(_project, fullgraph=True)


@lru_cache(maxsize=1)
def _ascend_project():
    from torch_npu.dynamo import torchair

    # TorchAir's AOT partitioner needs static shapes for this recomputed graph.
    return torch.compile(
        _project, backend=torchair.get_npu_backend(), fullgraph=True, dynamic=False
    )


def _run_compiled(input_logits, num_iterations, eps):
    project = (
        _ascend_project() if input_logits.device.type == "npu" else _compiled_project
    )
    # Each workload may specialize both a no-grad and a grad-enabled graph.
    # Keep enough cached graphs for the suite, with no silent eager fallback.
    with dynamo_config.patch(recompile_limit=2 * len(CASES) * len(DTYPES)):
        return project(input_logits, num_iterations, eps)


class _FusedSinkhornReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_logits, num_iterations, eps):
        ctx.save_for_backward(input_logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return _run_compiled(input_logits.detach(), num_iterations, eps)

    @staticmethod
    def backward(ctx, grad_output):
        (input_logits,) = ctx.saved_tensors
        with torch.enable_grad():
            logits = input_logits.detach().requires_grad_(True)
            projected = _run_compiled(logits, ctx.num_iterations, ctx.eps)
            projected.backward(grad_output)
        return logits.grad, None, None


def reference(input_logits, num_iterations, eps):
    return _FusedSinkhornReference.apply(input_logits, num_iterations, eps)


@pytest.mark.fused_sinkhorn
@pytest.mark.parametrize("shape,num_iterations,eps", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_sinkhorn(shape, num_iterations, eps, dtype):
    candidate = get_candidate(OVERRIDE_KEY, reference=reference)

    input_logits, _, _ = make_inputs(
        shape, dtype, flag_train.device, num_iterations, eps
    )
    candidate_input = input_logits.clone().requires_grad_()
    reference_input = input_logits.clone().requires_grad_()
    actual = candidate(candidate_input, num_iterations, eps)
    expected = reference(reference_input, num_iterations, eps)
    assert_close(actual, expected, expected.dtype)
    assert_equal(candidate_input, input_logits)
    assert (
        actual.untyped_storage().data_ptr()
        != candidate_input.untyped_storage().data_ptr()
    )

    grad_output = (
        torch.linspace(-0.8, 0.9, expected.numel()).reshape(expected.shape).to(expected)
    )
    actual_grad = torch.autograd.grad(actual, candidate_input, grad_output.clone())[0]
    expected_grad = torch.autograd.grad(expected, reference_input, grad_output)[0]
    assert_close(actual_grad, expected_grad, expected_grad.dtype)
    assert_equal(candidate_input, input_logits)


def test_fused_sinkhorn_wide_matrix():
    x = torch.randn((1, 1025, 1025), device=flag_train.device, requires_grad=True)
    actual = flag_train.megatron.fused_sinkhorn(x, 1)
    matrix = x.softmax(dim=-1) + 1e-6
    expected = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-6)
    upstream = torch.randn_like(actual)
    assert_close(actual, expected, expected.dtype)
    (actual_grad,) = torch.autograd.grad(actual, x, upstream)
    (expected_grad,) = torch.autograd.grad(expected, x, upstream)
    assert_close(actual_grad, expected_grad, expected_grad.dtype)


def test_fused_sinkhorn_saves_only_logits():
    x = torch.randn((8, 8, 8), device=flag_train.device, requires_grad=True)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = flag_train.megatron.fused_sinkhorn(x, 20)
    assert len(saved) == 1
    assert saved[0].data_ptr() == x.data_ptr()
    assert torch.isfinite(torch.autograd.grad(actual.sum(), x)[0]).all()
