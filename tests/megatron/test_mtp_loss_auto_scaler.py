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


"""Check forward aliasing and backward-time state of each MTP graph.

Explicit per-graph Megatron MTP loss scaling for FlagOS requirement 6471.
"""

import pytest
import torch

import flag_train
from flag_train.reference_candidate import get_candidate
from flag_train.testing import assert_equal

OP_NAME = "mtp_loss_auto_scaler"
OVERRIDE_KEY = f"megatron.{OP_NAME}"
SHAPES = ((2, 17), (128, 256), (1, 1), (257, 127), (128, 1024), (2, 256, 4096))
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
LOSS_SCALE = 0.375
# Keep output shapes unique across scales for shape-keyed --base-data.
SCALE_VARIANTS = (((3, 19), 0.0), ((129, 257), 2.0))
WORKLOADS = tuple((shape, LOSS_SCALE) for shape in SHAPES) + SCALE_VARIANTS


def make_inputs(shape, dtype, device, scale=LOSS_SCALE):
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.tensor(2.0, dtype=torch.float32, device=device),
        torch.tensor(scale, dtype=torch.float32, device=device),
    )


class _MTPLossAutoScalerReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, mtp_loss, loss_scale):
        ctx.save_for_backward(mtp_loss)
        # A mutable scale must avoid saved-tensor version checking.
        ctx.loss_scale = loss_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (mtp_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(mtp_loss) * ctx.loss_scale, None


def reference(output, mtp_loss, loss_scale):
    return _MTPLossAutoScalerReference.apply(output, mtp_loss, loss_scale)


def _candidate():
    return get_candidate(OVERRIDE_KEY, reference=reference)


def _build_graph(fn, shape, dtype, initial_scale):
    output, mtp_loss, scale = make_inputs(
        shape, dtype, flag_train.device, initial_scale
    )
    output.requires_grad_()
    mtp_loss.requires_grad_()
    original_output = output.detach().clone()
    original_loss = mtp_loss.detach().clone()
    result = fn(output, mtp_loss, scale)
    assert result.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    assert_equal(result, original_output)
    assert_equal(output, original_output)
    assert_equal(mtp_loss, original_loss)
    assert_equal(scale, scale.new_tensor(initial_scale))
    upstream = torch.linspace(-2, 3, output.numel(), device=output.device)
    upstream = upstream.reshape(output.shape).to(output.dtype)
    return result, output, mtp_loss, scale, upstream


def _check_backward(graph, expected_scale):
    result, output, mtp_loss, scale, upstream = graph
    originals = tuple(tensor.detach().clone() for tensor in (output, mtp_loss, scale))
    grad_output, grad_loss = torch.autograd.grad(result, (output, mtp_loss), upstream)
    assert_equal(grad_output, upstream)
    assert_equal(grad_loss, mtp_loss.new_tensor(expected_scale))
    for tensor, original in zip((output, mtp_loss, scale), originals):
        assert_equal(tensor, original)


@pytest.mark.mtp_loss_auto_scaler
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [1.0, LOSS_SCALE])
def test_mtp_loss_auto_scaler(shape, dtype, scale):
    _check_backward(_build_graph(_candidate(), shape, dtype, scale), scale)
    _check_backward(_build_graph(reference, shape, dtype, scale), scale)


@pytest.mark.mtp_loss_auto_scaler
@pytest.mark.parametrize("shape,scale", SCALE_VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mtp_loss_auto_scaler_scale_variants(shape, scale, dtype):
    _check_backward(_build_graph(_candidate(), shape, dtype, scale), scale)
    _check_backward(_build_graph(reference, shape, dtype, scale), scale)


@pytest.mark.mtp_loss_auto_scaler
@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["a-first", "b-first"])
def test_mtp_loss_auto_scaler_live_graphs(order):
    for fn in (_candidate(), reference):
        graph_a = _build_graph(fn, SHAPES[0], torch.float32, 2.0)
        graph_b = _build_graph(fn, SHAPES[0], torch.float32, 5.0)
        graph_a[3].fill_(3.0)
        graph_b[3].fill_(7.0)
        graphs, updated_scales = (graph_a, graph_b), (3.0, 7.0)
        for index in order:
            _check_backward(graphs[index], updated_scales[index])


@pytest.mark.mtp_loss_auto_scaler
def test_mtp_loss_auto_scaler_scale_is_nondifferentiable():
    for fn in (_candidate(), reference):
        inputs = make_inputs(SHAPES[0], torch.float32, flag_train.device)
        for tensor in inputs:
            tensor.requires_grad_()
        result = fn(*inputs)
        _, _, grad_scale = torch.autograd.grad(
            result, inputs, torch.ones_like(result), allow_unused=True
        )
        assert grad_scale is None
