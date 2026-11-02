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
"""Performance benchmark for Megatron's scaled masked softmax backward.

The baseline is Megatron-LM's ``ScaledMaskedSoftmax.backward``, the operator
this port is a port *of*. When ``megatron.core.fusions.fused_softmax`` cannot
supply it, its one-line method body is vendored below and run against the same
Apex CUDA extension. When that extension is unavailable, the benchmark warns
and falls back to ``masked_backward_ref``, the closed form
``scale * y * (dy - sum(dy * y))``. ``_BASELINE`` records which one was
measured.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). Both implementations read
``output_grads`` and ``softmax_results`` and write the gradient, so the reported
GB/s is 3x the tensor size over the measured latency. The operands are passed
directly rather than measured through ``autograd``: the operator is the
backward kernel, not a forward whose gradient is taken.
"""

import importlib.metadata
import importlib.util
import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import base

_SCALE = 0.125
_EXTENSION = "scaled_masked_softmax_cuda"

# ``[b, np, sq, sk]`` attention gradients. The sweep walks the key axis through
# the range Megatron's fused kernel accepts (16 < sk <= 4096); the largest case
# is ~2 GiB per fp16 tensor.
_SHAPES = [
    (8, 16, 128, 128),
    (8, 16, 256, 256),
    (8, 16, 512, 512),
    (8, 16, 1024, 1024),
    (4, 16, 2048, 2048),
    (4, 16, 4096, 4096),
]

# The upstream operator targets FP16/BF16 activations; BF16 only where the
# runtime reports it.
_DTYPES = [torch.float16]
if flag_train.runtime.device.support_bf16:
    _DTYPES.append(torch.bfloat16)


def masked_backward_ref(output_grads, softmax_results, scale):
    """Closed form ``scale * y * (dy - sum(dy * y))``; the fallback baseline."""
    dy = output_grads.to(torch.float32)
    y = softmax_results.to(torch.float32)
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return (scale * y * (dy - dot)).to(output_grads.dtype)


def _megatron_version():
    for distribution in ("megatron-core", "megatron"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


class _SavedForBackward:
    """Stand-in for the context ``ScaledMaskedSoftmax.backward`` takes.

    Megatron's forward saves ``(softmax_results, scale_t)``; the backward only
    reads ``ctx.saved_tensors``, so a two-attribute object runs the copied body
    -- and the installed one -- without a forward pass in the timed region.
    """

    def __init__(self, softmax_results, scale):
        self.saved_tensors = (softmax_results, torch.tensor([scale]))


def _vendored_backward(output_grads, softmax_results, scale):
    """Copied from ``ScaledMaskedSoftmax.backward`` (minus the ``ctx`` plumbing)."""
    import scaled_masked_softmax_cuda

    ctx = _SavedForBackward(softmax_results, scale)
    softmax_results, scale_t = ctx.saved_tensors
    input_grads = scaled_masked_softmax_cuda.backward(
        output_grads, softmax_results, scale_t[0]
    )
    # As in the upstream body: one answer per forward input (inputs, mask, scale).
    return input_grads, None, None


def _load_megatron_baseline():
    """Megatron's backward, or ``None`` when this install cannot provide one."""
    if importlib.util.find_spec(_EXTENSION) is None:
        warnings.warn(
            f"Megatron-LM's fused masked softmax backward needs the Apex "
            f"'{_EXTENSION}' extension, which is not installed. Falling back to "
            f"masked_backward_ref (torch).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    version = _megatron_version()
    try:
        from megatron.core.fusions.fused_softmax import ScaledMaskedSoftmax
    except Exception as exc:  # pragma: no cover - depends on the installation
        warnings.warn(
            "megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax could not "
            f"be imported ({exc!r}); using the vendored baseline "
            f"(installed megatron-core: {version or 'absent'}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return "vendored"
    return ScaledMaskedSoftmax


_MEGATRON = _load_megatron_baseline()
if _MEGATRON is None:
    _BASELINE = "masked_backward_ref (torch)"
elif _MEGATRON == "vendored":
    _BASELINE = (
        "Megatron-LM ScaledMaskedSoftmax.backward (vendored body, "
        f"megatron-core {_megatron_version() or 'absent'})"
    )
else:
    _BASELINE = f"megatron-core {_megatron_version()} ScaledMaskedSoftmax"


def _baseline_backward(output_grads, softmax_results, scale):
    if _MEGATRON is None:
        return masked_backward_ref(output_grads, softmax_results, scale)
    if _MEGATRON == "vendored":
        return _vendored_backward(output_grads, softmax_results, scale)[0]
    ctx = _SavedForBackward(softmax_results, scale)
    return _MEGATRON.backward(ctx, output_grads)[0]


class ScaledMaskedSoftmaxBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES

    def set_shapes(self, shape_file=None):
        # ``benchmark/core_shapes.yaml`` has no entry for this operator, and the
        # explicit list keeps memory bounded on CI GPUs.
        self.shapes = list(self.DEFAULT_SHAPES)

    def set_more_shapes(self):
        return []

    def set_more_metrics(self):
        # Memory-bound: GB/s explains the latency better than a FLOP count.
        return ["gbps"]

    def get_gbps(self, args, latency):
        dy, y = args[0], args[1]
        io_bytes = 2 * dy.numel() * dy.element_size()  # dy and y are read
        io_bytes += y.numel() * y.element_size()  # the gradient is written
        return io_bytes * 1e-9 / (latency * 1e-3)


def masked_backward_input_fn(shape, dtype, device):
    # A plausible forward output: non-negative rows that sum to one.
    y = torch.rand(shape, dtype=dtype, device=device)
    y = y / y.sum(dim=-1, keepdim=True)
    yield torch.randn(shape, dtype=dtype, device=device), y, _SCALE


def torch_op(output_grads, softmax_results, scale):
    """Baseline, chosen at import time. See the module docstring."""
    return _baseline_backward(output_grads, softmax_results, scale)


def train_op(output_grads, softmax_results, scale):
    """The operator under test."""
    return flag_train.megatron.scaled_masked_softmax_cuda_backward(
        output_grads, softmax_results, scale
    )


@pytest.mark.scaled_masked_softmax_cuda_backward
def test_scaled_masked_softmax_cuda_backward_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = ScaledMaskedSoftmaxBackwardBenchmark(
        input_fn=masked_backward_input_fn,
        op_name="scaled_masked_softmax_cuda_backward",
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
