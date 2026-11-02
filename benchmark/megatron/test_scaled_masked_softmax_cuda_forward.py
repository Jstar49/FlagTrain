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
"""Performance benchmark for Megatron's scaled masked softmax forward.

The baseline is Megatron-LM's ``ScaledMaskedSoftmax.forward``, the operator
this port is a port *of*. When ``megatron.core.fusions.fused_softmax`` cannot
supply it, its one-line method body is vendored below and run against the same
Apex CUDA extension. When that extension is unavailable, the benchmark warns
and falls back to ``masked_softmax_ref``, the plain-torch scale + sentinel +
softmax composition. ``_BASELINE`` records which one was measured.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). Both implementations read the
scores and the mask and write the probabilities, so the reported GB/s counts
those three streams over the measured latency.
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
_MASKED_SCORE = -10000.0
_EXTENSION = "scaled_masked_softmax_cuda"

# ``[b, np, sq, sk]`` attention scores. The sweep walks the key axis through
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


def masked_softmax_ref(inputs, mask, scale):
    """Torch composition of the forward; the fallback baseline."""
    scores = inputs.to(torch.float32) * scale
    masked = mask.bool().expand_as(scores)
    scores = scores.masked_fill(masked, _MASKED_SCORE)
    row_max = scores.max(dim=-1, keepdim=True).values
    numerator = torch.exp(scores - row_max)
    result = numerator / numerator.sum(dim=-1, keepdim=True)
    result = result.masked_fill(masked, 0.0)
    result = torch.where(
        masked.all(dim=-1, keepdim=True), torch.zeros_like(result), result
    )
    return result.to(inputs.dtype)


def _megatron_version():
    for distribution in ("megatron-core", "megatron"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _vendored_forward(inputs, mask, scale):
    """Copied from ``ScaledMaskedSoftmax.forward``."""
    import scaled_masked_softmax_cuda

    scale_t = torch.tensor([scale])
    return scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])


def _load_megatron_baseline():
    """Megatron's forward, or ``None`` when this install cannot provide one."""
    if importlib.util.find_spec(_EXTENSION) is None:
        warnings.warn(
            f"Megatron-LM's fused masked softmax forward needs the Apex "
            f"'{_EXTENSION}' extension, which is not installed. Falling back to "
            f"masked_softmax_ref (torch).",
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
    _BASELINE = "masked_softmax_ref (torch)"
elif _MEGATRON == "vendored":
    _BASELINE = (
        "Megatron-LM ScaledMaskedSoftmax.forward (vendored body, "
        f"megatron-core {_megatron_version() or 'absent'})"
    )
else:
    _BASELINE = f"megatron-core {_megatron_version()} ScaledMaskedSoftmax"


def _baseline_forward(inputs, mask, scale):
    if _MEGATRON is None:
        return masked_softmax_ref(inputs, mask, scale)
    if _MEGATRON == "vendored":
        return _vendored_forward(inputs, mask, scale)
    return _MEGATRON.apply(inputs, mask, scale)


class ScaledMaskedSoftmaxForwardBenchmark(base.GenericBenchmark):
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
        scores, mask = args[0], args[1]
        io_bytes = 2 * scores.numel() * scores.element_size()  # read and write
        io_bytes += mask.numel() * mask.element_size()  # the mask is read
        return io_bytes * 1e-9 / (latency * 1e-3)


def masked_forward_input_fn(shape, dtype, device):
    scores = torch.randn(shape, dtype=dtype, device=device)
    # Hide the second half of the keys: a padding mask, realistic for attention.
    mask = torch.zeros(
        (shape[0], 1, shape[2], shape[3]), dtype=torch.bool, device=device
    )
    mask[..., shape[3] // 2 :] = True
    yield scores, mask, _SCALE


def torch_op(scores, mask, scale):
    """Baseline, chosen at import time. See the module docstring."""
    return _baseline_forward(scores, mask, scale)


def train_op(scores, mask, scale):
    """The operator under test."""
    return flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, scale)


@pytest.mark.scaled_masked_softmax_cuda_forward
def test_scaled_masked_softmax_cuda_forward_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = ScaledMaskedSoftmaxForwardBenchmark(
        input_fn=masked_forward_input_fn,
        op_name="scaled_masked_softmax_cuda_forward",
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
