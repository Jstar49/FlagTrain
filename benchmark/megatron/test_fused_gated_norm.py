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
"""Performance benchmark for the fused output RMSNorm/SiLU gating operator.

The baseline is Megatron-LM's ``fused_gated_norm`` on the backends in
``_MCORE_BASELINE_VENDORS`` when the installed megatron-core provides it -- the
kernel this port is a port *of* -- and ``fused_gated_norm_ref``, a plain-torch
composition of the same contract, everywhere else and when that import fails.
``_BASELINE`` records which one was measured.

Both take the same ``(x, gate, weight)`` tuple, so there is nothing to
pre-compute outside the timed region; both are forward-only. Latency is reported
with GB/s, since the operation is memory-bound.
"""

import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import base

# GDN (Gated Delta Net) attention-output shapes, ``[batch, sequence, heads, dim]``.
# dim is a power of two (the kernel addresses it with ``tl.arange(0, D)``). The
# sweep walks the axes that change the kernel's cost: a decode step (one row),
# short-to-long prefill sequences, batch and head counts, and both head widths
# GDN uses. Sizes stay within CI GPU memory (the largest is ~67 MB per fp16
# tensor).
_GATED_NORM_SHAPES = [
    (1, 1, 16, 128),
    (2, 1024, 16, 128),
    (2, 2048, 16, 128),
    (2, 4096, 16, 128),
    (2, 8192, 16, 128),
    (8, 1024, 16, 128),
    (16, 512, 16, 128),
    (2, 1024, 8, 64),
    (2, 1024, 32, 128),
    (2, 1024, 16, 256),
    (2, 512, 16, 512),
]

# The variants are measured on a representative subset of the shapes above, as
# extra rows of the same dtype table; the variant travels as a fourth operand and
# labels the Size Detail column.
_VARIANT_SHAPES = [
    (1, 1, 16, 128),
    (2, 1024, 16, 128),
    (2, 1024, 16, 256),
]
_VARIANTS = ("zero_centered", "fp32_gate", "strided")

_EPS = 1e-5

# The upstream operator targets FP16/BF16 activations; FP16 is always available,
# BF16 only where the runtime reports it.
_DTYPES = [torch.float16]
if flag_train.runtime.device.support_bf16:
    _DTYPES.append(torch.bfloat16)


def fused_gated_norm_ref(x, gate, weight, eps, zero_centered=False):
    """Plain-torch composition of the same contract; the fallback baseline."""
    w = weight.to(torch.float32) + (1.0 if zero_centered else 0.0)
    xf = x.to(torch.float32)
    inv = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) / x.shape[-1] + eps)
    norm = (xf * inv * w).to(x.dtype).to(torch.float32)
    gate_f = gate.to(torch.float32)
    return (norm * (gate_f * torch.sigmoid(gate_f))).to(x.dtype)


# Backends whose baseline is Megatron-LM's kernel; it cannot run elsewhere.
_MCORE_BASELINE_VENDORS = {"nvidia"}


def _load_mcore_fused_gated_norm():
    """Megatron-LM's fused_gated_norm, or ``None`` when it is not the baseline."""
    if flag_train.vendor_name not in _MCORE_BASELINE_VENDORS:
        return None
    try:
        from megatron.core.fusions.fused_gated_norm import fused_gated_norm

        return fused_gated_norm
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} uses Megatron-LM's fused_gated_norm as "
            f"its baseline when megatron-core provides it, but the import "
            f"failed: {exc!r}. Falling back to fused_gated_norm_ref.",
            stacklevel=2,
        )
        return None


# Resolved once, so the first-use Triton JIT compile is not counted in the
# measurement.
_mcore_fused_gated_norm = _load_mcore_fused_gated_norm()
_BASELINE = (
    "megatron-core fused_gated_norm"
    if _mcore_fused_gated_norm is not None
    else "fused_gated_norm_ref (torch)"
)


class FusedGatedNormBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _GATED_NORM_SHAPES

    def set_shapes(self, shape_file=None):
        # Three input tensors per case plus an output; the explicit list keeps
        # memory bounded on CI GPUs.
        self.shapes = list(self.DEFAULT_SHAPES)

    def set_more_shapes(self):
        return []

    def set_more_metrics(self):
        # Memory-bound: GB/s explains the latency better than a FLOP count.
        return ["gbps"]

    def get_gbps(self, args, latency):
        # The fourth operand is the variant label, not a tensor to move.
        x, gate, weight = args[:3]
        io_bytes = 2 * x.numel() * x.element_size()  # x is read, out is written
        io_bytes += gate.numel() * gate.element_size()
        io_bytes += weight.numel() * weight.element_size()
        return io_bytes * 1e-9 / (latency * 1e-3)


def _make_operands(shape, dtype, device, variant):
    """Build ``(x, gate, weight, variant)`` for one measured configuration."""
    dim = shape[-1]
    # ``fp32_gate`` keeps the activation narrow but the gate and weight FP32.
    wide = torch.float32 if variant == "fp32_gate" else dtype
    if variant == "strided":
        # A ``[batch, sequence, heads, dim]`` view obtained by transposing the
        # base allocation, so the kernel reads it through strides.
        base_shape = (shape[0], shape[2], shape[1], dim)
        x = torch.randn(base_shape, dtype=dtype, device=device).permute(0, 2, 1, 3)
        gate = torch.randn(base_shape, dtype=wide, device=device).permute(0, 2, 1, 3)
    else:
        x = torch.randn(shape, dtype=dtype, device=device)
        gate = torch.randn(shape, dtype=wide, device=device)
    return x, gate, torch.randn((dim,), dtype=wide, device=device), variant


def fused_gated_norm_input_fn(shape, dtype, device):
    """The plain case for every shape, plus labeled variants where configured."""
    yield _make_operands(shape, dtype, device, "regular")
    if shape in _VARIANT_SHAPES:
        for variant in _VARIANTS:
            yield _make_operands(shape, dtype, device, variant)


def torch_op(x, gate, weight, variant):
    """Baseline, chosen by platform. See the module docstring."""
    impl = _mcore_fused_gated_norm or fused_gated_norm_ref
    return impl(x, gate, weight, _EPS, variant == "zero_centered")


def train_op(x, gate, weight, variant):
    """The operator under test."""
    return flag_train.megatron.fused_gated_norm(
        x, gate, weight, _EPS, variant == "zero_centered"
    )


@pytest.mark.fused_gated_norm
def test_fused_gated_norm_perf():
    print(f"\nBaseline: {_BASELINE}")
    bench = FusedGatedNormBenchmark(
        input_fn=fused_gated_norm_input_fn,
        op_name="fused_gated_norm",
        torch_op=torch_op,
        dtypes=_DTYPES,
        # The local ratio is a real comparison when the baseline is Megatron's
        # kernel, so --base-data's cross-hardware figure does not apply.
        native_baseline=_mcore_fused_gated_norm is not None,
    )
    bench.set_train(train_op)
    bench.run()
