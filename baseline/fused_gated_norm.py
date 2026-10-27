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

"""NVIDIA baseline for fused_gated_norm.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns Megatron-LM's
``megatron.core.fusions.fused_gated_norm.fused_gated_norm`` -- the Triton
``gated_norm_fwd`` kernel FlagTrain's operator ports -- and raises when
Megatron is unavailable instead of falling back to a torch composition.

Shapes, dtypes, the ``EPS`` value and the input construction (including the
``zero_centered``, ``fp32_gate`` and ``strided`` variants the benchmark
measures) are copied from
``FlagTrain/benchmark/megatron/test_fused_gated_norm.py``. Megatron is imported
lazily on the first ``native()`` call, so ops discovery does not trigger it.
The Triton kernel is compiled by the first measured call, outside do_bench's
timed region and the NCU NVTX range.
"""

import torch

OP_NAME = "fused_gated_norm"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fused_gated_norm.py::_GATED_NORM_SHAPES
_SHAPES = [
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

# benchmark/megatron/test_fused_gated_norm.py::_VARIANT_SHAPES / _VARIANTS
# The same tensor shapes are measured with these extra variants.
_VARIANT_SHAPES = [
    (1, 1, 16, 128),
    (2, 1024, 16, 128),
    (2, 1024, 16, 256),
]
_VARIANTS = ("zero_centered", "fp32_gate", "strided")

_EPS = 1e-5

_MEGATRON_FUSED_GATED_NORM = None


def _megatron_fused_gated_norm():
    """Megatron-LM's fused_gated_norm; raise when it cannot be imported."""
    global _MEGATRON_FUSED_GATED_NORM
    if _MEGATRON_FUSED_GATED_NORM is None:
        try:
            from megatron.core.fusions.fused_gated_norm import fused_gated_norm
        except Exception as exc:
            raise RuntimeError(
                "fused_gated_norm baseline requires Megatron-LM's "
                "megatron.core.fusions.fused_gated_norm.fused_gated_norm, "
                f"which could not be imported: {exc!r}. Install megatron-core "
                "on this machine."
            ) from exc
        _MEGATRON_FUSED_GATED_NORM = fused_gated_norm
    return _MEGATRON_FUSED_GATED_NORM


def torch_op(x, gate, weight, variant):
    """Megatron's Triton fused RMSNorm/SiLU gating; raises without Megatron."""
    return _megatron_fused_gated_norm()(
        x, gate, weight, _EPS, variant == "zero_centered"
    )


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_fused_gated_norm()
    return torch_op


def grid():
    bindings = [{"shape": shape, "variant": "regular"} for shape in _SHAPES]
    bindings.extend(
        {"shape": shape, "variant": variant}
        for shape in _VARIANT_SHAPES
        for variant in _VARIANTS
    )
    return bindings


def _make_operands(shape, dtype, device, variant):
    """Build ``(x, gate, weight)`` for one configuration as the benchmark does."""
    dim = shape[-1]
    # ``fp32_gate`` keeps the activation narrow but the gate and weight FP32.
    wide = torch.float32 if variant == "fp32_gate" else dtype
    if variant == "strided":
        # A [batch, sequence, heads, dim] view of a transposed allocation, so
        # the kernel reads it through strides.
        base_shape = (shape[0], shape[2], shape[1], dim)
        x = torch.randn(base_shape, dtype=dtype, device=device).permute(0, 2, 1, 3)
        gate = torch.randn(base_shape, dtype=wide, device=device).permute(0, 2, 1, 3)
    else:
        x = torch.randn(shape, dtype=dtype, device=device)
        gate = torch.randn(shape, dtype=wide, device=device)
    weight = torch.randn((dim,), dtype=wide, device=device)
    return x, gate, weight, variant


def build_inputs(binding, dtype, device):
    args = _make_operands(binding["shape"], dtype, device, binding["variant"])
    return args, {}


def key_shape(binding):
    # The regular case follows the simple-op convention (the first tensor's
    # shape). The benchmark's variants share that shape, so they carry a
    # suffix to keep their records distinct.
    shape = binding["shape"]
    variant = binding["variant"]
    if variant == "regular":
        return list(shape)
    return "x".join(str(dim) for dim in shape) + "_" + variant


def config(binding, dtype):
    batch, sequence, heads, dim = binding["shape"]
    variant = binding["variant"]
    wide = "torch.float32" if variant == "fp32_gate" else str(dtype)
    return {
        "inputs": {
            "x": {"shape": [batch, sequence, heads, dim], "dtype": str(dtype)},
            "gate": {"shape": [batch, sequence, heads, dim], "dtype": wide},
            "weight": {"shape": [dim], "dtype": wide},
            "eps": {"scalar": _EPS},
            "zero_centered": {"scalar": variant == "zero_centered"},
        },
        "dims": {
            "batch": batch,
            "sequence": sequence,
            "heads": heads,
            "dim": dim,
        },
    }
