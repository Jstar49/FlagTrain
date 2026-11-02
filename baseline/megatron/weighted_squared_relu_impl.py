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

"""NVIDIA baseline for weighted_squared_relu_impl (forward).

Self-contained: depends only on torch and megatron. ``native()`` returns
Megatron-LM's ``weighted_squared_relu_impl`` and raises when Megatron is
unavailable instead of falling back to a torch composition.

The benchmark measures three argument combinations of the same interface, so
they share one ``OP_NAME`` and are told apart by a suffix on the shape key (as
``fused_gated_norm`` does). Shapes, dtypes, the clamp scale and the inputs are
copied from ``benchmark/megatron/test_weighted_squared_relu_impl.py``.

Megatron's ``@jit_fuser`` wrappers are unwrapped in place, so NCU profiles the
eager baseline rather than Inductor kernels. Megatron is imported lazily on the
first ``native()`` call, so ops discovery does not trigger it.
"""

import math

import torch

OP_NAME = "weighted_squared_relu_impl"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_weighted_squared_relu_impl.py::_CORE_SHAPES
_CORE_SHAPES = [
    (1024, 4096),
    (4096, 4096),
    (4, 2048, 4096),
    (8192, 11008),
]

# benchmark/megatron/test_weighted_squared_relu_impl.py::_COMPREHENSIVE_SHAPES
_COMPREHENSIVE_SHAPES = [
    (1, 4096),
    (32768, 2048),
]

_SHAPES = _CORE_SHAPES + _COMPREHENSIVE_SHAPES

# benchmark/megatron/test_weighted_squared_relu_impl.py::_VARIANTS
_VARIANTS = (
    "weighted_squared_relu",
    "clamped_squared_relu",
    "weighted_clamped_squared_relu",
)

_CLAMP_SCALE = 5.0

_MEGATRON_WEIGHTED_SQUARED_RELU_IMPL = None


def _megatron_weighted_squared_relu_impl():
    """Megatron-LM's weighted_squared_relu_impl; raise when it cannot be loaded."""
    global _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL
    if _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL is None:
        try:
            import megatron.core.jit as megatron_jit

            # The collector profiles eager kernels, not Inductor output.
            megatron_jit.disable_jit_fuser()
            import megatron.core.fusions.fused_weighted_squared_relu as module
        except Exception as exc:
            raise RuntimeError(
                "weighted_squared_relu_impl baseline requires Megatron-LM's "
                "megatron.core.fusions.fused_weighted_squared_relu"
                ".weighted_squared_relu_impl, which could not be imported: "
                f"{exc!r}. Install megatron-core on this machine."
            ) from exc
        # megatron.core imports this module before disable_jit_fuser() can run,
        # so unwrap the compiled functions the public *_impl reads from globals.
        for name in (
            "_tanh_relu_over_scale",
            "weighted_squared_relu",
            "clamped_squared_relu",
            "weighted_clamped_squared_relu",
        ):
            kernel = getattr(module, name)
            setattr(module, name, getattr(kernel, "_torchdynamo_orig_callable", kernel))
        _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL = module.weighted_squared_relu_impl
    return _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL


def torch_op(x, weights=None, clamp_scale=None):
    """Megatron's weighted squared ReLU forward; raises without Megatron."""
    return _megatron_weighted_squared_relu_impl()(x, weights, clamp_scale)


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_weighted_squared_relu_impl()
    return torch_op


def grid():
    return [
        {"shape": shape, "variant": variant}
        for shape in _SHAPES
        for variant in _VARIANTS
    ]


def _weights(shape, device):
    """One fp32 router probability per token, as the benchmark builds it."""
    num_tokens = math.prod(shape[:-1])
    return torch.rand((num_tokens, 1), dtype=torch.float32, device=device)


def _make_operands(shape, dtype, device, variant):
    """Build ``(x, weights, clamp_scale)`` for one measured configuration."""
    if variant == "clamped_squared_relu":
        x = torch.randn(shape, dtype=dtype, device=device)
        return x, None, _CLAMP_SCALE
    weights = _weights(shape, device)
    if variant == "weighted_squared_relu":
        x = torch.randn(shape, dtype=dtype, device=device)
        return x, weights, None
    # The benchmark scales the input by the clamp before the weighted variant.
    x = torch.randn(shape, dtype=dtype, device=device) * _CLAMP_SCALE
    return x, weights, _CLAMP_SCALE


def build_inputs(binding, dtype, device):
    args = _make_operands(binding["shape"], dtype, device, binding["variant"])
    return args, {}


def key_shape(binding):
    # The variants share the first tensor's shape, so the key carries the label
    # the benchmark passes as a fourth operand.
    shape = binding["shape"]
    return "x".join(str(dim) for dim in shape) + "_" + binding["variant"]


def config(binding, dtype):
    shape = binding["shape"]
    variant = binding["variant"]
    num_tokens = math.prod(shape[:-1])
    inputs = {
        "x": {"shape": list(shape), "dtype": str(dtype)},
        # A scalar for the benchmark's signature fallback: the three variants
        # share tensor shapes, so the record is chosen by this label.
        "variant": {"scalar": variant},
    }
    if variant != "clamped_squared_relu":
        inputs["weights"] = {
            "shape": [num_tokens, 1],
            "dtype": str(torch.float32),
        }
    if variant != "weighted_squared_relu":
        inputs["clamp_scale"] = {"scalar": _CLAMP_SCALE}
    return {
        "inputs": inputs,
        "dims": {
            "num_tokens": num_tokens,
            "hidden": shape[-1],
            "variant": variant,
        },
    }
