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

"""NVIDIA baseline for fused_pad_routing_map.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns Megatron-LM's
``megatron.core.fusions.fused_pad_routing_map.fused_pad_routing_map`` -- the
Triton ``_pad_routing_map_kernel`` FlagTrain's operator ports -- and raises when
Megatron is unavailable instead of falling back to a torch composition.

Shapes, the boolean top-k input construction and the single measured dtype are
copied from ``FlagTrain/benchmark/megatron/test_fused_pad_routing_map.py``.
Each shape is a deployment ``(num_tokens, num_experts, top_k, pad_multiple)``.

Megatron decorates the function with ``@jit_fuser`` (``torch.compile``). Like
``quick_gelu.py`` this module disables the fuser and unwraps the original
callable, so NCU profiles Megatron's own Triton kernel and the ATen
transpose/cast rather than Inductor output. Megatron is imported lazily on the
first ``native()`` call, so ops discovery does not trigger it.
"""

import torch

OP_NAME = "fused_pad_routing_map"
DTYPES = [torch.bool]
IS_INPLACE = False

# benchmark/megatron/test_fused_pad_routing_map.py::_PAD_ROUTING_MAP_CONFIGS
# (num_tokens, num_experts, top_k, pad_multiple)
_SHAPES = [
    (4096, 8, 2, 256),
    (16384, 8, 2, 256),
    (8192, 16, 4, 16),
    (4096, 64, 8, 32),
    (8192, 128, 8, 256),
    (8192, 32, 8, 16),
    (32768, 8, 2, 256),
    (32768, 256, 8, 256),
    (4097, 64, 8, 256),
]

_MEGATRON_FUSED_PAD_ROUTING_MAP = None


def _megatron_fused_pad_routing_map():
    """Megatron-LM's fused_pad_routing_map; raise when it cannot be imported."""
    global _MEGATRON_FUSED_PAD_ROUTING_MAP
    if _MEGATRON_FUSED_PAD_ROUTING_MAP is None:
        try:
            import megatron.core.jit as megatron_jit

            # The collector profiles native kernels, not Inductor output.
            megatron_jit.disable_jit_fuser()
            from megatron.core.fusions.fused_pad_routing_map import (
                fused_pad_routing_map,
            )
        except Exception as exc:
            raise RuntimeError(
                "fused_pad_routing_map baseline requires Megatron-LM's "
                "megatron.core.fusions.fused_pad_routing_map"
                ".fused_pad_routing_map, which could not be imported: "
                f"{exc!r}. Install megatron-core on this machine."
            ) from exc
        # megatron.core's package __init__ imports the fusions module before
        # disable_jit_fuser() can run, so the decorator may already have wrapped
        # the function with torch.compile; unwrap the original callable.
        _MEGATRON_FUSED_PAD_ROUTING_MAP = getattr(
            fused_pad_routing_map, "_torchdynamo_orig_callable", fused_pad_routing_map
        )
    return _MEGATRON_FUSED_PAD_ROUTING_MAP


def torch_op(routing_map, pad_multiple):
    """Megatron's Triton routing-map padding; raises without Megatron."""
    return _megatron_fused_pad_routing_map()(routing_map, pad_multiple)


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_fused_pad_routing_map()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def _make_routing_map(num_tokens, num_experts, top_k, device):
    """A boolean top-k routing map, as the benchmark builds it."""
    shape = (num_tokens, num_experts)
    logits = torch.randn(shape, dtype=torch.float32, device=device)
    indices = logits.topk(min(top_k, num_experts), dim=-1).indices
    routing_map = torch.zeros(shape, dtype=torch.bool, device=device)
    routing_map.scatter_(1, indices, True)
    return routing_map


def build_inputs(binding, dtype, device):
    num_tokens, num_experts, top_k, pad_multiple = binding["shape"]
    routing_map = _make_routing_map(num_tokens, num_experts, top_k, device)
    return (routing_map, pad_multiple), {}


def key_shape(binding):
    # First tensor: the [num_tokens, num_experts] map; unique per deployment.
    num_tokens, num_experts, _, _ = binding["shape"]
    return [num_tokens, num_experts]


def config(binding, dtype):
    num_tokens, num_experts, top_k, pad_multiple = binding["shape"]
    return {
        "inputs": {
            "routing_map": {
                "shape": [num_tokens, num_experts],
                "dtype": str(dtype),
            },
            "pad_multiple": {"scalar": pad_multiple},
        },
        "dims": {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "top_k": top_k,
        },
    }
