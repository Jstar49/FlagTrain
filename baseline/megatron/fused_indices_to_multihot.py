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

"""NVIDIA baseline for fused_indices_to_multihot.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns Megatron-LM's
``megatron.core.fusions.fused_indices_converter.fused_indices_to_multihot`` --
the Triton kernel FlagTrain's operator ports -- and raises when Megatron is
unavailable instead of falling back to a torch composition.

Shapes, the dtype sweep and the input construction are copied from
``FlagTrain/benchmark/megatron/test_fused_indices_to_multihot.py``. The
converter is gated behind Megatron's experimental-flag switch, so the module
sets it before reading the function, exactly as the benchmark does. Megatron is
imported lazily on the first ``native()`` call, so ops discovery does not
trigger it.
"""

import torch

OP_NAME = "fused_indices_to_multihot"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fused_indices_to_multihot.py::_SHAPES
# (num_tokens, top_k, num_experts). The Ascend-only filtering done by the
# benchmark does not apply to the NVIDIA collector.
_SHAPES = [
    (4096, 2, 8),
    (4096, 6, 8),
    (4096, 4, 15),
    (2048, 8, 16),
    (16384, 8, 32),
    (8192, 1, 128),
    (65536, 8, 32),
    (131072, 8, 16),
    (65536, 2, 128),
    (256, 8, 32),
]

_NATIVE = None


def _megatron_fused_indices_to_multihot():
    """Megatron-LM's converter; raise when it cannot be imported."""
    global _NATIVE
    if _NATIVE is None:
        try:
            from megatron.core.config import set_experimental_flag
            from megatron.core.fusions import fused_indices_converter

            set_experimental_flag(True)
            _NATIVE = fused_indices_converter.fused_indices_to_multihot
        except Exception as exc:
            raise RuntimeError(
                "fused_indices_to_multihot baseline requires Megatron-LM's "
                "megatron.core.fusions.fused_indices_converter"
                ".fused_indices_to_multihot, which could not be imported: "
                f"{exc!r}. Install megatron-core on this machine."
            ) from exc
    return _NATIVE


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    return _megatron_fused_indices_to_multihot()


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    """``(indices, probs, num_experts)`` as the benchmark builds them."""
    num_tokens, topk, num_experts = binding["shape"]
    scores = torch.rand(num_tokens, num_experts, dtype=torch.float32, device=device)
    indices = scores.topk(topk, dim=1).indices.to(torch.int64)
    probs = torch.rand(num_tokens, topk, dtype=dtype, device=device)
    return (indices, probs, num_experts), {}


def key_shape(binding):
    # The first tensor is [num_tokens, top_k]; num_experts is only an int
    # argument. Use a semantic key so lookup falls back to the tensor-signature
    # match config() describes.
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    num_tokens, topk, num_experts = binding["shape"]
    return {
        "inputs": {
            "indices": {"shape": [num_tokens, topk], "dtype": "torch.int64"},
            "probs": {"shape": [num_tokens, topk], "dtype": str(dtype)},
            "num_of_local_experts": {"scalar": num_experts},
        },
        "outputs": {
            "multihot_indices": {
                "shape": [num_tokens, num_experts],
                "dtype": "torch.bool",
            },
            "probs_in_multihot": {
                "shape": [num_tokens, num_experts],
                "dtype": str(dtype),
            },
        },
        "dims": {
            "num_tokens": num_tokens,
            "top_k": topk,
            "num_experts": num_experts,
        },
    }
