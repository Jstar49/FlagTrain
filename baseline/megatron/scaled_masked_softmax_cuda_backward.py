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

"""NVIDIA baseline for scaled_masked_softmax_cuda_backward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``.

``native()`` returns Megatron-LM's ``ScaledMaskedSoftmax.backward``
(``megatron/core/fusions/fused_softmax.py``) -- the operator
``FlagTrain/src/flag_train/megatron/scaled_masked_softmax_cuda_backward.py``
ports -- whose whole body is one call into Megatron-LM's Apex CUDA extension
``scaled_masked_softmax_cuda``::

    softmax_results, scale_t = ctx.saved_tensors
    return scaled_masked_softmax_cuda.backward(
        output_grads, softmax_results, scale_t[0]
    ), None, None

The baseline is Megatron's ``ScaledMaskedSoftmax`` when megatron-core provides
it, and otherwise the same body vendored below against the same extension. The
Apex extension is a hard dependency. Without it this module raises instead of
measuring the benchmark's ``masked_backward_ref`` torch composition, because the
record is meant to describe Megatron's fused kernel.

The operands are passed directly rather than obtained by differentiating a
forward: the operator is the backward kernel. Its output is ``input_grads``;
upstream's ``None`` placeholders for ``mask`` and ``scale`` are dropped.
``ctx`` is stood in for by ``_SavedForBackward``, which holds the
``(softmax_results, scale_t)`` pair the upstream forward saves, so no forward
pass lands in the timed region.

Shapes, dtypes and the input construction are copied from
``FlagTrain/benchmark/megatron/test_scaled_masked_softmax_cuda_backward.py``
(``_SHAPES`` and ``masked_backward_input_fn``).
"""

import importlib.util
import warnings

import torch

OP_NAME = "scaled_masked_softmax_cuda_backward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_scaled_masked_softmax_cuda_backward.py::_SHAPES
# [b, np, sq, sk] attention gradients.
_SHAPES = [
    (8, 16, 128, 128),
    (8, 16, 256, 256),
    (8, 16, 512, 512),
    (8, 16, 1024, 1024),
    (4, 16, 2048, 2048),
    (4, 16, 4096, 4096),
]

_SCALE = 0.125
_EXTENSION = "scaled_masked_softmax_cuda"


def _require_extension():
    if importlib.util.find_spec(_EXTENSION) is None:
        raise RuntimeError(
            "scaled_masked_softmax_cuda_backward baseline requires Megatron-LM's "
            f"Apex CUDA extension '{_EXTENSION}', which is not installed. The "
            "record describes that fused kernel, so this module does not fall "
            "back to the benchmark's masked_backward_ref (torch) composition. "
            "Install Megatron-LM's Apex extensions on this machine."
        )


class _SavedForBackward:
    """Stand-in for the context ``ScaledMaskedSoftmax.backward`` takes.

    Megatron's forward saves ``(softmax_results, scale_t)``; the backward only
    reads ``ctx.saved_tensors``, so a two-attribute object runs the copied body
    -- and the installed one -- without a forward pass in the timed region.
    """

    def __init__(self, softmax_results, scale):
        self.saved_tensors = (softmax_results, torch.tensor([scale]))


def _vendored_backward(output_grads, softmax_results, scale):
    """``ScaledMaskedSoftmax.backward``'s body, when megatron-core lacks it.

    Upstream answers one value per forward input -- ``(input_grads, None,
    None)``, the ``None``s standing for ``mask`` and ``scale``, which have no
    gradient. Here only the gradient is the operator's output.
    """
    import scaled_masked_softmax_cuda

    softmax_results, scale_t = _SavedForBackward(softmax_results, scale).saved_tensors
    return scaled_masked_softmax_cuda.backward(
        output_grads, softmax_results, scale_t[0]
    )


_SCALED_MASKED_SOFTMAX = None
_NATIVE = None


def _class_backward(output_grads, softmax_results, scale):
    """``ScaledMaskedSoftmax.backward`` behind the saved-tensor context.

    The gradient is the first of the three answers; the other two are ``None``
    for the forward's ``mask`` and ``scale`` arguments.
    """
    return _SCALED_MASKED_SOFTMAX.backward(
        _SavedForBackward(softmax_results, scale), output_grads
    )[0]


def _megatron_backward():
    """Megatron's baseline backward, resolved once on first use.

    Imported lazily: ``megatron.core``'s ``__init__`` pulls in the whole core
    tree, which the ops-discovery pass must not trigger.
    """
    global _NATIVE, _SCALED_MASKED_SOFTMAX
    if _NATIVE is None:
        _require_extension()
        try:
            from megatron.core.fusions.fused_softmax import ScaledMaskedSoftmax
        except Exception as exc:  # megatron-core absent, or without the class
            warnings.warn(
                "megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax could "
                f"not be imported ({exc!r}); using the vendored body against "
                "the same Apex extension.",
                RuntimeWarning,
                stacklevel=2,
            )
            _NATIVE = _vendored_backward
        else:
            _SCALED_MASKED_SOFTMAX = ScaledMaskedSoftmax
            _NATIVE = _class_backward
    return _NATIVE


def torch_op(output_grads, softmax_results, scale):
    """Megatron's scaled masked softmax gradient, on the Apex CUDA kernel."""
    return _megatron_backward()(output_grads, softmax_results, scale)


def native():
    # Resolve eagerly so an unavailable extension or megatron-core fails here
    # instead of being discovered mid-collection.
    _megatron_backward()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    """``(output_grads, softmax_results, scale)`` as the benchmark builds them."""
    shape = binding["shape"]
    # A plausible forward output: non-negative rows that sum to one.
    softmax_results = torch.rand(shape, dtype=dtype, device=device)
    softmax_results = softmax_results / softmax_results.sum(dim=-1, keepdim=True)
    output_grads = torch.randn(shape, dtype=dtype, device=device)
    return (output_grads, softmax_results, _SCALE), {}


def key_shape(binding):
    # The first tensor's shape determines the operator; everything else follows.
    return list(binding["shape"])


def config(binding, dtype):
    b, np_, sq, sk = binding["shape"]
    return {
        "inputs": {
            "output_grads": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "softmax_results": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "scale": {"scalar": _SCALE},
        },
        "dims": {"b": b, "np": np_, "sq": sq, "sk": sk},
    }
