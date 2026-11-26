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

"""NVIDIA baseline for fused_h_aggregate_backward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns ``torch.autograd.grad``
through Megatron-LM's ``native_h_aggregate``
(``megatron.core.transformer.hyper_connection``) -- the reference whose
derivative ``FlagTrain/src/flag_train/megatron/fused_h_aggregate_backward.py``
is -- and raises when it cannot be imported instead of falling back to a torch
composition.

Shapes, dtypes and input construction are copied from
``FlagTrain/benchmark/megatron/test_fused_h_aggregate_backward.py``.

``native_h_aggregate`` carries ``@torch.compile`` upstream. The collector keeps
that compiled callable, matching what Megatron pays by default.

Only the backward is timed: the forward graph is built outside the profiled
region, as the benchmark's baseline does, because the operator under test takes
the forward's saved ``x`` and ``h_pre`` rather than recomputing them.

``native_h_aggregate`` arrived with the mHC stages in megatron-core 0.20; an
older install has no ``hyper_connection`` module and is reported as such rather
than being replaced by a transcription.
"""

import torch
import torch._functorch.config
from torch._dynamo import config as dynamo_config

# The collector re-runs the same backward graph for every timed repetition
# (retain_graph=True). Compiled AOTAutograd backward donates its saved buffers
# by default and then refuses retain_graph=True, so disable donation here, as
# the benchmark does.
torch._functorch.config.donated_buffer = False

# The collector profiles the backward under an NVTX range, but
# torch.autograd.grad runs its backward kernels on autograd worker threads,
# which do not inherit that thread-local range. Keep autograd on the calling
# thread so Nsight Compute profiles the whole backward instead of nothing.
try:
    torch.autograd.set_multithreading_enabled(False)
except AttributeError:
    pass

OP_NAME = "fused_h_aggregate_backward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fused_h_aggregate_backward.py::_BENCHMARK_SHAPES
# (s, b, n, C): x is [s, b, n, C], h_pre is [s, b, n], grad_output is [s, b, C];
# the gradients are [s, b, n, C] and [s, b, n].
_SHAPES = [
    (1, 1, 2, 256),
    (1, 1, 4, 4096),
    (8, 16, 4, 4096),
    (64, 8, 4, 4096),
    (32, 32, 4, 4096),
    (128, 32, 4, 4096),
    (256, 32, 4, 4096),
    (256, 32, 2, 4096),
    (256, 32, 8, 4096),
    (256, 32, 4, 7168),
]

_MEGATRON_NATIVE_H_AGGREGATE = None
_GRAPH = None


def _megatron_native_h_aggregate():
    """Megatron-LM's ``native_h_aggregate``; raise when unavailable."""
    global _MEGATRON_NATIVE_H_AGGREGATE
    if _MEGATRON_NATIVE_H_AGGREGATE is None:
        try:
            from megatron.core.transformer.hyper_connection import native_h_aggregate
        except Exception as exc:
            raise RuntimeError(
                "fused_h_aggregate_backward baseline requires Megatron-LM's "
                "megatron.core.transformer.hyper_connection.native_h_aggregate "
                "(megatron-core >= 0.20), which could not be imported: "
                f"{exc!r}. Install megatron-core from its main branch on this "
                "machine."
            ) from exc
        _MEGATRON_NATIVE_H_AGGREGATE = native_h_aggregate
    return _MEGATRON_NATIVE_H_AGGREGATE


def _build_graph(x, h_pre):
    """Forward graph for one input set, keyed on the inputs' identity.

    The originals are held so a freed tensor's address cannot be handed to the
    next case's allocation and silently reuse this graph.
    """
    global _GRAPH
    x_leaf = x.detach().clone().requires_grad_(True)
    h_leaf = h_pre.detach().clone().requires_grad_(True)
    out = _megatron_native_h_aggregate()(x_leaf, h_leaf)
    _GRAPH = {
        "x": x,
        "h_pre": h_pre,
        "x_leaf": x_leaf,
        "h_leaf": h_leaf,
        "out": out,
    }
    return _GRAPH


def torch_op(grad_output, x, h_pre):
    """Megatron's autograd backward of ``native_h_aggregate``; raises without it."""
    graph = _GRAPH
    if graph is None or graph["x"] is not x or graph["h_pre"] is not h_pre:
        graph = _build_graph(x, h_pre)
    return torch.autograd.grad(
        graph["out"],
        (graph["x_leaf"], graph["h_leaf"]),
        grad_output,
        retain_graph=True,
    )


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_native_h_aggregate()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    s, b, n, C = binding["shape"]
    grad_output = torch.randn((s, b, C), dtype=dtype, device=device)
    x = torch.randn((s, b, n, C), dtype=dtype, device=device)
    h_pre = torch.randn((s, b, n), dtype=dtype, device=device)
    # Forward graph, outside the profiled region.
    _build_graph(x, h_pre)
    args = (grad_output, x, h_pre)
    # Compile this shape/dtype (forward and backward) before timing.
    with dynamo_config.patch(
        recompile_limit=len(grid()) * len(DTYPES) + 8,
        fail_on_recompile_limit_hit=True,
    ):
        torch_op(*args)
    return args, {}


def key_shape(binding):
    # The grad_output tensor is [s, b, C], which cannot tell n apart; x is
    # [s, b, n, C] and determines the whole shape.
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    s, b, n, C = binding["shape"]
    return {
        "inputs": {
            "grad_output": {"shape": [s, b, C], "dtype": str(dtype)},
            "x": {"shape": [s, b, n, C], "dtype": str(dtype)},
            "h_pre": {"shape": [s, b, n], "dtype": str(dtype)},
        },
        "dims": {"s": s, "b": b, "n": n, "C": C},
    }
