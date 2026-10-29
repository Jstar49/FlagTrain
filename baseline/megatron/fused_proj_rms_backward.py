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

"""NVIDIA baseline for fused_proj_rms_backward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. The baseline is ``torch.autograd.grad``
through Megatron-LM's ``native_proj_rms``
(``megatron.core.transformer.hyper_connection``) -- the reference
``FlagTrain/src/flag_train/megatron/fused_proj_rms_backward.py``
differentiates -- and raises when it cannot be imported instead of falling
back to a torch composition.

Shapes, dtypes and input construction are copied from
``FlagTrain/benchmark/megatron/test_fused_proj_rms_backward.py``. As in that
benchmark, the forward graph is built outside the timed region and every
repetition only runs ``torch.autograd.grad``. ``build_inputs`` builds the graph
so the collector's NCU subprocess, which rebuilds the operands right before the
profiled call, also keeps the forward kernels out of the NVTX range. The
operator's saved scale ``r`` comes from the reference forward; the baseline's
own graph recomputes its scale from ``x``, exactly as the benchmark's baseline
does.

``native_proj_rms`` carries ``@torch.compile`` upstream. The collector profiles
native kernels rather than Inductor output, so the decorator is unwrapped back
to the eager function (``_torchdynamo_orig_callable``) and the eager autograd
derivative is measured. The benchmark keeps the compile (with buffer donation
off, so the graph can be re-run) because that is what Megatron pays; the eager
derivative is measurably slower, so this record is the conservative one.

``native_proj_rms`` arrived with the mHC proj_rms stage in megatron-core 0.20;
an older install has no ``hyper_connection`` module and is reported as such
rather than being replaced by a transcription.
"""

import math

import torch

OP_NAME = "fused_proj_rms_backward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fused_proj_rms_backward.py::_BENCHMARK_SHAPES
# (M, N, K): x is [M, K], weight is [N, K], grad_proj is [M, N], grad_r is
# [M, 1], r is [M, 1]; the outputs are grad_x [M, K] and grad_weight [N, K].
_SHAPES = [
    (1, 24, 4096),
    (128, 24, 4096),
    (256, 24, 4096),
    (1024, 24, 4096),
    (4096, 24, 4096),
    (8192, 24, 8192),
    (4096, 80, 4096),
    (4096, 288, 4096),
    (4096, 1024, 4096),
    (8192, 4096, 8192),
]

_EPS = 1e-6

_MEGATRON_NATIVE_PROJ_RMS = None
_GRAPH = None


def _megatron_native_proj_rms():
    """Megatron-LM's eager ``native_proj_rms``; raise when it is unavailable."""
    global _MEGATRON_NATIVE_PROJ_RMS
    if _MEGATRON_NATIVE_PROJ_RMS is None:
        try:
            from megatron.core.transformer.hyper_connection import native_proj_rms
        except Exception as exc:
            raise RuntimeError(
                "fused_proj_rms_backward baseline requires Megatron-LM's "
                "megatron.core.transformer.hyper_connection.native_proj_rms "
                "(megatron-core >= 0.20), which could not be imported: "
                f"{exc!r}. Install megatron-core from its main branch on this "
                "machine."
            ) from exc
        # Upstream decorates the function with @torch.compile directly, so
        # disable_jit_fuser() cannot help; unwrap the original eager callable
        # the collector should profile instead of Inductor output.
        _MEGATRON_NATIVE_PROJ_RMS = getattr(
            native_proj_rms, "_torchdynamo_orig_callable", native_proj_rms
        )
    return _MEGATRON_NATIVE_PROJ_RMS


def _build_graph(x, weight):
    """Megatron's forward graph for one input set, built outside the timing.

    The leaves are copies because the harness reuses the tensors it hands to
    the operator under test, and ``requires_grad_`` on them would make that
    call build a graph it never needs.
    """
    global _GRAPH
    x_leaf = x.detach().clone().requires_grad_(True)
    weight_leaf = weight.detach().clone().requires_grad_(True)
    proj, r = _megatron_native_proj_rms()(x_leaf, weight_leaf, _EPS)
    _GRAPH = {
        "x": x,
        "weight": weight,
        "x_leaf": x_leaf,
        "weight_leaf": weight_leaf,
        "proj": proj,
        "r": r,
    }
    return _GRAPH


def torch_op(x, weight, grad_proj, grad_r, r):
    """Megatron's autograd backward of ``native_proj_rms``.

    ``r`` is the operator's saved forward scale; this baseline recomputes its
    own from ``x`` inside the graph, as the benchmark's baseline does.
    """
    graph = _GRAPH
    if graph is None or graph["x"] is not x or graph["weight"] is not weight:
        graph = _build_graph(x, weight)
    return torch.autograd.grad(
        (graph["proj"], graph["r"]),
        (graph["x_leaf"], graph["weight_leaf"]),
        (grad_proj, grad_r),
        retain_graph=True,
    )


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_native_proj_rms()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    M, N, K = binding["shape"]
    x = torch.randn((M, K), dtype=dtype, device=device)
    # Megatron initialises the mapping with unit-variance rows scaled by
    # 1/sqrt(K), which keeps the products O(1) at every K; a raw randn weight
    # would make them grow with K and the comparison dtype-dependent.
    weight = torch.randn((N, K), dtype=torch.float32, device=device)
    weight = (weight / math.sqrt(max(K, 1))).to(dtype)
    grad_proj = torch.randn((M, N), dtype=dtype, device=device)
    grad_r = torch.randn((M, 1), dtype=dtype, device=device)
    # Forward graph and saved scale, both outside the timed/profiled region.
    r = _build_graph(x, weight)["r"]
    return (x, weight, grad_proj, grad_r, r), {}


def key_shape(binding):
    # The first tensor is [M, K], which cannot tell (M, N, K) apart: N lives
    # only in weight's and grad_proj's shapes. Use a semantic key so lookup
    # falls back to the tensor-signature match config() describes.
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    M, N, K = binding["shape"]
    return {
        "inputs": {
            "x": {"shape": [M, K], "dtype": str(dtype)},
            "weight": {"shape": [N, K], "dtype": str(dtype)},
            "grad_proj": {"shape": [M, N], "dtype": str(dtype)},
            "grad_r": {"shape": [M, 1], "dtype": str(dtype)},
            "r": {"shape": [M, 1], "dtype": str(dtype)},
        },
        "dims": {"M": M, "N": N, "K": K},
    }
