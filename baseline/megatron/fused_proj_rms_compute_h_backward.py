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

"""NVIDIA baseline for fused_proj_rms_compute_h_backward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. The baseline is ``torch.autograd.grad``
through Megatron-LM's ``megatron.core.fusions.fused_mhc_kernels.
fused_proj_rms_compute_h`` -- the public forward that
``FlagTrain/src/flag_train/megatron/fused_proj_rms_compute_h_backward.py``
differentiates. It raises when that module cannot be imported instead of
falling back to a torch composition.

Shapes, dtypes and input construction are copied from
``FlagTrain/benchmark/megatron/test_fused_proj_rms_compute_h_backward.py``.
As in that benchmark, the forward graph is built outside the timed region and
every repetition only runs ``torch.autograd.grad``. ``build_inputs`` builds the
graph so the collector's NCU subprocess, which rebuilds the operands right
before the profiled call, also keeps the forward kernels out of the NVTX
range.

Megatron 0.21 wraps the torch path of this forward in ``@torch.compile``. The
graph is built with Dynamo disabled, so the collector profiles the eager ATen
backward kernels rather than Inductor output; on a cuTile-capable install the
public forward dispatches to the cuTile kernel and its fused backward is what
``autograd.grad`` runs.
"""

import torch

OP_NAME = "fused_proj_rms_compute_h_backward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fused_proj_rms_compute_h_backward.py::_SHAPES
# (M, n, K): x is [M, K], weight is [N, K] with N = n * n + 2 * n.
_SHAPES = [
    (1024, 4, 16384),
    (4096, 4, 16384),
    (4096, 4, 28672),
    (8192, 4, 8192),
    (4096, 2, 8192),
    (16384, 4, 4096),
]

# mhc_init_gating_factor; the static bias is initialized to zero.
_GATING_FACTOR = 0.01

_MEGATRON_FORWARD = None
_GRAPH = None


def _megatron_forward():
    """Megatron-LM's fused_proj_rms_compute_h; raise when it is unavailable."""
    global _MEGATRON_FORWARD
    if _MEGATRON_FORWARD is None:
        try:
            from megatron.core.fusions.fused_mhc_kernels import fused_proj_rms_compute_h
        except Exception as exc:
            raise RuntimeError(
                "fused_proj_rms_compute_h_backward baseline requires "
                "Megatron-LM's megatron.core.fusions.fused_mhc_kernels, which "
                f"could not be imported: {exc!r}. Install a Megatron-LM build "
                "that ships the mHC kernels (megatron-core 0.19.2 does not)."
            ) from exc
        _MEGATRON_FORWARD = fused_proj_rms_compute_h
    return _MEGATRON_FORWARD


def _build_graph(x, weight, alpha_pre, alpha_post, alpha_res, bias, n):
    """Megatron's forward graph for one input set, built outside the timing.

    The leaves are copies because the harness reuses the tensors it hands to
    the operator under test, and ``requires_grad_`` on them would make that
    call build a graph it never needs.
    """
    global _GRAPH
    x_leaf = x.detach().clone().requires_grad_(True)
    weight_leaf = weight.detach().clone().requires_grad_(True)
    alpha_pre_leaf = alpha_pre.detach().clone().requires_grad_(True)
    alpha_post_leaf = alpha_post.detach().clone().requires_grad_(True)
    alpha_res_leaf = alpha_res.detach().clone().requires_grad_(True)
    bias_leaf = bias.detach().clone().requires_grad_(True)

    import torch._dynamo

    with torch._dynamo.config.patch(disable=True):
        h_pre, h_post, h_res, r = _megatron_forward()(
            x_leaf,
            weight_leaf,
            alpha_pre_leaf,
            alpha_post_leaf,
            alpha_res_leaf,
            bias_leaf,
            n,
        )
    _GRAPH = {
        "x": x,
        "weight": weight,
        "alpha_pre": alpha_pre,
        "alpha_post": alpha_post,
        "alpha_res": alpha_res,
        "bias": bias,
        "x_leaf": x_leaf,
        "weight_leaf": weight_leaf,
        "alpha_pre_leaf": alpha_pre_leaf,
        "alpha_post_leaf": alpha_post_leaf,
        "alpha_res_leaf": alpha_res_leaf,
        "bias_leaf": bias_leaf,
        "h_pre": h_pre,
        "h_post": h_post,
        "h_res": h_res,
        "r": r,
    }
    return _GRAPH


def torch_op(
    x,
    weight,
    grad_h_pre,
    grad_h_post,
    grad_h_res,
    h_pre,
    h_post,
    h_res,
    proj,
    r,
    grad_r_ext,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias,
    n,
):
    """Megatron's autograd backward of ``fused_proj_rms_compute_h``.

    The saved forward tensors in the argument tuple are informational; the
    autograd graph recomputes its own from ``x``/``weight``, exactly as the
    benchmark's baseline does.
    """
    graph = _GRAPH
    if (
        graph is None
        or graph["x"] is not x
        or graph["weight"] is not weight
        or graph["alpha_pre"] is not alpha_pre
        or graph["alpha_post"] is not alpha_post
        or graph["alpha_res"] is not alpha_res
        or graph["bias"] is not bias
    ):
        graph = _build_graph(x, weight, alpha_pre, alpha_post, alpha_res, bias, n)
    return torch.autograd.grad(
        (graph["h_pre"], graph["h_post"], graph["h_res"], graph["r"]),
        (
            graph["x_leaf"],
            graph["weight_leaf"],
            graph["alpha_pre_leaf"],
            graph["alpha_post_leaf"],
            graph["alpha_res_leaf"],
            graph["bias_leaf"],
        ),
        (grad_h_pre, grad_h_post, grad_h_res, grad_r_ext),
        retain_graph=True,
    )


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_forward()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    M, n, K = binding["shape"]
    N = n * n + 2 * n
    torch.manual_seed(0)
    x = torch.empty(M, K, dtype=dtype, device=device).uniform_(-0.1, 0.1)
    weight = torch.empty(N, K, dtype=torch.float32, device=device).uniform_(-0.1, 0.1)
    alpha = torch.full((1,), _GATING_FACTOR, dtype=torch.float32, device=device)
    bias = torch.zeros(N, dtype=torch.float32, device=device)

    # The mHC module keeps the mapping parameters in fp32, so the forward's
    # outputs (and their upstream gradients) are fp32 for every activation
    # dtype in DTYPES.
    matmul_dtype = torch.promote_types(dtype, weight.dtype)
    h_dtype = torch.promote_types(matmul_dtype, torch.float32)
    torch.manual_seed(1)
    grad_h_pre = torch.empty(M, n, dtype=h_dtype, device=device).uniform_(-0.1, 0.1)
    grad_h_post = torch.empty(M, n, dtype=h_dtype, device=device).uniform_(-0.1, 0.1)
    grad_h_res = torch.empty(M, N - 2 * n, dtype=h_dtype, device=device).uniform_(
        -0.1, 0.1
    )
    grad_r_ext = torch.empty(M, 1, dtype=matmul_dtype, device=device).uniform_(
        -0.1, 0.1
    )

    # Forward graph outside the timed/profiled region; its outputs are the
    # saved tensors the operator under test receives.
    graph = _build_graph(x, weight, alpha, alpha, alpha, bias, n)
    proj = torch.matmul(x.to(matmul_dtype), weight.to(matmul_dtype).t())
    # h_res is a slice of Megatron's [M, N] intermediate; the operator takes
    # the contiguous tensors the real forward returns.
    args = (
        x,
        weight,
        grad_h_pre,
        grad_h_post,
        grad_h_res,
        graph["h_pre"].contiguous(),
        graph["h_post"].contiguous(),
        graph["h_res"].contiguous(),
        proj,
        graph["r"].contiguous(),
        grad_r_ext,
        alpha,
        alpha,
        alpha,
        bias,
        n,
    )
    return args, {}


def key_shape(binding):
    # (M, n, K) is unambiguous: N is derived from n.
    return list(binding["shape"])


def config(binding, dtype):
    M, n, K = binding["shape"]
    N = n * n + 2 * n
    matmul_dtype = str(torch.promote_types(dtype, torch.float32))
    h_dtype = str(torch.promote_types(dtype, torch.float32))
    return {
        "inputs": {
            "x": {"shape": [M, K], "dtype": str(dtype)},
            "weight": {"shape": [N, K], "dtype": "torch.float32"},
            "grad_h_pre": {"shape": [M, n], "dtype": h_dtype},
            "grad_h_post": {"shape": [M, n], "dtype": h_dtype},
            "grad_h_res": {"shape": [M, N - 2 * n], "dtype": h_dtype},
            "h_pre": {"shape": [M, n], "dtype": h_dtype},
            "h_post": {"shape": [M, n], "dtype": h_dtype},
            "h_res": {"shape": [M, N - 2 * n], "dtype": h_dtype},
            "proj": {"shape": [M, N], "dtype": matmul_dtype},
            "r": {"shape": [M, 1], "dtype": matmul_dtype},
            "grad_r_ext": {"shape": [M, 1], "dtype": matmul_dtype},
            # build_inputs passes each alpha the way the benchmark does, as a
            # torch.full((1,)) tensor. The lookup matches the runtime tensor
            # signature exactly, so they are shapes here, not scalars.
            "alpha_pre": {"shape": [1], "dtype": "torch.float32"},
            "alpha_post": {"shape": [1], "dtype": "torch.float32"},
            "alpha_res": {"shape": [1], "dtype": "torch.float32"},
            "bias": {"shape": [N], "dtype": "torch.float32"},
            "n": {"scalar": n},
        },
        "outputs": {
            "grad_x": {"shape": [M, K], "dtype": str(dtype)},
            "grad_weight": {"shape": [N, K], "dtype": "torch.float32"},
            "grad_alpha_pre": {"shape": [1], "dtype": "torch.float32"},
            "grad_alpha_post": {"shape": [1], "dtype": "torch.float32"},
            "grad_alpha_res": {"shape": [1], "dtype": "torch.float32"},
            "grad_bias": {"shape": [N], "dtype": "torch.float32"},
        },
        "dims": {"M": M, "n": n, "K": K, "N": N},
    }
