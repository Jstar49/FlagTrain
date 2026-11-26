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

"""NVIDIA baseline for fused_proj_rms_forward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns Megatron-LM's
``native_proj_rms`` (``megatron.core.transformer.hyper_connection``) -- the
reference implementation ``FlagTrain/src/flag_train/megatron/
fused_proj_rms_forward.py`` ports -- and raises when it cannot be imported
instead of falling back to a torch composition.

Shapes, dtypes and input construction are copied from
``FlagTrain/benchmark/megatron/test_fused_proj_rms_forward.py``.

``native_proj_rms`` carries ``@torch.compile`` upstream. The collector keeps
that compiled callable, matching what Megatron pays by default.

``native_proj_rms`` arrived with the mHC proj_rms stage in megatron-core 0.20;
an older install has no ``hyper_connection`` module and is reported as such
rather than being replaced by a transcription.
"""

import math

import torch
from torch._dynamo import config as dynamo_config

OP_NAME = "fused_proj_rms_forward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fused_proj_rms_forward.py::_BENCHMARK_SHAPES
# (M, N, K): x is [M, K], weight is [N, K], proj is [M, N], r is [M, 1].
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


def _megatron_native_proj_rms():
    """Megatron-LM's ``native_proj_rms``; raise when it is unavailable."""
    global _MEGATRON_NATIVE_PROJ_RMS
    if _MEGATRON_NATIVE_PROJ_RMS is None:
        try:
            from megatron.core.transformer.hyper_connection import native_proj_rms
        except Exception as exc:
            raise RuntimeError(
                "fused_proj_rms_forward baseline requires Megatron-LM's "
                "megatron.core.transformer.hyper_connection.native_proj_rms "
                "(megatron-core >= 0.20), which could not be imported: "
                f"{exc!r}. Install megatron-core from its main branch on this "
                "machine."
            ) from exc
        _MEGATRON_NATIVE_PROJ_RMS = native_proj_rms
    return _MEGATRON_NATIVE_PROJ_RMS


def torch_op(x, weight, eps):
    """Megatron's ``native_proj_rms``; raises without Megatron."""
    return _megatron_native_proj_rms()(x, weight, eps)


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_native_proj_rms()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def _make_operands(M, N, K, dtype, device):
    """``(x, weight)`` exactly as the benchmark builds them."""
    x = torch.randn((M, K), dtype=dtype, device=device)
    # Megatron initialises the mapping with unit-variance rows scaled by
    # 1/sqrt(K), which keeps proj O(1) at every K; a raw randn weight would
    # make the projection grow with K and the comparison dtype-dependent.
    weight = torch.randn((N, K), dtype=torch.float32, device=device)
    weight = (weight / math.sqrt(max(K, 1))).to(dtype)
    return x, weight


def build_inputs(binding, dtype, device):
    x, weight = _make_operands(*binding["shape"], dtype, device)
    args = (x, weight, _EPS)
    # Compile this shape/dtype before the collector starts timing.
    with dynamo_config.patch(
        recompile_limit=len(grid()) * len(DTYPES) + 8,
        fail_on_recompile_limit_hit=True,
    ):
        torch_op(*args)
    return args, {}


def key_shape(binding):
    # The first tensor is [M, K], which cannot tell (M, N, K) apart: N lives
    # only in weight's shape. Use a semantic key so lookup falls back to the
    # tensor-signature match config() describes.
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    M, N, K = binding["shape"]
    return {
        "inputs": {
            "x": {"shape": [M, K], "dtype": str(dtype)},
            "weight": {"shape": [N, K], "dtype": str(dtype)},
            "eps": {"scalar": _EPS},
        },
        "dims": {"M": M, "N": N, "K": K},
    }
