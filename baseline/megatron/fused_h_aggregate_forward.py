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

"""NVIDIA baseline for fused_h_aggregate_forward.

Self-contained module; depends only on torch and megatron, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns Megatron-LM's
``native_h_aggregate`` (``megatron.core.transformer.hyper_connection``) -- the
reference implementation ``FlagTrain/src/flag_train/megatron/
fused_h_aggregate_forward.py`` ports -- and raises when it cannot be imported
instead of falling back to a torch composition.

Shapes, dtypes and input construction are copied from
``FlagTrain/benchmark/megatron/test_fused_h_aggregate_forward.py``.

``native_h_aggregate`` carries ``@torch.compile`` upstream. The collector
profiles native kernels rather than Inductor output, so the decorator is
unwrapped back to the eager function (``_torchdynamo_orig_callable``) -- the
same body the benchmark's correctness suite compares against. The benchmark
keeps the compile because that is what Megatron pays; this module measures the
eager implementation underneath it, like ``quick_gelu.py`` and
``fused_pad_routing_map.py``.

``native_h_aggregate`` arrived with the mHC stages in megatron-core 0.20; an
older install has no ``hyper_connection`` module and is reported as such rather
than being replaced by a transcription.
"""

import torch

OP_NAME = "fused_h_aggregate_forward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_fused_h_aggregate_forward.py::_BENCHMARK_SHAPES
# (s, b, n, C): x is [s, b, n, C], h_pre is [s, b, n], out is [s, b, C].
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


def _megatron_native_h_aggregate():
    """Megatron-LM's eager ``native_h_aggregate``; raise when unavailable."""
    global _MEGATRON_NATIVE_H_AGGREGATE
    if _MEGATRON_NATIVE_H_AGGREGATE is None:
        try:
            from megatron.core.transformer.hyper_connection import native_h_aggregate
        except Exception as exc:
            raise RuntimeError(
                "fused_h_aggregate_forward baseline requires Megatron-LM's "
                "megatron.core.transformer.hyper_connection.native_h_aggregate "
                "(megatron-core >= 0.20), which could not be imported: "
                f"{exc!r}. Install megatron-core from its main branch on this "
                "machine."
            ) from exc
        # Upstream decorates the function with @torch.compile directly, so
        # disable_jit_fuser() cannot help; unwrap the original eager callable
        # the collector should profile instead of Inductor output.
        _MEGATRON_NATIVE_H_AGGREGATE = getattr(
            native_h_aggregate, "_torchdynamo_orig_callable", native_h_aggregate
        )
    return _MEGATRON_NATIVE_H_AGGREGATE


def torch_op(x, h_pre):
    """Megatron's eager ``native_h_aggregate``; raises without Megatron."""
    return _megatron_native_h_aggregate()(x, h_pre)


def native():
    # Resolve eagerly so an unavailable Megatron fails here instead of being
    # discovered mid-collection.
    _megatron_native_h_aggregate()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def _make_operands(s, b, n, C, dtype, device):
    """``(x, h_pre)`` exactly as the benchmark builds them."""
    x = torch.randn((s, b, n, C), dtype=dtype, device=device)
    h_pre = torch.randn((s, b, n), dtype=dtype, device=device)
    return x, h_pre


def build_inputs(binding, dtype, device):
    s, b, n, C = binding["shape"]
    x, h_pre = _make_operands(s, b, n, C, dtype, device)
    return (x, h_pre), {}


def key_shape(binding):
    # The first tensor is [s, b, n, C], which determines the whole shape.
    return "x".join(str(dim) for dim in binding["shape"])


def config(binding, dtype):
    s, b, n, C = binding["shape"]
    return {
        "inputs": {
            "x": {"shape": [s, b, n, C], "dtype": str(dtype)},
            "h_pre": {"shape": [s, b, n], "dtype": str(dtype)},
        },
        "dims": {"s": s, "b": b, "n": n, "C": C},
    }
