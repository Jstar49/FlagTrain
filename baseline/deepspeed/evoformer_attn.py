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

"""NVIDIA baseline for evoformer_attn.

Self-contained module; depends only on torch and deepspeed. ``native()`` returns
``DS4Sci_EvoformerAttention``; when DeepSpeed is unavailable it raises instead
of collecting a baseline or falling back to a torch composition. The input
construction is copied from
``FlagTrain/benchmark/deepspeed/test_evoformer_attn.py``.

DeepSpeed is loaded lazily on the first call, so importing the module does not
trigger its JIT (the JIT would prepend the directory containing
``evoformer_attn.so`` to ``sys.path`` and could shadow an ops module of the same
name).
"""

import torch

OP_NAME = "evoformer_attn"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

# benchmark/deepspeed/test_evoformer_attn.py::_EVOFORMER_SHAPES
_SHAPES = [
    (1, 32, 128, 4, 32),
    (1, 128, 256, 4, 32),
    (1, 256, 256, 4, 32),
    (1, 512, 256, 8, 8),
    (2, 64, 256, 8, 32),
    (1, 64, 384, 8, 64),
    (1, 1024, 256, 4, 32),
    (1, 512, 64, 4, 32),
    (1, 8, 1024, 4, 32),
    (1, 64, 64, 8, 64),
    (1, 2048, 256, 4, 32),
    (1, 512, 384, 8, 32),
    (1, 4, 2048, 4, 32),
]


_DEEPSPEED_EVOFORMER = None


def _load_deepspeed_evoformer():
    try:
        from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
        from deepspeed.ops.op_builder import EvoformerAttnBuilder

        EvoformerAttnBuilder().load()
        return DS4Sci_EvoformerAttention
    except Exception as exc:
        raise RuntimeError(
            "evoformer_attn baseline requires DeepSpeed's "
            f"DS4Sci_EvoformerAttention, which could not be loaded: {exc!r}. "
            "Build deepspeed with CUTLASS >= 3.1 (`pip install nvidia-cutlass`)."
        ) from exc


def _deepspeed_evoformer():
    global _DEEPSPEED_EVOFORMER
    if _DEEPSPEED_EVOFORMER is None:
        _DEEPSPEED_EVOFORMER = _load_deepspeed_evoformer()
    return _DEEPSPEED_EVOFORMER


def torch_op(q, k, v, bias1, bias2):
    """DeepSpeed Evoformer attention; raises when it cannot be loaded."""
    # A fresh list per call: upstream's entry appends to the one it is handed.
    return _deepspeed_evoformer()(q, k, v, [bias1, bias2])


def native():
    # Load eagerly so an unavailable DeepSpeed fails here instead of being
    # discovered mid-collection (and silently skipped).
    _deepspeed_evoformer()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    batch, pairs, seq_len, heads, head_size = binding["shape"]

    generator = torch.Generator(device="cpu").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(
            dtype=dtype, device=device
        )

    q = randn(batch, pairs, seq_len, heads, head_size)
    k = randn(batch, pairs, seq_len, heads, head_size)
    v = randn(batch, pairs, seq_len, heads, head_size)

    mask = torch.randint(0, 2, (batch, pairs, 1, 1, seq_len), generator=generator).to(
        device=device
    )
    bias1 = (1e9 * (mask - 1)).to(dtype)
    bias2 = randn(batch, 1, heads, seq_len, seq_len)

    return (q, k, v, bias1, bias2), {}


def key_shape(binding):
    # First tensor: q [batch, pairs, seq_len, heads, head_size]; unique.
    return list(binding["shape"])
