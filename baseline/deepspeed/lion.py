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

"""NVIDIA baseline for multi_tensor_lion.

Self-contained like ``ops/vllm/``: depends only on torch and deepspeed, and does
not import ``flag_train`` or ``benchmark``. ``native()`` returns DeepSpeed's
``multi_tensor_lion``; when DeepSpeed is unavailable it raises instead of
collecting a baseline or falling back to a torch reference. DeepSpeed is loaded
lazily on the first call, so ops discovery does not trigger its JIT.

The input construction follows
``FlagTrain/benchmark/deepspeed/test_lion.py::lion_input_fn``: ``gs``/``ps``/``ms``
each hold ``n_tensors`` tensors of length ``numel``.
"""

import torch

OP_NAME = "multi_tensor_lion"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
IS_INPLACE = True

# benchmark/deepspeed/test_lion.py::_LION_SHAPES
_SHAPES = [
    (8, 1024),
    (64, 1024),
    (256, 1024),
    (1024, 1024),
    (8, 1048576),
    (64, 1048576),
    (256, 1048576),
]

_LR = 1e-4
_BETA1 = 0.9
_BETA2 = 0.99
_WEIGHT_DECAY = 0.01
_CHUNK_SIZE = 2048
_STEP = 1


_DEEPSPEED_LION = None


def _deepspeed_lion():
    """DeepSpeed's multi_tensor_lion; raise when it cannot be loaded."""
    global _DEEPSPEED_LION
    if _DEEPSPEED_LION is None:
        try:
            from deepspeed.ops.op_builder import FusedLionBuilder

            _DEEPSPEED_LION = FusedLionBuilder().load().multi_tensor_lion
        except Exception as exc:
            raise RuntimeError(
                "multi_tensor_lion baseline requires DeepSpeed's FusedLion, "
                f"which could not be loaded: {exc!r}. Install deepspeed and "
                "ensure its CUDA extension builds on this machine."
            ) from exc
    return _DEEPSPEED_LION


_NOOP_FLAG = None


def _noop_flag():
    global _NOOP_FLAG
    if _NOOP_FLAG is None:
        _NOOP_FLAG = torch.zeros(1, dtype=torch.int32)
    return _NOOP_FLAG


def _call(op, gs, ps, ms):
    return op(
        _CHUNK_SIZE,
        _noop_flag(),
        [gs, ps, ms],
        _LR,
        _BETA1,
        _BETA2,
        _STEP,
        _WEIGHT_DECAY,
    )


def native():
    """DeepSpeed FusedLion; raises when it cannot be loaded."""
    lion = _deepspeed_lion()

    def torch_op(gs, ps, ms):
        return _call(lion, gs, ps, ms)

    return torch_op


def grid():
    return [{"n_tensors": n, "numel": u} for n, u in _SHAPES]


def build_inputs(binding, dtype, device):
    n = binding["n_tensors"]
    numel = binding["numel"]
    gs = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n)]
    ps = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n)]
    ms = [torch.zeros((numel,), dtype=dtype, device=device) for _ in range(n)]
    return (gs, ps, ms), {}


def key_shape(binding):
    return f"n{binding['n_tensors']}_numel{binding['numel']}"


def config(binding, dtype):
    n = binding["n_tensors"]
    numel = binding["numel"]
    inputs = {}
    for prefix in ("gs", "ps", "ms"):
        for index in range(n):
            inputs[f"{prefix}_{index}"] = {
                "shape": [numel],
                "dtype": str(dtype),
            }
    return {"inputs": inputs, "dims": {"n_tensors": n, "numel": numel}}
