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

"""NVIDIA baseline for multi_tensor_adam.

Self-contained like ``ops/vllm/``: depends only on torch and deepspeed, and does
not import ``flag_train`` or ``benchmark``. ``native()`` returns DeepSpeed's
``multi_tensor_adam``; when DeepSpeed is unavailable it raises instead of
collecting a baseline or falling back to a torch reference. DeepSpeed is loaded
lazily on the first call, so ops discovery does not trigger its JIT.

The input construction follows
``FlagTrain/benchmark/deepspeed/test_multi_tensor_adam.py::multi_tensor_adam_input_fn``.
"""

import torch

OP_NAME = "multi_tensor_adam"
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
IS_INPLACE = True

# benchmark/deepspeed/test_multi_tensor_adam.py::_MULTI_TENSOR_ADAM_SHAPES
_SHAPES = [
    (16, 1024),
    (16, 4096),
    (16, 16384),
    (16, 65536),
    (16, 262144),
    (16, 1048576),
]

_LR = 1e-2
_BETA1 = 0.9
_BETA2 = 0.999
_EPS = 1e-8
_STEP = 1
_MODE = 1  # decoupled weight decay (AdamW)
_BIAS_CORRECTION = 1
_WEIGHT_DECAY = 0.1
_CHUNK_SIZE = 2048 * 32

_DEEPSPEED_ADAM = None


def _deepspeed_adam():
    """DeepSpeed's multi_tensor_adam; raise when it cannot be loaded."""
    global _DEEPSPEED_ADAM
    if _DEEPSPEED_ADAM is None:
        try:
            from deepspeed.ops.op_builder import FusedAdamBuilder

            _DEEPSPEED_ADAM = FusedAdamBuilder().load().multi_tensor_adam
        except Exception as exc:
            raise RuntimeError(
                "multi_tensor_adam baseline requires DeepSpeed's FusedAdam, "
                f"which could not be loaded: {exc!r}. Install deepspeed and "
                "ensure its CUDA extension builds on this machine."
            ) from exc
    return _DEEPSPEED_ADAM


_NOOP_FLAG = None


def _noop_flag():
    global _NOOP_FLAG
    if _NOOP_FLAG is None:
        _NOOP_FLAG = torch.zeros(1, dtype=torch.int32)
    return _NOOP_FLAG


def _call(op, gs, ps, ms, vs):
    return op(
        _CHUNK_SIZE,
        _noop_flag(),
        [gs, ps, ms, vs],
        _LR,
        _BETA1,
        _BETA2,
        _EPS,
        _STEP,
        _MODE,
        _BIAS_CORRECTION,
        _WEIGHT_DECAY,
    )


def native():
    """DeepSpeed FusedAdam; raises when it cannot be loaded."""
    adam = _deepspeed_adam()

    def torch_op(gs, ps, ms, vs):
        return _call(adam, gs, ps, ms, vs)

    return torch_op


def grid():
    return [{"n_tensors": n, "numel": u} for n, u in _SHAPES]


def build_inputs(binding, dtype, device):
    n = binding["n_tensors"]
    numel = binding["numel"]
    gs = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n)]
    ps = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n)]
    ms = [torch.zeros((numel,), dtype=dtype, device=device) for _ in range(n)]
    vs = [torch.zeros((numel,), dtype=dtype, device=device) for _ in range(n)]
    return (gs, ps, ms, vs), {}


def key_shape(binding):
    return f"n{binding['n_tensors']}_numel{binding['numel']}"


def config(binding, dtype):
    n = binding["n_tensors"]
    numel = binding["numel"]
    inputs = {}
    for prefix in ("gs", "ps", "ms", "vs"):
        for index in range(n):
            inputs[f"{prefix}_{index}"] = {
                "shape": [numel],
                "dtype": str(dtype),
            }
    return {"inputs": inputs, "dims": {"n_tensors": n, "numel": numel}}
