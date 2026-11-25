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

"""NVIDIA backward baseline using Megatron-LM's MTPLossAutoScaler."""

from functools import lru_cache

import torch

OP_NAME = "mtp_loss_auto_scaler"
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
IS_INPLACE = False

_CASES = (
    ((2, 17), 0.375),
    ((128, 256), 0.375),
    ((1, 1), 0.375),
    ((257, 127), 0.375),
    ((128, 1024), 0.375),
    ((2, 256, 4096), 0.375),
    ((3, 19), 0.0),
    ((129, 257), 2.0),
)


@lru_cache(maxsize=1)
def _upstream():
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    return MTPLossAutoScaler


def _backward(result, output, mtp_loss, grad_output, scale):
    scaler = _upstream()
    previous_scale = scaler.main_loss_backward_scale
    scaler.set_loss_scale(scale)
    try:
        return torch.autograd.grad(
            result, (output, mtp_loss), grad_output, retain_graph=True
        )
    finally:
        scaler.set_loss_scale(previous_scale)


def native():
    _upstream()
    return _backward


def grid():
    return [{"shape": shape, "loss_scale": scale} for shape, scale in _CASES]


def build_inputs(binding, dtype, device):
    with torch.enable_grad():
        output = torch.randn(
            binding["shape"], dtype=dtype, device=device, requires_grad=True
        )
        mtp_loss = torch.tensor(
            2.0, dtype=torch.float32, device=device, requires_grad=True
        )
        scale = torch.tensor(binding["loss_scale"], dtype=torch.float32, device=device)
        result = _upstream().apply(output, mtp_loss)
    grad_output = torch.randn_like(output)
    return (result, output, mtp_loss, grad_output, scale), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    return {
        "inputs": {
            "output": {"shape": list(binding["shape"]), "dtype": str(dtype)},
            "mtp_loss": {"shape": [], "dtype": str(torch.float32)},
            "loss_scale": {"shape": [], "dtype": str(torch.float32)},
        },
        "loss_scale": binding["loss_scale"],
        "stage": "backward",
    }
