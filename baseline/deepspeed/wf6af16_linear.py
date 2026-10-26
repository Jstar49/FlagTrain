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

"""NVIDIA baseline for wf6af16_linear.

Self-contained module; depends only on torch and deepspeed, and does not import
``flag_train`` or ``benchmark``. ``native()`` returns DeepSpeed's
``cuda_wf6af16_linear``; when DeepSpeed is unavailable (or the device is not
Ampere) it raises instead of collecting a baseline or falling back to a torch
reference.

The input construction and FP6 packing are copied from
``FlagTrain/benchmark/deepspeed/test_wf6af16_linear.py``. DeepSpeed is loaded
lazily on the first call, so ops discovery does not trigger its JIT.
"""

import torch

OP_NAME = "wf6af16_linear"
DTYPES = [torch.float16]
IS_INPLACE = False

# benchmark/deepspeed/test_wf6af16_linear.py::_WF6AF16_SHAPES
_SHAPES = [
    (1, 4096, 4096),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (16, 14336, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 14336),
    (256, 8192, 8192),
    (512, 4096, 4096),
    (1024, 4096, 4096),
]


def _fp6_e3m2_table(device=None):
    """All 64 values representable in FP6-E3M2, indexed by the 6-bit code."""
    codes = torch.arange(64, dtype=torch.int32)
    sign = torch.where((codes >> 5) & 1 == 1, -1.0, 1.0)
    e = ((codes >> 2) & 7).to(torch.float32)
    m = (codes & 3).to(torch.float32)
    mag = torch.where(
        (codes >> 2) & 7 == 0,
        m * 0.0625,
        torch.exp2(e - 3.0) * (1.0 + m * 0.25),
    )
    return (sign * mag).to(device)


def _pack_weights_fp6(codes):
    """Pack FP6 codes into the 2-bit and 4-bit slices the kernel consumes."""
    out_channels, in_channels = codes.shape

    top2 = codes >> 4
    low4 = codes & 0xF

    t = top2.reshape(out_channels, in_channels // 4, 4).to(torch.uint8)
    weights_2bit = (t[..., 0] << 6) | (t[..., 1] << 4) | (t[..., 2] << 2) | t[..., 3]

    low = low4.reshape(out_channels, in_channels // 2, 2)
    weights_4bit = (low[..., 0] << 4) | low[..., 1]

    return weights_2bit.contiguous(), weights_4bit.contiguous()


def _load_deepspeed_wf6af16():
    """DeepSpeed's cuda_wf6af16_linear; raise when it cannot run here."""
    try:
        capability = torch.cuda.get_device_capability()
    except Exception as exc:
        raise RuntimeError(
            "wf6af16_linear baseline requires an NVIDIA GPU: " f"{exc!r}"
        ) from exc
    if capability != (8, 0):
        raise RuntimeError(
            "wf6af16_linear baseline requires DeepSpeed's cuda_wf6af16_linear, "
            f"whose kernel hard-codes compute capability 8.0; this device is "
            f"{capability}."
        )

    try:
        from deepspeed.ops.op_builder import InferenceCoreBuilder

        inf_module = InferenceCoreBuilder().load()
        inf_module.create_handle()
    except Exception as exc:
        raise RuntimeError(
            "wf6af16_linear baseline requires DeepSpeed's cuda_wf6af16_linear, "
            f"which could not be loaded: {exc!r}. Install deepspeed and ensure "
            "its InferenceCore CUDA extension builds on this machine."
        ) from exc

    def deepspeed_wf6af16(hidden_states, weights_2bit, weights_4bit, scales):
        out_channels = scales.shape[0]
        tokens, in_channels = hidden_states.shape
        output = torch.empty(
            (tokens, out_channels), dtype=torch.float16, device=hidden_states.device
        )
        workspace = torch.empty(
            (1, out_channels, tokens),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        inf_module.cuda_wf6af16_linear(
            output,
            hidden_states,
            weights_2bit,
            weights_4bit,
            scales,
            workspace,
            out_channels,
            tokens,
            in_channels,
            1,
        )
        return output

    deepspeed_wf6af16.preprocess_weight = inf_module.preprocess_weight
    return deepspeed_wf6af16


_DEEPSPEED_WF6 = None


def _deepspeed_wf6af16():
    global _DEEPSPEED_WF6
    if _DEEPSPEED_WF6 is None:
        _DEEPSPEED_WF6 = _load_deepspeed_wf6af16()
    return _DEEPSPEED_WF6


def torch_op(hidden, weights_2bit, weights_4bit, scales, ds_w2, ds_w4):
    """DeepSpeed W6A16 linear; raises when it cannot be loaded."""
    return _deepspeed_wf6af16()(hidden, ds_w2, ds_w4, scales)


def native():
    # Load eagerly so an unavailable DeepSpeed fails here instead of being
    # discovered mid-collection (and silently skipped).
    _deepspeed_wf6af16()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    tokens, in_channels, out_channels = binding["shape"]
    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 64, (out_channels, in_channels), generator=gen).to(
        torch.uint8
    )
    scales = (torch.rand(out_channels, generator=gen) + 0.5).to(torch.float16)
    hidden = torch.randn(tokens, in_channels, generator=gen).to(dtype)

    weights_2bit, weights_4bit = _pack_weights_fp6(codes.to(device))

    baseline = _deepspeed_wf6af16()
    table = _fp6_e3m2_table()
    fake_fp16 = table[codes.long()].to(torch.float16)
    ds_w2, ds_w4 = baseline.preprocess_weight(fake_fp16)
    ds_w2 = ds_w2.to(device)
    ds_w4 = ds_w4.to(device)

    return (
        hidden.to(device),
        weights_2bit,
        weights_4bit,
        scales.to(device),
        ds_w2,
        ds_w4,
    ), {}


def key_shape(binding):
    # First yielded tensor: hidden [tokens, in_channels]; unique per shape.
    tokens, in_channels, _ = binding["shape"]
    return [tokens, in_channels]
