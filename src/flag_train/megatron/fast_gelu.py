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
"""Fast GELU activation (forward only).

``fast_gelu(x) = 0.5 * x * (1 + tanh(0.7978845608 * x * (1 + 0.044715 * x^2)))``

The operator returns a plain tensor without an autograd node: no backward is
provided here, and a backward operator is expected to be contributed separately.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

BLOCK = 1024
_FAST_GELU_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_input(tensor, name, dtypes):
    if tensor.dtype not in dtypes:
        wanted = " / ".join(str(dtype) for dtype in dtypes)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


@libentry()
@triton.jit
def fast_gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tle.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = (
        0.5
        * x
        * (1.0 + tl_extra_shim.tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))
    )
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


def fast_gelu(x):
    logger.debug("TRAIN FAST_GELU")
    _check_input(x, "fast_gelu", _FAST_GELU_DTYPES)
    if x.numel() == 0:
        return torch.empty_like(x)
    x_flat = x.contiguous().view(-1)
    out = torch.empty_like(x_flat)
    grid = (triton.cdiv(x.numel(), BLOCK),)
    with torch_device_fn.device(x.device):
        fast_gelu_kernel[grid](x_flat, out, x.numel(), BLOCK=BLOCK)
    return out.view_as(x)
