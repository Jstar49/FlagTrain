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

import logging
from dataclasses import replace

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.utils import pointwise_dynamic
from flag_train.utils.codegen_config_utils import get_codegen_config
from flag_train.utils.triton_lang_helper import tl_extra_shim

logger = logging.getLogger(__name__)

_CONFIG = get_codegen_config()
if runtime_device.vendor_name == "ascend":
    _CONFIG = replace(_CONFIG, max_tile_size=4096, max_grid_size=(512, 1, 1))


@pointwise_dynamic(
    is_tensor=[True, False],
    dtypes=[None, float],
    # Store the FP32 result in x's dtype, which truncates it for int32.
    promotion_methods=[(0, "NO_OPMATH")],
    config=_CONFIG,
)
@triton.jit
def _tanh_soft_clamp_kernel(x, scale):
    x_fp32 = x.to(tl.float32)
    scale_fp32 = scale.to(tl.float32)
    inv_scale = 1.0 / scale_fp32
    return scale_fp32 * tl_extra_shim.tanh(x_fp32 * inv_scale)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, "NO_OPMATH")],
    config=_CONFIG,
)
@triton.jit
def _tanh_soft_clamp_int32_tensor_scale_kernel(x, scale):
    return scale * tl_extra_shim.tanh(x.to(tl.float32) / scale)


def tanh_soft_clamp(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Clamp a floating-point or int32 tensor with a scalar or broadcastable scale."""
    logger.debug("TRAIN TANH_SOFT_CLAMP")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"tanh_soft_clamp expects a tensor, got {type(x).__name__}")
    if x.dtype != torch.int32 and not x.is_floating_point():
        raise TypeError(
            f"tanh_soft_clamp only supports floating-point or int32 tensors, got {x.dtype}"
        )
    if x.device.type != runtime_device.name:
        raise ValueError(
            f"tanh_soft_clamp only supports {runtime_device.name} tensors, "
            f"got {x.device.type}"
        )
    if isinstance(scale, torch.Tensor):
        if (
            x.dtype == torch.int32
            and scale.device == x.device
            and scale.layout == torch.strided
            and not scale.is_complex()
        ):
            # Match the fused division before integer truncation. A zero-dimensional
            # scale does not promote a higher-rank FP32 activation to FP64.
            scale_dtype = torch.float32
            if scale.ndim or x.ndim == 0:
                scale_dtype = torch.promote_types(scale_dtype, scale.dtype)
            return _tanh_soft_clamp_int32_tensor_scale_kernel(
                x.contiguous(), scale.to(scale_dtype).contiguous()
            )
        return (scale * torch.tanh(x.float() / scale)).to(x.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    return _tanh_soft_clamp_kernel(x, float(scale))
