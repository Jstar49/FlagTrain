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

import torch
import triton

from flag_train.megatron.tanh_soft_clamp import (
    _NUM_WARPS,
    _finish,
    _launch_context,
    _prepare,
    _tanh_soft_clamp_kernel,
    _tanh_soft_clamp_walk_kernel,
)
from flag_train.runtime.backend._ascend.utils import CORE_NUM

logger = logging.getLogger(__name__)


_ASCEND_BLOCK = 16384
_NARROW_ASCEND_BLOCK = 8192
_NARROW_DTYPES = (torch.int8, torch.uint8, torch.bool)


def tanh_soft_clamp(x, scale):
    """Megatron's ``(scale * tanh(x.float() / scale)).to(x.dtype)`` on Ascend.

    Validation, broadcasting, CPU staging and the result copy are shared with
    the portable implementation through ``_prepare``; only the launch differs.
    """
    logger.debug("TRAIN_ASCEND TANH_SOFT_CLAMP")
    prep = _prepare(x, scale)
    if not prep.n:
        return _finish(prep)

    if prep.n <= 16384:
        # Launch-bound sizes: a handful of wide programs beats one program per
        # core (measured; 4096 elements: 1.33 us at 4 programs against 2.18 us
        # at 32).
        target = max(1, prep.n // 4)
        block = min(2048, max(64, 1 << (target - 1).bit_length()))
        grid = (triton.cdiv(prep.n, block),)
        kernel = _tanh_soft_clamp_kernel
    else:
        lanes = (prep.n + CORE_NUM - 1) // CORE_NUM
        widest = (
            _NARROW_ASCEND_BLOCK if prep.out_dtype in _NARROW_DTYPES else _ASCEND_BLOCK
        )
        block = min(widest, max(128, 1 << (lanes - 1).bit_length()))
        tiles = (prep.n + block - 1) // block
        grid = (min(tiles, CORE_NUM),)
        # The walk only pays when one program owns several tiles.
        kernel = (
            _tanh_soft_clamp_kernel
            if tiles <= CORE_NUM
            else _tanh_soft_clamp_walk_kernel
        )

    with _launch_context(prep.result_device, prep.compute_device):
        kernel[grid](
            prep.x_flat,
            prep.s_arg,
            prep.out_flat,
            prep.n,
            SCALE=prep.scalar,
            SCALE_MODE=prep.mode,
            BLOCK=block,
            num_warps=_NUM_WARPS,
            num_stages=1,
        )
    return _finish(prep)
