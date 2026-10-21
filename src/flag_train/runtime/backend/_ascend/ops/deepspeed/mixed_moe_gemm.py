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
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mixed_moe_gemm_ascend_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    scale_ptr,
    bias_ptr,
    row_ends_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_IM: tl.constexpr,
    STRIDE_IK: tl.constexpr,
    STRIDE_OM: tl.constexpr,
    STRIDE_ON: tl.constexpr,
    NUM_BITS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    expert = tl.program_id(2).to(tl.int64)
    first_row = tl.load(row_ends_ptr + expert - 1, expert > 0, other=0)
    last_row = tl.load(row_ends_ptr + expert)
    if tile_m * BLOCK_M < last_row - first_row:
        rows = first_row + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        row_mask = (rows < last_row) & (rows < M)
        col_mask = cols < N
        scales = tl.load(scale_ptr + expert * N + cols, col_mask, other=0).to(
            tl.float32
        )
        packed_n: tl.constexpr = (N + 1) // 2 if NUM_BITS == 4 else N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        k_tiles: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
        for block_k in range(0, k_tiles):
            ks = block_k * BLOCK_K + offsets_k
            inputs = tl.load(
                input_ptr + rows[:, None] * STRIDE_IM + ks[None, :] * STRIDE_IK,
                row_mask[:, None] & (ks[None, :] < K),
                other=0,
            )
            weight_cols = cols // 2 if NUM_BITS == 4 else cols
            packed = tl.load(
                weight_ptr
                + expert * K * packed_n
                + ks[:, None] * packed_n
                + weight_cols[None, :],
                (ks[:, None] < K) & col_mask[None, :],
                other=0,
            ).to(tl.int32)
            if NUM_BITS == 4:
                nibbles = (packed >> ((cols[None, :] % 2) * 4)) & 15
                packed = (nibbles ^ 8) - 8
            weights = (packed.to(tl.float32) * scales[None, :]).to(inputs.dtype)
            al.compile_hint(inputs, "dot_pad_only_k")
            al.compile_hint(weights, "dot_pad_only_k")
            accumulator = tl.dot(inputs, weights, acc=accumulator)
        if HAS_BIAS:
            biases = tl.load(bias_ptr + expert * N + cols, col_mask, other=0).to(
                tl.float32
            )
            accumulator += biases[None, :]
        if ACTIVATION == 0:
            accumulator = (
                0.5
                * accumulator
                * (1.0 + tl_extra_shim.erf(accumulator * 0.7071067811865476))
            )
        elif ACTIVATION == 1:
            accumulator = tl.maximum(accumulator, 0.0)
        elif ACTIVATION == 2:
            accumulator = accumulator * tl.sigmoid(accumulator)
        tl.store(
            output_ptr + rows[:, None] * STRIDE_OM + cols[None, :] * STRIDE_ON,
            accumulator,
            row_mask[:, None] & col_mask[None, :],
        )


def mixed_moe_gemm(
    output,
    hidden_states,
    weights,
    scales,
    biases,
    total_rows_before_expert,
    num_bits,
    activation=6,
):
    """Use Ascend Cube tiles with dense device buffers for strided views."""
    # Import lazily because vendor operators are discovered while the public
    # DeepSpeed package is still being initialized.
    logger.debug("TRAIN_ASCEND MIXED_MOE_GEMM")
    from flag_train.deepspeed.mixed_moe_gemm import _validate_inputs

    m, n, k, experts = _validate_inputs(
        output,
        hidden_states,
        weights,
        scales,
        biases,
        total_rows_before_expert,
        num_bits,
        activation,
    )
    if m == 0 or n == 0:
        return output
    with torch_device_fn.device(hidden_states.device):
        # Cube loads require dense matrix rows; preserve strided public views
        # by using dense device buffers only when a view needs them.
        inputs = hidden_states.contiguous()
        result = (
            output
            if output.is_contiguous()
            else torch.empty_like(output, memory_format=torch.contiguous_format)
        )
        mixed_moe_gemm_ascend_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64), experts)](
            result,
            inputs,
            weights,
            scales,
            biases,
            total_rows_before_expert,
            m,
            n,
            k,
            inputs.stride(0),
            inputs.stride(1),
            result.stride(0),
            result.stride(1),
            NUM_BITS=num_bits,
            ACTIVATION=int(activation),
            HAS_BIAS=biases is not None,
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=64,
        )
        if result is not output:
            output.copy_(result)
    return output
