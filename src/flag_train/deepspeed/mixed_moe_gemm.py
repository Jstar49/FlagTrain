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

from flag_train import runtime
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mixed_moe_gemm_kernel(
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
    tile = tle.program_id(0)
    expert = tle.program_id(1)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile % tiles_n
    first_row = tl.load(row_ends_ptr + expert - 1, expert > 0, other=0)
    last_row = tl.load(row_ends_ptr + expert)
    # A rectangular expert grid avoids relying on cross-lane integer scans
    # in vendor compilers. Empty expert/tile pairs exit without touching output.
    if tile_m * BLOCK_M < last_row - first_row:
        rows = first_row + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        row_mask = (rows < last_row) & (rows < M)
        col_mask = cols < N
        expert_offset = expert.to(tl.int64)
        scales = tl.load(scale_ptr + expert_offset * N + cols, col_mask, other=0).to(
            tl.float32
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        packed_n = tl.cdiv(N, 2) if NUM_BITS == 4 else N
        for block_k in range(tl.cdiv(K, BLOCK_K)):
            ks = block_k * BLOCK_K + offsets_k
            inputs = tl.load(
                input_ptr + rows[:, None] * STRIDE_IM + ks[None, :] * STRIDE_IK,
                row_mask[:, None] & (ks[None, :] < K),
                other=0,
            )
            weight_cols = cols // 2 if NUM_BITS == 4 else cols
            weight_offsets = (
                expert_offset * K * packed_n
                + ks[:, None].to(tl.int64) * packed_n
                + weight_cols[None, :]
            )
            quantized = tl.load(
                weight_ptr + weight_offsets,
                (ks[:, None] < K) & col_mask[None, :],
                other=0,
            ).to(tl.int32)
            if NUM_BITS == 4:
                nibbles = (quantized >> ((cols[None, :] % 2) * 4)) & 15
                quantized = (nibbles ^ 8) - 8
            weights = (quantized.to(tl.float32) * scales[None, :]).to(inputs.dtype)
            accumulator += tl.dot(inputs, weights)

        if HAS_BIAS:
            biases = tl.load(bias_ptr + expert_offset * N + cols, col_mask, other=0).to(
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

        output_offsets = rows[:, None] * STRIDE_OM + cols[None, :] * STRIDE_ON
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            row_mask[:, None] & col_mask[None, :],
        )


def _validate_inputs(
    output,
    hidden_states,
    weights,
    scales,
    biases,
    total_rows_before_expert,
    num_bits,
    activation,
):
    supported_vendors = ("nvidia", "hygon", "metax", "thead", "iluvatar", "ascend")
    if runtime.device.vendor_name not in supported_vendors:
        raise RuntimeError("mixed_moe_gemm does not support this accelerator backend")
    if num_bits not in (4, 8):
        raise ValueError("num_bits must be 4 or 8")
    if activation not in (0, 1, 2, 6):
        raise ValueError("activation must be GELU=0, RELU=1, SILU=2, or IDENTITY=6")
    if hidden_states.ndim != 2 or output.ndim != 2 or weights.ndim != 3:
        raise ValueError("input/output must be 2D and weights must be 3D")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("hidden_states must have dtype float16 or bfloat16")
    if output.dtype != hidden_states.dtype:
        raise TypeError("output and hidden_states must have the same dtype")
    weight_dtypes = (torch.int8, torch.uint8) if num_bits == 4 else (torch.int8,)
    if weights.dtype not in weight_dtypes:
        raise TypeError("weights must be signed int8, or packed int8/uint8 for 4 bits")

    m, k = hidden_states.shape
    n = output.shape[1]
    num_experts = weights.shape[0]
    packed_n = triton.cdiv(n, 2) if num_bits == 4 else n
    if num_experts == 0:
        raise ValueError("at least one expert is required")
    if output.shape[0] != m or weights.shape[1:] != (k, packed_n):
        raise ValueError("input, output, and quantized weight dimensions do not match")
    if scales.shape != (num_experts, n) or scales.dtype != hidden_states.dtype:
        raise ValueError("scales must have shape [experts, N] and the input dtype")
    if (
        total_rows_before_expert.shape != (num_experts,)
        or total_rows_before_expert.dtype != torch.int64
    ):
        raise ValueError("row ends must have shape [experts] and dtype int64")
    if biases is not None:
        if biases.shape != (num_experts, n) or biases.dtype != hidden_states.dtype:
            raise ValueError("biases must have shape [experts, N] and the input dtype")

    tensors = [output, hidden_states, weights, scales, total_rows_before_expert]
    if biases is not None:
        tensors.append(biases)
    if hidden_states.device.type != runtime.device.name or any(
        t.device != hidden_states.device for t in tensors
    ):
        raise ValueError("all tensors must be on the same accelerator device")
    contiguous_tensors = [weights, scales, total_rows_before_expert]
    if biases is not None:
        contiguous_tensors.append(biases)
    if any(not t.is_contiguous() for t in contiguous_tensors):
        raise ValueError("weights, scales, biases, and row ends must be contiguous")
    return m, n, k, num_experts


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
    """Weight-only quantized MoE GEMM, writing into ``output``.

    Follows the argument order of DeepSpeed's ``mixed_moe_gemm`` binding.
    NVIDIA, Hygon, MetaX, T-Head, and Iluvatar use the generic Triton kernel;
    Ascend uses a separate Cube kernel. Activations are already ordered
    by expert; this operator does not route tokens or combine top-k expert outputs.

    Args:
        output: FP16/BF16 output, shape [tokens, N], returned by identity.
        hidden_states: FP16/BF16 input, shape [tokens, K].
        weights: Contiguous signed int8 [experts, K, N] for 8 bits, or
            int8/uint8 [experts, K, ceil(N / 2)] for 4 bits. INT4 uses
            two's-complement nibbles, with the even N column in the low nibble.
            These are logical row-major weights, not CUTLASS-preprocessed bytes.
        scales: Contiguous [experts, N], in the input dtype. Dequantized
            weights are rounded to the input dtype before multiplication.
        biases: Optional contiguous [experts, N], in the input dtype.
        total_rows_before_expert: Contiguous int64 inclusive cumulative row
            ends, shape [experts]. Values must be nondecreasing, in [0, tokens],
            with the final value equal to tokens. Repeated ends allow empty
            experts. These value preconditions are not checked on the host,
            so calling the operator does not synchronize the device.
        num_bits: 4 or 8; these are integer formats, not floating-point FP4/FP8.
        activation: DeepSpeed ActivationType value: GELU=0 (erf), RELU=1,
            SILU=2, IDENTITY=6. Bias and activation use FP32 accumulators.

    Input/output strides are supported for nonoverlapping views. All tensors
    must be on the same accelerator device. This is a forward-only inference
    operator; output must not overlap any input. The generic kernel allocates no
    intermediate tensors. Ascend uses dense device buffers for strided views.
    Neither implementation provides autograd.
    """
    logger.debug("TRAIN MIXED_MOE_GEMM")
    m, n, k, num_experts = _validate_inputs(
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

    # A 32-row minimum keeps matrix fragments valid on all five backends.
    block_m = 32 if m <= 64 * num_experts else 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, 64), num_experts)
    with torch_device_fn.device(hidden_states.device):
        mixed_moe_gemm_kernel[grid](
            output,
            hidden_states,
            weights,
            scales,
            biases,
            total_rows_before_expert,
            m,
            n,
            k,
            hidden_states.stride(0),
            hidden_states.stride(1),
            output.stride(0),
            output.stride(1),
            NUM_BITS=num_bits,
            ACTIVATION=int(activation),
            HAS_BIAS=biases is not None,
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
    return output
