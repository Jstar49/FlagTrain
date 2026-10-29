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
"""Triton implementation of DeepSpeed's multi_tensor_adam operator.

Batches the Adam/AdamW update of many parameter tensors into a single kernel
launch, amortizing launch overhead across the whole parameter set. Ports
``csrc/adam/multi_tensor_adam.cu`` -- numerically identical, but the per-tensor
metadata lives in device tensors instead of a fixed-size kernel argument
struct, so a single launch is not limited to 48 tensors / 320 blocks.

Two Adam variants share one kernel, chosen at compile time:

* ``mode == 0`` -- L2 regularization (``torch.optim.Adam``): the weight decay
  is folded into the gradient before the moment update.
* ``mode == 1`` -- decoupled weight decay (``torch.optim.AdamW``): the decay
  is applied to the parameter directly.

``bias_correction`` follows DeepSpeed: when 0 the correction factors are 1.0,
when 1 they are ``1 - beta**step`` computed on the host and passed as scalars.

Supports fp32, fp16 and bf16. DeepSpeed's CUDA op dispatches only fp16/fp32/fp64
via ``DISPATCH_DOUBLE_FLOAT_AND_HALF`` and does not cover bf16; the Triton port
does, and the reference test exercises all three.

Host-side overhead matters as much as the kernel here. The operator is called
once per optimizer step with the *same* tensors, so the address arrays and the
block-to-(tensor, offset) map are cached keyed by the tensors' addresses and
sizes. Without that cache a Python-level rebuild dominates the runtime and the
kernel's advantage disappears. The cache key includes both ``data_ptr`` and
``numel``: two tensors that share an address but differ in size cannot collide,
and the common case -- an optimizer stepping over a fixed set of parameters --
hits every time. A caller that frees and reallocates a tensor of the same
address *and* the same size between steps would get stale metadata; that is not
a pattern the optimizers in this repository use.
"""

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def multi_tensor_adam_kernel(
    ptrs_g,
    ptrs_p,
    ptrs_m,
    ptrs_v,
    sizes,
    block_to_tensor,
    block_to_offset,
    beta1,
    beta2,
    bc1,
    bc2,
    epsilon,
    lr,
    decay,
    BLOCK_SIZE: tl.constexpr,
    MODE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """One program handles ``BLOCK_SIZE`` contiguous elements of one tensor.

    ``block_to_tensor[pid]`` and ``block_to_offset[pid]`` pick the tensor and
    the in-tensor offset for this program, mirroring the ``block_to_tensor`` /
    ``block_to_chunk`` mapping DeepSpeed builds on the host. All arithmetic is
    fp32 regardless of the storage dtype, matching ``MATH_T = float``.
    """
    pid = tle.program_id(0)

    tensor_loc = tl.load(block_to_tensor + pid)
    offset = tl.load(block_to_offset + pid)
    n = tl.load(sizes + tensor_loc)

    g_ptr = tl.load(ptrs_g + tensor_loc).to(tl.pointer_type(DTYPE))
    p_ptr = tl.load(ptrs_p + tensor_loc).to(tl.pointer_type(DTYPE))
    m_ptr = tl.load(ptrs_m + tensor_loc).to(tl.pointer_type(DTYPE))
    v_ptr = tl.load(ptrs_v + tensor_loc).to(tl.pointer_type(DTYPE))

    offs = offset + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    if MODE == 0:
        g = g + decay * p
        m_new = beta1 * m + (1.0 - beta1) * g
        v_new = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m_new / bc1
        v_hat = v_new / bc2
        update = m_hat / (tl.sqrt(v_hat) + epsilon)
        p_new = p - lr * update
    else:
        m_new = beta1 * m + (1.0 - beta1) * g
        v_new = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m_new / bc1
        v_hat = v_new / bc2
        update = m_hat / (tl.sqrt(v_hat) + epsilon) + decay * p
        p_new = p - lr * update

    tl.store(p_ptr + offs, p_new.to(DTYPE), mask=mask)
    tl.store(m_ptr + offs, m_new.to(DTYPE), mask=mask)
    tl.store(v_ptr + offs, v_new.to(DTYPE), mask=mask)


_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}

_BLOCK_SIZE = 1024

# Metadata cache, keyed by (addresses, sizes, block_size). See the module
# docstring for the assumptions this makes.
_METADATA_CACHE = {}


def _build_metadata_arrays(sizes_list, block_size):
    """Vectorised ``(sizes, block_to_tensor, block_to_offset)`` construction.

    Avoids the Python double loop the CUDA op does on the host: at 16 tensors
    per launch that loop is a rounding error, but optimisers routinely step
    hundreds, and the loop then costs more than the kernel it schedules.
    """
    sizes_np = np.asarray(sizes_list, dtype=np.int64)
    blocks_per_tensor = (sizes_np + block_size - 1) // block_size
    total_blocks = int(blocks_per_tensor.sum())

    block_to_tensor = np.repeat(
        np.arange(len(sizes_list), dtype=np.int32), blocks_per_tensor
    )
    starts = np.concatenate(([0], np.cumsum(blocks_per_tensor)[:-1]))
    within = np.arange(total_blocks, dtype=np.int64) - np.repeat(
        starts, blocks_per_tensor
    )
    block_to_offset = within * block_size
    return sizes_np, block_to_tensor, block_to_offset


def _get_metadata(g_list, p_list, m_list, v_list, block_size, device):
    """Return ``(ptrs_g, ptrs_p, ptrs_m, ptrs_v, sizes, b2t, b2o)`` on ``device``.

    Cached: a steady-state optimizer step reuses the same tensor objects every
    iteration, so rebuilding these on every call would push the operator's cost
    onto the host and mask whatever the kernel saves.
    """
    key = (
        tuple(t.data_ptr() for t in g_list),
        tuple(t.data_ptr() for t in p_list),
        tuple(t.data_ptr() for t in m_list),
        tuple(t.data_ptr() for t in v_list),
        tuple(t.numel() for t in g_list),
        block_size,
    )
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    sizes_np, b2t_np, b2o_np = _build_metadata_arrays(
        [t.numel() for t in p_list], block_size
    )

    ptrs_g = np.asarray([t.data_ptr() for t in g_list], dtype=np.int64)
    ptrs_p = np.asarray([t.data_ptr() for t in p_list], dtype=np.int64)
    ptrs_m = np.asarray([t.data_ptr() for t in m_list], dtype=np.int64)
    ptrs_v = np.asarray([t.data_ptr() for t in v_list], dtype=np.int64)

    cached = (
        torch.from_numpy(ptrs_g).to(device),
        torch.from_numpy(ptrs_p).to(device),
        torch.from_numpy(ptrs_m).to(device),
        torch.from_numpy(ptrs_v).to(device),
        torch.from_numpy(sizes_np).to(device),
        torch.from_numpy(b2t_np).to(device),
        torch.from_numpy(b2o_np).to(device),
    )
    _METADATA_CACHE[key] = cached
    return cached


def multi_tensor_adam(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    """Batched Adam/AdamW step over ``tensor_lists = [g_list, p_list, m_list, v_list]``.

    Signature mirrors DeepSpeed's ``multi_tensor_adam_cuda`` so it can drop into
    ``FusedAdam.step`` through ``MultiTensorApply`` unchanged. ``chunk_size``
    is accepted for compatibility but not used; the Triton kernel chooses its own
    element-per-program granularity via ``_BLOCK_SIZE``.
    """
    logger.debug("TRAIN MULTI_TENSOR_ADAM")
    g_list, p_list, m_list, v_list = tensor_lists
    if not g_list:
        return

    device = g_list[0].device
    dtype = g_list[0].dtype
    assert (
        dtype in _DTYPE_MAP
    ), f"multi_tensor_adam only supports fp16/bf16/fp32, got {dtype}"

    if bias_correction == 1:
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
    else:
        bc1 = 1.0
        bc2 = 1.0

    ptrs_g, ptrs_p, ptrs_m, ptrs_v, sizes, block_to_tensor, block_to_offset = (
        _get_metadata(g_list, p_list, m_list, v_list, _BLOCK_SIZE, device)
    )

    grid = (block_to_tensor.numel(),)

    with torch_device_fn.device(device):
        multi_tensor_adam_kernel[grid](
            ptrs_g,
            ptrs_p,
            ptrs_m,
            ptrs_v,
            sizes,
            block_to_tensor,
            block_to_offset,
            float(beta1),
            float(beta2),
            float(bc1),
            float(bc2),
            float(epsilon),
            float(lr),
            float(weight_decay),
            BLOCK_SIZE=_BLOCK_SIZE,
            MODE=int(mode),
            DTYPE=_DTYPE_MAP[dtype],
        )
