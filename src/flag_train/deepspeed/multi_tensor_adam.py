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

Supports fp64, fp32, fp16 and bf16, matching the raw DeepSpeed op's
``DISPATCH_DOUBLE_FLOAT_AND_HALF`` dispatch (which includes ``BFloat16`` despite
its name). fp64 is rejected when the active backend reports no fp64 support.

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
import numbers

import numpy as np
import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# DeepSpeed's DISPATCH_DOUBLE_FLOAT_AND_HALF also includes BFloat16.
_SUPPORTED_DTYPES = (
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
)


def _check_chunk_size(chunk_size, op_name):
    """Validate the upstream ``chunk_size`` argument.

    The Triton port accepts the argument for interface parity but ignores it:
    results do not depend on DeepSpeed's launch segmentation. Validation still
    rejects values DeepSpeed itself could not divide by.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral):
        raise TypeError(
            f"{op_name} chunk_size must be an integer, got "
            f"{type(chunk_size).__name__}"
        )
    if chunk_size <= 0:
        raise ValueError(f"{op_name} chunk_size must be positive, got {chunk_size}")


def _unpack_tensor_lists(tensor_lists, depth, op_name):
    """Return the ``depth`` tensor lists after checking the outer sequence."""
    if not isinstance(tensor_lists, (list, tuple)):
        raise TypeError(
            f"{op_name} tensor_lists must be a list or tuple of {depth} lists"
        )
    if len(tensor_lists) != depth:
        raise RuntimeError(
            f"{op_name} expects {depth} tensor lists, got {len(tensor_lists)}"
        )
    for index, tensors in enumerate(tensor_lists):
        if not isinstance(tensors, (list, tuple)):
            raise TypeError(
                f"{op_name} tensor_lists[{index}] must be a list or tuple of tensors"
            )
    return tensor_lists


def _is_contiguous(tensor):
    return tensor.is_contiguous() or tensor.is_contiguous(
        memory_format=torch.channels_last
    )


def _validate_tensor_lists(tensor_lists, op_name, expected_device=None):
    """Validate DeepSpeed's raw multi-tensor calling contract.

    ``tensor_lists`` has already been unpacked by :func:`_unpack_tensor_lists`.
    Returns the shared ``(dtype, device)``.

    A single dtype is required. DeepSpeed dispatches its raw op from the first
    tensor and only defines behaviour for one dtype; its optimizer classes
    group by dtype before calling the op. Rejecting a mixed call here is a
    deliberate, explicit version of that assumption rather than silent
    reinterpretation of the other tensors.
    """
    first_list = tensor_lists[0]
    if len(first_list) == 0:
        raise RuntimeError(f"{op_name} requires at least one tensor per list")

    for index, tensors in enumerate(tensor_lists):
        if len(tensors) != len(first_list):
            raise RuntimeError(
                f"{op_name} tensor lists must have the same length; "
                f"tensor_lists[0] has {len(first_list)}, "
                f"tensor_lists[{index}] has {len(tensors)}"
            )
        for position, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{op_name} tensor_lists[{index}][{position}] must be a Tensor"
                )

    reference = first_list[0]
    dtype = reference.dtype
    device = reference.device
    if expected_device is not None and device.type != expected_device:
        raise RuntimeError(
            f"{op_name} only supports {expected_device} tensors, got {device.type}"
        )

    for index, tensors in enumerate(tensor_lists):
        for position, tensor in enumerate(tensors):
            where = f"tensor_lists[{index}][{position}]"
            if tensor.dtype != dtype:
                raise RuntimeError(
                    f"{op_name} requires one dtype per call; {where} is "
                    f"{tensor.dtype}, expected {dtype}"
                )
            if tensor.device != device:
                raise RuntimeError(
                    f"{op_name} requires all tensors on one device; {where} is "
                    f"on {tensor.device}, expected {device}"
                )
            expected_numel = first_list[position].numel()
            if tensor.numel() != expected_numel:
                raise RuntimeError(
                    f"{op_name} {where} has {tensor.numel()} elements; "
                    f"tensor_lists[0][{position}] has {expected_numel}"
                )
            if not _is_contiguous(tensor):
                raise RuntimeError(
                    f"{op_name} {where} must be contiguous or channels_last "
                    "contiguous"
                )

    return dtype, device


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
    torch.float64: tl.float64,
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
    ``FusedAdam.step`` through ``MultiTensorApply`` unchanged. The raw-op input
    contract is validated: four equally long, non-empty lists; one dtype; one
    device; matching numels; and contiguous or channels_last storage.
    ``chunk_size`` is accepted for compatibility but not used; the Triton kernel
    chooses its own element-per-program granularity via ``_BLOCK_SIZE``.
    ``noop_flag`` is accepted and ignored, matching DeepSpeed's commented-out
    noop check.
    """
    logger.debug("TRAIN MULTI_TENSOR_ADAM")
    _check_chunk_size(chunk_size, "multi_tensor_adam")
    g_list, p_list, m_list, v_list = _unpack_tensor_lists(
        tensor_lists, 4, "multi_tensor_adam"
    )
    dtype, device = _validate_tensor_lists(
        (g_list, p_list, m_list, v_list),
        "multi_tensor_adam",
        expected_device=runtime_device.name,
    )
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            "multi_tensor_adam only supports fp64/fp32/fp16/bf16, got " f"{dtype}"
        )
    if dtype == torch.float64 and not runtime_device.support_fp64:
        raise RuntimeError(
            f"multi_tensor_adam cannot run fp64 on {runtime_device.name}: "
            "the backend reports no fp64 support"
        )

    if bias_correction == 1:
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
    else:
        bc1 = 1.0
        bc2 = 1.0

    ptrs_g, ptrs_p, ptrs_m, ptrs_v, sizes, block_to_tensor, block_to_offset = (
        _get_metadata(g_list, p_list, m_list, v_list, _BLOCK_SIZE, device)
    )

    total_blocks = int(block_to_tensor.numel())
    if total_blocks == 0:
        # DeepSpeed's multi_tensor_apply launches nothing when every tensor is
        # empty; return before asking Triton for a zero-sized grid.
        return
    grid = (total_blocks,)

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
