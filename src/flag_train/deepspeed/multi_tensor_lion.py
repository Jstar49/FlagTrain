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
import numbers

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


# Launch configuration swept on H20: BLOCK_SIZE 2048 with 16 warps keeps the
# per-thread work at 4 elements, matching the ILP=4 of DeepSpeed's CUDA kernel.
# Larger per-thread counts collapse occupancy and the kernel falls off a cliff.
_BLOCK_SIZE = 2048
# Metax caps a block at 512 threads (8 warps x 64 lanes), so use 8 warps
# there; 2048 elements over 512 threads keeps the same ILP=4 per thread
# as the H20 config (16 warps x 32 lanes).
_NUM_WARPS = 8 if runtime_device.vendor_name == "metax" else 16

# Ascend rejects kernel launches with coreDim > 65535, so cap the grid and let
# each program stride over multiple blocks. On platforms with a larger limit
# (nvidia: 2**31) the cap is never hit and the loop below runs a single
# iteration, so this costs nothing where it is not needed.
_MAX_GRID = 65535


@libentry()
@triton.jit
def lion_fused_kernel(
    meta_ptr,
    n_tensors,
    n_blocks,
    lr,
    beta1,
    beta2,
    weight_decay,
    anchor,
    BLOCK_SIZE: tl.constexpr,
):
    """One Lion update block for one tensor, addressed through the meta buffer.

    Mirrors ``multi_tensor_lion_cuda`` in DeepSpeed: fp32 intermediate math,
    ``c = beta1*m + (1-beta1)*g``, ``update = c > 0 ? -lr : +lr`` (c == 0 takes
    +lr), ``p = p*(1 - lr*weight_decay) + update``, ``m = beta2*m + (1-beta2)*g``.
    A single launch covers every chunk of every tensor: each program looks its
    (tensor, chunk) pair up in the device-side meta buffer instead of receiving
    raw pointers as kernel arguments. The grid is capped at _MAX_GRID, so a
    program strides over several blocks when there are more blocks than the
    backend allows in one launch.
    """
    # meta (int64) layout:
    #   [0:n]                 g base addresses
    #   [n:2n]                p base addresses
    #   [2n:3n]               m base addresses
    #   [3n:4n]               numel of each tensor
    #   [4n + pid]            block -> tensor id
    #   [4n + n_blocks + pid] block -> chunk id within tensor
    num_programs = tl.num_programs(0)
    for pid in range(tle.program_id(0), n_blocks, num_programs):
        t = tl.load(meta_ptr + 4 * n_tensors + pid)
        c = tl.load(meta_ptr + 4 * n_tensors + n_blocks + pid)
        n = tl.load(meta_ptr + 3 * n_tensors + t)

        dtype = anchor.dtype.element_ty
        g_base = tl.load(meta_ptr + t).to(tl.pointer_type(dtype))
        p_base = tl.load(meta_ptr + n_tensors + t).to(tl.pointer_type(dtype))
        m_base = tl.load(meta_ptr + 2 * n_tensors + t).to(tl.pointer_type(dtype))

        # Hints so the compiler can vectorize the loads/stores: the chunk start
        # is a multiple of BLOCK_SIZE and the lane offsets are contiguous.
        start = tl.multiple_of(c * BLOCK_SIZE, BLOCK_SIZE)
        idx = tl.max_contiguous(start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
        mask = idx < n

        # Match DeepSpeed's CUDA kernel: fp32 intermediate math (MATH_T = float).
        g = tl.load(g_base + idx, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_base + idx, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_base + idx, mask=mask, other=0.0).to(tl.float32)

        after_decay = 1.0 - lr * weight_decay
        cc = beta1 * m + (1.0 - beta1) * g
        upd = tl.where(
            cc > 0, -lr, lr
        )  # C++ semantics: c > 0 ? -lr : +lr (c==0 -> +lr)
        p_new = p * after_decay + upd
        m_new = beta2 * m + (1.0 - beta2) * g

        tl.store(p_base + idx, p_new.to(dtype), mask=mask)
        tl.store(m_base + idx, m_new.to(dtype), mask=mask)


def _build_metadata(g_list, p_list, m_list, block_size):
    """Pack every pointer/numel/block-mapping array into ONE int64 buffer.

    Packing means the device side costs a single H2D copy per rebuild instead
    of one small copy per array. The buffer is a pure function of the tensor
    set (addresses, numels, block_size), which is what makes it cacheable.
    """
    dev = p_list[0].device
    n_tensors = len(g_list)

    numels = [t.numel() for t in g_list]
    b2t = []
    b2c = []
    for i, n in enumerate(numels):
        nchunks = (n + block_size - 1) // block_size
        b2t.extend([i] * nchunks)
        b2c.extend(range(nchunks))
    ptrs = (
        [t.data_ptr() for t in g_list]
        + [t.data_ptr() for t in p_list]
        + [t.data_ptr() for t in m_list]
    )

    meta_cpu = torch.tensor(ptrs + numels + b2t + b2c, dtype=torch.int64)
    meta = meta_cpu.pin_memory().to(dev, non_blocking=True)
    return meta, n_tensors, len(b2t), g_list[0]


# Device-meta cache keyed by the exact tensor set. A hit always yields
# bit-identical content: the key contains every address and numel the buffer
# is built from, so even tensors freed and re-created at the same addresses
# with the same shapes produce the same key AND the same meta. Training loops
# call the optimizer on a fixed tensor set every step, so after the first step
# every call is a hit and the metadata cost disappears from the hot path.
_META_CACHE = {}


def _get_metadata(g_list, p_list, m_list, block_size):
    key = (
        g_list[0].dtype,
        block_size,
        tuple(t.data_ptr() for t in g_list),
        tuple(t.data_ptr() for t in p_list),
        tuple(t.data_ptr() for t in m_list),
        tuple(t.numel() for t in g_list),
    )
    entry = _META_CACHE.get(key)
    if entry is None:
        if len(_META_CACHE) >= 64:
            # Bound the device memory held by cached metas. Real workloads use
            # one tensor set, so an eviction here only costs one rebuild.
            cur = next(iter(_META_CACHE))
            del _META_CACHE[cur]
        entry = _build_metadata(g_list, p_list, m_list, block_size)
        _META_CACHE[key] = entry
    return entry


def multi_tensor_lion(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    step,
    weight_decay,
):
    """Fused Lion optimizer step over a list of parameter tensors.

    Faithful port of DeepSpeed's ``multi_tensor_lion`` CUDA operator. For each
    tensor, with gradient ``g``, parameter ``p`` and first moment ``m``::

        c      = beta1 * m + (1 - beta1) * g
        update = -lr if c > 0 else +lr      # sign update; c == 0 takes +lr
        p      = p * (1 - lr * weight_decay) + update
        m      = beta2 * m + (1 - beta2) * g

    Lion replaces Adam's adaptive scaling with the sign of the momentum, so the
    update magnitude is always exactly ``lr`` per element; it needs no second
    moment, which halves the optimizer-state memory.

    All chunks of all tensors are fused into a single kernel launch, addressed
    through a device-side meta buffer that is cached across calls (see
    ``_get_metadata``). The raw-op contract requires one dtype across the whole
    call; DeepSpeed's ``FusedLion.step`` groups by dtype *above* the op, so this
    layer deliberately does not group mixed dtypes.

    Args:
        chunk_size (int): kept for interface parity with DeepSpeed; this
            implementation always launches with BLOCK_SIZE=2048 and the result
            does not depend on the upstream launch segmentation.
        noop_flag (Tensor): kept for interface parity; DeepSpeed's own noop
            check is commented out in its source, so the flag is ignored here
            as well.
        tensor_lists (list): ``[grads, params, exp_avgs]`` -- three equally
            long, non-empty lists of contiguous (or channels_last) tensors
            with matching numels, one dtype and one device across the call.
        lr (float): learning rate; also the per-element update magnitude.
        beta1 (float): momentum coefficient used to build the update direction.
        beta2 (float): momentum coefficient for the exp_avg update itself.
        step (int): optimizer step; Lion has no bias correction, so the value
            is unused and accepted only for interface parity.
        weight_decay (float): decoupled weight decay coefficient.
    """
    logger.debug("TRAIN LION")
    _check_chunk_size(chunk_size, "multi_tensor_lion")
    g_list, p_list, m_list = _unpack_tensor_lists(tensor_lists, 3, "multi_tensor_lion")
    dtype, device = _validate_tensor_lists(
        (g_list, p_list, m_list),
        "multi_tensor_lion",
        expected_device=runtime_device.name,
    )
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            "multi_tensor_lion only supports fp64/fp32/fp16/bf16, got " f"{dtype}"
        )
    if dtype == torch.float64 and not runtime_device.support_fp64:
        raise RuntimeError(
            f"multi_tensor_lion cannot run fp64 on {runtime_device.name}: "
            "the backend reports no fp64 support"
        )

    meta, n_tensors, n_blocks, anchor = _get_metadata(
        g_list, p_list, m_list, _BLOCK_SIZE
    )
    if n_blocks == 0:
        return
    with torch_device_fn.device(device):
        lion_fused_kernel[(min(n_blocks, _MAX_GRID),)](
            meta,
            n_tensors,
            n_blocks,
            lr,
            beta1,
            beta2,
            weight_decay,
            anchor=anchor,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
