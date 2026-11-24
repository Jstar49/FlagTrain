"""Ascend multi_tensor_lion: 8192-lane tiles, walked when the program count
would otherwise be large. A program costs tens of ns whatever it carries
here, so the generic 2048-lane masked shape was bounded by its program
count, not by memory. Arithmetic and interface are the generic
operator's; ``_ascend/deepspeed/__init__.py`` registers this by name.
"""

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from flag_train.deepspeed.multi_tensor_lion import (
    _SUPPORTED_DTYPES,
    _check_chunk_size,
    _unpack_tensor_lists,
    _validate_tensor_lists,
)
from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# Measured plateau; a 16384-lane fp32 tile overflows the 192 KB UB.
_BLOCK_MAX = 8192
_BLOCK_BYTES = 32768
_BLOCK_MIN = 512

# ~32768 lanes of work (about 0.6 MB over five fp32 streams) per program.
_LANES_PER_PROGRAM = 32768
_MIN_PROGRAMS = 32
_MAX_WALK_PROGRAMS = 2048


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _block_for(numel, itemsize):
    """Tile for one tensor (0 when empty): a power-of-two divisor of ``numel``,
    capped, so the block grid is exact and the kernel runs unmasked.
    """
    if numel == 0:
        return 0
    cap = min(_BLOCK_MAX, _BLOCK_BYTES // itemsize)
    low = numel & -numel
    if low >= _BLOCK_MIN:
        return min(low, cap)
    return min(cap, max(_BLOCK_MIN, _next_pow2(numel)))


def _plan_launch(numels, block):
    """``(grid, walk, full)`` for a group sharing ``block``: about 32768 lanes of
    work per program, at least 32 programs, at most 2048.
    """
    n_blocks = sum((n + block - 1) // block for n in numels)
    full = all(n % block == 0 for n in numels)
    if n_blocks == 0:
        return 0, False, full
    per_program = max(1, _LANES_PER_PROGRAM // block)
    wanted = -(-n_blocks // per_program)
    grid = min(max(wanted, _MIN_PROGRAMS), _MAX_WALK_PROGRAMS, n_blocks)
    return grid, grid < n_blocks, full


@libentry()
@triton.jit
def lion_block_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    lr,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complete Lion block from pre-offset pointers: unmasked, one runtime
    scalar. ``c = b1*m + (1-b1)*g``; ``p = p*(1-lr*wd) + (c>0 ? -lr : lr)``;
    ``m = b2*m + (1-b2)*g``.
    """
    dtype = anchor.dtype.element_ty
    i = tl.program_id(0)
    g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
    p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
    m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))

    offs = tl.arange(0, BLOCK)
    g = tl.load(g_ptr + offs).to(tl.float32)
    p = tl.load(p_ptr + offs).to(tl.float32)
    m = tl.load(m_ptr + offs).to(tl.float32)

    cc = BETA1 * m + (1.0 - BETA1) * g
    m_new = BETA2 * m + (1.0 - BETA2) * g
    tl.store(m_ptr + offs, m_new.to(dtype))

    update = tl.where(cc > 0, -lr, lr)
    tl.store(p_ptr + offs, (p * (1.0 - lr * WEIGHT_DECAY) + update).to(dtype))


@libentry()
@triton.jit
def lion_walk_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    lr,
    n_blocks,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``lion_block_kernel`` with each program walking several blocks; the trip
    count stays a runtime value so the loop is not unrolled.
    """
    dtype = anchor.dtype.element_ty
    n_programs = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    for i in range(tl.program_id(0), n_blocks, n_programs):
        g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
        p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
        m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))

        g = tl.load(g_ptr + offs).to(tl.float32)
        p = tl.load(p_ptr + offs).to(tl.float32)
        m = tl.load(m_ptr + offs).to(tl.float32)

        cc = BETA1 * m + (1.0 - BETA1) * g
        m_new = BETA2 * m + (1.0 - BETA2) * g
        tl.store(m_ptr + offs, m_new.to(dtype))

        update = tl.where(cc > 0, -lr, lr)
        tl.store(p_ptr + offs, (p * (1.0 - lr * WEIGHT_DECAY) + update).to(dtype))


@libentry()
@triton.jit
def lion_partial_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    starts,
    remains,
    lr,
    n_blocks,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Blocks of tensors whose size is not a multiple of ``BLOCK``: ``remain``
    masks the tail, and the loaded ``start`` keeps a runtime term in the
    index (a bare ``arange`` on a loaded base miscompiles here).
    """
    dtype = anchor.dtype.element_ty
    n_programs = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    for i in range(tl.program_id(0), n_blocks, n_programs):
        mask = lanes < tl.load(remains + i)
        start = tl.multiple_of(tl.load(starts + i), BLOCK)
        offs = tl.max_contiguous(start + lanes, BLOCK)
        g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
        p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
        m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))

        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        cc = BETA1 * m + (1.0 - BETA1) * g
        m_new = BETA2 * m + (1.0 - BETA2) * g
        tl.store(m_ptr + offs, m_new.to(dtype), mask=mask)

        update = tl.where(cc > 0, -lr, lr)
        tl.store(
            p_ptr + offs, (p * (1.0 - lr * WEIGHT_DECAY) + update).to(dtype), mask=mask
        )


def _build_metadata(g_list, p_list, m_list, block, partial):
    """Per-block pointer tables: exact tiles point at their own start, partial
    ones keep the base plus the block's offset and remaining count.
    """
    device = p_list[0].device
    numels = np.fromiter((t.numel() for t in g_list), dtype=np.int64, count=len(g_list))
    blocks_per_tensor = (numels + block - 1) // block
    total = int(blocks_per_tensor.sum())
    owner = np.repeat(np.arange(len(numels), dtype=np.int64), blocks_per_tensor)
    first_block = np.concatenate(([0], np.cumsum(blocks_per_tensor)[:-1]))
    within = np.arange(total, dtype=np.int64) - np.repeat(
        first_block, blocks_per_tensor
    )
    starts = within * block

    byte_offsets = 0 if partial else starts * g_list[0].element_size()

    def tables(tensors):
        base = np.fromiter(
            (t.data_ptr() for t in tensors), dtype=np.int64, count=len(tensors)
        )
        return torch.from_numpy(base[owner] + byte_offsets).to(device)

    g_ptrs = tables(g_list)
    p_ptrs = tables(p_list)
    m_ptrs = tables(m_list)
    if not partial:
        return g_ptrs, p_ptrs, m_ptrs, None, None
    starts_dev = torch.from_numpy(starts).to(device)
    remain_dev = torch.from_numpy(numels[owner] - starts).to(device)
    return g_ptrs, p_ptrs, m_ptrs, starts_dev, remain_dev


# Keyed by the exact tensor set, so an optimizer loop hits every step after the first.
_META_CACHE = {}


def _get_metadata(g_list, p_list, m_list, block, partial):
    key = (
        block,
        partial,
        g_list[0].dtype,
        tuple(t.data_ptr() for t in g_list),
        tuple(t.data_ptr() for t in p_list),
        tuple(t.data_ptr() for t in m_list),
        tuple(t.numel() for t in g_list),
    )
    entry = _META_CACHE.get(key)
    if entry is None:
        if len(_META_CACHE) >= 64:
            cur = next(iter(_META_CACHE))
            del _META_CACHE[cur]
        entry = _build_metadata(g_list, p_list, m_list, block, partial)
        _META_CACHE[key] = entry
    return entry


def _launch_group(triples, block, n_blocks, grid, walk, full, lr, params):
    g_list = [t[0] for t in triples]
    p_list = [t[1] for t in triples]
    m_list = [t[2] for t in triples]
    anchor = g_list[0]
    # An 8192-lane walking tile only fits the UB single-buffered.
    stage = {"num_stages": 1} if block >= _BLOCK_MAX else {}

    if full and not walk:
        g_ptrs, p_ptrs, m_ptrs, _, _ = _get_metadata(
            g_list, p_list, m_list, block, False
        )
        with torch_device_fn.device(anchor.device):
            lion_block_kernel[(n_blocks,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                lr,
                anchor,
                BETA1=params[0],
                BETA2=params[1],
                WEIGHT_DECAY=params[2],
                BLOCK=block,
            )
        return

    partial = not full
    g_ptrs, p_ptrs, m_ptrs, starts, remains = _get_metadata(
        g_list, p_list, m_list, block, partial
    )
    with torch_device_fn.device(anchor.device):
        if partial:
            lion_partial_kernel[(grid,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                starts,
                remains,
                lr,
                n_blocks,
                anchor,
                BETA1=params[0],
                BETA2=params[1],
                WEIGHT_DECAY=params[2],
                BLOCK=block,
                **stage,
            )
        else:
            lion_walk_kernel[(grid,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                lr,
                n_blocks,
                anchor,
                BETA1=params[0],
                BETA2=params[1],
                WEIGHT_DECAY=params[2],
                BLOCK=block,
                **stage,
            )


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
    """Fused Lion step over ``[grads, params, exp_avgs]`` (Ascend).

    Same contract as ``flag_train.deepspeed.multi_tensor_lion``; ``lr`` is
    the only runtime scalar, so the betas and weight decay are baked and a
    caller that changes one pays one recompile. ``chunk_size`` is accepted for
    interface parity and ignored.
    """
    logger.debug("TRAIN_ASCEND MULTI_TENSOR_LION")
    _check_chunk_size(chunk_size, "multi_tensor_lion")
    g_list, p_list, m_list = _unpack_tensor_lists(tensor_lists, 3, "multi_tensor_lion")
    dtype, _ = _validate_tensor_lists(
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

    groups = {}
    itemsize = g_list[0].element_size()
    for g, p, m in zip(g_list, p_list, m_list):
        block = _block_for(g.numel(), itemsize)
        if block == 0:
            continue
        groups.setdefault((dtype, block), []).append((g, p, m))

    params = (float(beta1), float(beta2), float(weight_decay))
    lr = float(lr)
    for (_, block), triples in groups.items():
        numels = [t[0].numel() for t in triples]
        n_blocks = sum((n + block - 1) // block for n in numels)
        grid, walk, full = _plan_launch(numels, block)
        if grid == 0:
            continue
        _launch_group(triples, block, n_blocks, grid, walk, full, lr, params)
