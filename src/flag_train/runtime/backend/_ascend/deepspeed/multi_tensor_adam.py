"""Ascend multi_tensor_adam: 8192-lane tiles, walked when the program count
would otherwise be large. A program costs tens of ns whatever it carries
here, so the generic 1024-lane masked shape left the memory path
unreached. Arithmetic and interface are the generic operator's;
``_ascend/deepspeed/__init__.py`` registers this by name.
"""

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from flag_train.deepspeed.multi_tensor_adam import (
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

# ~16384 lanes of work (about 0.5 MB over seven fp32 streams) per program.
_LANES_PER_PROGRAM = 16384
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
    """``(grid, walk, full)`` for a group sharing ``block``: about 16384 lanes of
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
def adam_block_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    v_ptrs,
    lr,
    bc1,
    bc2,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    EPSILON: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complete Adam block from pre-offset pointers: unmasked, ``lr``/``bc1``/
    ``bc2`` the only runtime scalars. ``p -= lr * m_hat / (sqrt(v_hat)+eps)``
    with ``MODE == 1`` adding ``decay*p`` inside the update.
    """
    dtype = anchor.dtype.element_ty
    i = tl.program_id(0)
    g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
    p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
    m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))
    v_ptr = tl.load(v_ptrs + i).to(tl.pointer_type(dtype))

    offs = tl.arange(0, BLOCK)
    g = tl.load(g_ptr + offs).to(tl.float32)
    p = tl.load(p_ptr + offs).to(tl.float32)
    m = tl.load(m_ptr + offs).to(tl.float32)
    v = tl.load(v_ptr + offs).to(tl.float32)

    if MODE == 0:
        g = g + WEIGHT_DECAY * p
        m_new = BETA1 * m + (1.0 - BETA1) * g
        v_new = BETA2 * v + (1.0 - BETA2) * g * g
        update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON)
        p_new = p - lr * update
    else:
        m_new = BETA1 * m + (1.0 - BETA1) * g
        v_new = BETA2 * v + (1.0 - BETA2) * g * g
        update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON) + WEIGHT_DECAY * p
        p_new = p - lr * update

    tl.store(p_ptr + offs, p_new.to(dtype))
    tl.store(m_ptr + offs, m_new.to(dtype))
    tl.store(v_ptr + offs, v_new.to(dtype))


@libentry()
@triton.jit
def adam_walk_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    v_ptrs,
    lr,
    bc1,
    bc2,
    n_blocks,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    EPSILON: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``adam_block_kernel`` with each program walking several blocks; the trip
    count stays a runtime value so the loop is not unrolled.
    """
    dtype = anchor.dtype.element_ty
    n_programs = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    for i in range(tl.program_id(0), n_blocks, n_programs):
        g_ptr = tl.load(g_ptrs + i).to(tl.pointer_type(dtype))
        p_ptr = tl.load(p_ptrs + i).to(tl.pointer_type(dtype))
        m_ptr = tl.load(m_ptrs + i).to(tl.pointer_type(dtype))
        v_ptr = tl.load(v_ptrs + i).to(tl.pointer_type(dtype))

        g = tl.load(g_ptr + offs).to(tl.float32)
        p = tl.load(p_ptr + offs).to(tl.float32)
        m = tl.load(m_ptr + offs).to(tl.float32)
        v = tl.load(v_ptr + offs).to(tl.float32)

        if MODE == 0:
            g = g + WEIGHT_DECAY * p
            m_new = BETA1 * m + (1.0 - BETA1) * g
            v_new = BETA2 * v + (1.0 - BETA2) * g * g
            update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON)
            p_new = p - lr * update
        else:
            m_new = BETA1 * m + (1.0 - BETA1) * g
            v_new = BETA2 * v + (1.0 - BETA2) * g * g
            update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON) + WEIGHT_DECAY * p
            p_new = p - lr * update

        tl.store(p_ptr + offs, p_new.to(dtype))
        tl.store(m_ptr + offs, m_new.to(dtype))
        tl.store(v_ptr + offs, v_new.to(dtype))


@libentry()
@triton.jit
def adam_partial_kernel(
    g_ptrs,
    p_ptrs,
    m_ptrs,
    v_ptrs,
    starts,
    remains,
    lr,
    bc1,
    bc2,
    n_blocks,
    anchor,
    BETA1: tl.constexpr,
    BETA2: tl.constexpr,
    EPSILON: tl.constexpr,
    WEIGHT_DECAY: tl.constexpr,
    MODE: tl.constexpr,
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
        v_ptr = tl.load(v_ptrs + i).to(tl.pointer_type(dtype))

        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        if MODE == 0:
            g = g + WEIGHT_DECAY * p
            m_new = BETA1 * m + (1.0 - BETA1) * g
            v_new = BETA2 * v + (1.0 - BETA2) * g * g
            update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON)
            p_new = p - lr * update
        else:
            m_new = BETA1 * m + (1.0 - BETA1) * g
            v_new = BETA2 * v + (1.0 - BETA2) * g * g
            update = (m_new / bc1) / (tl.sqrt(v_new / bc2) + EPSILON) + WEIGHT_DECAY * p
            p_new = p - lr * update

        tl.store(p_ptr + offs, p_new.to(dtype), mask=mask)
        tl.store(m_ptr + offs, m_new.to(dtype), mask=mask)
        tl.store(v_ptr + offs, v_new.to(dtype), mask=mask)


def _build_metadata(g_list, p_list, m_list, v_list, block, partial):
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
    v_ptrs = tables(v_list)
    if not partial:
        return g_ptrs, p_ptrs, m_ptrs, v_ptrs, None, None
    starts_dev = torch.from_numpy(starts).to(device)
    remain_dev = torch.from_numpy(numels[owner] - starts).to(device)
    return g_ptrs, p_ptrs, m_ptrs, v_ptrs, starts_dev, remain_dev


# Keyed by the exact tensor set, so an optimizer loop hits every step after the first.
_META_CACHE = {}


def _get_metadata(g_list, p_list, m_list, v_list, block, partial):
    key = (
        block,
        partial,
        g_list[0].dtype,
        tuple(t.data_ptr() for t in g_list),
        tuple(t.data_ptr() for t in p_list),
        tuple(t.data_ptr() for t in m_list),
        tuple(t.data_ptr() for t in v_list),
        tuple(t.numel() for t in g_list),
    )
    entry = _META_CACHE.get(key)
    if entry is None:
        if len(_META_CACHE) >= 64:
            cur = next(iter(_META_CACHE))
            del _META_CACHE[cur]
        entry = _build_metadata(g_list, p_list, m_list, v_list, block, partial)
        _META_CACHE[key] = entry
    return entry


def _launch_group(triples, block, n_blocks, grid, walk, full, values, consts):
    g_list, p_list, m_list, v_list = zip(*triples)
    anchor = g_list[0]
    # An 8192-lane walking tile only fits the UB single-buffered.
    stage = {"num_stages": 1} if block >= _BLOCK_MAX else {}

    if full and not walk:
        g_ptrs, p_ptrs, m_ptrs, v_ptrs, _, _ = _get_metadata(
            g_list, p_list, m_list, v_list, block, False
        )
        with torch_device_fn.device(anchor.device):
            adam_block_kernel[(n_blocks,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                v_ptrs,
                values[0],
                values[1],
                values[2],
                anchor,
                BETA1=consts[0],
                BETA2=consts[1],
                EPSILON=consts[2],
                WEIGHT_DECAY=consts[3],
                MODE=consts[4],
                BLOCK=block,
            )
        return

    partial = not full
    g_ptrs, p_ptrs, m_ptrs, v_ptrs, starts, remains = _get_metadata(
        g_list, p_list, m_list, v_list, block, partial
    )
    with torch_device_fn.device(anchor.device):
        if partial:
            adam_partial_kernel[(grid,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                v_ptrs,
                starts,
                remains,
                values[0],
                values[1],
                values[2],
                n_blocks,
                anchor,
                BETA1=consts[0],
                BETA2=consts[1],
                EPSILON=consts[2],
                WEIGHT_DECAY=consts[3],
                MODE=consts[4],
                BLOCK=block,
                **stage,
            )
        else:
            adam_walk_kernel[(grid,)](
                g_ptrs,
                p_ptrs,
                m_ptrs,
                v_ptrs,
                values[0],
                values[1],
                values[2],
                n_blocks,
                anchor,
                BETA1=consts[0],
                BETA2=consts[1],
                EPSILON=consts[2],
                WEIGHT_DECAY=consts[3],
                MODE=consts[4],
                BLOCK=block,
                **stage,
            )


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
    """Batched Adam/AdamW step over ``[g, p, m, v]`` (Ascend).

    Same contract as ``flag_train.deepspeed.multi_tensor_adam``; betas,
    epsilon, weight decay and mode are baked, ``lr``/``bc1``/``bc2`` are
    runtime. ``chunk_size`` is accepted for interface parity and ignored.
    """
    logger.debug("TRAIN_ASCEND MULTI_TENSOR_ADAM")
    _check_chunk_size(chunk_size, "multi_tensor_adam")
    g_list, p_list, m_list, v_list = _unpack_tensor_lists(
        tensor_lists, 4, "multi_tensor_adam"
    )
    dtype, _ = _validate_tensor_lists(
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

    groups = {}
    itemsize = g_list[0].element_size()
    for g, p, m, v in zip(g_list, p_list, m_list, v_list):
        block = _block_for(g.numel(), itemsize)
        if block == 0:
            continue
        groups.setdefault((dtype, block), []).append((g, p, m, v))

    values = (float(lr), float(bc1), float(bc2))
    consts = (
        float(beta1),
        float(beta2),
        float(epsilon),
        float(weight_decay),
        int(mode),
    )
    for (_, block), triples in groups.items():
        numels = [t[0].numel() for t in triples]
        n_blocks = sum((n + block - 1) // block for n in numels)
        grid, walk, full = _plan_launch(numels, block)
        if grid == 0:
            continue
        _launch_group(triples, block, n_blocks, grid, walk, full, values, consts)
