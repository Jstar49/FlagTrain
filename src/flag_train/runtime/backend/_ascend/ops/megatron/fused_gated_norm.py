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
"""Ascend-tuned fused RMSNorm/SiLU gating.

The generic implementation is ``flag_train.megatron.fused_gated_norm`` and is
what every backend without an override runs. The arithmetic here is that
module's, unchanged -- same casts, same rounding, same checks, and the strided
addressing is imported from it rather than rewritten. Only the launch shape
differs, because the generic one is priced for a machine where a program is
cheap.

**A program costs ~73 ns here whatever it carries.** Sweeping rows per program
over (2, 1024, 16, 128): 32768 programs of one row took 2.40 ms, 1024 programs
of 32 rows 73 us, 512 programs of 64 rows 47 us -- the same 73 ns per program at
three different tile sizes. With 40 AI cores that is a wave-latency cost paid
per program, and the generic launcher hands this kernel 8192 of them. It ran
that shape at 42 GB/s against a copy of the same tensors at 2.5 TB/s.

So the grid is the core count and each program walks the tiles inside the
kernel. That alone takes the shape from 572 us to 36 us.

**The tile is 8192 elements.** With the walk in place the tile only has to cover
the latency of its own loads. ``BT = 8192 // D`` is where that saturates and the
largest that compiles -- one power of two up (128 rows at D=128, 64 at D=256, 32
at D=512) overflows the 192 KB unified buffer at compile time. Measured across
D = 16..512 the optimum is exactly that rule. The masked tile costs the same as
an even one (36.0 against 35.8 us), so this kernel masks unconditionally instead
of carrying the generic ``EVEN_ROWS`` split.

**What is left is the row reduction.** It is 9.5 us of the 30.6 us that shape
takes at BT=64, measured by deleting it while keeping the loads live; every
further fp32 pass over the tile is ~1 us, against a vector unit retiring 64
fp32 lanes a cycle, so the elementwise arithmetic is already at the hardware
rate. Three ways to make the reduction cheaper are all worse and were measured:
reducing axis 0 of a (D, BT) tile (86-568 us -- the transposed gather is the
cost), reshaping to (BT, D/8, 8) and summing twice (3.5x slower), and a Cube
matvec ``tl.dot(x*x, ones)`` (72 us, and wrong). It stays as it is.

**Strided operands keep the generic addressing**, and it costs nothing: 41.9 us
through the shared ``_row_offset`` against 41.8 us inlined. What matters is that
the strides reach it as ``constexpr`` -- the generic passes ``X_STRIDES`` that
way already -- so ``token % SEQUENCE`` and ``head * stride_h`` fold to shifts
and masks. With the same strides passed as runtime scalars the kernel takes
135 us. The tile is halved for them (``_STRIDED_TILE_ELEMS``), because a tile
that spans four tokens instead of two pays for the extra boundaries.

**A stride that is not 1 on the innermost axis is copied instead.** That layout
turns every load into a gather down the head axis, which this device prices at
~4.8 ms for (2, 1024, 16, 128) against 21 us to materialise the tensor
contiguously; the same tile also overflows the unified buffer past 1024
elements. A copy is a semantic no-op, so the kernel never sees that layout.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.megatron.fused_gated_norm import _check_arguments, _row_offset
from flag_train.runtime import torch_device_fn
from flag_train.runtime.backend._ascend.utils import CORE_NUM
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# Elements in one program's tile; see the module docstring for why this size
# and why BT is capped.
_TILE_ELEMS = 8192
# A strided operand pays for the tile's row offsets, and a tile that crosses
# more token boundaries pays more: 41.5 us against 51.2 at 8192 elements.
_STRIDED_TILE_ELEMS = 4096
_MAX_ROWS_PER_PROGRAM = 128


@libentry()
@triton.jit
def _gated_norm_fwd(
    x,
    gate,
    weight,
    out,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SEQUENCE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    GATE_STRIDES: tl.constexpr,
    X_CONTIGUOUS: tl.constexpr,
    GATE_CONTIGUOUS: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    BT: tl.constexpr,
    NBLK: tl.constexpr,
    NPROG: tl.constexpr,
    I64: tl.constexpr,
):
    """Persistent RMSNorm + SiLU gating: one program walks many row tiles."""
    pid = tl.program_id(0)
    d = tl.arange(0, D)
    r = tl.arange(0, BT)
    w = tl.load(weight + d).to(tl.float32)
    if ZERO_CENTERED:
        w += 1

    for blk in tl.range((NBLK + NPROG - 1) // NPROG):
        idx = pid + blk * NPROG
        if I64:
            idx = idx.to(tl.int64)
        row = idx * BT + r
        # The walk overshoots NBLK on the last pass, so both the row count and
        # the tile index bound the access.
        valid = (row[:, None] < ROWS) & (idx < NBLK)

        if X_CONTIGUOUS:
            x_off = row[:, None] * D + d[None, :]
        else:
            x_off = (
                _row_offset(
                    row, SEQUENCE, HEADS, X_STRIDES[0], X_STRIDES[1], X_STRIDES[2]
                )[:, None]
                + d[None, :] * X_STRIDES[3]
            )
        if GATE_CONTIGUOUS:
            gate_off = row[:, None] * D + d[None, :]
        else:
            gate_off = (
                _row_offset(
                    row,
                    SEQUENCE,
                    HEADS,
                    GATE_STRIDES[0],
                    GATE_STRIDES[1],
                    GATE_STRIDES[2],
                )[:, None]
                + d[None, :] * GATE_STRIDES[3]
            )

        xv = tl.load(x + x_off, valid, 0).to(tl.float32)
        gate_raw = tl.load(gate + gate_off, valid, 0)
        # Scaling by the reciprocal, not dividing: D is a power of two, so the
        # two agree bit for bit and the multiply is one pass fewer.
        inv = tl.rsqrt(tl.sum(xv * xv, 1) * (1.0 / D) + EPS)
        # Materialise RMSNorm in the activation dtype before FP32 gating.
        norm = (xv * inv[:, None] * w[None, :]).to(x.dtype.element_ty).to(tl.float32)
        gv = gate_raw.to(tl.float32)
        res = norm * (gv * tl.sigmoid(gv))
        # ``out`` is always contiguous, whatever the inputs' layout is.
        tl.store(out + row[:, None] * D + d[None, :], res, valid)


def _tile_rows(dim, rows, strided):
    """Rows per program: a full tile, split when the input is tiny.

    Below the tile size the mask would cover most of the tile, so halving keeps
    at least two programs busy -- 8 rows over 16 measures 1.48 us against 1.68
    for one program of 16.
    """
    elems = _STRIDED_TILE_ELEMS if strided else _TILE_ELEMS
    rows_per_tile = min(_MAX_ROWS_PER_PROGRAM, max(1, elems // dim))
    return min(rows_per_tile, max(1, triton.next_power_of_2(rows) // 2))


def _index_is_wide(x, gate, span):
    """Whether 32-bit offsets can reach every element this launch touches."""
    for t in (x, gate):
        if not t.is_contiguous():
            span = max(
                span, sum((n - 1) * abs(s) for n, s in zip(t.shape, t.stride())) + 1
            )
    return span >= (1 << 31)


@torch.compiler.disable
def fused_gated_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Run fused RMSNorm/SiLU gating (forward only).

    Same signature, arithmetic and returned tensor as
    ``flag_train.megatron.fused_gated_norm``; only the launch differs -- see this
    module's docstring.
    """
    logger.debug("TRAIN_ASCEND FUSED_GATED_NORM")

    dim = _check_arguments(x, gate, weight)

    batch, sequence, heads, _ = x.shape
    rows = batch * sequence * heads
    x_contiguous = x.is_contiguous()
    gate_contiguous = gate.is_contiguous()
    # See the docstring: an innermost stride other than 1 gathers, and a copy is
    # an order of magnitude cheaper.
    if not x_contiguous and x.stride(-1) != 1:
        x, x_contiguous = x.contiguous(), True
    if not gate_contiguous and gate.stride(-1) != 1:
        gate, gate_contiguous = gate.contiguous(), True

    block_rows = _tile_rows(dim, rows, not (x_contiguous and gate_contiguous))
    n_blocks = triton.cdiv(rows, block_rows)
    grid = min(CORE_NUM, n_blocks)

    # A fresh contiguous tensor: ``empty_like`` warns when handed a view.
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _gated_norm_fwd[(grid,)](
            x,
            gate,
            weight,
            out,
            ROWS=rows,
            HEADS=heads,
            D=dim,
            SEQUENCE=sequence,
            X_STRIDES=x.stride(),
            GATE_STRIDES=gate.stride(),
            X_CONTIGUOUS=x_contiguous,
            GATE_CONTIGUOUS=gate_contiguous,
            EPS=eps,
            ZERO_CENTERED=zero_centered,
            BT=block_rows,
            NBLK=n_blocks,
            NPROG=grid,
            I64=_index_is_wide(x, gate, n_blocks * block_rows * dim),
        )
    return out
