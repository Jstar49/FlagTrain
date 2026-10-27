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
"""Fused output RMSNorm / SiLU gating (GDN output-norm fusion).

Forward-only port of ``megatron.core.fusions.fused_gated_norm``, adapted to
the FlagTrain vendor-abstraction layer.  Semantics match the upstream kernel:

* output RMSNorm over the trailing ``dim``, with an optional ``zero_centered``
  weight offset (``w -> w + 1``);
* SiLU gating by a same-shaped ``gate`` tensor;
* the normalised activation is materialised in the activation dtype *before*
  the FP32 gating multiply, preserving the rounding behaviour of the unfused
  ``RMSNorm -> x * silu(gate)`` pipeline.

The operator handles arbitrary (possibly non-contiguous) ``[batch, sequence,
heads, dim]`` views, addressing logical rows through the per-dimension strides.
It has no backward: it is the forward/inference path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry

logger = logging.getLogger(__name__)

# Row count past which a 128-wide head is large enough to saturate DRAM; from
# there on the wider one-warp tile measures faster than the smaller two-warp
# one (see ``_launch_config``).
_LARGE_ROWS = 1 << 17


@triton.jit
def _row_offset(
    row, SEQUENCE: tl.constexpr, HEADS: tl.constexpr, stride_b, stride_s, stride_h
):
    """Address logical ``[batch, sequence, head]`` rows without copying views."""
    token = row // HEADS
    return (
        (token // SEQUENCE) * stride_b
        + (token % SEQUENCE) * stride_s
        + (row % HEADS) * stride_h
    )


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
    EVEN_ROWS: tl.constexpr,
    I64: tl.constexpr,
):
    """Apply output RMSNorm and SiLU gating."""
    if I64:
        row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    else:
        row = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    off = row[:, None] * D + d[None, :]
    if X_CONTIGUOUS:
        x_ptr = x + off
    else:
        xrow = _row_offset(
            row, SEQUENCE, HEADS, X_STRIDES[0], X_STRIDES[1], X_STRIDES[2]
        )
        x_ptr = x + xrow[:, None] + d[None, :] * X_STRIDES[3]
    if GATE_CONTIGUOUS:
        gate_ptr = gate + off
    else:
        grow = _row_offset(
            row, SEQUENCE, HEADS, GATE_STRIDES[0], GATE_STRIDES[1], GATE_STRIDES[2]
        )
        gate_ptr = gate + grow[:, None] + d[None, :] * GATE_STRIDES[3]

    # Issue the gate load together with the activation load, before the RMS
    # reduction.  The two input streams then overlap in flight instead of the
    # gate waiting behind the reduction.
    if EVEN_ROWS:
        xv = tl.load(x_ptr, eviction_policy="evict_first").to(tl.float32)
        gate_raw = tl.load(gate_ptr, eviction_policy="evict_first")
    else:
        valid = row[:, None] < ROWS
        xv = tl.load(x_ptr, valid, 0, eviction_policy="evict_first").to(tl.float32)
        gate_raw = tl.load(gate_ptr, valid, 0, eviction_policy="evict_first")

    w = tl.load(weight + d).to(tl.float32)
    if ZERO_CENTERED:
        w += 1
    inv = tl.rsqrt(tl.sum(xv * xv, 1) / D + EPS)
    # Materialise RMSNorm in the activation dtype before FP32 gating.
    norm = (xv * inv[:, None] * w[None, :]).to(x.dtype.element_ty).to(tl.float32)
    gv = gate_raw.to(tl.float32)
    res = norm * (gv * tl.sigmoid(gv))
    if EVEN_ROWS:
        tl.store(out + off, res)
    else:
        tl.store(out + off, res, valid)


def _launch_config(dim: int, rows: int) -> tuple[int, int]:
    """Pick ``(rows per program, warps per program)`` for one launch.

    The kernel is bandwidth-bound, so the winning tile keeps a row's reduction
    inside one warp while giving each thread enough contiguous elements to
    vectorise: one row per program for ``dim >= 512`` (two for ``dim == 256``),
    eight rows for 64-wide heads, and, for 128-wide heads, the wider one-warp
    tile only once the tensors are large enough to be DRAM-bound.  The
    boundaries are measured rather than derived.
    """
    if dim <= 64:
        return 8, 2
    if dim == 128:
        return (8, 1) if rows >= _LARGE_ROWS else (4, 2)
    if dim == 256:
        return 2, 1
    # One row per program, about 512 columns per warp for wider heads.
    return 1, min(8, max(1, dim // 512))


def _check_arguments(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor) -> int:
    """Validate the operands; return the head dimension.

    Shared with backend overrides so a specialised implementation cannot come to
    accept something the generic one rejects, or the reverse.
    """
    if x.ndim != 4 or gate.shape != x.shape:
        raise ValueError("fused_gated_norm: gate must match x's 4-D shape")
    if x.device.type != runtime_device.name:
        raise ValueError(
            f"fused_gated_norm: x must be on the {runtime_device.name} device"
        )
    if gate.device != x.device:
        raise ValueError("fused_gated_norm: gate must be on x's device")
    dim = x.shape[-1]
    if dim & (dim - 1):
        raise ValueError("fused_gated_norm: requires a power-of-two head dimension")
    if (
        weight.shape != (dim,)
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "fused_gated_norm: weight must be a contiguous (dim,) tensor on the input device"
        )
    return dim


@torch.compiler.disable
def fused_gated_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Run fused RMSNorm/SiLU gating (forward only).

    Args:
        x: Core attention output, ``[batch, sequence, heads, dim]``.
        gate: SiLU gate, same shape as ``x`` (``x.dtype`` or ``torch.float32``).
        weight: Contiguous RMSNorm weight, shape ``(dim,)``.
        eps: RMSNorm epsilon.
        zero_centered: If ``True``, apply ``weight + 1`` before gating.
    """
    logger.debug("TRAIN FUSED_GATED_NORM")

    dim = _check_arguments(x, gate, weight)

    batch, sequence, heads, _ = x.shape
    rows = batch * sequence * heads
    x_contiguous = x.is_contiguous()
    gate_contiguous = gate.is_contiguous()
    block_rows, num_warps = _launch_config(dim, rows)
    grid = triton.cdiv(rows, block_rows)
    # 32-bit index arithmetic halves the size of the offset tile.  It is only
    # valid when no address computed for the padded block can pass 2**31
    # elements: guaranteed for plain contiguous operands of this size, not for
    # arbitrary strided views.
    padded_elems = grid * block_rows * dim
    index_i64 = padded_elems >= (1 << 31) or not (x_contiguous and gate_contiguous)
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
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
            EVEN_ROWS=rows % block_rows == 0,
            I64=index_i64,
            num_warps=num_warps,
        )
    return out
