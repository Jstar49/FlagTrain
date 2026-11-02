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
"""Performance benchmark for the fused routing-map padding operator.

The baseline is picked by what the platform can run, following the same rule
``benchmark/deepspeed/test_blocked_flash.py`` picks DeepSpeed's kernel by:

* on a backend in ``_NV_BASELINE_VENDORS``, Megatron-LM's own
  ``fused_pad_routing_map`` **is** the baseline. It is the kernel this port is a
  port *of*, which makes it the only baseline that answers "what did the port
  cost", and it is *required*: if the installed package cannot provide it the
  module raises rather than timing something else and reporting it under that
  name;
* on any other backend the baseline is ``pad_routing_map_ref``, the plain-torch
  transcription of Megatron's ``moe_utils.pad_routing_map``. It is not a
  competitor, and the speedup there says how far the kernel is from *a* correct
  implementation rather than from the best one.

The same kernel is held to the torch composition by the functional suite
(``tests/megatron/test_fused_pad_routing_map.py``) on every NVIDIA run, so the
timed baseline is not an unchecked one. That check is not repeated here: it
belongs to the tests, and this file only times the kernel.

Both baselines take the same ``(routing_map, pad_multiple)`` tuple, so there is
nothing to pre-compute outside the timed region. The input is a boolean top-k map
-- the operator's real input in Megatron's MoE dispatcher -- and both sides return
int32 ``[num_tokens, num_experts]``.

The shapes are deployments rather than bare tensor sizes: each row of
``_PAD_ROUTING_MAP_CONFIGS`` is a real model's expert count and top-k, the token
count it feeds through one dispatch, and the alignment its quantization path
asks for (Megatron's ``get_align_size_for_quantization``: 256 for TE's
grouped-tensor MLP, 16 for blockwise FP8, 32 for MXFP8/FP4). One synthetic row
crosses the kernel's tile boundary.
"""

import importlib.metadata

import pytest
import torch

import flag_train
from flag_train.megatron import fused_pad_routing_map

from .. import base

# Real deployments, as ``(num_tokens, num_experts, top_k, pad_multiple)``.
#
# ``num_experts`` is the routing map's expert dimension -- the global count with
# the all-to-all dispatcher, the local slice with flex/DeepEP -- and
# ``pad_multiple`` is what Megatron's ``get_align_size_for_quantization`` returns
# for that deployment: 256 for TE's grouped-tensor MLP, 16 for blockwise FP8,
# 32 for MXFP8 and FP4. ``num_tokens`` spans one micro-batch of Megatron's own
# MoE performance proxies (4096) through the larger maps the same models feed in
# production. The last row is synthetic: a non-power-of-two row that still fits
# one masked tile, next to the multi-tile paths the 16K/32K rows walk.
_PAD_ROUTING_MAP_CONFIGS = [
    (4096, 8, 2, 256),  # Mixtral 8x7B (Megatron's MIXTRAL_PROXY), TE grouped MLP
    (16384, 8, 2, 256),  # Mixtral, four micro-batches
    (8192, 16, 4, 16),  # DBRX 132B, blockwise FP8
    (4096, 64, 8, 32),  # OLMoE 1B-7B, MXFP8
    (8192, 128, 8, 256),  # Qwen3 235B-A22B, TE grouped MLP
    (8192, 32, 8, 16),  # DeepSeek-V3 on EP=8: one rank's local expert slice
    (32768, 8, 2, 256),  # Mixtral, a large dispatch map
    (32768, 256, 8, 256),  # DeepSeek-V3's full 256-expert map
    (4097, 64, 8, 256),  # synthetic: non-power-of-two row, one masked tile
]

# ---------------------------------------------------------------------------
# Reference implementation
#
# Megatron's ``megatron.core.transformer.moe.moe_utils.pad_routing_map``,
# transcribed, then cast to int32 to match the operator's output dtype.
#
# !! KEEP IN SYNC with tests/megatron/test_fused_pad_routing_map.py !!
#
# This is a second, byte-identical copy of the reference that file uses as its
# primary oracle. The duplication is deliberate -- a reference belongs to the
# tests that check against it, not to the operator package, so that a checked
# implementation is never code the operator ships. The cost is that the two
# copies can drift: change one and you must change the other, or this benchmark
# will measure against a different contract than the tests accept.
# ---------------------------------------------------------------------------


def pad_routing_map_ref(routing_map, pad_multiple):
    """Reference for ``fused_pad_routing_map``, composed from torch ops."""
    if routing_map.shape[0] == 0:
        return routing_map

    work = routing_map.clone().transpose(0, 1)  # [num_experts, num_tokens]

    num_ones = work.sum(dim=1)
    num_to_pad = (-num_ones) % pad_multiple

    is_zero = work == 0
    zero_ranks = torch.cumsum(is_zero.int(), dim=1)
    work[zero_ranks <= num_to_pad.unsqueeze(1)] = 1

    return work.transpose(0, 1).int()


# ---------------------------------------------------------------------------
# Baseline, by platform
#
# * on a backend in ``_NV_BASELINE_VENDORS`` Megatron-LM's kernel **is** the
#   baseline -- it is the operator this port is a port of. If it cannot be
#   loaded there the module raises rather than falling back, so a missing
#   baseline is reported instead of silently measuring something else. The
#   functional suite holds that kernel to the torch composition; this file only
#   times it;
# * elsewhere the plain-torch composition, which is not a competitor.
# ---------------------------------------------------------------------------
_NV_BASELINE_VENDORS = {"nvidia"}


def _load_mcore_pad_routing_map():
    """Megatron-LM's ``fused_pad_routing_map``, or ``None`` on another backend.

    Imported from the installed ``megatron-core`` package, the way the
    blocked_flash benchmark imports DeepSpeed's kernel: the baseline is the
    operator itself, not a copy of it that could drift. Importing
    ``megatron.core`` runs its ``__init__`` and pulls in the whole core tree, so
    it happens once at module import, behind the vendor gate.
    """
    if flag_train.vendor_name not in _NV_BASELINE_VENDORS:
        return None

    try:
        from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

        return fused_pad_routing_map
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use Megatron-LM's fused_pad_routing_map "
            f"as its baseline, but megatron-core could not provide it: {exc!r}. "
            f"Install megatron-core for this platform, or drop the backend from "
            f"_NV_BASELINE_VENDORS."
        ) from exc


# Resolved once, at module import time.
_mcore_pad_routing_map = _load_mcore_pad_routing_map()

# The baseline's version, recorded in the report line: a speedup means little
# without the version of the kernel it was measured against.
_BASELINE = (
    "megatron fused_pad_routing_map "
    f"(megatron-core {importlib.metadata.version('megatron-core')})"
    if _mcore_pad_routing_map is not None
    else "pad_routing_map_ref (torch)"
)


class PadRoutingMapBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _PAD_ROUTING_MAP_CONFIGS
    DEFAULT_SHAPE_DESC = "num_tokens, num_experts, top_k, pad_multiple"

    def set_shapes(self, shape_file=None):
        # Each row is a deployment, not a bare tensor size; the generic
        # 1D/2D/3D extra shapes do not apply.
        self.shapes = list(_PAD_ROUTING_MAP_CONFIGS)

    def set_more_shapes(self):
        return []

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        """Boundary traffic: read the bool map, write the int32 padded map.

        The operator's internal ``.int()`` working copy is not counted, so this
        is the I/O the contract implies rather than every byte the kernels move.
        """
        num_tokens, num_experts = args[0].shape
        io_bytes = num_tokens * num_experts * (args[0].element_size() + 4)
        return io_bytes * 1e-9 / (latency * 1e-3)


def _make_routing_map(num_tokens, num_experts, top_k, device=None):
    """A boolean top-k routing map, the operator's real input.

    Logits are independent uniform draws exactly as a router's would be before
    training; the map keeps the top ``top_k`` per token, so expert loads are
    uneven in the same way a real batch's are.
    """
    device = flag_train.device if device is None else device
    shape = (num_tokens, num_experts)
    logits = torch.randn(shape, dtype=torch.float32, device=device)
    indices = logits.topk(min(top_k, num_experts), dim=-1).indices
    routing_map = torch.zeros(shape, dtype=torch.bool, device=device)
    routing_map.scatter_(1, indices, True)
    return routing_map


def pad_routing_map_input_fn(shape, dtype, device):
    num_tokens, num_experts, top_k, pad_multiple = shape
    yield _make_routing_map(num_tokens, num_experts, top_k, device), pad_multiple


def torch_op(routing_map, pad_multiple):
    """Baseline, chosen by platform. See the module docstring."""
    baseline = _mcore_pad_routing_map or pad_routing_map_ref
    return baseline(routing_map, pad_multiple)


def train_op(routing_map, pad_multiple):
    """The operator under test."""
    return fused_pad_routing_map(routing_map, pad_multiple)


@pytest.mark.fused_pad_routing_map
def test_fused_pad_routing_map_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = PadRoutingMapBenchmark(
        input_fn=pad_routing_map_input_fn,
        op_name="fused_pad_routing_map",
        torch_op=torch_op,
        # The dispatcher's map is boolean, so that is the contract measured here.
        dtypes=[torch.bool],
        # Megatron's own kernel is a real local baseline; the torch transcription
        # is not, and --base-data still applies to it.
        native_baseline=_mcore_pad_routing_map is not None,
    )
    bench.set_train(train_op)
    bench.run()
