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
  ``fused_pad_routing_map`` **is** the baseline when it can be loaded. It is the
  kernel this port is a port *of*, which makes it the only baseline that
  answers "what did the port cost"; a load failure is reported rather than
  hidden behind another measurement;
* on any other backend, or when the kernel cannot be loaded, there is no
  on-machine baseline. A plain-torch transcription of Megatron's
  ``moe_utils.pad_routing_map`` is not a competitor and is not timed: a ratio
  against it says how far the kernel is from *a* correct implementation rather
  than from the best one. ``torch_op`` is ``None``, so the framework reports
  Train latency only and ``--base-data`` is the only comparison.

The same kernel is held to the torch composition by the functional suite
(``tests/megatron/test_fused_pad_routing_map.py``) on every NVIDIA run, so the
timed baseline is not an unchecked one. That check is not repeated here: it
belongs to the tests, and this file only times the kernel.

The baseline takes the same ``(routing_map, pad_multiple)`` tuple as the
operator, so there is nothing to pre-compute outside the timed region. The input
is a top-k map -- boolean, the operator's real input in Megatron's MoE
dispatcher, and also int32, the same map after the cast a dispatcher may already
have applied (which takes the integer counting path) -- and both sides return
int32 ``[num_tokens, num_experts]``.

Most rows of ``_PAD_ROUTING_MAP_CONFIGS`` are deployments rather than bare tensor
sizes: a real model's expert count and top-k, the token count it feeds through
one dispatch, and the alignment its quantization path asks for (Megatron's
``get_align_size_for_quantization``: 256 for TE's grouped-tensor MLP, 16 for
blockwise FP8, 32 for MXFP8/FP4). Synthetic rows add the axes the deployments do
not cover: tile boundaries, a ragged expert tile, a launch-bound map, and a row
long enough to walk the partial-count sum more than once.
"""

import importlib.metadata
import warnings

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
# production. The 4097 row is synthetic: a non-power-of-two row that still fits
# one masked tile, next to the multi-tile paths the 16K/32K rows walk.
#
# The remaining synthetic rows cover axes the deployments do not: a
# non-power-of-two expert count whose last expert tile is masked, a row long
# enough to walk the partial-count sum more than once, and two decode-sized
# dispatches. The decode rows were added after the cross-hardware records were
# taken, so ``vs Base`` reports them as missing until
# ``tools/op_flagtrain_perf_baseline.json`` is regenerated; their latency is
# measured either way. One token reaches the single-chunk flip path and a
# 128-token micro-batch the small one -- neither is exercised by the deployment
# rows above, which start at 4096 tokens.
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
    (4096, 48, 8, 32),  # synthetic: non-power-of-two expert tile
    (131073, 8, 2, 32),  # synthetic: 1025 transpose tiles, partial-count loop
    (1, 64, 8, 32),  # decode: one token, the single-chunk flip
    (128, 64, 8, 32),  # decode: one micro-batch
]

# ---------------------------------------------------------------------------
# Baseline, by platform
#
# * on a backend in ``_NV_BASELINE_VENDORS`` Megatron-LM's kernel **is** the
#   baseline -- it is the operator this port is a port of. If it cannot be
#   loaded there the module raises rather than falling back, so a missing
#   baseline is reported instead of silently measuring something else. The
#   functional suite holds that kernel to the torch composition; this file only
#   times it;
# * elsewhere there is no baseline: the plain-torch composition is not a
#   competitor, so ``torch_op`` is None and ``--base-data`` is the comparison.
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
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use Megatron-LM's "
            f"fused_pad_routing_map as its baseline, but megatron-core could "
            f"not provide it: {exc!r}. This run has no on-machine baseline; a "
            f"torch transcription is not timed, so --base-data is the only "
            f"comparison.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


# Resolved once, at module import time. ``None`` is the benchmark harness's
# "no on-machine baseline" sentinel; the harness checks ``torch_op is not None``
# before timing anything.
torch_op = _load_mcore_pad_routing_map()

# The baseline's version, recorded in the report line: a speedup means little
# without the version of the kernel it was measured against.
_BASELINE = (
    "megatron fused_pad_routing_map "
    f"(megatron-core {importlib.metadata.version('megatron-core')})"
    if torch_op is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)


# The dtypes timed. bool is the dispatcher's map and the reference record set;
# int32 is the same map after the cast a dispatcher may already have applied,
# and it takes the integer counting path (fp32 accumulation is the boolean
# one). Its rows have no cross-hardware record yet, so ``vs Base`` reports them
# missing until the baseline file is regenerated.
_DTYPES = [torch.bool, torch.int32]


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
        """Boundary traffic: read the input map, write the int32 padded map.

        The operator's internal ``.int()`` working copy is not counted, so this
        is the I/O the contract implies rather than every byte the kernels move.
        """
        num_tokens, num_experts = args[0].shape
        io_bytes = num_tokens * num_experts * (args[0].element_size() + 4)
        return io_bytes * 1e-9 / (latency * 1e-3)


def _make_routing_map(num_tokens, num_experts, top_k, dtype=torch.bool, device=None):
    """A top-k routing map, in the dispatcher's bool form or its int32 form.

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
    return routing_map.to(dtype)


def pad_routing_map_input_fn(shape, dtype, device):
    num_tokens, num_experts, top_k, pad_multiple = shape
    yield _make_routing_map(
        num_tokens, num_experts, top_k, dtype=dtype, device=device
    ), pad_multiple


def train_op(routing_map, pad_multiple):
    """The operator under test."""
    return fused_pad_routing_map(routing_map, pad_multiple)


@pytest.mark.fused_pad_routing_map
def test_fused_pad_routing_map_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = PadRoutingMapBenchmark(
        input_fn=pad_routing_map_input_fn,
        op_name="fused_pad_routing_map",
        # None off the vendored backends: a torch composition is not timed as
        # a baseline, so --base-data supplies the comparison.
        torch_op=torch_op,
        # bool is the dispatcher's map; int32 is the same map after its cast and
        # takes the integer counting path.
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
