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
import warnings

import pytest
import torch

import flag_train

from .. import base, consts


def _load_megatron_baseline():
    """Megatron's Triton kernel, or ``None`` when it cannot run here."""
    try:
        from megatron.core.config import set_experimental_flag
        from megatron.core.fusions import fused_indices_converter

        set_experimental_flag(True)
        return fused_indices_converter.fused_indices_to_multihot
    except Exception as exc:
        warnings.warn(
            "Megatron-LM's fused_indices_to_multihot Triton kernel is "
            f"unavailable ({exc!r}). This run has no on-machine baseline: a "
            "torch composition is not timed, so --base-data is the only "
            "comparison.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


# ``None`` is the benchmark harness's "no on-machine baseline" sentinel; the
# harness checks ``torch_op is not None`` before timing anything.
if flag_train.vendor_name == "ascend":
    # Megatron's native kernel is not adapted for Ascend: some shapes cannot
    # run (one program per token overflows coreDim at 65536 tokens; top-k or
    # expert width 1 does not compile), and where it runs it is about 150x
    # slower than the candidate. A vs-native speedup would measure that port
    # mismatch, so drop the on-machine baseline and compare through the
    # normalized --base-data instead.
    torch_op = None
else:
    torch_op = _load_megatron_baseline()

_SHAPES = [
    (4096, 2, 8),
    (4096, 6, 8),
    (4096, 4, 15),
    (2048, 8, 16),
    (16384, 8, 32),
    (8192, 1, 128),
    (65536, 8, 32),
    (131072, 8, 16),
    (65536, 2, 128),
    (256, 8, 32),
]

_BASELINE = (
    "megatron fused_indices_to_multihot (Triton)"
    if torch_op is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)


class _FusedIndicesToMultihotBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)


def fused_indices_to_multihot_input_fn(shape, dtype, device):
    num_tokens, topk, num_experts = shape
    scores = torch.rand(num_tokens, num_experts, dtype=torch.float32, device=device)
    indices = scores.topk(topk, dim=1).indices.to(torch.int64)
    probs = torch.rand(num_tokens, topk, dtype=dtype, device=device)
    yield indices, probs, num_experts


@pytest.mark.fused_indices_to_multihot
def test_fused_indices_to_multihot():
    print(f"\nBaseline: {_BASELINE}")
    bench = _FusedIndicesToMultihotBenchmark(
        input_fn=fused_indices_to_multihot_input_fn,
        op_name="fused_indices_to_multihot",
        # Megatron's IndicesToMultihot is a Triton kernel, not a torch
        # composition, so it is a real on-machine baseline when it loads; a
        # torch composition is never timed as one.
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_train(flag_train.megatron.fused_indices_to_multihot)
    bench.run()
