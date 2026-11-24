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
"""Performance benchmark for the multi_tensor_adam operator.

The baseline is DeepSpeed's ``multi_tensor_adam`` on the backends in
``_DEEPSPEED_BASELINE_VENDORS``. When it cannot be built, or on any other
backend, there is no on-machine baseline: a plain-torch composition is not a
competitor, so it is not timed, ``torch_op`` is ``None``, and ``--base-data``
is the only comparison. The ``Baseline:`` line reports which case ran.

Unlike LAMB, each benchmark case runs the optimizer over a *batch* of parameter
tensors rather than a single one. That is the whole point of the operator: the
per-tensor launch cost is amortized across the batch, so the relevant axis is
``(n_tensors, numel_per_tensor)`` and not just ``numel``.

fp32, fp16 and (when the backend supports it) bf16 are benchmarked. DeepSpeed's
raw CUDA op dispatches fp64/fp32/fp16/bf16 via
``DISPATCH_DOUBLE_FLOAT_AND_HALF``, whose macro includes ``BFloat16`` despite
its name; ``FusedAdam`` groups bf16 tensors into that same op, so bf16 is an
apples-to-apples baseline.
"""

import warnings

import pytest
import torch

import flag_train

from .. import base

# One-dimensional parameter tensors of realistic optimizer sizes. The operator
# is a pointwise update over each tensor, so numel is the only shape dimension
# that matters; ``_N_TENSORS`` below decides how many such tensors share one
# launch.
_MULTI_TENSOR_ADAM_SHAPES = [
    (1024,),
    (4096,),
    (16384,),
    (65536,),
    (262144,),
    (1048576,),
]

# Number of parameter tensors folded into a single operator call. Real optimizer
# steps batch hundreds of small tensors; 16 keeps the tensor count realistic
# without making each benchmark case allocate hundreds of buffers.
_N_TENSORS = 16

# Fixed hyper-parameters shared by both implementations so the comparison is
# apples-to-apples. These mirror DeepSpeed's FusedAdam defaults.
_LR = 1e-2
_BETA1 = 0.9
_BETA2 = 0.999
_EPS = 1e-8
_STEP = 1
_MODE = 1  # decoupled weight decay (AdamW), DeepSpeed's FusedAdam default
_BIAS_CORRECTION = 1
_WEIGHT_DECAY = 0.1

# DeepSpeed hard-codes 2048 * 32 in FusedAdam; the Triton op ignores it but the
# signature keeps it for drop-in compatibility.
_CHUNK_SIZE = 2048 * 32


# Backends whose baseline is DeepSpeed. multi_tensor_adam ships as a CUDA op
# builder, so only a backend that can compile and execute one can host it. A
# load failure is reported, not raised: with no runnable kernel there is no
# on-machine baseline, and the module resolves that below.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_multi_tensor_adam():
    """Harness-ready multi_tensor_adam, or ``None`` where it is not the baseline.

    ``FusedAdamBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under ``torch_extensions``.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.ops.op_builder import FusedAdamBuilder

        multi_tensor_adam = FusedAdamBuilder().load().multi_tensor_adam
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use DeepSpeed's "
            f"multi_tensor_adam as its baseline, but it could not be loaded: "
            f"{exc!r}. This run has no on-machine baseline; a torch "
            f"composition is not timed, so --base-data is the only comparison.",
            stacklevel=2,
        )
        return None

    def torch_op(g, p, m, v):
        return _call(multi_tensor_adam, g, p, m, v)

    return torch_op


# Resolved once, so the first-use JIT compile is not counted in the measurement;
# ``None`` is the benchmark harness's "no on-machine baseline" sentinel.
torch_op = _load_deepspeed_multi_tensor_adam()

_BASELINE = (
    "deepspeed multi_tensor_adam"
    if torch_op is not None
    else "none (no on-machine baseline; --base-data is the only comparison)"
)


class MultiTensorAdamBenchmark(base.GenericBenchmark):
    """One case per parameter-tensor size, ``_N_TENSORS`` tensors per case."""

    DEFAULT_SHAPES = _MULTI_TENSOR_ADAM_SHAPES
    DEFAULT_SHAPE_DESC = f"numel per tensor ({_N_TENSORS} tensors per case)"

    def set_shapes(self, shape_file=None):
        # Each ``shape`` is the numel of a single parameter tensor; the case
        # holds ``_N_TENSORS`` of them so the fused launch is exercised with a
        # realistic tensor count. Keep the shapes explicit rather than pulling
        # in the generic 2D/3D sweep, which does not apply to a flattened
        # optimizer update.
        self.shapes = list(_MULTI_TENSOR_ADAM_SHAPES)

    def set_more_shapes(self):
        return []

    def record_shapes(self, *args, **kwargs):
        # The inputs are (grads, params, exp_avgs, exp_avg_sqs), four lists of
        # _N_TENSORS tensors of one size each, so the default deep_parse prints
        # that same ``torch.Size`` 4 * _N_TENSORS times per row and says nothing
        # extra. Compress to the case description; the number of lists is fixed
        # by the operator's signature and is implied by ``multi_tensor_adam``.
        gs = args[0]
        return f"{len(gs)} tensors x {list(gs[0].shape)}"

    def baseline_shape_detail(self, *args, **kwargs):
        """The default structured form, for ``--base-data`` matching.

        ``record_shapes`` above is the compact report line; the baseline lookup
        needs the per-tensor dimensions to build its signature.
        """
        return super().record_shapes(*args, **kwargs)


def multi_tensor_adam_input_fn(shape, dtype, device):
    """Yield ``(g_list, p_list, m_list, v_list)`` with ``_N_TENSORS`` tensors each.

    ``m`` and ``v`` are zero-initialised, matching FusedAdam's state
    initialisation. ``g`` and ``p`` are freshly allocated for every case.
    """
    g_list = [torch.randn(shape, dtype=dtype, device=device) for _ in range(_N_TENSORS)]
    p_list = [torch.randn(shape, dtype=dtype, device=device) for _ in range(_N_TENSORS)]
    m_list = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(_N_TENSORS)]
    v_list = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(_N_TENSORS)]
    yield g_list, p_list, m_list, v_list


_NOOP = None


def _call(op, g, p, m, v):
    global _NOOP
    if _NOOP is None:
        _NOOP = torch.zeros((1,), dtype=torch.int32, device=g[0].device)
    return op(
        _CHUNK_SIZE,
        _NOOP,
        [g, p, m, v],
        _LR,
        _BETA1,
        _BETA2,
        _EPS,
        _STEP,
        _MODE,
        _BIAS_CORRECTION,
        _WEIGHT_DECAY,
    )


def train_op(g, p, m, v):
    """The operator under test."""
    return _call(flag_train.deepspeed.multi_tensor_adam, g, p, m, v)


@pytest.mark.multi_tensor_adam
def test_multi_tensor_adam_perf():
    print(f"\nBaseline: {_BASELINE}")

    bench = MultiTensorAdamBenchmark(
        input_fn=multi_tensor_adam_input_fn,
        op_name="multi_tensor_adam",
        # None when DeepSpeed is unavailable: a torch composition is not timed
        # as a baseline, so --base-data supplies the comparison.
        torch_op=torch_op,
        # DeepSpeed's raw op supports bf16 (its dispatch macro includes
        # BFloat16), so bf16 is an apples-to-apples case where the backend
        # supports it.
        dtypes=[
            torch.float32,
            torch.float16,
            *([torch.bfloat16] if flag_train.runtime.device.support_bf16 else []),
        ],
    )
    bench.set_train(train_op)
    bench.run()
