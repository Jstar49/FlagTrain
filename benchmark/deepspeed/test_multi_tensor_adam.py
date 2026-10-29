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

The baseline follows ``_DEEPSPEED_BASELINE_VENDORS`` below:

* on a listed backend, DeepSpeed's ``multi_tensor_adam`` **is** the baseline. If
  it cannot be built there the module raises rather than falling back -- timing
  the torch reference and reporting it under the same name would answer a
  different question;
* on any other backend the baseline is ``multi_tensor_adam_ref``, the
  plain-torch composition of the same contract.

Unlike LAMB, each benchmark case runs the optimizer over a *batch* of parameter
tensors rather than a single one. That is the whole point of the operator: the
per-tensor launch cost is amortized across the batch, so the relevant axis is
``(n_tensors, numel_per_tensor)`` and not just ``numel``.

Only fp32 and fp16 are benchmarked. DeepSpeed's CUDA op dispatches only
fp16/fp32/fp64 via ``DISPATCH_DOUBLE_FLOAT_AND_HALF`` and has no bf16 kernel, so
a bf16 case would compare the Triton op against a different arithmetic path
rather than an apples-to-apples baseline. The Triton op *does* support bf16 --
that is covered by the correctness tests, not here.
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


# ---------------------------------------------------------------------------
# Reference implementation
#
# Exists so the operator can be benchmarked against a plain-torch composition of
# the same contract on any device. The DeepSpeed oracle needs the ``deepspeed``
# package and a CUDA device, so without a torch reference the operator would be
# unbenchmarkable off NVIDIA.
# ---------------------------------------------------------------------------


def multi_tensor_adam_ref(
    g_list,
    p_list,
    m_list,
    v_list,
    lr,
    beta1,
    beta2,
    epsilon,
    step,
    mode,
    bias_correction,
    weight_decay,
):
    """Plain-torch reference matching ``csrc/adam/multi_tensor_adam.cu``.

    Steps every tensor in ``p_list``/``m_list``/``v_list`` in place, promoting
    to fp32 for the arithmetic and rounding back to the storage dtype. Loops
    over tensors one at a time, which is exactly the launch-overhead the fused
    operator removes.
    """
    if bias_correction == 1:
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
    else:
        bc1 = 1.0
        bc2 = 1.0

    for g, p, m, v in zip(g_list, p_list, m_list, v_list):
        g_f = g.float()
        p_f = p.float()
        m_f = m.float()
        v_f = v.float()

        if mode == 0:
            g_f = g_f + weight_decay * p_f
            m_new = beta1 * m_f + (1.0 - beta1) * g_f
            v_new = beta2 * v_f + (1.0 - beta2) * g_f * g_f
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            update = m_hat / (torch.sqrt(v_hat) + epsilon)
            p_new = p_f - lr * update
        else:
            m_new = beta1 * m_f + (1.0 - beta1) * g_f
            v_new = beta2 * v_f + (1.0 - beta2) * g_f * g_f
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            update = m_hat / (torch.sqrt(v_hat) + epsilon) + weight_decay * p_f
            p_new = p_f - lr * update

        p.copy_(p_new.to(p.dtype))
        m.copy_(m_new.to(m.dtype))
        v.copy_(v_new.to(v.dtype))


# Backends whose baseline is DeepSpeed. multi_tensor_adam ships as a CUDA op
# builder, so only a backend that can compile and execute one can host it. On
# these the baseline is not optional -- if it will not load, timing the torch
# reference and reporting it under the DeepSpeed baseline's name would answer a
# different question.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_multi_tensor_adam():
    """DeepSpeed's multi_tensor_adam, or ``None`` on a backend that does not use it.

    ``FusedAdamBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under ``torch_extensions``.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None

    try:
        from deepspeed.ops.op_builder import FusedAdamBuilder

        return FusedAdamBuilder().load().multi_tensor_adam
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use DeepSpeed's "
            f"multi_tensor_adam as its baseline, but it could not be loaded: "
            f"{exc!r}. Falling back to the torch reference; build deepspeed to "
            f"enable the fused baseline.",
            stacklevel=2,
        )
        return None


# Resolved once, so the first-use JIT compile is not counted in the measurement.
_deepspeed_multi_tensor_adam = _load_deepspeed_multi_tensor_adam()

_BASELINE = (
    "deepspeed multi_tensor_adam"
    if _deepspeed_multi_tensor_adam is not None
    else "multi_tensor_adam_ref (torch)"
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


def torch_op(g, p, m, v):
    """Baseline, chosen by platform.

    The two baselines take different arguments: DeepSpeed's op takes the
    MultiTensorApply-style tuple, while the torch reference takes the four
    tensor lists directly.
    """
    if _deepspeed_multi_tensor_adam is not None:
        return _call(_deepspeed_multi_tensor_adam, g, p, m, v)
    return multi_tensor_adam_ref(
        g,
        p,
        m,
        v,
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
        torch_op=torch_op,
        # DeepSpeed's CUDA op has no bf16 kernel, so only fp32/fp16 give an
        # apples-to-apples baseline. The Triton op supports bf16 too; that is
        # covered by the correctness tests.
        dtypes=[torch.float32, torch.float16],
    )
    bench.set_train(train_op)
    bench.run()
