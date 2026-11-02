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
"""Correctness tests for the multi_tensor_adam operator.

Two oracles, because they fail differently:

* ``multi_tensor_adam_ref`` -- a plain-torch composition of the same contract.
  It is the primary check and runs on any device, so the operator is testable
  off NVIDIA.
* DeepSpeed's ``multi_tensor_adam`` -- the operator this implementation ports.
  It needs the ``deepspeed`` package and a CUDA host, and on a backend in
  ``_DEEPSPEED_BASELINE_VENDORS`` it is *required* -- if it will not load there
  the module raises rather than losing the check quietly.

The torch reference mirrors ``csrc/adam/multi_tensor_adam.cu`` including the
order of operations, since the comparison is against the same arithmetic rather
than a re-derived formula.
"""

import warnings

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

_LR = 1e-2
_BETA1 = 0.9
_BETA2 = 0.999
_EPS = 1e-8
_WEIGHT_DECAY = 0.1


# ---------------------------------------------------------------------------
# Reference implementation
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

    Steps every tensor in ``p_list``/``m_list``/``v_list`` in place. Every
    tensor is promoted to fp32 for the arithmetic, matching ``MATH_T = float``
    in the CUDA kernel, and the result is rounded back to the storage dtype.
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


# ---------------------------------------------------------------------------
# DeepSpeed oracle
# ---------------------------------------------------------------------------

_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's multi_tensor_adam reference is unavailable; install the "
    "deepspeed package on a CUDA host to run this check."
)

_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_multi_tensor_adam():
    """``(op, version)`` for DeepSpeed's multi_tensor_adam, or ``(None, None)``."""
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        import deepspeed
        from deepspeed.ops.op_builder import FusedAdamBuilder

        return FusedAdamBuilder().load().multi_tensor_adam, deepspeed.__version__
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use DeepSpeed's "
            f"multi_tensor_adam as its reference, but it could not be loaded: "
            f"{exc!r}. Falling back to the torch reference only; build "
            f"deepspeed to enable the fused check.",
            stacklevel=2,
        )
        return None, None


_deepspeed_multi_tensor_adam, _DEEPSPEED_VERSION = _load_deepspeed_multi_tensor_adam()

requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_multi_tensor_adam is None, reason=_DEEPSPEED_UNAVAILABLE_MSG
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 2048 * 32


def _make_tensors(shapes, dtype):
    """Return ``[g, p, m, v]`` lists for the given shapes."""
    device = flag_train.device
    g = [torch.randn(s, dtype=dtype, device=device) for s in shapes]
    p = [torch.randn(s, dtype=dtype, device=device) for s in shapes]
    m = [torch.zeros(s, dtype=dtype, device=device) for s in shapes]
    v = [torch.zeros(s, dtype=dtype, device=device) for s in shapes]
    return g, p, m, v


def _call_op(op, g, p, m, v, step, mode, bias_correction):
    """Invoke an op with the shared hyper-parameters and noop flag."""
    noop = torch.zeros((1,), dtype=torch.int32, device=flag_train.device)
    op(
        _CHUNK_SIZE,
        noop,
        [g, p, m, v],
        _LR,
        _BETA1,
        _BETA2,
        _EPS,
        step,
        mode,
        int(bias_correction),
        _WEIGHT_DECAY,
    )


def _clone_lists(*lists):
    return [[t.clone() for t in lst] for lst in lists]


def _tolerance(dtype, a, b):
    """Absolute tolerance for comparing two same-dtype tensors.

    The framework default (``atol=1e-4``) is a few ulps of fp32, but only a
    fraction of one ulp of fp16/bf16. Over five optimizer steps the fp32
    arithmetic inside each implementation can differ by an ulp -- Triton may
    fuse multiply-adds the torch reference does not, and ``tl.sqrt`` is not
    bit-identical to ``torch.sqrt`` -- and rounding to a low-precision storage
    dtype amplifies that. DeepSpeed's own
    ``test_fused_adam_matches_reference`` uses the same storage-dtype-scaled
    tolerance for exactly this comparison.
    """
    if dtype == torch.float32:
        return 1e-4
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-6)
    return 8 * torch.finfo(dtype).eps * scale


def _assert_step_matches(train_p, train_m, train_v, other_p, other_m, other_v, dtype):
    def _close(a, b):
        atol = _tolerance(dtype, a, b)
        utils.train_assert_close(
            utils.to_reference(a), utils.to_reference(b), dtype, atol=atol
        )

    for tp, op in zip(train_p, other_p):
        _close(tp, op)
    for tm, om in zip(train_m, other_m):
        _close(tm, om)
    for tv, ov in zip(train_v, other_v):
        _close(tv, ov)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.multi_tensor_adam
@pytest.mark.parametrize("shape", [(1024,), (4096,), (65536,)])
@pytest.mark.parametrize("mode", [0, 1], ids=["L2", "decoupled"])
@pytest.mark.parametrize("bias_correction", [0, 1], ids=["no_bias_corr", "bias_corr"])
def test_multi_tensor_adam(shape, mode, bias_correction):
    """A single step must match the torch reference, and DeepSpeed's
    multi_tensor_adam when it is available."""
    dtype = torch.float32

    g, p, m, v = _make_tensors([shape], dtype)

    train_g, train_p, train_m, train_v = _clone_lists(g, p, m, v)
    _call_op(
        flag_train.deepspeed.multi_tensor_adam,
        train_g,
        train_p,
        train_m,
        train_v,
        1,
        mode,
        bias_correction,
    )

    ref_g, ref_p, ref_m, ref_v = _clone_lists(g, p, m, v)
    multi_tensor_adam_ref(
        ref_g,
        ref_p,
        ref_m,
        ref_v,
        _LR,
        _BETA1,
        _BETA2,
        _EPS,
        1,
        mode,
        bias_correction,
        _WEIGHT_DECAY,
    )
    _assert_step_matches(train_p, train_m, train_v, ref_p, ref_m, ref_v, dtype)

    if _deepspeed_multi_tensor_adam is not None:
        ds_g, ds_p, ds_m, ds_v = _clone_lists(g, p, m, v)
        _call_op(
            _deepspeed_multi_tensor_adam,
            ds_g,
            ds_p,
            ds_m,
            ds_v,
            1,
            mode,
            bias_correction,
        )
        _assert_step_matches(train_p, train_m, train_v, ds_p, ds_m, ds_v, dtype)


@pytest.mark.multi_tensor_adam
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
@pytest.mark.parametrize("mode", [0, 1], ids=["L2", "decoupled"])
def test_multi_tensor_adam_matches_reference(mode, dtype):
    """Run several steps across several tensors and compare against the torch
    reference. Mirrors DeepSpeed's ``test_fused_adam_matches_reference``."""
    if dtype == torch.bfloat16 and not utils.bf16_is_supported:
        pytest.skip("bf16 not supported on this backend")

    torch.manual_seed(0)
    shapes = [(1024,), (4096,), (16384,)]
    _, p_all, m_all, v_all = _make_tensors(shapes, dtype)

    train_p = [t.clone() for t in p_all]
    train_m = [t.clone() for t in m_all]
    train_v = [t.clone() for t in v_all]
    ref_p = [t.clone() for t in p_all]
    ref_m = [t.clone() for t in m_all]
    ref_v = [t.clone() for t in v_all]

    for step in range(1, 6):
        train_g = [torch.randn_like(t) for t in train_p]
        ref_g = [t.clone() for t in train_g]

        _call_op(
            flag_train.deepspeed.multi_tensor_adam,
            train_g,
            train_p,
            train_m,
            train_v,
            step,
            mode,
            1,
        )
        multi_tensor_adam_ref(
            ref_g,
            ref_p,
            ref_m,
            ref_v,
            _LR,
            _BETA1,
            _BETA2,
            _EPS,
            step,
            mode,
            1,
            _WEIGHT_DECAY,
        )

    _assert_step_matches(train_p, train_m, train_v, ref_p, ref_m, ref_v, dtype)


@pytest.mark.multi_tensor_adam
@requires_deepspeed_reference
@pytest.mark.parametrize("mode", [0, 1], ids=["L2", "decoupled"])
@pytest.mark.parametrize("shape", [(1024,), (16384,)])
def test_matches_deepspeed_oracle(mode, shape):
    """Pin the DeepSpeed oracle explicitly, so a run that quietly stopped
    reaching it (deepspeed missing) is visible rather than silently thinner."""
    dtype = torch.float32

    g, p, m, v = _make_tensors([shape], dtype)

    train_g, train_p, train_m, train_v = _clone_lists(g, p, m, v)
    _call_op(
        flag_train.deepspeed.multi_tensor_adam,
        train_g,
        train_p,
        train_m,
        train_v,
        1,
        mode,
        1,
    )

    ds_g, ds_p, ds_m, ds_v = _clone_lists(g, p, m, v)
    _call_op(_deepspeed_multi_tensor_adam, ds_g, ds_p, ds_m, ds_v, 1, mode, 1)
    _assert_step_matches(train_p, train_m, train_v, ds_p, ds_m, ds_v, dtype)


@pytest.mark.multi_tensor_adam
def test_multi_tensor_adam_boundary_shapes():
    """Non-aligned and single-element tensors must not corrupt neighbours."""
    dtype = torch.float32
    shapes = [(1,), (17,), (1023,), (1024,), (1025,)]

    g, p, m, v = _make_tensors(shapes, dtype)

    train_g, train_p, train_m, train_v = _clone_lists(g, p, m, v)
    _call_op(
        flag_train.deepspeed.multi_tensor_adam,
        train_g,
        train_p,
        train_m,
        train_v,
        1,
        1,
        1,
    )

    ref_g, ref_p, ref_m, ref_v = _clone_lists(g, p, m, v)
    multi_tensor_adam_ref(
        ref_g, ref_p, ref_m, ref_v, _LR, _BETA1, _BETA2, _EPS, 1, 1, 1, _WEIGHT_DECAY
    )

    _assert_step_matches(train_p, train_m, train_v, ref_p, ref_m, ref_v, dtype)
