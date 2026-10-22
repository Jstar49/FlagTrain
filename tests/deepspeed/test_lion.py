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
"""Correctness tests for the multi_tensor_lion operator.

Oracle selection:

* The DEFAULT oracle is DeepSpeed's ``multi_tensor_lion`` -- the operator this
  implementation ports. Where the ``deepspeed`` package is available, the
  operator is verified against it directly.
* Where DeepSpeed is unavailable on the host, the oracle falls back to
  ``lion_ref``, a plain-torch composition of the same contract. Under
  ``--ref cpu`` it genuinely executes on the CPU, which makes it an
  independent oracle rather than a second reading of the same device
  arithmetic; see ``_reference_copy``.
* ``test_reference_matches_deepspeed`` pins the two oracles against each
  other on hosts that have both, so a fallback run still rests on a reference
  that is itself validated against DeepSpeed.

The operator under test is always invoked through its public interface,
``flag_train.deepspeed.multi_tensor_lion``, so the tests also exercise the
registration path a user would actually call.

On a backend in ``_DEEPSPEED_BASELINE_VENDORS`` the DeepSpeed oracle is
*required* -- if it will not load there the module raises rather than losing
the check quietly. Its version is recorded in ``_DEEPSPEED_VERSION``, as
tests/deepspeed/README.md asks for.
"""

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# Hyper-parameters shared by both implementations, mirroring the values
# DeepSpeed's FusedLion is exercised with.
_LR = 1e-4
_BETA1 = 0.9
_BETA2 = 0.99
_WEIGHT_DECAY = 0.01
_CHUNK_SIZE = 2048

# Lion's update is sign(c) * lr with c = beta1*m + (1-beta1)*g, so two
# implementations whose fp32 rounding paths differ (FMA contraction or not)
# can disagree on the sign of c when it sits within a few ulp of zero
# (~1e-7), flipping the update by 2*lr on an isolated element. Skip elements
# below this window in the p comparison: 1e-6 is far above fp32 rounding
# noise and far below the 2e-4 jump it masks out.
_SIGN_BOUNDARY = 1e-6

# ---------------------------------------------------------------------------
# Reference implementation
#
# Exists so the operator can be checked against a plain-torch composition of
# the same contract on any device. The DeepSpeed oracle needs the ``deepspeed``
# package and a CUDA device, so without a torch reference the operator would be
# untestable off NVIDIA.
# ---------------------------------------------------------------------------


def _reference_copy(tensor):
    """An independent copy of ``tensor``, on the device the reference must use.

    ``--ref cpu`` pushes the reference *computation* onto the CPU, not merely its
    result, so every operand ``lion_ref`` is handed comes from here and no kernel
    of the reference runs on the accelerator. That is what makes it an
    independent oracle -- a reference evaluated by the same device arithmetic it
    is checking can only confirm the arithmetic agrees with itself.
    ``lion_ref`` checks this rather than trusting it; the comparison helpers move
    the operator's own output across afterwards.

    ``lion_ref`` steps its parameters in place, so the copy is not optional: on a
    non-CPU reference ``to_reference`` returns its argument unchanged, and the
    reference would then corrupt the operator's operands.
    """
    return utils.to_reference(tensor).clone()


def lion_ref(
    chunk_size,
    noop_flag,
    tensor_lists,
    lr,
    beta1,
    beta2,
    step,
    weight_decay,
):
    """Reference for multi_tensor_lion, composed from plain torch ops.

    Steps ``p``/``m`` in place with fp32 intermediate math, matching the
    operator's contract including the c == 0 -> +lr tie-break, since the two
    are compared bit-for-bit-ish rather than through a fuzzy formula.
    """
    if utils.TO_CPU:
        # ``--ref cpu`` promises the reference *computation* is on the CPU, not
        # only its result. Asserting the precondition here rather than at the
        # call sites means a test that forgets ``_reference_copy`` fails loudly
        # instead of quietly becoming a second reading of the device arithmetic.
        for name, tensors in zip(("grads", "params", "exp_avgs"), tensor_lists):
            for operand in tensors:
                assert operand.device.type == "cpu", (
                    f"--ref cpu must run the reference on the CPU; a {name} "
                    f"operand is on {operand.device}"
                )

    g_list, p_list, m_list = tensor_lists
    for g, p, m in zip(g_list, p_list, m_list):
        gf = g.to(torch.float32)
        pf = p.to(torch.float32)
        mf = m.to(torch.float32)
        c = beta1 * mf + (1 - beta1) * gf
        upd = torch.where(c > 0, torch.full_like(c, -lr), torch.full_like(c, lr))
        p_new = pf * (1 - lr * weight_decay) + upd
        m_new = beta2 * mf + (1 - beta2) * gf
        p.copy_(p_new.to(p.dtype))
        m.copy_(m_new.to(m.dtype))


_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's multi_tensor_lion reference is unavailable; install the "
    "deepspeed package on a CUDA host to run this check."
)

# Backends whose reference is DeepSpeed. multi_tensor_lion ships as a CUDA op
# builder, so only a backend that can compile and execute one can host it. On
# these the reference is not optional -- a missing one is an environment fault,
# and skipping quietly would thin the suite without saying so.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia", "hygon"}


def _load_deepspeed_lion():
    """``(op, version)`` for DeepSpeed's multi_tensor_lion, or ``(None, None)``.

    ``FusedLionBuilder`` JIT-compiles the CUDA source shipped inside the
    ``deepspeed`` package, then reuses the build cached under
    ``torch_extensions``.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        import deepspeed
        from deepspeed.ops.op_builder import FusedLionBuilder

        return FusedLionBuilder().load().multi_tensor_lion, deepspeed.__version__
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use DeepSpeed's multi_tensor_lion "
            f"as its reference, but it could not be loaded: {exc!r}. Build "
            f"deepspeed, or drop the backend from _DEEPSPEED_BASELINE_VENDORS."
        ) from exc


# Resolved once, at module import time.
_deepspeed_lion, _DEEPSPEED_VERSION = _load_deepspeed_lion()

# The torch reference is always available, so only the DeepSpeed checks skip.
requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_lion is None, reason=_DEEPSPEED_UNAVAILABLE_MSG
)


def _oracle():
    """The default oracle: DeepSpeed's op where available, else ``lion_ref``."""
    return _deepspeed_lion if _deepspeed_lion is not None else lion_ref


def _oracle_lists(gs, ps, ms):
    """Operand copies for the default oracle, on the device it must use.

    DeepSpeed runs on the accelerator, so plain clones; ``lion_ref`` may need
    CPU operands under ``--ref cpu``.
    """
    if _deepspeed_lion is not None:
        return _clone_lists(gs, ps, ms)
    return _reference_lists(gs, ps, ms)


def _oracle_atol(dtype):
    """The DeepSpeed oracle is a cross-implementation check (see ``_ds_atol``);
    ``lion_ref`` shares our arithmetic exactly, so the strict default holds."""
    return _ds_atol(dtype) if _deepspeed_lion is not None else 1e-4


def _noop_flag():
    """The flag DeepSpeed's op takes but (with the check commented out) ignores."""
    return torch.zeros(1, dtype=torch.int32)


def _step(op, g_list, p_list, m_list, step):
    """Run one in-place multi-tensor step through ``op``.

    Every implementation updates its tensors in place, so callers pass tensors
    they own.
    """
    return op(
        _CHUNK_SIZE,
        _noop_flag(),
        [g_list, p_list, m_list],
        _LR,
        _BETA1,
        _BETA2,
        step,
        _WEIGHT_DECAY,
    )


def _make_triplet(n_tensors, numel, dtype, m_init="randn"):
    """Fresh (grads, params, exp_avgs) sharing one dtype."""
    device = flag_train.device
    gs = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    ps = [torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)]
    if m_init == "zeros":
        ms = [
            torch.zeros((numel,), dtype=dtype, device=device) for _ in range(n_tensors)
        ]
    else:
        ms = [
            torch.randn((numel,), dtype=dtype, device=device) for _ in range(n_tensors)
        ]
    return gs, ps, ms


def _clone_lists(gs, ps, ms):
    """Every implementation steps in place, so each runs on its own copies."""
    return ([t.clone() for t in gs], [t.clone() for t in ps], [t.clone() for t in ms])


def _reference_lists(gs, ps, ms):
    """Copies on the device the reference must use (the CPU under --ref cpu)."""
    return (
        [_reference_copy(t) for t in gs],
        [_reference_copy(t) for t in ps],
        [_reference_copy(t) for t in ms],
    )


def _ds_atol(dtype):
    """Atol for comparisons against the DeepSpeed oracle. With fp16/bf16 the
    two independent builds can contract the fp32 FMAs differently and land on
    opposite sides of a rounding boundary, flipping the last ulp on isolated
    elements. The torch-reference checks pin our arithmetic exactly, so a
    couple-of-ulps atol is appropriate for this cross-implementation check."""
    return 1e-4 if dtype == torch.float32 else 2e-3


def _sign_safe_mask(g_list, m_list):
    """Per-element mask of where Lion's sign decision is unambiguous.

    Computed from the PRE-STEP gradients and momentums in fp32. Elements with
    |c| < _SIGN_BOUNDARY are excluded from the p comparison: there a one-ulp
    difference in c between two correct implementations flips the update by
    2*lr, which no element-wise atol can absorb without also opening the door
    to real errors on every other element.
    """
    masks = []
    for g, m in zip(g_list, m_list):
        c = _BETA1 * m.to(torch.float32) + (1 - _BETA1) * g.to(torch.float32)
        masks.append(c.abs() >= _SIGN_BOUNDARY)
    return masks


def _assert_step_matches(
    train_ps, train_ms, other_ps, other_ms, atol=1e-4, p_masks=None
):
    """Both the updated params and the updated moments must agree, per tensor.

    ``p_masks`` (from ``_sign_safe_mask``) restricts the p comparison to
    elements where the sign of c is unambiguous. The m comparison always
    covers every element: m's update is continuous arithmetic with no sign
    jump, so it pins the operator's math exactly and the mask can hide no
    real bug there.
    """
    for i, (train_p, other_p) in enumerate(zip(train_ps, other_ps)):
        a = utils.to_reference(train_p)
        b = utils.to_reference(other_p)
        if p_masks is not None:
            mask = utils.to_reference(p_masks[i])
            a, b = a[mask], b[mask]
        utils.train_assert_close(a, b, train_p.dtype, atol=atol)
    for train_m, other_m in zip(train_ms, other_ms):
        utils.train_assert_close(
            utils.to_reference(train_m),
            utils.to_reference(other_m),
            train_m.dtype,
            atol=atol,
        )


@pytest.mark.multi_tensor_lion
@pytest.mark.parametrize("n_tensors,numel", [(1, 1024), (4, 4096), (16, 16384)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lion(n_tensors, numel, dtype):
    """A single Lion step must match the default oracle: DeepSpeed's
    multi_tensor_lion where available, the torch reference otherwise."""
    gs, ps, ms = _make_triplet(n_tensors, numel, dtype)
    # Must be computed before any _step call: every implementation steps in
    # place.
    p_masks = _sign_safe_mask(gs, ms)

    train_gs, train_ps, train_ms = _clone_lists(gs, ps, ms)
    _step(flag_train.deepspeed.multi_tensor_lion, train_gs, train_ps, train_ms, 1)

    oracle_gs, oracle_ps, oracle_ms = _oracle_lists(gs, ps, ms)
    _step(_oracle(), oracle_gs, oracle_ps, oracle_ms, 1)
    _assert_step_matches(
        train_ps,
        train_ms,
        oracle_ps,
        oracle_ms,
        atol=_oracle_atol(dtype),
        p_masks=p_masks,
    )


@pytest.mark.multi_tensor_lion
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_lion_matches_reference(dtype):
    """Run several Lion steps and compare against the torch reference.

    Multiple parameter tensors, fresh gradients each step, accumulating first
    moments, with the operator under test compared to the reference after all
    steps. Repeating the call on the same tensor set also exercises the
    metadata cache on every step after the first.
    """
    torch.manual_seed(0)
    n_tensors, numel, n_steps = 4, 4096, 5

    _, ps, ms = _make_triplet(n_tensors, numel, dtype, m_init="zeros")
    train_ps = [p.clone() for p in ps]
    train_ms = [m.clone() for m in ms]
    # Same values as the operator's operands -- ``to_reference`` copies rather
    # than converts -- but on whichever device the reference is meant to run.
    ref_ps = [_reference_copy(p) for p in ps]
    ref_ms = [_reference_copy(m) for m in ms]

    for step in range(1, n_steps + 1):
        grads = [
            torch.randn((numel,), dtype=dtype, device=flag_train.device)
            for _ in range(n_tensors)
        ]
        # Lion reads the gradient but never writes it, so the operator and the
        # reference may be handed the same values.
        _step(flag_train.deepspeed.multi_tensor_lion, grads, train_ps, train_ms, step)
        _step(lion_ref, [_reference_copy(g) for g in grads], ref_ps, ref_ms, step)

    _assert_step_matches(train_ps, train_ms, ref_ps, ref_ms)


@pytest.mark.multi_tensor_lion
def test_lion_mixed_dtypes():
    """A single call may mix dtypes; the implementation groups by dtype and
    launches once per group (mirroring FusedLion.step). DeepSpeed's op itself
    is always called per dtype group, so the torch reference is the oracle."""
    gs32, ps32, ms32 = _make_triplet(2, 4096, torch.float32)
    gs16, ps16, ms16 = _make_triplet(2, 4096, torch.float16)
    gs, ps, ms = gs32 + gs16, ps32 + ps16, ms32 + ms16

    train_gs, train_ps, train_ms = _clone_lists(gs, ps, ms)
    _step(flag_train.deepspeed.multi_tensor_lion, train_gs, train_ps, train_ms, 1)

    ref_gs, ref_ps, ref_ms = _reference_lists(gs, ps, ms)
    _step(lion_ref, ref_gs, ref_ps, ref_ms, 1)
    _assert_step_matches(train_ps, train_ms, ref_ps, ref_ms)


@pytest.mark.multi_tensor_lion
@requires_deepspeed_reference
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matches_deepspeed_oracle(dtype):
    """Pin the DeepSpeed oracle explicitly over multiple steps, so a run that
    quietly stopped reaching it (deepspeed missing) is visible rather than
    silently thinner."""
    gs, ps, ms = _make_triplet(8, 16384, dtype)

    train_gs, train_ps, train_ms = _clone_lists(gs, ps, ms)
    ds_gs, ds_ps, ds_ms = _clone_lists(gs, ps, ms)
    unsafe = None
    for step in range(1, 4):
        # The momentum evolves each step, so an element well clear of the sign
        # boundary at step k may sit on it at step k+1; union the boundary
        # elements of every step and exclude them all from the p comparison.
        step_mask = _sign_safe_mask(train_gs, train_ms)
        if unsafe is None:
            unsafe = [~m for m in step_mask]
        else:
            unsafe = [u | ~m for u, m in zip(unsafe, step_mask)]
        _step(
            flag_train.deepspeed.multi_tensor_lion, train_gs, train_ps, train_ms, step
        )
        _step(_deepspeed_lion, ds_gs, ds_ps, ds_ms, step)
    p_masks = [~u for u in unsafe]
    _assert_step_matches(
        train_ps, train_ms, ds_ps, ds_ms, atol=_ds_atol(dtype), p_masks=p_masks
    )


@pytest.mark.multi_tensor_lion
@requires_deepspeed_reference
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_matches_deepspeed(dtype):
    """Cross-check the two oracles against each other.

    Hosts without DeepSpeed fall back to ``lion_ref`` as the oracle; that
    fallback is only meaningful if ``lion_ref`` itself agrees with DeepSpeed's
    operator, so pin the two against each other where both exist.
    """
    gs, ps, ms = _make_triplet(8, 16384, dtype)
    p_masks = _sign_safe_mask(gs, ms)

    ds_gs, ds_ps, ds_ms = _clone_lists(gs, ps, ms)
    _step(_deepspeed_lion, ds_gs, ds_ps, ds_ms, 1)

    ref_gs, ref_ps, ref_ms = _reference_lists(gs, ps, ms)
    _step(lion_ref, ref_gs, ref_ps, ref_ms, 1)

    _assert_step_matches(
        ref_ps, ref_ms, ds_ps, ds_ms, atol=_ds_atol(dtype), p_masks=p_masks
    )
