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
"""Correctness tests for the scaled masked softmax backward.

The operator is ``ScaledMaskedSoftmax.backward`` in Megatron-LM's
``megatron/core/fusions/fused_softmax.py``. Three oracles are used:

* ``torch.autograd`` run over the plain-torch scale + mask + softmax
  composition, the strongest check because it does not restate the gradient
  formula;
* the closed form ``scale * y * (dy - sum(dy * y))``, which pins the contract
  for hand-built ``softmax_results`` (including rows the mask zeroed out);
* Megatron-LM's fused path, i.e. the Apex ``scaled_masked_softmax_cuda``
  extension. Installing ``megatron-core`` does not ship that extension, so this
  comparison is skipped when the CUDA module itself is not importable.

``ScaledMaskedSoftmax`` is taken from the installed ``megatron-core`` whenever
it can be imported; otherwise the method body is vendored below and used as the
Megatron baseline. That is feasible because the body is one call into the Apex
extension; ``test_vendored_baseline_matches_torch`` holds the copy to the closed
form so it cannot silently drift.
"""

import importlib.metadata
import importlib.util
import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import accuracy_utils as utils

_SCALE = 0.125
_EXTENSION = "scaled_masked_softmax_cuda"
_HAVE_EXTENSION = importlib.util.find_spec(_EXTENSION) is not None
needs_extension = pytest.mark.skipif(
    not _HAVE_EXTENSION,
    reason="Megatron's Apex scaled-softmax CUDA extension is not installed",
)

_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)
_EXT_DTYPES = [torch.float16] + ([torch.bfloat16] if utils.bf16_is_supported else [])

# ``[b, np, sq, sk]``: a single row, odd sizes, padding masks that mask whole
# rows, and the sizes the benchmark reports.
_SHAPES = [
    (1, 1, 1, 1),
    (1, 1, 1, 5),
    (2, 3, 4, 4),
    (2, 3, 5, 7),
    (1, 2, 16, 16),
    (2, 4, 31, 31),
    (2, 8, 64, 64),
    (4, 8, 128, 128),
    (2, 8, 128, 256),
    (1, 8, 1024, 1024),
]
_BENCH_SHAPES = [
    (8, 16, 128, 128),
    (8, 16, 256, 256),
    (8, 16, 512, 512),
    (8, 16, 1024, 1024),
    (8, 16, 2048, 2048),
    (4, 16, 4096, 4096),
]


# ---------------------------------------------------------------------------
# Megatron baseline
# ---------------------------------------------------------------------------


def _megatron_version():
    for distribution in ("megatron-core", "megatron"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _load_megatron_native():
    """Installed Megatron's class, or ``None`` when it cannot be imported."""
    version = _megatron_version()
    try:
        from megatron.core.fusions.fused_softmax import ScaledMaskedSoftmax
    except Exception as exc:  # pragma: no cover - depends on the installation
        warnings.warn(
            "megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax could not "
            f"be imported ({exc!r}); using the vendored baseline "
            f"(installed megatron-core: {version or 'absent'}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return version, ScaledMaskedSoftmax


class _SavedForBackward:
    """Stand-in for the autograd context ``ScaledMaskedSoftmax.backward`` takes.

    Megatron's forward saves ``(softmax_results, scale_t)``; the backward only
    reads ``ctx.saved_tensors``, so a two-attribute object is enough to run the
    copied body outside of autograd.
    """

    def __init__(self, softmax_results, scale):
        self.saved_tensors = (softmax_results, torch.tensor([scale]))


def _vendored_forward(inputs, mask, scale):
    """Copied from ``ScaledMaskedSoftmax.forward``."""
    import scaled_masked_softmax_cuda

    scale_t = torch.tensor([scale])
    return scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])


def _vendored_backward(output_grads, softmax_results, scale):
    """Copied from ``ScaledMaskedSoftmax.backward`` (minus the ``ctx`` plumbing)."""
    import scaled_masked_softmax_cuda

    ctx = _SavedForBackward(softmax_results, scale)
    softmax_results, scale_t = ctx.saved_tensors
    input_grads = scaled_masked_softmax_cuda.backward(
        output_grads, softmax_results, scale_t[0]
    )
    return input_grads, None, None


_NATIVE = _load_megatron_native()


def _megatron_backward(output_grads, softmax_results, scale):
    """Run Megatron's backward body, native when importable, else vendored.

    Both paths only read ``ctx.saved_tensors``, so the same stand-in context
    drives the installed class and the copy. Like every autograd backward, it
    answers with one entry per forward input -- ``(input_grads, None, None)``
    for ``(inputs, mask, scale)`` -- so the gradient is the first element.
    """
    ctx = _SavedForBackward(softmax_results, scale)
    if _NATIVE is None:
        grads = _vendored_backward(output_grads, softmax_results, scale)
    else:
        grads = _NATIVE[1].backward(ctx, output_grads)
    return grads[0]


if not _HAVE_EXTENSION:
    _BASELINE_SOURCE = "unavailable (Apex scaled-softmax extension missing)"
elif _NATIVE is None:
    _BASELINE_SOURCE = (
        "vendored megatron/core/fusions/fused_softmax.py "
        f"(installed megatron-core: {_megatron_version() or 'absent'})"
    )
else:
    _BASELINE_SOURCE = f"installed megatron-core {_NATIVE[0]}"


# ---------------------------------------------------------------------------
# Torch references
# ---------------------------------------------------------------------------


def masked_softmax_ref(scores, mask, scale):
    """Upstream forward: ``-10000`` sentinel, fp32, fully masked rows zeroed."""
    scores = scores.to(torch.float32) * scale
    # The mask is built on the device while ``--ref cpu`` has already moved the
    # scores to the host, and ``masked_fill`` refuses to mix the two, so the
    # mask follows the scores.
    masked = mask.to(scores.device).bool().expand_as(scores)
    scores = scores.masked_fill(masked, -10000.0)
    row_max = scores.max(dim=-1, keepdim=True).values
    numerator = torch.exp(scores - row_max)
    denominator = numerator.sum(dim=-1, keepdim=True)
    result = numerator / denominator
    result = torch.where(row_max == -10000.0, torch.zeros_like(result), result)
    return result.to(scores.dtype)


def masked_backward_ref(output_grads, softmax_results, scale):
    """Closed form ``scale * y * (dy - sum(dy * y))`` over the key axis."""
    dy = output_grads.to(torch.float32)
    y = softmax_results.to(torch.float32)
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return (scale * y * (dy - dot)).to(output_grads.dtype)


def masked_backward_autograd(scores, mask, output_grads, scale):
    """``torch.autograd`` through the reference forward, upcast to fp32."""
    scores = scores.to(torch.float32).requires_grad_(True)
    y = masked_softmax_ref(scores, mask, scale)
    (grad,) = torch.autograd.grad(y, scores, output_grads.to(torch.float32))
    return grad


def _reference(tensor):
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


def _padding_mask(shape):
    """A ``[b, 1, sq, sk]`` mask that hides a suffix of keys and some whole rows."""
    b, _, sq, sk = shape
    mask = torch.zeros((b, 1, sq, sk), dtype=torch.bool, device=flag_train.device)
    if sk > 1:
        mask[..., 0, max(sk // 2, 1) :] = True
    if sq > 1:
        mask[:, 0, -1, :] = True  # a fully masked row
        mask[:, 0, -2, sk // 2 :] = True
    return mask


def _softmax_like(tensor):
    """Uniform probabilities: a valid softmax output for hand-built gradients."""
    return torch.full_like(tensor, 1.0 / tensor.shape[-1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.scaled_masked_softmax_cuda_backward
def test_baseline_source():
    """Record which Megatron baseline this run can check against."""
    assert _BASELINE_SOURCE
    print(f"\nMegatron baseline: {_BASELINE_SOURCE}")


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("shape", _SHAPES + _BENCH_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_gradient_matches_autograd(shape, dtype):
    utils.init_seed(0)
    scores = torch.randn(shape, dtype=torch.float32, device=flag_train.device)
    mask = _padding_mask(shape)
    y = masked_softmax_ref(scores, mask, _SCALE).to(dtype)
    dy = torch.randn(shape, dtype=dtype, device=flag_train.device)

    dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, _SCALE)

    assert dx.shape == dy.shape
    assert dx.dtype == dtype
    assert dx.is_contiguous()
    # The oracle runs where the reference runs: on the host under ``--ref cpu``.
    expected = masked_backward_autograd(
        _reference(scores), mask, _reference(dy), _SCALE
    )
    _assert_close(dx, expected, dtype)


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_closed_form(shape, dtype):
    """Hand-built probabilities, including the all-zero rows the mask leaves."""
    utils.init_seed(1)
    dy = torch.randn(shape, dtype=dtype, device=flag_train.device)
    y = masked_softmax_ref(
        torch.randn(shape, dtype=torch.float32, device=flag_train.device),
        _padding_mask(shape),
        _SCALE,
    ).to(dtype)

    dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, _SCALE)
    _assert_close(dx, masked_backward_ref(_reference(dy), _reference(y), _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_masked_positions_get_zero_gradient(dtype):
    utils.init_seed(2)
    shape = (2, 4, 16, 16)
    mask = _padding_mask(shape)
    y = masked_softmax_ref(
        torch.randn(shape, dtype=torch.float32, device=flag_train.device), mask, _SCALE
    ).to(dtype)
    dy = torch.randn(shape, dtype=dtype, device=flag_train.device)

    dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, _SCALE)

    assert torch.count_nonzero(dx[mask.expand_as(dx)]) == 0
    fully_masked = torch.zeros(shape, dtype=torch.bool, device=dx.device)
    fully_masked[:, 0, -1, :] = True
    assert torch.count_nonzero(dx[fully_masked]) == 0


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scale_and_values(dtype):
    """Scale 0 gives zero gradient; a large scale stays finite."""
    shape = (2, 2, 8, 8)
    y = _softmax_like(torch.empty(shape, dtype=dtype, device=flag_train.device))
    dy = torch.randn(shape, dtype=dtype, device=flag_train.device)
    for scale in (0.0, -0.5, 1.0, 8.0):
        dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, scale)
        assert torch.isfinite(dx).all()
        _assert_close(
            dx, masked_backward_ref(_reference(dy), _reference(y), scale), dtype
        )
    dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, 0.0)
    assert torch.count_nonzero(dx) == 0


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_dim_scale(dtype):
    """Megatron passes ``scale_t[0]``, a 0-dim tensor, not a Python float."""
    shape = (2, 2, 8, 8)
    y = _softmax_like(torch.empty(shape, dtype=dtype, device=flag_train.device))
    dy = torch.randn(shape, dtype=dtype, device=flag_train.device)
    scale = torch.tensor([_SCALE], dtype=torch.float32)

    dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, scale[0])
    _assert_close(dx, masked_backward_ref(_reference(dy), _reference(y), _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_noncontiguous(dtype):
    utils.init_seed(3)
    base_y = torch.randn((4, 2, 8, 8), dtype=dtype, device=flag_train.device)
    base_dy = torch.randn((4, 2, 8, 8), dtype=dtype, device=flag_train.device)
    y = base_y.transpose(0, 1)  # [2, 4, 8, 8], non-contiguous
    dy = base_dy.transpose(0, 1)
    assert not y.is_contiguous() and not dy.is_contiguous()

    dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, _SCALE)
    assert dx.is_contiguous()
    _assert_close(dx, masked_backward_ref(_reference(dy), _reference(y), _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_empty_batch(dtype):
    dy = torch.empty((0, 2, 4, 4), dtype=dtype, device=flag_train.device)
    y = torch.empty_like(dy)
    dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, _SCALE)
    assert dx.shape == dy.shape and dx.numel() == 0


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_does_not_mutate_operands(dtype):
    """The Apex kernel is in-place on ``output_grads``; this port is not."""
    shape = (2, 2, 8, 8)
    y = _softmax_like(torch.empty(shape, dtype=dtype, device=flag_train.device))
    dy = torch.randn(shape, dtype=dtype, device=flag_train.device)
    y_copy, dy_copy = y.clone(), dy.clone()

    flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, _SCALE)
    torch.testing.assert_close(y, y_copy)
    torch.testing.assert_close(dy, dy_copy)


@needs_extension
@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize(
    "shape",
    [
        (4, 8, 16, 16),
        (4, 8, 32, 64),
        (4, 8, 64, 64),
        (2, 8, 128, 256),
        (2, 8, 1024, 1024),
    ],
)
@pytest.mark.parametrize("dtype", _EXT_DTYPES)
def test_matches_megatron(shape, dtype):
    """The Triton kernel against the exact Apex call Megatron makes."""
    utils.init_seed(4)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = _padding_mask(shape)
    # Build ``softmax_results`` with the upstream forward so both sides start
    # from Megatron's own probabilities.
    y = _vendored_forward(scores, mask, _SCALE)
    dy = torch.randn(shape, dtype=dtype, device=flag_train.device)

    dx = flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, _SCALE)
    # The Apex backward overwrites ``output_grads``; give it a private copy.
    expected = _megatron_backward(dy.clone(), y, _SCALE)

    _assert_close(dx, utils.to_reference(expected), dtype)


@needs_extension
@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize("dtype", _EXT_DTYPES)
def test_vendored_baseline_matches_torch(dtype):
    """The copied Megatron body agrees with the closed form, not just with us."""
    utils.init_seed(5)
    shape = (4, 8, 32, 32)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = _padding_mask(shape)
    y = _vendored_forward(scores, mask, _SCALE)
    dy = torch.randn(shape, dtype=dtype, device=flag_train.device)

    dx, _, _ = _vendored_backward(dy.clone(), y, _SCALE)
    expected = masked_backward_ref(_reference(dy), _reference(y), _SCALE)
    _assert_close(dx, expected, dtype)


@pytest.mark.scaled_masked_softmax_cuda_backward
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("ndim", "4-D"),
        ("dtype", "needs a"),
        ("shape", "matching shapes"),
        ("dtype-mismatch", "matching dtypes"),
        ("cpu", "not one on the CPU"),
        ("key-limit", "exceeds Megatron"),
        ("scale", "one element"),
    ],
)
def test_rejects_invalid_arguments(case, match):
    dy = torch.randn((2, 2, 4, 4), dtype=torch.float32, device=flag_train.device)
    y = _softmax_like(dy)
    if case == "ndim":
        dy, y = dy.reshape(2, 2, 16), y.reshape(2, 2, 16)
    elif case == "dtype":
        dy, y = dy.double(), y.double()
    elif case == "shape":
        y = _softmax_like(torch.empty((2, 2, 4, 8), dtype=dy.dtype, device=dy.device))
    elif case == "dtype-mismatch":
        y = y.half()
    elif case == "cpu":
        y = y.cpu()
    elif case == "key-limit":
        dy = torch.empty((1, 1, 1, 4097), dtype=torch.float16, device=dy.device)
        y = _softmax_like(dy)
    elif case == "scale":
        with pytest.raises(ValueError, match=match):
            flag_train.megatron.scaled_masked_softmax_cuda_backward(
                dy, y, torch.ones(2)
            )
        return
    with pytest.raises(ValueError, match=match):
        flag_train.megatron.scaled_masked_softmax_cuda_backward(dy, y, _SCALE)
