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
"""Correctness tests for the scaled masked softmax forward.

The operator is ``ScaledMaskedSoftmax.forward`` in Megatron-LM's
``megatron/core/fusions/fused_softmax.py``. Two oracles are used:

* the plain-torch composition of the same contract (scale, ``-10000`` sentinel,
  softmax, zeroed fully masked rows), always available and run on every
  shape/dtype/mask, upcast by ``to_reference(upcast=True)`` (and by ``--ref
  cpu``);
* Megatron-LM's fused path, i.e. the Apex ``scaled_masked_softmax_cuda``
  extension. Installing ``megatron-core`` does not ship that extension, so this
  comparison is skipped when the CUDA module itself is not importable.

``ScaledMaskedSoftmax`` is taken from the installed ``megatron-core`` whenever
it can be imported; otherwise the method body is vendored below and used as the
Megatron baseline. That is feasible because the body is one call into the Apex
extension; ``test_vendored_baseline_matches_torch`` holds the copy to the
composition so it cannot silently drift.
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
_MASKED_SCORE = -10000.0
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

# ``[b, np, sq, sk]``: a single score, odd sizes, query and key lengths that
# differ, a full attention head, and the sizes the benchmark reports.
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


def _vendored_forward(inputs, mask, scale):
    """Copied from ``ScaledMaskedSoftmax.forward``."""
    import scaled_masked_softmax_cuda

    scale_t = torch.tensor([scale])
    return scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])


_NATIVE = _load_megatron_native()


def _megatron_forward(inputs, mask, scale):
    """Megatron's forward body, native when importable, else vendored."""
    if _NATIVE is None:
        return _vendored_forward(inputs, mask, scale)
    return _NATIVE[1].apply(inputs, mask, scale)


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
# Torch reference
# ---------------------------------------------------------------------------


def masked_softmax_ref(inputs, mask, scale):
    """Upstream forward: ``-10000`` sentinel, fp32, fully masked rows zeroed.

    The reduction runs in the input's own wide dtype when it is already fp32 or
    fp64, and in fp32 for the half-precision inputs, as upstream does.
    """
    if inputs.dtype not in (torch.float32, torch.float64):
        inputs = inputs.to(torch.float32)
    scores = inputs * scale
    # The mask is built on the device while ``--ref cpu`` has already moved the
    # scores to the host, and ``masked_fill`` refuses to mix the two, so the
    # mask follows the scores.
    masked = mask.to(scores.device).bool().expand_as(scores)
    scores = scores.masked_fill(masked, _MASKED_SCORE)
    row_max = scores.max(dim=-1, keepdim=True).values
    numerator = torch.exp(scores - row_max)
    result = numerator / numerator.sum(dim=-1, keepdim=True)
    result = result.masked_fill(masked, 0.0)
    result = torch.where(
        masked.all(dim=-1, keepdim=True), torch.zeros_like(result), result
    )
    return result.to(inputs.dtype)


def _reference(tensor):
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


# Query rows per reference chunk. The benchmark-sized shapes are far too large
# for one upcast fp64 reference tensor, and the softmax factorises over the
# query axis, so the oracle is evaluated chunk by chunk instead.
_REFERENCE_CHUNK = 256


def _assert_forward_close(out, scores, mask, dtype, scale=_SCALE):
    """Compare ``out`` against the torch reference, chunked to bound memory."""
    sq = scores.shape[2]
    for start in range(0, sq, _REFERENCE_CHUNK):
        stop = min(start + _REFERENCE_CHUNK, sq)
        expected = masked_softmax_ref(
            _reference(scores[:, :, start:stop]),
            mask[..., start:stop, :],
            scale,
        )
        _assert_close(out[:, :, start:stop], expected, dtype)


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


def _causal_mask(shape):
    """A ``[1, 1, sq, sk]`` decoder mask: keys strictly after the query."""
    _, _, sq, sk = shape
    return (
        torch.ones((sq, sk), dtype=torch.bool, device=flag_train.device)
        .triu(1)[None, None]
        .contiguous()
    )


def _random_mask(shape):
    """A ``[b, 1, sq, sk]`` dropout-style mask that keeps the first key."""
    b, _, sq, sk = shape
    mask = torch.rand((b, 1, sq, sk), device=flag_train.device) < 0.3
    mask[..., 0] = False
    return mask


def _shared_mask(shape):
    """A one-batch mask that every batch entry broadcasts."""
    b, _, sq, sk = shape
    return _random_mask((1, 1, sq, sk)) if b > 1 else _random_mask(shape)


def _apex_batches_per_block(shape):
    """The blocks-per-batch unit Apex's forward asserts divisibility by."""
    import scaled_masked_softmax_cuda

    b, np_, sq, sk = shape
    return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np_)


_MASK_BUILDERS = [_padding_mask, _causal_mask, _random_mask, _shared_mask]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.scaled_masked_softmax_cuda_forward
def test_baseline_source():
    """Record which Megatron baseline this run can check against."""
    assert _BASELINE_SOURCE
    print(f"\nMegatron baseline: {_BASELINE_SOURCE}")


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("shape", _SHAPES + _BENCH_SHAPES)
@pytest.mark.parametrize("build_mask", _MASK_BUILDERS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward(shape, build_mask, dtype):
    utils.init_seed(0)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = build_mask(shape)

    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)

    assert out.shape == scores.shape
    assert out.dtype == dtype
    assert out.is_contiguous()
    _assert_forward_close(out, scores, mask, dtype)

    # Masked keys are exactly zero, and every row that keeps a key is a
    # distribution over the keys it keeps.
    masked = mask.expand_as(out)
    assert torch.count_nonzero(out[masked]) == 0
    attended_rows = ~masked.all(dim=-1)
    sums = out.to(torch.float32).sum(dim=-1)[attended_rows]
    tolerance = 1e-2 if dtype is torch.bfloat16 else 2e-3
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=tolerance, rtol=0)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fully_masked_rows_are_zero(dtype):
    """A row that masks every key has no distribution: upstream answers zeros."""
    shape = (2, 2, 8, 8)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = torch.zeros((2, 1, 8, 8), dtype=torch.bool, device=flag_train.device)
    mask[:, 0, -1, :] = True  # one fully masked row per batch entry
    mask[:, 0, 0, :] = True

    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[:, :, -1, :]) == 0
    assert torch.count_nonzero(out[:, :, 0, :]) == 0
    _assert_close(out, masked_softmax_ref(_reference(scores), mask, _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_scale_and_values(dtype):
    """Scale 0, negative scale, and saturated scores stay finite."""
    shape = (2, 4, 4, 4)
    scores = torch.randn(shape, dtype=torch.float32, device=flag_train.device)
    scores[..., 0], scores[..., -1] = -10.0, 10.0
    scores = scores.to(dtype)
    mask = _causal_mask(shape)
    for scale in (0.0, -0.5, 1.0, 8.0):
        out = flag_train.megatron.scaled_masked_softmax_cuda_forward(
            scores, mask, scale
        )
        assert torch.isfinite(out).all()
        _assert_close(out, masked_softmax_ref(_reference(scores), mask, scale), dtype)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_dim_scale(dtype):
    """Megatron passes ``scale_t[0]``, a 0-dim tensor, not a Python float."""
    shape = (2, 2, 8, 8)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = _padding_mask(shape)
    scale = torch.tensor([_SCALE], dtype=torch.float32)

    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, scale[0])
    _assert_close(out, masked_softmax_ref(_reference(scores), mask, _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_noncontiguous(dtype):
    utils.init_seed(1)
    base = torch.randn((2, 4, 8, 8), dtype=dtype, device=flag_train.device)
    scores = base.transpose(2, 3)  # [2, 4, 8, 8], non-contiguous
    mask = torch.zeros((2, 1, 8, 8), dtype=torch.bool, device=base.device)
    mask[:, 0, :, 4:] = True
    mask = mask.transpose(2, 3)  # [2, 1, 8, 8], non-contiguous
    assert not scores.is_contiguous() and not mask.is_contiguous()

    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)
    assert out.is_contiguous()
    _assert_close(out, masked_softmax_ref(_reference(scores), mask, _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_empty_batch(dtype):
    scores = torch.empty((0, 2, 4, 4), dtype=dtype, device=flag_train.device)
    mask = torch.empty((1, 1, 4, 4), dtype=torch.bool, device=flag_train.device)
    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)
    assert out.shape == scores.shape and out.numel() == 0


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_does_not_mutate_operands(dtype):
    shape = (2, 2, 8, 8)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = _padding_mask(shape)
    scores_copy, mask_copy = scores.clone(), mask.clone()

    flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)
    torch.testing.assert_close(scores, scores_copy)
    torch.testing.assert_close(mask, mask_copy)


@needs_extension
@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 16, 16),
        (1, 2, 64, 64),
        (2, 4, 128, 128),
        (1, 8, 128, 256),
        (2, 4, 256, 512),
        (1, 4, 512, 512),
    ],
)
@pytest.mark.parametrize("dtype", _EXT_DTYPES)
def test_matches_megatron(shape, dtype):
    """The Triton kernel against the exact Apex call Megatron makes."""
    # Apex's forward tiles the query axis by a key-length dependent unit.
    assert shape[2] % _apex_batches_per_block(shape) == 0
    utils.init_seed(2)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = _random_mask(shape)

    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)
    expected = _megatron_forward(scores, mask, _SCALE)

    _assert_close(out, utils.to_reference(expected), dtype)


@needs_extension
@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _EXT_DTYPES)
def test_vendored_baseline_matches_torch(dtype):
    """The copied Megatron body agrees with the composition, not just with us."""
    utils.init_seed(3)
    shape = (2, 4, 64, 64)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = _padding_mask(shape)

    out = _vendored_forward(scores, mask, _SCALE)
    _assert_close(out, masked_softmax_ref(_reference(scores), mask, _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("ndim", "4-D"),
        ("dtype", "needs a"),
        ("mask-ndim", "mask needs a 4-D"),
        ("mask-dtype", "bool"),
        ("mask-batch", "b or 1"),
        ("mask-shape", "must match the scores"),
        ("mask-device", "one device"),
        ("cpu", "not one on the CPU"),
        ("key-limit", "exceeds Megatron"),
        ("scale", "one element"),
    ],
)
def test_rejects_invalid_arguments(case, match):
    scores = torch.randn((2, 2, 4, 4), dtype=torch.float32, device=flag_train.device)
    mask = _padding_mask(scores.shape)
    args = (scores, mask, _SCALE)
    if case == "ndim":
        args = (scores.reshape(2, 2, 16), mask, _SCALE)
    elif case == "dtype":
        args = (scores.double(), mask, _SCALE)
    elif case == "mask-ndim":
        args = (scores, mask.reshape(2, 16), _SCALE)
    elif case == "mask-dtype":
        args = (scores, mask.to(torch.uint8), _SCALE)
    elif case == "mask-batch":
        args = (scores, mask[:1].repeat(3, 1, 1, 1), _SCALE)
    elif case == "mask-shape":
        args = (scores, mask[..., :2], _SCALE)
    elif case == "mask-device":
        args = (scores, mask.cpu(), _SCALE)
    elif case == "cpu":
        args = (scores.cpu(), mask.cpu(), _SCALE)
    elif case == "key-limit":
        big = torch.empty((1, 1, 1, 4097), dtype=torch.float16, device=scores.device)
        args = (big, torch.zeros_like(big, dtype=torch.bool), _SCALE)
    elif case == "scale":
        args = (scores, mask, torch.ones(2))
    with pytest.raises(ValueError, match=match):
        flag_train.megatron.scaled_masked_softmax_cuda_forward(*args)
