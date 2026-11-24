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
import inspect
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
# Hygon's apex-das computes the full-mask ``scale_value`` but never multiplies
# it into the stored result, so a fully masked row comes back as a uniform
# distribution instead of zeros. That is a vendor bug in the HIP port, not in
# FlagTrain; skip the oracle-calibration case there until it is fixed. The
# kernel's own full-mask contract is covered by
# ``test_fully_masked_rows_are_zero``.
hygon_apex_full_mask = pytest.mark.skipif(
    flag_train.vendor_name == "hygon",
    reason="Hygon Apex HIP port omits upstream's full-mask scale_value multiply",
)

_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)
_EXT_DTYPES = [torch.float16] + ([torch.bfloat16] if utils.bf16_is_supported else [])

# ``[b, np, sq, sk]`` shapes Megatron selects for the fused path:
# ``16 < sk <= 4096``, ``sq/sk`` divisible by 4, ``b * np`` divisible by 4,
# and ``sq`` divisible by Apex's batch-per-block for the key length. The key
# lengths cover the batch-per-block tiers from 8 down to 4; query/key lengths
# differ, and the benchmark shapes are appended below.
_SHAPES = [
    (2, 2, 8, 20),
    (1, 4, 24, 24),
    (2, 2, 16, 32),
    (2, 2, 32, 48),
    (2, 2, 64, 96),
    (2, 8, 64, 128),
    (4, 8, 128, 128),
    (2, 8, 128, 256),
    (2, 2, 64, 4096),
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
    # Materialise the mask before broadcasting: torch_musa's ``masked_fill``
    # returns wrong values when the mask is a non-contiguous view (e.g. a
    # transposed mask), even though the scores it fills may be contiguous.
    # A contiguous source expanded to stride 0 in the head dimension is read
    # correctly.
    masked = mask.to(scores.device).bool().contiguous().expand_as(scores)
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
def test_public_interface_matches_megatron():
    """The call shape is the Apex entry Megatron's forward invokes."""
    params = inspect.signature(
        flag_train.megatron.scaled_masked_softmax_cuda_forward
    ).parameters
    assert list(params) == ["inputs", "mask", "scale"]
    assert all(param.default is inspect.Parameter.empty for param in params.values())

    shape = (2, 4, 32, 32)
    scores = torch.randn(shape, dtype=torch.float16, device=flag_train.device)
    mask = _padding_mask(shape)
    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)
    assert isinstance(out, torch.Tensor)
    assert out.shape == scores.shape and out.dtype == scores.dtype


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
    # distribution over the keys it keeps. The select is chunked over the
    # query axis like the reference above: torch_npu's boolean indexing
    # allocates a workspace proportional to the mask's numel (~48 bytes per
    # element whatever the True count), which is ~49 GiB on the largest
    # benchmark shape.
    masked = mask.expand_as(out)
    for start in range(0, out.shape[2], _REFERENCE_CHUNK):
        stop = start + _REFERENCE_CHUNK
        assert torch.count_nonzero(out[:, :, start:stop][masked[:, :, start:stop]]) == 0
    attended_rows = ~masked.all(dim=-1)
    sums = out.to(torch.float32).sum(dim=-1)[attended_rows]
    tolerance = 1e-2 if dtype is torch.bfloat16 else 2e-3
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=tolerance, rtol=0)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fully_masked_rows_are_zero(dtype):
    """A row that masks every key has no distribution: upstream answers zeros."""
    shape = (2, 2, 32, 32)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = torch.zeros((2, 1, 32, 32), dtype=torch.bool, device=flag_train.device)
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
    shape = (2, 4, 16, 32)
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
    shape = (2, 2, 32, 32)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = _padding_mask(shape)
    scale = torch.tensor([_SCALE], dtype=torch.float32)

    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, scale[0])
    _assert_close(out, masked_softmax_ref(_reference(scores), mask, _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_noncontiguous(dtype):
    utils.init_seed(1)
    base = torch.randn((2, 4, 64, 32), dtype=dtype, device=flag_train.device)
    scores = base.transpose(2, 3)  # [2, 4, 32, 64], non-contiguous
    mask = torch.zeros((2, 1, 64, 32), dtype=torch.bool, device=base.device)
    mask[:, 0, :, 4:] = True
    mask = mask.transpose(2, 3)  # [2, 1, 32, 64], non-contiguous
    assert not scores.is_contiguous() and not mask.is_contiguous()

    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)
    assert out.is_contiguous()
    _assert_close(out, masked_softmax_ref(_reference(scores), mask, _SCALE), dtype)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_uint8_mask(dtype):
    """Apex reads the mask as bytes; a uint8 0/1 mask is the same contract."""
    utils.init_seed(2)
    shape = (2, 4, 16, 32)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    mask = _padding_mask(shape)

    expected = flag_train.megatron.scaled_masked_softmax_cuda_forward(
        scores, mask, _SCALE
    )
    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(
        scores, mask.to(torch.uint8), _SCALE
    )
    torch.testing.assert_close(out, expected)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.int32, torch.float32])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mask_dtype_is_read_as_bytes(mask_dtype, dtype):
    """Apex casts mask storage to uint8; only bytes equal to 1 are masked."""
    utils.init_seed(3)
    shape = (2, 4, 16, 32)
    scores = torch.randn(shape, dtype=dtype, device=flag_train.device)
    packed = _padding_mask(shape).to(mask_dtype)
    raw = packed.view(torch.uint8).reshape(-1)[: packed.numel()].view(packed.shape)
    expected = flag_train.megatron.scaled_masked_softmax_cuda_forward(
        scores, raw == 1, _SCALE
    )

    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, packed, _SCALE)
    torch.testing.assert_close(out, expected)


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_empty_batch(dtype):
    scores = torch.empty((0, 2, 8, 64), dtype=dtype, device=flag_train.device)
    mask = torch.empty((1, 1, 8, 64), dtype=torch.bool, device=flag_train.device)
    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)
    assert out.shape == scores.shape and out.numel() == 0


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_heads_returns_empty(dtype):
    """Apex's zero-sized head grid is a no-op; the output is empty."""
    scores = torch.empty((2, 0, 32, 32), dtype=dtype, device=flag_train.device)
    mask = torch.empty((2, 1, 32, 32), dtype=torch.bool, device=flag_train.device)
    out = flag_train.megatron.scaled_masked_softmax_cuda_forward(scores, mask, _SCALE)
    assert out.shape == scores.shape and out.numel() == 0


@pytest.mark.scaled_masked_softmax_cuda_forward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_does_not_mutate_operands(dtype):
    shape = (2, 2, 32, 32)
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
        (2, 2, 16, 20),
        (2, 2, 32, 32),
        (2, 4, 64, 64),
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
@hygon_apex_full_mask
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
    ("case", "match", "error"),
    [
        ("ndim", "4D", RuntimeError),
        ("dtype", "Only fp16 and bf16", RuntimeError),
        ("mask-ndim", "4D", RuntimeError),
        ("mask-batch", "pad_batches", RuntimeError),
        ("mask-shape", r"mask\.size\(3\)", RuntimeError),
        ("mask-device", "one device", RuntimeError),
        ("cpu", "CPU", RuntimeError),
        ("key-limit", "key_seq_len", RuntimeError),
        ("key-limit-wide", "key_seq_len <= 16384", RuntimeError),
        ("sq-one", "query_seq_len > 1", RuntimeError),
        ("sk-zero", "16 < sk <= 4096", RuntimeError),
        ("sk-lower", "16 < sk <= 4096", RuntimeError),
        ("sq-multiple-of-4", r"sq % 4 == 0", RuntimeError),
        ("sk-multiple-of-4", r"sk % 4 == 0", RuntimeError),
        ("attn-batches", r"attn_batches % 4 == 0", RuntimeError),
        ("sq-divisibility", "batches_per_block", RuntimeError),
        ("scale", "one-element", TypeError),
    ],
)
def test_rejects_invalid_arguments(case, match, error):
    scores = torch.randn((2, 2, 32, 32), dtype=torch.float16, device=flag_train.device)
    mask = _padding_mask(scores.shape)
    args = (scores, mask, _SCALE)
    if case == "ndim":
        args = (scores.reshape(2, 2, 1024), mask, _SCALE)
    elif case == "dtype":
        args = (scores.float(), mask, _SCALE)
    elif case == "mask-ndim":
        args = (scores, mask.reshape(2, 1024), _SCALE)
    elif case == "mask-batch":
        args = (scores, mask[:1].repeat(3, 1, 1, 1), _SCALE)
    elif case == "mask-shape":
        args = (scores, mask[..., :4], _SCALE)
    elif case == "mask-device":
        args = (scores, mask.cpu(), _SCALE)
    elif case == "cpu":
        args = (scores.cpu(), mask.cpu(), _SCALE)
    elif case == "key-limit":
        big = torch.empty((2, 2, 32, 4097), dtype=torch.float16, device=scores.device)
        mask = torch.zeros((2, 1, 32, 4097), dtype=torch.bool, device=scores.device)
        args = (big, mask, _SCALE)
    elif case == "key-limit-wide":
        big = torch.empty((2, 2, 32, 16385), dtype=torch.float16, device=scores.device)
        mask = torch.zeros((2, 1, 32, 16385), dtype=torch.bool, device=scores.device)
        args = (big, mask, _SCALE)
    elif case == "sq-one":
        args = (
            torch.empty((2, 2, 1, 32), dtype=torch.float16, device=scores.device),
            torch.empty((2, 1, 1, 32), dtype=torch.bool, device=scores.device),
            _SCALE,
        )
    elif case == "sk-zero":
        args = (
            torch.empty((2, 2, 32, 0), dtype=torch.float16, device=scores.device),
            torch.empty((2, 1, 32, 0), dtype=torch.bool, device=scores.device),
            _SCALE,
        )
    elif case == "sk-lower":
        args = (
            torch.empty((2, 2, 32, 16), dtype=torch.float16, device=scores.device),
            torch.empty((2, 1, 32, 16), dtype=torch.bool, device=scores.device),
            _SCALE,
        )
    elif case == "sq-multiple-of-4":
        args = (
            torch.empty((2, 2, 10, 32), dtype=torch.float16, device=scores.device),
            torch.empty((2, 1, 10, 32), dtype=torch.bool, device=scores.device),
            _SCALE,
        )
    elif case == "sk-multiple-of-4":
        args = (
            torch.empty((2, 2, 8, 18), dtype=torch.float16, device=scores.device),
            torch.empty((2, 1, 8, 18), dtype=torch.bool, device=scores.device),
            _SCALE,
        )
    elif case == "attn-batches":
        args = (
            torch.empty((1, 2, 32, 32), dtype=torch.float16, device=scores.device),
            torch.empty((1, 1, 32, 32), dtype=torch.bool, device=scores.device),
            _SCALE,
        )
    elif case == "sq-divisibility":
        args = (
            torch.empty((2, 2, 12, 64), dtype=torch.float16, device=scores.device),
            torch.empty((2, 1, 12, 64), dtype=torch.bool, device=scores.device),
            _SCALE,
        )
    elif case == "scale":
        args = (scores, mask, torch.ones(2))
    with pytest.raises(error, match=match):
        flag_train.megatron.scaled_masked_softmax_cuda_forward(*args)
