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
"""Correctness tests for the evoformer attention forward.

Two oracles, because they fail differently:

* ``DS4Sci_EvoformerAttention`` -- DeepSpeed's own operator, the thing this port
  claims to replace. On a backend in ``_DEEPSPEED_BASELINE_VENDORS`` it is
  *expected*: if it will not load there the module warns and falls back to the
  torch reference so the suite still runs, rather than failing outright. Its
  version is recorded in ``_DEEPSPEED_VERSION``, as
  tests/deepspeed/README.md asks for.
* ``_reference`` -- the same contract composed from plain torch: no tiling, no
  online softmax, no running max. It runs on any device, so the operator is
  testable off NVIDIA.

The reference is not implied by the DeepSpeed oracle: both could be wrong the
same way, and the reference is the only one that can disagree with a *reading*
of the contract rather than with one implementation of it.

Only the forward is covered, because only the forward is ported -- upstream's
``attention_bwd`` and its trainable entry are absent, not stubbed, so there is
nothing here to test them against.

Cases are seeded, so a failure reproduces rather than depending on the draw.
"""

import warnings

import pytest
import torch

import flag_train
import flag_train.deepspeed

from .. import accuracy_utils as utils

# The absolute tolerance is the reference implementation's, not this repository's
# default: upstream holds its own kernel to `eps = 1e-2` (fp16) / `5e-2` (bf16).
# Matching that is the requirement; a tighter number would be measuring the
# rounding that no fp16 tensor-core attention can avoid (the softmax
# probabilities have to be rounded into the compute dtype before the PV dot).
# It is passed to ``train_assert_close`` as its ``atol``; the relative part of
# the comparison stays the house one. ``lse`` is fp32 whatever the operands are
# (the operator's contract), so it is compared under the fp32 entry.
_ATOL = {torch.float16: 1e-2, torch.bfloat16: 5e-2}

# NGC images ship TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which silently runs every
# fp32 matmul in tf32 -- ten mantissa bits, ~5e-4 relative error. That error
# lands on the *reference*, not on the kernel, so left enabled the suite measures
# the reference and reports it against the operator. benchmark/base.py already
# switches it off; the tests have to as well.
torch.backends.cuda.matmul.allow_tf32 = False

# The reference is evaluated in fp32 rather than fp64. With tf32 off, fp32
# accumulation is ~1e-7 relative -- four orders below the 1e-2 the operator is
# held to -- while the score tensor of the largest case here is 537M entries, so
# fp64 would cost 4 GB per intermediate for no measurable gain.
_REF_DTYPE = torch.float32

# The reference materialises an ``[L, L]`` matrix per (pair, head); the biggest
# case here is (1, 2048, 256, 4, 32) -- 537M fp32 entries, 2 GB, and softmax adds
# another copy. It is evaluated in slices of at most this many entries instead,
# which the operator's math allows because a pair slice is independent of every
# other: the biases broadcast *into* the slice, never across it. Smaller cases
# fit in one slice, so there is only the one path.
_REFERENCE_SCORE_BUDGET = 1 << 26  # 67M entries, 268 MB of fp32


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference(q, k, v, bias1=None, bias2=None):
    """The contract, composed from plain torch.

    ``[B, N, L, H, D]`` in, ``(out, lse)`` out, with the attention matrix laid out
    as ``[B, N, H, L, L]`` -- which is where both biases broadcast, and the reason
    they have the shapes they do: ``bias1`` has a singleton head axis, so one
    value per key position reaches every query row and head; ``bias2`` has a
    singleton pair axis, so one matrix per head is shared by every pair.

    The scale is applied to the scores before the biases are added, which is the
    order upstream's epilogue uses (``accum * scale + bias``).

    Evaluated in slices along the flattened (batch, pair) axis -- see
    ``_REFERENCE_SCORE_BUDGET``. The slices are independent, so this is the same
    arithmetic as doing it in one go, not an approximation of it.

    Query and key lengths are read separately, so a shorter key set is allowed
    here even though the operator does not accept one (it reads ``num_queries =
    num_keys`` off the query). That is what lets the padding test express "the
    masked key is not there" as a reference over the remaining keys, rather than
    as a second reading of what masking means.
    """
    dtype = q.dtype
    q_len = q.shape[-3]
    kv_len = k.shape[-3]
    heads, head_size = q.shape[-2], q.shape[-1]

    queries = q.reshape(-1, q_len, heads, head_size).to(dtype)
    keys = k.reshape(-1, kv_len, heads, head_size).to(dtype)
    values = v.reshape(-1, kv_len, heads, head_size).to(dtype)

    # bias1 is [B, N, 1, 1, Lk]: one row of key positions per slice. bias2 is
    # [B, 1, H, Lq, Lk]: the same matrix for every pair of a batch, which is why
    # a slice has to be mapped back to its batch rather than indexed by slice.
    b1_rows = None if bias1 is None else bias1.reshape(-1, kv_len).to(dtype)
    b2_rows = (
        None if bias2 is None else bias2.reshape(-1, heads, q_len, kv_len).to(dtype)
    )
    pairs = q.shape[-4] if q.dim() >= 4 else 1

    per_slice = heads * q_len * kv_len
    step = max(1, _REFERENCE_SCORE_BUDGET // max(per_slice, 1))

    outs, lses = [], []
    for start in range(0, queries.shape[0], step):
        # Clamped, not `start + step`: the last slice is the short one, and the
        # batch index below is built from this range. A slice that ran past the
        # end would index bias2 past its batch -- which is what a first attempt
        # at this did, on the small shapes, where one slice covers everything.
        stop = min(start + step, queries.shape[0])
        chunk = slice(start, stop)

        qh = queries[chunk].transpose(-2, -3)
        kh = keys[chunk].transpose(-2, -3)
        vh = values[chunk].transpose(-2, -3)

        scores = qh @ kh.transpose(-1, -2) / (head_size**0.5)
        if b1_rows is not None:
            scores = scores + b1_rows[chunk][:, None, None, :]
        if b2_rows is not None:
            batch_of_slice = torch.arange(start, stop, device=q.device) // pairs
            scores = scores + b2_rows[batch_of_slice]

        lses.append(torch.logsumexp(scores, dim=-1))
        p = torch.softmax(scores, dim=-1)
        outs.append((p @ vh).transpose(-2, -3))

    out = torch.cat(outs).reshape(q.shape)
    lse = torch.cat(lses).reshape(*q.shape[:-3], heads, q_len)
    return out, lse


def _empty(like):
    """The "no bias" sentinel both implementations accept: an empty tensor."""
    return torch.empty((0,), dtype=like.dtype, device=like.device)


def _assert_close(name, res, ref, dtype, atol):
    """Compare ``res`` against ``ref`` through the repository's own helper.

    ``utils.train_assert_close`` is what the other operators' functional suites
    compare with (see ``tests/deepspeed/test_lamb.py``), so the operator is held
    to the house comparison rather than to one written here: it checks ``res`` is
    already ``dtype``, casts ``ref`` to it, and takes the relative tolerance from
    ``flag_train.testing.RESOLUTION``.

    Both sides go through ``utils.to_reference`` first, which is how
    ``test_lamb.py`` calls it and not merely a convenience: under ``--ref cpu``
    that is what puts the comparison on the CPU, and ``train_assert_close``
    requires the reference to be there already (``utils.to_cpu`` asserts it).
    Most callers here pass a reference that ``_reference_operands`` already moved
    and for which this is then a no-op, but the DeepSpeed cross-check and
    ``test_absent_bias_is_empty_tensor`` compare two *device* results, and
    without it they would trip that assert as soon as ``--ref cpu`` is passed.

    The one thing *not* taken from there is ``atol``, which is passed in. The
    operator is held to the absolute tolerance upstream's own test asserts rather
    than the house default (see ``_ATOL``), because that is the requirement this
    port took on -- the same split ``test_blocked_flash.py`` makes, and for the
    same reason.

    ``name`` only labels the failure: the helper reports the offending elements,
    not which of a test's several comparisons they came from.
    """
    try:
        utils.train_assert_close(
            utils.to_reference(res), utils.to_reference(ref), dtype, atol=atol
        )
    except AssertionError as exc:
        raise AssertionError(f"{name}: {exc}") from exc


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# ``(batch, pairs, seq_len, heads, head_size)``.
#
# The first two are upstream's own test cases. The rest are chosen from what this
# operator is rather than from a shape sweep, one axis each:
#
#   * ``N`` (pairs) is what makes it evoformer attention instead of attention:
#     ``bias2`` is shared across it and ``bias1`` is indexed by it, and the two
#     axes are flattened together for the kernel, so the cases vary B and N
#     *separately* -- a swap of the two would survive ``B == N``.
#   * ``L`` meets four boundaries: the 32 the lse is padded to, the 64 of both
#     tiles, "not a multiple of either", and 16 -- the floor upstream asserts and
#     this port does not, so the shapes just below it are part of the contract
#     too (``test_sequences_below_upstreams_floor`` walks them individually).
#   * ``D`` is the head size, capped at 64 upstream; the case that matters is the
#     one where ``D`` is not a power of two, because that is the only time the
#     kernel's head-size mask masks anything (at D = 8/16/32/64 the tile width is
#     exactly D and the mask folds away at compile time).
#   * ``B*N`` sets the grid's third dimension, so the two ends of its range are
#     both here: 2 (one slice, one query tile) and 512.
_SHAPES = [
    (1, 256, 256, 4, 32),
    (1, 512, 256, 8, 8),
    (2, 3, 64, 4, 32),
    (1, 1, 17, 1, 64),
    (1, 4, 65, 2, 16),
    # The old floor itself, so both the forward and the lse padding below 16
    # (16 of the 32 padded columns are +inf here) are covered at this size too.
    (1, 4, 16, 2, 32),
    (1, 4, 100, 2, 24),
    # lse needs no padding at all when L is already a multiple of 32.
    (1, 4, 32, 2, 32),
    # D = 40: tile width 64 against a 40-wide head, so the mask masks, and
    # L = 48 leaves both tiles partial.
    (1, 4, 48, 2, 40),
    # B != N, one query tile, both biases per-index.
    (3, 2, 64, 8, 8),
    # The other end of the L/N aspect ratio: long sequence, two pairs.
    (1, 2, 512, 4, 32),
]


def _mask_bias(mask, dtype):
    """Turn a 0/1 keep-mask into the additive ``bias1`` a padding mask is.

    Upstream's own test builds it as ``1e9 * (mask - 1)``, which in fp16/bf16 is
    -inf (1e9 overflows both) -- so a masked key is excluded *exactly*, not
    merely made unlikely. That is the property the padding tests below rely on.
    """
    return (1e9 * (mask.to(torch.float32) - 1)).to(dtype)


def _inputs(shape, dtype, seed=0, bias1=True, bias2=True):
    """Seeded operands. The biases are the ones upstream's own test builds.

    ``bias1`` is where a key-padding mask lives, so the test builds one the way
    OpenFold does -- a large negative value on the masked positions -- rather
    than drawing it from a normal, which would leave the operator's handling of
    a masked key column unexercised.
    """
    batch, pairs, seq_len, heads, head_size = shape
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*s):
        return torch.randn(*s, generator=generator, dtype=torch.float32).to(
            dtype=dtype, device=flag_train.device
        )

    q = randn(batch, pairs, seq_len, heads, head_size)
    k = randn(batch, pairs, seq_len, heads, head_size)
    v = randn(batch, pairs, seq_len, heads, head_size)

    b1 = None
    if bias1:
        mask = torch.randint(0, 2, (batch, pairs, 1, 1, seq_len), generator=generator)
        b1 = _mask_bias(mask.to(device=flag_train.device), dtype)
    b2 = None
    if bias2:
        b2 = randn(batch, 1, heads, seq_len, seq_len)
    return q, k, v, b1, b2


def _reference_operands(q, k, v, bias1, bias2):
    """The same values on whichever device the reference is meant to run, upcast.

    ``--ref cpu`` pushes the reference *computation* onto the CPU, not merely its
    result, so every operand it is handed comes from here.
    """
    conv = lambda t: None if t is None else utils.to_reference(t).to(_REF_DTYPE)
    return conv(q), conv(k), conv(v), conv(bias1), conv(bias2)


# ---------------------------------------------------------------------------
# DeepSpeed oracle
# ---------------------------------------------------------------------------

_DEEPSPEED_UNAVAILABLE_MSG = (
    "DeepSpeed's DS4Sci_EvoformerAttention is unavailable; on a CUDA backend it "
    "is the expected oracle. It JIT-compiles its CUTLASS kernels, so it needs a "
    "CUTLASS >= 3.1: `pip install nvidia-cutlass` (the builder resolves it from "
    "the cutlass_library package), or point CUTLASS_PATH at a checkout."
)

# Backends whose oracle is DeepSpeed. Its operator is a CUDA extension builder,
# so only a backend that can compile and run one can host it. On these the oracle
# is expected, not mandatory: if it fails to load we warn and fall back to the
# torch reference so the rest of the suite still runs, and the warning is what
# keeps the lost check from thinning quietly.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia"}

# Backends whose kernel walks the flattened (batch, pair) dimension over a grid
# axis of its own instead of placing it in the grid. Such a kernel is bounded by
# the walk's width rather than by the driver's 65535-per-axis limit, so it serves
# a slice count the generic kernel refuses. Ascend's is the only one: the walk is
# there to keep the launch under its Cube's 32768-program ceiling, and the same
# restructure happens to remove the z ceiling with it.
_WALKING_SLICE_VENDORS = {"ascend"}


def _load_deepspeed_evoformer():
    """``(public entry, version)`` for DeepSpeed's evoformer attention.

    The public entry rather than the raw ``attention`` kernel, because that is
    the name this port is meant to stand in for. It is also a superset of what is
    compared: the output it returns is the same one the kernel writes, and the
    autograd wiring around it is upstream's own business.

    On a backend in ``_DEEPSPEED_BASELINE_VENDORS`` a load failure is warned about
    rather than raised: the DeepSpeed check is dropped but the torch reference
    still runs, so a missing DS4Sci_EvoformerAttention no longer aborts the whole
    module.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        import deepspeed
        from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
        from deepspeed.ops.op_builder import EvoformerAttnBuilder

        EvoformerAttnBuilder().load()
        return DS4Sci_EvoformerAttention, deepspeed.__version__
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} is expected to use DeepSpeed's "
            f"DS4Sci_EvoformerAttention as its oracle, but it could not be "
            f"loaded: {exc!r}. Falling back to the torch reference only; build "
            f"deepspeed (and CUTLASS) to enable the DeepSpeed check.",
            stacklevel=2,
        )
        return None, None


# Resolved once, at module import time.
_deepspeed_evoformer, _DEEPSPEED_VERSION = _load_deepspeed_evoformer()

requires_deepspeed_reference = pytest.mark.skipif(
    _deepspeed_evoformer is None, reason=_DEEPSPEED_UNAVAILABLE_MSG
)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.evoformer_attn
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(
    "biases",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["both_biases", "bias1_only", "bias2_only", "no_bias"],
)
def test_forward_matches_reference(shape, dtype, biases):
    """The output and the lse must match the reference.

    The four bias combinations are not decoration: upstream compiles a separate
    kernel for each (``BroadcastNoLoad``/``BroadcastA``/``BroadcastB``), and this
    port mirrors that with two constexpr flags, so an unexercised combination is
    an unexercised kernel.
    """
    q, k, v, bias1, bias2 = _inputs(shape, dtype, bias1=biases[0], bias2=biases[1])
    out, lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2)

    rq, rk, rv, rb1, rb2 = _reference_operands(q, k, v, bias1, bias2)
    ref_out, ref_lse = _reference(rq, rk, rv, rb1, rb2)

    atol = _ATOL[dtype]
    _assert_close("out", out, ref_out, dtype, atol)
    # lse is padded to a multiple of 32; only the first L columns carry the
    # log-sum-exp, and they are the only ones the reference has.
    batch, pairs, seq_len, heads, _ = shape
    _assert_close(
        "lse",
        lse[..., :seq_len],
        ref_lse.reshape(batch * pairs, heads, seq_len),
        torch.float32,
        atol,
    )


@pytest.mark.evoformer_attn
@pytest.mark.parametrize("shape", _SHAPES)
def test_lse_contract(shape):
    """``lse`` is ``[B*N, H, ceil(L/32)*32]`` fp32, +inf past the sequence.

    The padding is not cosmetic: it is upstream's arrangement, made so that its
    backward can read past the sequence end without bounding every load. So this
    is part of the forward's output contract -- a caller handing this lse to
    upstream's backward gets the +inf it expects -- and a garbage or zero pad
    would be silently wrong there rather than here.
    """
    dtype = torch.float16
    q, k, v, bias1, bias2 = _inputs(shape, dtype)
    batch, pairs, seq_len, heads, _ = shape

    _, lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2)

    expected_len = (seq_len + 31) // 32 * 32
    assert lse.shape == (batch * pairs, heads, expected_len), lse.shape
    assert lse.dtype == torch.float32

    pad = lse[..., seq_len:]
    if pad.numel():
        assert bool(torch.isinf(pad).all()), "the padding must be +inf"

    rq, rk, rv, rb1, rb2 = _reference_operands(q, k, v, bias1, bias2)
    _, ref_lse = _reference(rq, rk, rv, rb1, rb2)
    _assert_close(
        "lse",
        lse[..., :seq_len],
        ref_lse.reshape(batch * pairs, heads, seq_len),
        torch.float32,
        _ATOL[dtype],
    )


@pytest.mark.evoformer_attn
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_absent_bias_is_empty_tensor(dtype):
    """``None`` and an empty tensor mean the same thing, and neither changes the result.

    Upstream spells "no bias" as an empty tensor and dispatches on ``size(0)``;
    accepting ``None`` as well is this port's convenience, so the two spellings
    have to agree.
    """
    shape = (1, 4, 64, 2, 32)
    q, k, v, bias1, bias2 = _inputs(shape, dtype)

    with_none = flag_train.deepspeed.evoformer_attn(q, k, v, None, None)
    with_empty = flag_train.deepspeed.evoformer_attn(q, k, v, _empty(q), _empty(q))
    # Bit-identical, so the house equality helper rather than the closeness one:
    # the two spellings of "no bias" have to select the same kernel and produce
    # the same bits, not merely agree to a tolerance. Both sides are the
    # *operator's* output, so both go through ``to_reference`` -- see
    # ``_assert_close`` for why that is required under ``--ref cpu``.
    utils.train_assert_equal(
        utils.to_reference(with_none[0]), utils.to_reference(with_empty[0])
    )
    utils.train_assert_equal(
        utils.to_reference(with_none[1]), utils.to_reference(with_empty[1])
    )

    # And an empty bias1 must not be the same as a present one.
    with_bias, _ = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2)
    assert not torch.equal(with_none[0], with_bias)


# ---------------------------------------------------------------------------
# What bias1 is for: padding
#
# bias1 exists because a padded batch needs one: it is the only bias with a
# single value per key position, and OpenFold's own use of it is a key mask. The
# tests above feed it a *random* mask, which shows the kernel reads it, but not
# that a masked key is excluded rather than merely unlikely. These two do, and
# neither needs the reference to be right about anything except plain torch.
# ---------------------------------------------------------------------------


@pytest.mark.evoformer_attn
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [32, 40])
def test_a_masked_key_contributes_nothing(dtype, head_size):
    """Masking a block of keys gives the same answer as deleting them.

    The reference is run on the operand with those keys removed, so this compares
    against attention over a shorter key set rather than against a second reading
    of the mask. A kernel that added ``bias1`` to the output instead of the
    scores, or that let a masked key keep a share of the softmax, would differ
    here while still matching the reference on the random-mask cases above --
    where both sides would be wrong together.
    """
    seq_len = 64
    shape = (1, 4, seq_len, 2, head_size)
    q, k, v, bias1, bias2 = _inputs(shape, dtype, bias1=False, bias2=False)

    # Keys 10..19 are padding.
    keep = list(range(10)) + list(range(20, seq_len))
    mask = torch.ones(1, 4, 1, 1, seq_len, dtype=torch.int64, device=flag_train.device)
    mask[..., 10:20] = 0
    bias1 = _mask_bias(mask, dtype)

    out, lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, None)

    rq, rk, rv, rb1, _ = _reference_operands(
        q, k[..., keep, :, :], v[..., keep, :, :], bias1[..., keep], None
    )
    ref_out, ref_lse = _reference(rq, rk, rv, rb1, None)

    atol = _ATOL[dtype]
    _assert_close("out", out, ref_out, dtype, atol)
    _assert_close(
        "lse",
        lse[..., :seq_len],
        ref_lse.reshape(4, 2, seq_len),
        torch.float32,
        _ATOL[dtype],
    )


@pytest.mark.evoformer_attn
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,cut", [((1, 4, 64, 2, 32), 40), ((1, 4, 100, 2, 24), 65)]
)
def test_padding_the_tail_matches_a_shorter_sequence(dtype, shape, cut):
    """A masked tail must leave the rows before it exactly as they were.

    This is the inference case the mask exists for: a batch padded to a common
    length, where every real row must come out as if the padding were not there.
    It is checked against *this operator* on the truncated operands -- a
    self-consistency check, so nothing depends on the reference being right, and
    it also means the shorter run's own lse padding is exercised at a second L.
    """
    seq_len = shape[2]
    q, k, v, _, bias2 = _inputs(shape, dtype)

    mask = torch.ones(
        *shape[:2], 1, 1, seq_len, dtype=torch.int64, device=flag_train.device
    )
    mask[..., cut:] = 0
    bias1 = _mask_bias(mask, dtype)

    out, lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2)

    # The same attention over the first ``cut`` tokens -- bias2 is L x L, so it
    # is truncated on both axes, which is what "the padding is not there" means.
    short = (shape[0], shape[1], cut, shape[3], shape[4])
    qs, ks, vs = (
        q[:, :, :cut].contiguous(),
        k[:, :, :cut].contiguous(),
        v[:, :, :cut].contiguous(),
    )
    b1s = bias1[..., :cut].contiguous()
    b2s = bias2[..., :cut, :cut].contiguous()
    short_out, short_lse = flag_train.deepspeed.evoformer_attn(qs, ks, vs, b1s, b2s)

    assert short_out.shape == short
    _assert_close("prefix out", out[:, :, :cut], short_out, dtype, _ATOL[dtype])
    _assert_close(
        "prefix lse", lse[..., :cut], short_lse[..., :cut], torch.float32, _ATOL[dtype]
    )


@pytest.mark.evoformer_attn
@pytest.mark.skipif(
    utils.TO_CPU,
    reason=(
        "the fp32 CPU reference costs ~30 s per case at these sizes; the point "
        "of these cases is scale on the device, which --ref cpu cannot exercise "
        "anyway (the small shapes already cover its arithmetic)"
    ),
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        # 2048 pairs -- four times the deepest case upstream's own test uses,
        # and the largest pair count this operator is likely to see (an MSA
        # stack). Everything about it is L-shaped: it is 537M score entries,
        # which is what the sliced reference exists for.
        (1, 2048, 256, 4, 32),
        # The other axis: 2048 tokens, so the online softmax runs over 32 key
        # tiles and the query grid over 32 tiles. The running max and sum carry
        # across all of them, which is where an online-softmax implementation
        # accumulates its drift.
        (1, 4, 2048, 4, 32),
    ],
)
def test_large_shapes_match_reference(shape, dtype):
    """The two large ends of each axis, against the same reference.

    These are here for the arithmetic that only shows up at size: the running
    max over many tiles (long L), and the same score matrix over many pairs
    (large N). Neither is a new code path -- which is the point, because the
    failure they would catch is one that only appears when a loop or an index
    runs long enough.

    Only the both-biases combination: the four combinations are there to cover
    the kernel's four branches, and the 10 shapes above already do that. What
    these add is size, and repeating the branch sweep at 537M score entries
    would buy the same branch coverage for minutes of runtime.
    """
    q, k, v, bias1, bias2 = _inputs(shape, dtype)
    out, lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2)

    rq, rk, rv, rb1, rb2 = _reference_operands(q, k, v, bias1, bias2)
    ref_out, ref_lse = _reference(rq, rk, rv, rb1, rb2)

    batch, pairs, seq_len, heads, _ = shape
    atol = _ATOL[dtype]
    _assert_close("out", out, ref_out, dtype, atol)
    _assert_close(
        "lse",
        lse[..., :seq_len],
        ref_lse.reshape(batch * pairs, heads, seq_len),
        torch.float32,
        atol,
    )


@pytest.mark.evoformer_attn
def test_the_pair_dimension_has_a_grid_ceiling():
    """``B*N`` is the kernel's grid z axis, and an axis is capped at 65535.

    The ceiling is upstream's as well -- its forward grid is
    ``(queries/64, heads, batches)`` -- so on a kernel that *puts* the flattened
    (batch, pair) dimension in the grid this is the same limit reported the same
    way upstream's launcher would report it, rather than the driver's
    "[CUDA]: invalid argument", which does not say which argument. Both sides of
    the boundary are checked: 65535 must run, because a limit that rejects one
    less than it allows would be worse than no check.

    A kernel that *walks* that dimension instead -- the Ascend one does -- is
    bounded by the walk's width rather than by the axis, so it serves past the
    ceiling and this becomes a test of that instead. Which of the two a backend
    is, is not something the caller can see, so the branch is on the vendor.

    The operands are deliberately tiny (L = 17, H = 1, D = 8): what is being
    tested is the grid, not the arithmetic, and 65536 slices of anything is
    enough to fill the axis.
    """
    dtype = torch.float16
    slices, seq_len = 65535, 17

    def operands(count):
        return [
            torch.randn(
                (1, count, seq_len, 1, 8), dtype=dtype, device=flag_train.device
            )
            for _ in range(3)
        ]

    q, k, v = operands(slices)
    out, lse = flag_train.deepspeed.evoformer_attn(q, k, v)
    assert out.shape == q.shape
    assert bool(torch.isfinite(out).all())

    q, k, v = operands(slices + 1)
    if flag_train.vendor_name in _WALKING_SLICE_VENDORS:
        out, lse = flag_train.deepspeed.evoformer_attn(q, k, v)
        assert out.shape == q.shape
        assert bool(torch.isfinite(out).all())
    else:
        with pytest.raises(ValueError, match="65535"):
            flag_train.deepspeed.evoformer_attn(q, k, v)


@pytest.mark.evoformer_attn
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 2, 3, 5, 8, 15, 16])
def test_sequences_below_upstreams_floor(seq_len, dtype):
    """``L`` below the 16 upstream asserts must be *served*, not refused.

    Upstream's forward asserts ``L > 16``. That is a statement about the kernel
    being ported -- its tiles are sized for it -- and not about this one, whose
    tiles are masked, so below the floor is the same arithmetic over fewer
    columns. Refusing it would make the port reject inputs upstream's API
    description does not mention, and the direction that breaks a drop-in
    replacement is refusing, not accepting.

    The whole range is walked rather than sampled, because the failures this
    would catch are not arithmetic: they are the masks and the lse padding. At
    ``L`` = 1 the sequence is a single key, and the padding the kernel writes
    past it is 31 of the 32 lse columns.

    No ``bias1`` here. It is a *key* mask, and at these lengths a random one
    zeroes a whole row often enough to matter -- at ``L`` = 1 it does so half the
    time -- which is the fully-masked-row NaN that
    ``test_a_fully_masked_query_row_is_nan`` pins on its own terms. Composing the
    two would test that contract here without saying so.
    """
    shape = (1, 4, seq_len, 2, 32)
    q, k, v, bias1, bias2 = _inputs(shape, dtype, bias1=False)
    out, lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2)

    rq, rk, rv, rb1, rb2 = _reference_operands(q, k, v, bias1, bias2)
    ref_out, ref_lse = _reference(rq, rk, rv, rb1, rb2)

    atol = _ATOL[dtype]
    _assert_close("out", out, ref_out, dtype, atol)
    batch, pairs, _, heads, _ = shape
    _assert_close(
        "lse",
        lse[..., :seq_len],
        ref_lse.reshape(batch * pairs, heads, seq_len),
        torch.float32,
        atol,
    )
    assert bool(torch.isinf(lse[..., seq_len:]).all()), "the padding must be +inf"


@pytest.mark.evoformer_attn
def test_more_heads_than_one_launch_can_hold():
    """Heads beyond a launch's width are split across launches, not refused.

    A kernel using the Cube cannot launch more than 32768 programs on this
    device, and the head axis is one of the two a launch spends that on (the
    other is the query tiles). So heads do not have a ceiling here -- past one
    launch's worth, the host splits the axis and launches once per chunk.

    ``L`` = 64, ``D`` = 8 makes the query one tile, so heads alone have to carry
    the launch past 32768: the chunk is 32768 wide and the second launch holds
    the single remaining head. That ragged tail is the part worth testing -- an
    implementation that only handled an exact multiple of the chunk would look
    right at 32768 and wrong here.
    """
    dtype = torch.float16
    batch, pairs, seq_len, heads, head_size = 1, 1, 64, 32769, 8
    shape = (batch, pairs, seq_len, heads, head_size)
    q, k, v, bias1, bias2 = _inputs(shape, dtype)
    out, lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2)

    assert out.shape == q.shape
    assert lse.shape == (batch * pairs, heads, seq_len)

    rq, rk, rv, rb1, rb2 = _reference_operands(q, k, v, bias1, bias2)
    ref_out, ref_lse = _reference(rq, rk, rv, rb1, rb2)
    _assert_close("out", out, ref_out, dtype, _ATOL[dtype])
    _assert_close(
        "lse",
        lse[..., :seq_len],
        ref_lse.reshape(batch * pairs, heads, seq_len),
        torch.float32,
        _ATOL[dtype],
    )


@pytest.mark.evoformer_attn
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a_fully_masked_query_row_is_nan(dtype):
    """A query row with every score at -inf produces NaN -- here and upstream.

    Worth pinning rather than fixing. ``softmax`` of a row whose every score is
    -inf is undefined, and both implementations reach the same undefined place:
    the running max is -inf, so ``exp2(-inf - -inf)`` is ``exp2(nan)``. A port
    that quietly turned those rows into zeros would be *changing the gradient a
    training loop sees* while looking tidier, so the NaN is part of the contract
    this port takes on -- and the rows that are not fully masked must still be
    right, which the second half of the test checks (a "just clamp it" fix would
    break that half instead).

    The row is masked with an actual ``-inf``, not with the ``1e9 * (mask - 1)``
    recipe upstream's test uses. That recipe is a mask only where it *overflows*:
    fp16 tops out at 65504 so -1e9 becomes -inf, but bf16 has fp32's exponent
    range, so -1e9 stays finite -- and a finite constant added to a whole row
    cancels in the softmax, which is to say it masks nothing. As a *key* mask the
    recipe works in both dtypes (the constant is per key position, not per row,
    so it does not cancel); ``test_a_masked_key_contributes_nothing`` runs it in
    both.

    ``bias2`` is what masks a row: ``bias1`` has one value per key position, so it
    cannot single out a row.
    """
    seq_len, cut = 64, 40
    shape = (1, 4, seq_len, 2, 32)
    q, k, v, bias1, bias2 = _inputs(shape, dtype)

    b2 = bias2.clone()
    b2[..., cut:, :] = float("-inf")

    out, lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, b2)

    assert bool(torch.isnan(out[:, :, cut:]).all()), "a fully masked row is NaN"
    assert bool(torch.isnan(lse[..., cut:]).all()), "and so is its lse"

    # Everything before the masked rows is unaffected and still correct.
    assert bool(torch.isfinite(out[:, :, :cut]).all())
    rq, rk, rv, rb1, rb2 = _reference_operands(q, k, v, bias1, b2)
    ref_out, _ = _reference(rq, rk, rv, rb1, rb2)
    _assert_close(
        "unmasked rows", out[:, :, :cut], ref_out[:, :, :cut], dtype, _ATOL[dtype]
    )


# ---------------------------------------------------------------------------
# DeepSpeed oracle, pinned
# ---------------------------------------------------------------------------


@pytest.mark.evoformer_attn
@requires_deepspeed_reference
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        # Upstream's own two cases.
        (1, 256, 256, 4, 32),
        (1, 512, 256, 8, 8),
        # And two of the boundaries: no lse padding at all, and a partial query
        # tile with a head size that is not a power of two. Both were checked
        # against DeepSpeed before being added here -- its `problem_size_0_m` is
        # `min(kQueriesPerBlock, num_queries)` rather than the tile's remaining
        # rows, which reads like it would overrun a partial tile; it does not.
        (1, 4, 32, 2, 32),
        (1, 4, 100, 2, 24),
    ],
)
def test_matches_deepspeed_oracle(shape, dtype):
    """Pin the DeepSpeed oracle explicitly, so a run that quietly stopped
    reaching it (deepspeed or CUTLASS missing) is visible rather than silently
    thinner.

    The output is compared at upstream's own tolerance -- this is the check that
    says the operator can stand in for the one it ports. Only the output: the
    gradients upstream's entry also produces come from a backward this port does
    not have.
    """
    q, k, v, bias1, bias2 = _inputs(shape, dtype)

    train_out, train_lse = flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2)

    # Upstream's entry takes the biases as a list, and appends to it when it is
    # short -- which is why the list is built inline rather than reused.
    ds_out = _deepspeed_evoformer(q, k, v, [bias1, bias2])

    atol = _ATOL[dtype]
    _assert_close("out vs deepspeed", train_out, ds_out, dtype, atol)

    # The lse has no counterpart on the public entry, so it is checked against
    # the reference here instead: a pin that only ever looked at `out` would not
    # notice a forward that stopped producing the second half of its contract.
    rq, rk, rv, rb1, rb2 = _reference_operands(q, k, v, bias1, bias2)
    _, ref_lse = _reference(rq, rk, rv, rb1, rb2)
    batch, pairs, seq_len, heads, _ = shape
    _assert_close(
        "lse vs reference",
        train_lse[..., :seq_len],
        ref_lse.reshape(batch * pairs, heads, seq_len),
        torch.float32,
        atol,
    )


# ---------------------------------------------------------------------------
# What the operator refuses
# ---------------------------------------------------------------------------


@pytest.mark.evoformer_attn
def test_rejects_inputs_outside_the_contract():
    """Every rejection names something upstream refuses too.

    None of these is this port inventing a limit: each is an assertion or a
    ``TORCH_CHECK`` in the code being ported, and the point of raising here is
    that the alternative is a Triton compile error or a silent wrong answer.

    What is *not* here is upstream's ``L > 16`` assert. That one is about the
    kernel being ported rather than about this one -- the tiles are masked, so a
    shorter ``L`` is the same arithmetic over fewer columns -- and this port
    serves it rather than repeating a limit it does not have. See
    ``test_sequences_below_upstreams_floor``.
    """
    dtype = torch.float16
    q, k, v, bias1, bias2 = _inputs((1, 4, 64, 2, 32), dtype)

    wide = torch.randn(1, 4, 64, 2, 128, dtype=dtype, device=flag_train.device)
    with pytest.raises(ValueError, match="head_size"):
        flag_train.deepspeed.evoformer_attn(wide, wide, wide)

    with pytest.raises(ValueError, match="same dtype"):
        flag_train.deepspeed.evoformer_attn(q, k, v, bias1.to(torch.float32), bias2)

    with pytest.raises(ValueError, match="bias1"):
        flag_train.deepspeed.evoformer_attn(q, k, v, bias1[:, :, 0], bias2)

    # A transposed L x L bias has the shape bias2 must have and is a different
    # matrix; only contiguity tells them apart.
    with pytest.raises(ValueError, match="contiguous"):
        flag_train.deepspeed.evoformer_attn(q, k, v, bias1, bias2.transpose(-1, -2))

    # Same shape, non-contiguous: a strided view of the same storage.
    strided = torch.randn(1, 4, 64, 4, 32, dtype=dtype, device=flag_train.device)
    with pytest.raises(ValueError, match="contiguous"):
        flag_train.deepspeed.evoformer_attn(
            strided[:, :, :, ::2, :], k, v, bias1, bias2
        )

    with pytest.raises(ValueError, match="at least 3-D"):
        flag_train.deepspeed.evoformer_attn(q[0, 0, 0], k[0, 0, 0], v[0, 0, 0])

    with pytest.raises(ValueError, match="device tensor"):
        flag_train.deepspeed.evoformer_attn(q.cpu(), k.cpu(), v.cpu())

    # A 3-D operand has no pair dimension for a bias to be indexed by.
    flat = q.reshape(-1, 64, 2, 32)
    with pytest.raises(ValueError, match="5-D"):
        flag_train.deepspeed.evoformer_attn(
            flat, flat, flat, bias1.reshape(-1, 64), bias2
        )


@pytest.mark.evoformer_attn
def test_unbiased_query_may_have_any_leading_shape():
    """With no bias there is nothing to index by, so 3-D and 4-D operands work.

    Upstream's ``_attention`` reshapes to ``[-1, L, H, D]`` and reads ``B`` and
    ``N`` off the first two dimensions only to advance the biases; with none, any
    leading shape is legal, and this port accepts the same ones.
    """
    dtype = torch.float16
    for shape in [(64, 2, 32), (3, 64, 2, 32), (1, 4, 64, 2, 32)]:
        q = torch.randn(shape, dtype=dtype, device=flag_train.device)
        out, lse = flag_train.deepspeed.evoformer_attn(q, q.clone(), q.clone())
        assert out.shape == shape
        assert lse.shape == (q.numel() // (64 * 2 * 32), 2, 64)
