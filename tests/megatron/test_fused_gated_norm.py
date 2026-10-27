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
"""Correctness tests for fused output RMSNorm/SiLU gating (forward only).

The oracle is Megatron-LM's ``fused_gated_norm`` on the backends in
``_MCORE_ORACLE_VENDORS`` when megatron-core provides it, and
``fused_gated_norm_ref`` -- a plain-torch composition of the same contract --
elsewhere and under ``--ref cpu``. ``test_mcore_fused_gated_norm_*`` checks that
oracle itself against the composition, so it is verified rather than trusted.

The reference mirrors the kernel's intermediate rounding: the normalised
activation is cast to ``x.dtype`` before the FP32 gating multiply.
"""

import warnings

import pytest
import torch

import flag_train
import flag_train.megatron

from .. import accuracy_utils as utils

_EPS = 1e-5

# FP16 tolerance: two ULPs. ``2**-9`` is two ulps relative at the bottom of an
# fp16 binade, the worst case; see ``_assert_close`` for why one ULP (the house
# table's ``1e-3``) is not enough for this operator.
_FP16_RTOL = 2**-9

# dim must be a power of two (the kernel addresses it with ``tl.arange(0, D)``).
# Besides ordinary GDN shapes, this covers one row (only the tail lane is valid),
# an odd row count (21 rows leaves a masked tail in the last program), and the
# widest head dimension the kernel supports.
_SHAPES = [
    (1, 1, 1, 64),
    (2, 8, 4, 64),
    (1, 7, 3, 128),
    (1, 32, 8, 128),
    (2, 16, 16, 256),
    (2, 4, 8, 512),
]

# The shape most of the focused tests below use.
_SHAPE = (2, 8, 4, 64)

# The sweep the performance suite reports latency for. Kept in sync with
# ``benchmark/megatron/test_fused_gated_norm.py`` so every benchmarked shape also
# has a correctness check.
_BENCHMARK_SHAPES = [
    (1, 1, 16, 128),
    (2, 1024, 16, 128),
    (2, 2048, 16, 128),
    (2, 4096, 16, 128),
    (2, 8192, 16, 128),
    (8, 1024, 16, 128),
    (16, 512, 16, 128),
    (2, 1024, 8, 64),
    (2, 1024, 32, 128),
    (2, 1024, 16, 256),
    (2, 512, 16, 512),
]

# FP32 exercises the exact-arithmetic path; BF16 only where the runtime has it.
_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    _DTYPES.append(torch.bfloat16)
_BENCHMARK_DTYPES = [dt for dt in _DTYPES if dt is not torch.float32]

# The variant tests cover both weight modes.
_ZERO_CENTERED = pytest.mark.parametrize(
    "zero_centered", [False, True], ids=["norm", "zero_centered"]
)


def fused_gated_norm_ref(x, gate, weight, eps, zero_centered=False):
    """Reference composed from plain torch ops, rounding included."""
    w = weight.to(torch.float32) + (1.0 if zero_centered else 0.0)
    xf = x.to(torch.float32)
    inv = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) / x.shape[-1] + eps)
    norm = (xf * inv * w).to(x.dtype).to(torch.float32)
    gate_f = gate.to(torch.float32)
    return (norm * (gate_f * torch.sigmoid(gate_f))).to(x.dtype)


# The upstream kernel is CUDA Triton, so only NVIDIA can execute it.
_MCORE_ORACLE_VENDORS = {"nvidia"}


def _load_mcore_fused_gated_norm():
    """Megatron-LM's fused_gated_norm, or ``None`` when it is not the oracle."""
    if flag_train.vendor_name not in _MCORE_ORACLE_VENDORS:
        return None
    try:
        from megatron.core.fusions.fused_gated_norm import fused_gated_norm

        return fused_gated_norm
    except Exception as exc:
        warnings.warn(
            f"{flag_train.vendor_name!r} can use Megatron-LM's fused_gated_norm "
            f"as its oracle when megatron-core provides it, but the import "
            f"failed: {exc!r}. Falling back to fused_gated_norm_ref.",
            stacklevel=2,
        )
        return None


_mcore_fused_gated_norm = _load_mcore_fused_gated_norm()
needs_mcore_oracle = pytest.mark.skipif(
    _mcore_fused_gated_norm is None,
    reason="megatron-core's fused_gated_norm is not available (NVIDIA-only oracle)",
)


def _oracle(x, gate, weight, eps, zero_centered=False):
    """The implementation the port is checked against; see the module docstring.

    The upstream kernel cannot run on a CPU, so ``--ref cpu`` keeps the torch
    composition as the oracle even where megatron-core is available.
    """
    if _mcore_fused_gated_norm is not None and not utils.TO_CPU:
        return _mcore_fused_gated_norm(x, gate, weight, eps, zero_centered)
    return fused_gated_norm_ref(x, gate, weight, eps, zero_centered)


def _operands(shape, dtype, gate_dtype=None, weight_dtype=None):
    """Random ``(x, gate, weight)`` for one case."""
    device = flag_train.device
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn(shape, dtype=gate_dtype or dtype, device=device),
        torch.randn((shape[-1],), dtype=weight_dtype or dtype, device=device),
    )


def _assert_close(out, expected, dtype):
    """Compare at the house tolerances, with FP16 widened to its two ULPs.

    FP16 is the tight one: the contract rounds the normalised activation into
    the activation dtype before the gating multiply and casts the product
    again, so two correct implementations that sum the power sum in a
    different order can land two ULPs apart -- one from the intermediate norm,
    one from the final cast. The house ``RESOLUTION`` table gives FP16
    ``rtol=1e-3``, one ULP, which only holds when both sides round identically;
    Megatron-LM's kernel against the torch composition already exceeds it.
    Measured over every shape below, our kernel, Megatron-LM's and the
    composition stay within exactly two ULPs, so that is the band here. The
    number belongs at this call site rather than in the house helper, which
    other operators share. Other dtypes keep the house tolerances through
    ``utils.train_assert_close``.
    """
    res = utils.to_reference(out)
    ref = utils.to_reference(expected)
    if dtype is not torch.float16:
        utils.train_assert_close(res, ref, dtype)
        return
    assert res.dtype == dtype
    torch.testing.assert_close(res, ref.to(dtype), rtol=_FP16_RTOL, atol=1e-4)


def _check(x, gate, weight, dtype, zero_centered=False, impl=None, ref=None):
    """Run the operator (``impl``) against ``ref`` and return its result."""
    out = (impl or flag_train.megatron.fused_gated_norm)(
        x, gate, weight, _EPS, zero_centered
    )
    expected = (ref or _oracle)(
        utils.to_reference(x),
        utils.to_reference(gate),
        utils.to_reference(weight),
        _EPS,
        zero_centered,
    )
    _assert_close(out, expected, dtype)
    return out


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@_ZERO_CENTERED
def test_fused_gated_norm_forward(shape, dtype, zero_centered):
    """Forward must match the oracle, including the rounding."""
    _check(*_operands(shape, dtype), dtype, zero_centered)


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("shape", _BENCHMARK_SHAPES)
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
def test_fused_gated_norm_benchmark_shape(shape, dtype):
    """Every shape whose latency the benchmark reports must be correct."""
    _check(*_operands(shape, dtype), dtype)


@pytest.mark.fused_gated_norm
@needs_mcore_oracle
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@_ZERO_CENTERED
def test_mcore_fused_gated_norm_forward(shape, dtype, zero_centered):
    """The upstream oracle must agree with the torch composition."""
    _check(
        *_operands(shape, dtype),
        dtype,
        zero_centered,
        impl=_mcore_fused_gated_norm,
        ref=fused_gated_norm_ref,
    )


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
@pytest.mark.parametrize("shape", [_SHAPE, (1, 7, 3, 128)])
@_ZERO_CENTERED
def test_fused_gated_norm_fp32_gate(shape, dtype, zero_centered):
    """The gate and weight may be FP32 under a lower-precision activation."""
    _check(*_operands(shape, dtype, torch.float32, torch.float32), dtype, zero_centered)


# layout -> (x contiguous?, gate contiguous?)
_LAYOUTS = {
    "transposed-both": (False, False),
    "transposed-x": (False, True),
    "transposed-gate": (True, False),
    "strided-rows": (False, False),
    "strided-last": (False, False),
}


def _strided_operands(layout, shape, dtype):
    """Build ``(x, gate, weight)`` with the requested layout."""
    batch, sequence, heads, dim = shape
    device = flag_train.device
    weight = torch.randn((dim,), dtype=dtype, device=device)

    if layout == "strided-last":
        # ``dim`` is the inner axis of the base allocation, so the last logical
        # dimension ends up with stride ``heads`` instead of 1.
        base = (batch, sequence, dim, heads)
        x, gate = (
            torch.randn(base, dtype=dtype, device=device).permute(0, 1, 3, 2)
            for _ in range(2)
        )
        return x, gate, weight

    if layout == "strided-rows":
        # A doubly-strided, offset slice: neither the row start nor the row-to-row
        # distance is the contiguous one.
        base = (2 * batch, 2 * sequence, heads, dim)
        x, gate = (
            torch.randn(base, dtype=dtype, device=device)[1::2, 1::2] for _ in range(2)
        )
        return x, gate, weight

    base = (batch, heads, sequence, dim)
    x, gate = (
        torch.randn(base, dtype=dtype, device=device).permute(0, 2, 1, 3)
        for _ in range(2)
    )
    if layout == "transposed-x":
        gate = gate.contiguous()
    elif layout == "transposed-gate":
        x = x.contiguous()
    return x, gate, weight


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_gated_norm_strided(layout, dtype):
    """Strided views must be read in place and produce the contiguous result."""
    x, gate, weight = _strided_operands(layout, _SHAPE, dtype)
    assert (x.is_contiguous(), gate.is_contiguous()) == _LAYOUTS[layout]
    assert _check(x, gate, weight, dtype).is_contiguous()


@pytest.mark.fused_gated_norm
@needs_mcore_oracle
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_mcore_fused_gated_norm_strided(layout):
    """The upstream oracle must also agree on strided views."""
    _check(
        *_strided_operands(layout, _SHAPE, torch.float32),
        torch.float32,
        impl=_mcore_fused_gated_norm,
        ref=fused_gated_norm_ref,
    )


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_gated_norm_zero_activation(dtype):
    """An all-zero activation must produce zeros, not ``0 * rsqrt(0)`` NaNs."""
    x, gate, weight = _operands(_SHAPE, dtype)
    out = _check(torch.zeros_like(x), gate, weight, dtype)
    assert out.count_nonzero().item() == 0


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_gated_norm_saturated_gate(dtype):
    """A gate far from zero drives SiLU into saturation; it must stay finite."""
    x, gate, weight = _operands(_SHAPE, dtype)
    out = _check(x * 10, gate * 50, weight, dtype)
    assert torch.isfinite(out).all().item()


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_gated_norm_output_contract(dtype):
    """The result keeps ``x``'s shape/dtype, is contiguous, and inputs are intact."""
    x, gate, weight = _operands(_SHAPE, dtype)
    before = x.clone(), gate.clone(), weight.clone()

    out = flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS)

    assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()
    for now, was in zip((x, gate, weight), before):
        # Under ``--ref cpu`` every comparison is on the CPU, and
        # ``train_assert_equal`` expects that of its operands; ``to_reference``
        # moves them there and is a no-op otherwise.
        utils.train_assert_equal(utils.to_reference(now), utils.to_reference(was))


@pytest.mark.fused_gated_norm
@pytest.mark.parametrize(
    ("bad_input", "match"),
    [
        ("x-ndim", "4-D shape"),
        ("gate-shape", "4-D shape"),
        ("head-dim", "power-of-two"),
        ("weight-shape", r"contiguous \(dim,\)"),
        ("weight-stride", r"contiguous \(dim,\)"),
    ],
)
def test_fused_gated_norm_rejects_invalid_input(bad_input, match):
    """Unsupported layouts fail with a diagnostic instead of reading out of bounds."""
    x, gate, weight = _operands(_SHAPE, torch.float32)
    if bad_input == "x-ndim":
        x, gate = x.reshape(-1, 4, 64), gate.reshape(-1, 4, 64)
    elif bad_input == "gate-shape":
        gate = gate[..., :32]
    elif bad_input == "head-dim":
        x, gate, weight = _operands((2, 8, 4, 96), torch.float32)
    elif bad_input == "weight-shape":
        weight = weight[:32]
    else:
        weight = _operands(_SHAPE, torch.float32)[2].repeat(2)[::2]

    with pytest.raises(ValueError, match=match):
        flag_train.megatron.fused_gated_norm(x, gate, weight, _EPS)
