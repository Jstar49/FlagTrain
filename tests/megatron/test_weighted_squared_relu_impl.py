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
"""Correctness tests for Megatron weighted squared ReLU (forward).

The reference is Megatron-LM's ``weighted_squared_relu_impl``, required on every
backend this operator ships on (``_MEGATRON_BASELINE_VENDORS``): the loader
raises when Megatron -- or the plain weighted call itself -- cannot run at all.
megatron-core before 0.20.0 has no ``clamp_scale``, so the clamped cases skip
there while the weighted ones still run against Megatron. One test also pins the
PyTorch composition the benchmark falls back to against Megatron's.

Comparisons run on ``utils.to_reference(..., upcast=True)`` operands (fp32 on
device, fp64 on CPU under ``--ref cpu``) with ``train_assert_close`` defaults.
"""

import importlib.util
import inspect
import math
import re
from importlib import metadata

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# Some vendor images ship a ``flash-attn-3`` wheel whose interface module is
# missing while its metadata is still registered; ``transformer_engine`` then
# takes its FlashAttention-3 branch and ``megatron.core`` fails to import.
# These tests use no attention, so when the module is really missing we make
# ``importlib.metadata`` report the distribution as absent. The shim is
# process-local, only rewrites that one distribution, and is a no-op on images
# where ``flash_attn_3.flash_attn_interface`` exists.
_FLASH_ATTN_3_DISTRIBUTIONS = {"flash-attn-3", "flash_attn_3"}


def _flash_attn_3_interface_missing():
    try:
        return importlib.util.find_spec("flash_attn_3.flash_attn_interface") is None
    except (ImportError, ValueError):
        return True


def _hide_broken_flash_attn_3():
    if not _flash_attn_3_interface_missing():
        return

    def normalize(name):
        return str(name).lower().replace("_", "-")

    original_version = metadata.version
    original_distribution = metadata.distribution
    original_metadata = metadata.metadata

    def version(name):
        if normalize(name) in _FLASH_ATTN_3_DISTRIBUTIONS:
            raise metadata.PackageNotFoundError(name)
        return original_version(name)

    def distribution(name):
        if normalize(name) in _FLASH_ATTN_3_DISTRIBUTIONS:
            raise metadata.PackageNotFoundError(name)
        return original_distribution(name)

    def metadata_lookup(name):
        if normalize(name) in _FLASH_ATTN_3_DISTRIBUTIONS:
            raise metadata.PackageNotFoundError(name)
        return original_metadata(name)

    metadata.version = version
    metadata.distribution = distribution
    metadata.metadata = metadata_lookup


# Must run before the Megatron loader below imports megatron.core.
_hide_broken_flash_attn_3()

# Both ranks of the interface, a hidden size that is not a multiple of the
# kernel block, and empty tensors.
_SHAPES = [
    (1, 1),
    (17, 1000),
    (16, 64),
    (4, 8, 64),
    (1024, 4096),
    (0, 8),
    (2, 0, 8),
]

_CLAMP_SCALE = 5.0

# The backends this operator ships on. Megatron-LM is the required functional
# baseline there; see ``_load_megatron_weighted_squared_relu``.
_MEGATRON_BASELINE_VENDORS = {
    "nvidia",
    "hygon",
    "iluvatar",
    "metax",
    "thead",
    "ascend",
}


# Set by ``_use_eager_megatron_reference``; see there.
_FORCE_EAGER_REFERENCE = False


def _use_eager_megatron_reference():
    """Run the reference eagerly when Megatron's jit fuser cannot compile.

    Inductor's pointwise kernel fails on some vendor backends (PPU inside
    ``ppu-llc``, MetaX after a CPU device graph), which would break every
    comparison below; NVIDIA keeps the compiled path. ``disable_jit_fuser``
    cannot undo the decorator ``fused_weighted_squared_relu`` applies at import
    time, so ``torch.compiler.set_stance`` is what actually forces eager
    execution; the first switch stays for builds that resolve the fuser lazily.
    """
    global _FORCE_EAGER_REFERENCE
    _FORCE_EAGER_REFERENCE = hasattr(torch.compiler, "set_stance")
    try:
        from megatron.core.jit import disable_jit_fuser
    except ImportError:
        return
    disable_jit_fuser()


def _megatron_core_version():
    """Return the installed megatron-core version, or ``""`` when unavailable."""
    try:
        return metadata.version("megatron-core")
    except metadata.PackageNotFoundError:
        return ""


def _megatron_has_clamp_scale(op):
    """Whether Megatron's reference accepts a third ``clamp_scale`` argument.

    megatron-core 0.20.0 added it; otherwise the signature is consulted, so a
    backported argument without a version bump is still recognised.
    """
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", _megatron_core_version())
    if match is not None:
        version = tuple(int(part or 0) for part in match.groups())
        if version >= (0, 20, 0):
            return True
    try:
        return "clamp_scale" in inspect.signature(op).parameters
    except (TypeError, ValueError):
        return False


def _require_megatron_runs(op):
    """Prove the baseline can run the plain weighted case on this backend.

    The clamped cases may skip, but a baseline that cannot do the weighted case
    describes no scenario at all, which is when this suite must raise.
    """
    x = torch.ones((2, 8), dtype=torch.float32, device=flag_train.device)
    weights = torch.ones((2, 1), dtype=torch.float32, device=flag_train.device)

    def call():
        if _megatron_has_clamp_scale(op):
            return op(x, weights, None)
        return op(x, weights)

    try:
        # First call into the reference, so run it under the same stance as the
        # comparisons below.
        if _FORCE_EAGER_REFERENCE:
            with torch.compiler.set_stance("force_eager"):
                call()
        else:
            call()
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use Megatron's "
            f"weighted_squared_relu_impl as its functional baseline, but it "
            f"cannot run even the plain weighted case on this backend: {exc!r}."
        ) from exc


def _load_megatron_weighted_squared_relu():
    """Megatron's reference, required on the backends this operator ships on.

    Mirrors ``_load_deepspeed_blocked_flash``: a baseline that cannot run
    anything raises rather than falling back to a torch composition.
    """
    if flag_train.vendor_name not in _MEGATRON_BASELINE_VENDORS:
        return None
    try:
        from megatron.core.fusions.fused_weighted_squared_relu import (
            weighted_squared_relu_impl,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use Megatron's "
            f"weighted_squared_relu_impl as its functional baseline, but it could "
            f"not be loaded: {exc!r}. Install Megatron-LM for this platform, or "
            f"drop the backend from _MEGATRON_BASELINE_VENDORS."
        ) from exc
    if flag_train.vendor_name != "nvidia":
        # ``_require_megatron_runs`` below is the first call into the reference,
        # so the fuser must be switched off before it runs.
        _use_eager_megatron_reference()
    _require_megatron_runs(weighted_squared_relu_impl)
    return weighted_squared_relu_impl


# Resolved once, at module import time.
megatron_weighted_squared_relu_impl = _load_megatron_weighted_squared_relu()

_MEGATRON_HAS_CLAMP_SCALE = (
    megatron_weighted_squared_relu_impl is not None
    and _megatron_has_clamp_scale(megatron_weighted_squared_relu_impl)
)


def _megatron_weighted_squared_relu(x, weights, clamp_scale):
    """Megatron's baseline for one case; clamped cases skip without ``clamp_scale``."""
    if megatron_weighted_squared_relu_impl is None:
        raise AssertionError(
            f"{flag_train.vendor_name!r} has no Megatron baseline loaded; this "
            f"suite compares against Megatron only."
        )
    if clamp_scale is not None and not _MEGATRON_HAS_CLAMP_SCALE:
        pytest.skip(
            f"megatron-core {_megatron_core_version() or '(unknown version)'} "
            f"exposes weighted_squared_relu_impl(input, weights) with no "
            f"clamp_scale argument, so the clamped scenario cannot be compared "
            f"against Megatron. Install megatron-core >= 0.20.0 (or a "
            f"Megatron-LM build whose weighted_squared_relu_impl accepts "
            f"clamp_scale) to cover it."
        )
    if _FORCE_EAGER_REFERENCE:
        with torch.compiler.set_stance("force_eager"):
            return _call_megatron_reference(x, weights, clamp_scale)
    return _call_megatron_reference(x, weights, clamp_scale)


def _call_megatron_reference(x, weights, clamp_scale):
    """Invoke Megatron's implementation through whichever interface it has."""
    if _MEGATRON_HAS_CLAMP_SCALE:
        return megatron_weighted_squared_relu_impl(x, weights, clamp_scale)
    return megatron_weighted_squared_relu_impl(x, weights)


def _weights_for(shape, dtype, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_tokens = math.prod(shape[:-1])
    # Router probabilities are non-negative and produced in fp32. Drawing on the
    # CPU keeps the values identical on every backend.
    return torch.rand(
        (num_tokens, 1), dtype=dtype, device="cpu", generator=generator
    ).to(flag_train.device)


def _check(x, weights, clamp_scale):
    y = flag_train.megatron.weighted_squared_relu_impl(x, weights, clamp_scale)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

    x_ref = utils.to_reference(x.detach(), upcast=True)
    weights_ref = (
        None if weights is None else utils.to_reference(weights.detach(), upcast=True)
    )
    # Megatron flattens with ``view``, which needs contiguous input; the values
    # are unchanged, so copy instead of dropping the non-contiguous case.
    if not x_ref.is_contiguous():
        x_ref = x_ref.contiguous()
    # Megatron's 3-D epilogue reshape to ``(B, S, -1)`` fails for 0 tokens; use
    # its 2-D entry point there.
    baseline_x = x_ref
    if x_ref.dim() == 3 and x_ref.numel() == 0:
        baseline_x = x_ref.reshape(-1, x_ref.shape[-1])
    expected = _megatron_weighted_squared_relu(baseline_x, weights_ref, clamp_scale)
    if expected.shape != x_ref.shape:
        expected = expected.reshape(x_ref.shape)
    utils.train_assert_close(utils.to_reference(y), expected, x.dtype)
    return y


@pytest.mark.weighted_squared_relu
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_squared_relu(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    weights = _weights_for(shape, torch.float32, seed=1)
    _check(x, weights, None)


@pytest.mark.weighted_squared_relu
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamped_squared_relu(shape, dtype):
    utils.init_seed(2)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check(x, None, _CLAMP_SCALE)


@pytest.mark.weighted_squared_relu
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_clamped_squared_relu(shape, dtype):
    utils.init_seed(3)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device) * 5.0
    weights = _weights_for(shape, torch.float32, seed=4)
    _check(x, weights, _CLAMP_SCALE)


@pytest.mark.weighted_squared_relu
@pytest.mark.parametrize("weights_dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_squared_relu_weight_dtype(dtype, weights_dtype):
    """Per-token weights may be fp32 while the activation is bf16/fp16."""
    utils.init_seed(5)
    shape = (8, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    weights = _weights_for(shape, weights_dtype, seed=6)
    _check(x, weights, None)


@pytest.mark.weighted_squared_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_squared_relu_values(dtype):
    """Negative, zero, tiny, and clamp-saturating inputs with varied weights."""
    values = [
        -50.0,
        -8.0,
        -1.0,
        -1e-6,
        0.0,
        1e-6,
        1e-3,
        0.5,
        1.0,
        5.0,
        8.0,
        50.0,
    ]
    x = torch.tensor(values, dtype=dtype, device=flag_train.device)
    x = x.repeat(4, 1)
    weights = torch.tensor(
        [[0.0], [0.5], [1.0], [2.0]], dtype=torch.float32, device=flag_train.device
    )
    _check(x, weights, None)
    _check(x, weights, _CLAMP_SCALE)


@pytest.mark.weighted_squared_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_squared_relu_noncontiguous(dtype):
    utils.init_seed(7)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    assert not x.is_contiguous()
    weight_base = torch.rand((64, 2), dtype=torch.float32, device=flag_train.device)
    weights = weight_base[:, :1]
    assert not weights.is_contiguous()
    _check(x, weights, None)
    _check(x, weights, _CLAMP_SCALE)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)
    assert not transposed.is_contiguous()
    transposed_weights = _weights_for(transposed.shape, torch.float32, seed=8)
    _check(transposed, transposed_weights, _CLAMP_SCALE)


@pytest.mark.weighted_squared_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_squared_relu_empty(dtype):
    for shape in ((0, 8), (2, 0, 8)):
        x = torch.empty(shape, dtype=dtype, device=flag_train.device)
        weights = torch.empty(
            (math.prod(shape[:-1]), 1), dtype=torch.float32, device=flag_train.device
        )
        y = _check(x, weights, None)
        assert y.numel() == 0
        _check(x, None, _CLAMP_SCALE)


@pytest.mark.weighted_squared_relu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weighted_squared_relu_does_not_modify_inputs(dtype):
    utils.init_seed(9)
    x = torch.randn((16, 64), dtype=dtype, device=flag_train.device)
    weights = _weights_for(x.shape, torch.float32, seed=10)
    x_before = x.clone()
    weights_before = weights.clone()
    flag_train.megatron.weighted_squared_relu_impl(x, weights, _CLAMP_SCALE)
    assert torch.equal(x, x_before)
    assert torch.equal(weights, weights_before)


@pytest.mark.weighted_squared_relu
def test_weighted_squared_relu_rejects_no_weights_and_no_clamp():
    x = torch.randn((16, 64), dtype=torch.bfloat16, device=flag_train.device)
    with pytest.raises(AssertionError):
        flag_train.megatron.weighted_squared_relu_impl(x, None, None)


@pytest.mark.weighted_squared_relu
def test_weighted_squared_relu_rejects_bad_weights_shape():
    x = torch.randn((16, 64), dtype=torch.float32, device=flag_train.device)
    for shape in ((16,), (16, 2), (1, 1)):
        weights = torch.rand(shape, dtype=torch.float32, device=flag_train.device)
        with pytest.raises(ValueError, match="shape"):
            flag_train.megatron.weighted_squared_relu_impl(x, weights)


@pytest.mark.weighted_squared_relu
def test_weighted_squared_relu_rejects_bad_input():
    weights = torch.rand((16, 1), dtype=torch.float32, device=flag_train.device)
    with pytest.raises(ValueError, match="tensor"):
        flag_train.megatron.weighted_squared_relu_impl(
            torch.zeros((16, 64), dtype=torch.int32, device=flag_train.device), weights
        )
    with pytest.raises(AssertionError):
        flag_train.megatron.weighted_squared_relu_impl(
            torch.randn((16,), dtype=torch.float32, device=flag_train.device),
            torch.rand((16, 1), dtype=torch.float32, device=flag_train.device),
        )


@pytest.mark.weighted_squared_relu
@pytest.mark.skipif(
    megatron_weighted_squared_relu_impl is None,
    reason="Megatron-LM is not installed",
)
@pytest.mark.parametrize("shape", [(16, 64), (4, 8, 64)], ids=["2d", "3d"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "clamp_scale",
    [None, _CLAMP_SCALE],
    ids=["plain", "clamped"],
)
def test_weighted_squared_relu_matches_megatron(shape, dtype, clamp_scale):
    utils.init_seed(11)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    if clamp_scale is not None:
        x = x * _CLAMP_SCALE
    weights = _weights_for(shape, torch.float32, seed=12)

    y = flag_train.megatron.weighted_squared_relu_impl(x, weights, clamp_scale)
    expected = _megatron_weighted_squared_relu(x, weights, clamp_scale)

    # ``to_reference`` also moves the reference to the CPU under ``--ref cpu``.
    utils.train_assert_close(y, utils.to_reference(expected), dtype)


def weighted_squared_relu_ref(x, weights=None, clamp_scale=None):
    """The PyTorch composition the benchmark falls back to.

    Copied verbatim from ``benchmark/megatron/test_weighted_squared_relu_impl.py``;
    keep the two in step. The test below pins it to Megatron's implementation.
    """
    ori_shape = x.shape
    flat = x.reshape(-1, ori_shape[-1])
    if clamp_scale is not None:
        t = torch.tanh(torch.relu(flat.float()) / clamp_scale)
        act = torch.pow(clamp_scale * t, 2)
    else:
        act = torch.pow(torch.relu(flat), 2)
    if weights is not None:
        act = act * weights
    return act.to(x.dtype).reshape(ori_shape)


# Two of the benchmark's core shapes, one 2-D and one 3-D. The comparison is
# elementwise, so its outcome does not depend on the size; the large MoE shape
# stays in the benchmark.
_COMPOSITION_SHAPES = [
    (1024, 4096),
    (4, 2048, 4096),
]

# The benchmark's three variants: per-token weights, the clamp alone, and both.
_COMPOSITION_VARIANTS = [
    pytest.param(True, None, id="weighted"),
    pytest.param(False, _CLAMP_SCALE, id="clamped"),
    pytest.param(True, _CLAMP_SCALE, id="weighted_clamped"),
]


@pytest.mark.weighted_squared_relu
@pytest.mark.skipif(
    megatron_weighted_squared_relu_impl is None,
    reason="the torch composition is only cross-checked where Megatron runs",
)
@pytest.mark.parametrize("shape", _COMPOSITION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("with_weights, clamp_scale", _COMPOSITION_VARIANTS)
def test_torch_composition_matches_megatron(shape, dtype, with_weights, clamp_scale):
    """The benchmark's torch fallback must reproduce Megatron's baseline.

    Cases Megatron's interface cannot express skip, like every other comparison.
    """
    utils.init_seed(13)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    if clamp_scale is not None:
        x = x * _CLAMP_SCALE
    weights = _weights_for(shape, torch.float32, seed=14) if with_weights else None

    expected = _megatron_weighted_squared_relu(x, weights, clamp_scale)
    actual = weighted_squared_relu_ref(x, weights, clamp_scale)
    utils.train_assert_close(actual, utils.to_reference(expected), dtype)
