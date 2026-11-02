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
"""Performance benchmark for Megatron weighted squared ReLU (forward).

The baseline is Megatron-LM's ``weighted_squared_relu_impl``, required on every
backend this operator ships on (``_MEGATRON_BASELINE_VENDORS``): like
``_load_deepspeed_blocked_flash``, the loader below raises when no scenario can
run against Megatron -- a missing import, or a plain weighted call that fails
here -- instead of timing against a PyTorch composition. megatron-core before
0.20.0 has no ``clamp_scale``, so on that interface the clamped rows are left
out and ``weighted_squared_relu`` still runs against Megatron. The test prints
the resolved baseline before timing.

The three variants (``weighted_squared_relu``, ``clamped_squared_relu`` and
``weighted_clamped_squared_relu``) share one ``op_name`` and are told apart by
a variant suffix on the shape key, with the variant passed as a fourth operand
for the baseline lookup. ``--level core`` runs the shapes below;
``--level comprehensive`` adds the extra shapes.
"""

import importlib.util
import inspect
import math
import re
import warnings
from importlib import metadata

import pytest
import torch

import flag_train

from .. import base

# Some vendor images ship a ``flash-attn-3`` wheel whose interface module is
# missing while its metadata is still registered; ``transformer_engine`` then
# takes its FlashAttention-3 branch and ``megatron.core`` fails to import.
# These benchmarks use no attention, so when the module is really missing we
# make ``importlib.metadata`` report the distribution as absent. The shim is
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

# Small, Megatron-MLP-sized, and a large MoE activation tensor (tokens, hidden).
_CORE_SHAPES = [
    (1024, 4096),
    (4096, 4096),
    (4, 2048, 4096),
    (8192, 11008),
]

_COMPREHENSIVE_SHAPES = [
    (1, 4096),
    (32768, 2048),
]

_CLAMP_SCALE = 5.0


def weighted_squared_relu_ref(x, weights=None, clamp_scale=None):
    # Flatten ``(..., hidden)`` to tokens like the kernels do, so the per-token
    # weights broadcast against the token dimension on 3D inputs too.
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


# The backends this operator ships on. Megatron-LM is the required baseline
# there; see ``_load_megatron_weighted_squared_relu``.
_MEGATRON_BASELINE_VENDORS = {
    "nvidia",
    "hygon",
    "iluvatar",
    "metax",
    "thead",
    "ascend",
}

# Set when the reference has to run eagerly on this backend (see
# ``_use_eager_megatron_reference``). This is what the baseline label below
# reports, and what makes the timed baseline the eager implementation.
_FORCE_EAGER_REFERENCE = False


def _use_eager_megatron_reference():
    """Time the eager Megatron reference when its jit fuser cannot compile.

    Inductor's pointwise kernel fails on some vendor backends (PPU inside
    ``ppu-llc``, MetaX after a CPU device graph), which would make the reference
    raise before it could be timed; NVIDIA keeps the compiled path.
    ``disable_jit_fuser`` cannot undo the decorator
    ``fused_weighted_squared_relu`` applies at import time, so
    ``torch.compiler.set_stance`` is what actually forces eager execution; the
    first switch stays for builds that resolve the fuser lazily.
    """
    global _FORCE_EAGER_REFERENCE
    _FORCE_EAGER_REFERENCE = hasattr(torch.compiler, "set_stance")
    try:
        from megatron.core.jit import disable_jit_fuser
    except ImportError:
        return
    disable_jit_fuser()


def _eager_reference(op):
    """Adapt ``op`` to run under the force-eager stance when it is needed."""
    if not _FORCE_EAGER_REFERENCE:
        return op

    def eager(*args, **kwargs):
        with torch.compiler.set_stance("force_eager"):
            return op(*args, **kwargs)

    return eager


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

    The clamped variants may be dropped, but a baseline that cannot do the
    weighted case describes no scenario at all, which is when this raises.
    """
    x = torch.ones((2, 8), dtype=torch.float32, device=flag_train.device)
    weights = torch.ones((2, 1), dtype=torch.float32, device=flag_train.device)

    def call():
        if _megatron_has_clamp_scale(op):
            return op(x, weights, None)
        return op(x, weights)

    try:
        # First call into the reference, so run it under the same stance as the
        # timed calls below.
        call = _eager_reference(call)
        call()
    except Exception as exc:
        raise RuntimeError(
            f"{flag_train.vendor_name!r} must use Megatron's "
            f"weighted_squared_relu_impl as its baseline, but it cannot run even "
            f"the plain weighted case on this backend: {exc!r}."
        ) from exc


def _load_megatron_weighted_squared_relu():
    """Megatron's ``weighted_squared_relu_impl``, required on the listed backends.

    Mirrors ``_load_deepspeed_blocked_flash``: a baseline that cannot run
    anything raises instead of falling back to the PyTorch composition.
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
            f"weighted_squared_relu_impl as its baseline, but it could not be "
            f"loaded: {exc!r}. Install Megatron-LM for this platform, or drop the "
            f"backend from _MEGATRON_BASELINE_VENDORS."
        ) from exc
    if flag_train.vendor_name != "nvidia":
        # ``_require_megatron_runs`` below is the first call into the reference,
        # so the fuser must be switched off before it runs.
        _use_eager_megatron_reference()
    _require_megatron_runs(weighted_squared_relu_impl)
    return weighted_squared_relu_impl


# Resolved once and printed by the test, like the ``_BASELINE`` tier in
# ``benchmark/deepspeed/test_blocked_flash.py``: the line says which reference
# the reported speedups were taken against.
_MEGATRON_WEIGHTED_SQUARED_RELU_IMPL = _load_megatron_weighted_squared_relu()

_MEGATRON_HAS_CLAMP_SCALE = (
    _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL is not None
    and _megatron_has_clamp_scale(_MEGATRON_WEIGHTED_SQUARED_RELU_IMPL)
)

if _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL is None:
    _BASELINE = "weighted_squared_relu_ref (torch)"
elif _MEGATRON_HAS_CLAMP_SCALE:
    _BASELINE = "megatron weighted_squared_relu_impl"
else:
    _BASELINE = (
        "megatron weighted_squared_relu_impl "
        "(clamped variants skipped: megatron-core has no clamp_scale)"
    )
if _FORCE_EAGER_REFERENCE:
    _BASELINE += " (eager; jit fuser disabled)"


def base_op():
    if _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL is None:
        # Only reachable on a backend outside _MEGATRON_BASELINE_VENDORS; the
        # platforms this operator ships on raise in the loader above instead.
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch weighted squared "
            "ReLU composition as the benchmark baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        return weighted_squared_relu_ref
    if _MEGATRON_HAS_CLAMP_SCALE:
        return _eager_reference(_MEGATRON_WEIGHTED_SQUARED_RELU_IMPL)

    warnings.warn(
        f"megatron-core {_megatron_core_version() or '(unknown version)'} has "
        f"no clamp_scale argument; the clamped variants are not benchmarked, "
        f"and the weighted variant keeps Megatron as its baseline",
        RuntimeWarning,
        stacklevel=2,
    )

    def megatron_without_clamp(x, weights=None, clamp_scale=None):
        # ``_VARIANTS`` drops the clamped rows on this interface, so the only
        # configuration that reaches here is the weighted one.
        assert clamp_scale is None
        return _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL(x, weights)

    return _eager_reference(megatron_without_clamp)


def _without_variant(op):
    """Adapt ``op(x, weights, clamp_scale)`` to the benchmark's 4-operand call."""

    def wrapped(x, weights=None, clamp_scale=None, variant=None):
        return op(x, weights, clamp_scale)

    return wrapped


def _dtypes():
    dtypes = [torch.float16, torch.float32]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


def _weights(shape, device):
    num_tokens = math.prod(shape[:-1])
    # Router probabilities: one fp32 scalar per token.
    return torch.rand((num_tokens, 1), dtype=torch.float32, device=device)


if _MEGATRON_WEIGHTED_SQUARED_RELU_IMPL is None or _MEGATRON_HAS_CLAMP_SCALE:
    # The torch fallback and a clamp-capable Megatron express all three.
    _VARIANTS = (
        "weighted_squared_relu",
        "clamped_squared_relu",
        "weighted_clamped_squared_relu",
    )
else:
    # This Megatron has no clamp_scale, so only the weighted variant has a
    # Megatron baseline. The clamped rows are left out of the table rather
    # than timed against torch.
    _VARIANTS = ("weighted_squared_relu",)


def _make_operands(shape, dtype, device, variant):
    """Build ``(x, weights, clamp_scale, variant)`` for one configuration."""
    if variant == "clamped_squared_relu":
        x = torch.randn(shape, dtype=dtype, device=device)
        return x, None, _CLAMP_SCALE, variant
    weights = _weights(shape, device)
    if variant == "weighted_squared_relu":
        x = torch.randn(shape, dtype=dtype, device=device)
        return x, weights, None, variant
    # The weighted_clamped variant scales the input by the clamp.
    x = torch.randn(shape, dtype=dtype, device=device) * _CLAMP_SCALE
    return x, weights, _CLAMP_SCALE, variant


def weighted_squared_relu_input_fn(shape, dtype, device):
    """Every variant for one shape, measured as rows of the same dtype table."""
    for variant in _VARIANTS:
        yield _make_operands(shape, dtype, device, variant)


class WeightedSquaredReluBenchmark(base.GenericBenchmark):
    """Shapes are fixed here: ``benchmark/core_shapes.yaml`` does not list this op.

    ``Benchmark.set_shapes`` opens that file, so an override is required for an
    operator that is not listed in one. This matches ``QuickGeluBenchmark``.
    """

    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)
        if base.Config.bench_level == base.consts.BenchLevel.COMPREHENSIVE:
            self.shapes += list(_COMPREHENSIVE_SHAPES)


@pytest.mark.weighted_squared_relu
def test_weighted_squared_relu_perf():
    # State the reference first, so a run whose numbers moved says whether the
    # baseline it moved against is the one the platform was supposed to pick.
    print(f"\nBaseline: {_BASELINE}")
    bench = WeightedSquaredReluBenchmark(
        input_fn=weighted_squared_relu_input_fn,
        op_name="weighted_squared_relu_impl",
        torch_op=_without_variant(base_op()),
        dtypes=_dtypes(),
    )
    bench.set_train(_without_variant(flag_train.megatron.weighted_squared_relu_impl))
    bench.run()
