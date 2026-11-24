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
"""Correctness tests for fused bias + SwiGLU (forward only).

The reference is Megatron-LM's own ``bias_swiglu_impl`` once megatron-core
ships the clamp / SiTU arguments (0.20), so the kernel is checked against the
implementation it ports. Older installs, and the ``--ref cpu`` run, compare
against ``bias_swiglu_ref``: the eager formula Megatron-LM checks its own
fusion against, SiLU (or SiTU) of the two halves with the bias added by
ordinary broadcasting. That formula covers every variant and is always
available, so the suite still runs without Megatron-LM.

Under ``--ref cpu`` the operands are moved to the CPU and upcast, so the
comparison is against a higher-precision CPU result. The Megatron reference
also sees upcast operands: its ``jit_fuser`` is a no-op outside CUDA, so adding
the bias in the input dtype would otherwise make the reference the inaccurate
side. Tolerances are ``train_assert_close``'s defaults.

Only the forward output is checked: the backward is a separate operator. The
``fp8_input_store`` and ``cpu_offload_input`` flags stay for Megatron-LM
signature compatibility and are ignored by the forward. The Triton kernel now
covers fp64 and mixed bias dtypes as well; a CPU tensor is copied to the
active accelerator, computed, and copied back. The operator is compared against the
same Megatron entry point when available, and against the eager reference
under ``--ref cpu``.
"""

import importlib.metadata
import re

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# megatron-core 0.20 is the first release with the clamp / SiTU arguments.
_MCORE_MIN_VERSION = (0, 20)


def _mcore_version():
    try:
        return importlib.metadata.version("megatron-core")
    except Exception:
        return "not installed"


_MCORE_VERSION = _mcore_version()


def _version_tuple(version):
    """``(major, minor)`` of a distribution version, or ``None`` without one."""
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", version)
    return None if match is None else (int(match.group(1)), int(match.group(2) or 0))


def _load_megatron_impl():
    """Installed ``bias_swiglu_impl`` when it has the clamp/SiTU arguments.

    Returns ``None`` below 0.20, where the eager formula is the reference.
    A version new enough to have the arguments but unable to import them is a
    broken install and raises instead of silently skipping Megatron.
    """
    version = _version_tuple(_MCORE_VERSION)
    if version is None or version < _MCORE_MIN_VERSION:
        return None
    try:
        from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
    except Exception as exc:
        raise RuntimeError(
            f"megatron-core {_MCORE_VERSION} should provide bias_swiglu_impl "
            f"with the clamp/SiTU arguments, but importing it failed: {exc!r}"
        ) from exc
    return bias_swiglu_impl


_megatron_impl = _load_megatron_impl()


def _keep_compile_in_process():
    """Compile the Megatron baseline without Inductor's worker processes.

    Inductor's async compile ships the compiled Triton kernel back from a
    worker process, so it has to pickle it; vendor Triton builds whose
    JITFunction carries an RLock cannot be pickled. One compile thread runs the
    task inline. This changes compile time only, not the reference values.
    """
    try:
        from torch._inductor import config as inductor_config
    except ImportError:
        return
    if getattr(inductor_config, "compile_threads", 1) != 1:
        inductor_config.compile_threads = 1


_keep_compile_in_process()

# Empty, unaligned (half = 3 and 17), and MLP-like ranks. The last dim is even.
_SHAPES = [
    (0, 8),
    (1, 2),
    (8, 6),
    (5, 34),
    (17, 32),
    (64, 128),
    (2, 8, 64),
    (4, 7, 32),
]

# Bias is orthogonal to the shapes and variants below, so it is sampled on a
# few representative cases only.
_BIAS_SHAPES = [
    (1, 2),
    (17, 32),
    (64, 128),
    (4, 7, 32),
]

# name -> (input std, operator kwargs). The std puts a meaningful share of a
# normal sample across the variant's threshold:
#   clamped 1.5 / 3.0 = 0.5 sigma  -> ~62% clamped
#   situ gate 4 / 5.0 = 0.8 sigma  -> ~42% in the tanh saturation region
#        linear 2.5 / 5.0 = 0.5 sigma -> ~62% soft-clamped
#   plain keeps the standard normal.
_VARIANTS = {
    "plain": (1.0, {}),
    "clamped": (3.0, {"clamp_value": 1.5}),
    "situ": (5.0, {"gate_clamp_scale": 4.0}),
    "situ_linear": (5.0, {"gate_clamp_scale": 4.0, "linear_clamp_scale": 2.5}),
}


def bias_swiglu_ref(
    x,
    bias=None,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    y = x if bias is None else x + bias
    dtype = y.dtype
    gate, linear = torch.chunk(y, 2, dim=-1)
    if gate_clamp_scale is not None:
        gate = gate_clamp_scale * torch.tanh(gate.float() / gate_clamp_scale)
        gate = gate * torch.sigmoid(torch.chunk(y, 2, dim=-1)[0])
        if linear_clamp_scale is not None:
            linear = linear_clamp_scale * torch.tanh(
                linear.float() / linear_clamp_scale
            )
        return (gate * linear).to(dtype)
    if clamp_value is not None and clamp_value > 0:
        gate, linear = torch.chunk(y.float(), 2, dim=-1)
        gate = gate.clamp(max=clamp_value)
        linear = linear.clamp(min=-clamp_value, max=clamp_value)
        return (torch.nn.functional.silu(gate) * linear).to(dtype)
    return torch.nn.functional.silu(gate) * linear


def _reference(tensor):
    if tensor is None:
        return None
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype, equal_nan=False, atol=1e-4):
    utils.train_assert_close(
        utils.to_reference(actual),
        expected,
        dtype,
        equal_nan=equal_nan,
        atol=atol,
    )


def _expected_output(x, bias, **kwargs):
    """Megatron-LM at 0.20+, the eager formula otherwise or under ``--ref cpu``.

    The Megatron side gets contiguous operands: its ``input.view(-1, hidden)``
    rejects the transposed inputs this suite deliberately feeds the operator.
    """
    if _megatron_impl is not None and not utils.TO_CPU:
        return _megatron_impl(
            x.contiguous(),
            None if bias is None else bias.contiguous(),
            **kwargs,
        )
    return bias_swiglu_ref(x, bias, **kwargs)


def _check_bias_swiglu(x, bias, atol=None, **kwargs):
    dtype = x.dtype
    y = flag_train.megatron.bias_swiglu_impl(x, bias, **kwargs)
    assert y.shape[-1] * 2 == x.shape[-1]
    assert y.shape[:-1] == x.shape[:-1]
    assert y.dtype == dtype

    if atol is None:
        atol = 2e-3 if dtype == torch.float16 else 1e-4
    if _megatron_impl is not None and not utils.TO_CPU:
        # Compare the port to the same-dtype upstream result: the high-precision
        # reference would charge the extra half-precision rounding of both
        # implementations to the tolerance.
        expected = _megatron_impl(
            x.contiguous(), None if bias is None else bias.contiguous(), **kwargs
        )
    else:
        expected = _expected_output(_reference(x), _reference(bias), **kwargs)
    _assert_close(y, expected, dtype, atol=atol)


def _check_noncontiguous(x, bias, **kwargs):
    """Compare a view-compatible non-contiguous input to the same oracle.

    Megatron accepts this layout, so use its own dtype-preserving result when
    available; the upcast reference is used only under ``--ref cpu``.
    """
    y = flag_train.megatron.bias_swiglu_impl(x, bias, **kwargs)
    assert y.dtype == x.dtype
    if _megatron_impl is not None and not utils.TO_CPU:
        expected = _megatron_impl(x, bias, **kwargs)
        # One fp16 ulp of Inductor's fused order is still the same contract.
        _assert_close(y, expected, x.dtype, atol=2e-3)
    else:
        _check_bias_swiglu(x, bias, atol=2e-3, **kwargs)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, None)


_EMPTY_3D_SHAPES = [(0, 4, 16), (2, 0, 16)]


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("shape", _EMPTY_3D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_3d_empty_raises(shape, dtype):
    """Match Megatron-LM, whose final ``view(..., -1)`` is ambiguous for 0 elements."""
    x = torch.empty(shape, dtype=dtype, device=flag_train.device)
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_swiglu_impl(x, None)

    if _megatron_impl is not None:
        with pytest.raises(RuntimeError):
            _megatron_impl(x, None)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("shape", _BIAS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_bias(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    bias = torch.randn(shape[-1], dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, bias)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_variants(variant, dtype):
    utils.init_seed(1)
    scale, kwargs = _VARIANTS[variant]
    x = torch.randn((16, 64), dtype=dtype, device=flag_train.device) * scale
    _check_bias_swiglu(x, None, **kwargs)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_bias_swiglu_impl_variants_bias(variant):
    utils.init_seed(1)
    scale, kwargs = _VARIANTS[variant]
    x = torch.randn((16, 64), dtype=torch.float16, device=flag_train.device) * scale
    bias = torch.randn((64,), dtype=torch.float16, device=flag_train.device)
    _check_bias_swiglu(x, bias, **kwargs)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_bias_shapes(dtype):
    utils.init_seed(4)
    x = torch.randn((8, 32), dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, torch.randn((1, 32), dtype=dtype, device=flag_train.device))
    _check_bias_swiglu(x, torch.randn((8, 32), dtype=dtype, device=flag_train.device))
    _check_bias_swiglu(
        x,
        torch.tensor(0.25, dtype=dtype, device=flag_train.device),
        gate_clamp_scale=3.0,
    )


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_clamp_boundary(dtype):
    """Values that land exactly on the clamp, just inside, and just outside."""
    clamp = 1.0
    gate = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float32
    )
    linear = torch.tensor(
        [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 1.5, 2.0], dtype=torch.float32
    )
    values = torch.cat((gate, linear)).to(dtype=dtype, device=flag_train.device)
    x = values.view(1, -1)
    bias = torch.zeros((x.shape[-1],), dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, bias, clamp_value=clamp)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_clamped_nan(dtype):
    """torch.clamp propagates NaN; the kernel must not replace it with a bound."""
    gate = torch.tensor(
        [float("nan"), 1.0, -2.0, 2.0], dtype=torch.float32, device=flag_train.device
    )
    linear = torch.tensor(
        [0.5, float("nan"), 3.0, -3.0], dtype=torch.float32, device=flag_train.device
    )
    x = torch.cat((gate, linear)).to(dtype).view(1, -1)

    y = flag_train.megatron.bias_swiglu_impl(x, None, clamp_value=2.0)
    assert torch.isnan(y[0, 0])  # gate is NaN
    assert torch.isnan(y[0, 1])  # linear is NaN

    expected = bias_swiglu_ref(_reference(x), None, clamp_value=2.0)
    _assert_close(y, expected, dtype, equal_nan=True)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_large_values(dtype):
    values = [-1000.0, -90.0, -20.0, -1.0, 0.0, 1.0, 20.0, 90.0, 1000.0, -8.0]
    # 10 gate values and 10 linear values.
    data = values + [-1000.0, -4.0, -1.0, 0.0, 0.5, 1.0, 4.0, 20.0, 90.0, 1000.0]
    x = torch.tensor(data, dtype=torch.float32, device=flag_train.device).to(dtype)
    x = x.view(1, -1)
    _check_bias_swiglu(x, None, gate_clamp_scale=5.0, linear_clamp_scale=3.0)
    _check_bias_swiglu(x, None)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_noncontiguous(dtype):
    utils.init_seed(5)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    assert x.shape == (64, 32)
    assert not x.is_contiguous()
    bias = torch.randn((64,), dtype=dtype, device=flag_train.device)[::2]
    assert bias.shape == (32,)
    assert not bias.is_contiguous()
    _check_noncontiguous(x, bias, clamp_value=2.0)

    # A layout whose view(-1, hidden) is not expressible: Megatron's own
    # entry point raises here too, so the port must not silently accept it.
    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)
    assert not transposed.is_contiguous()
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_swiglu_impl(transposed, None, gate_clamp_scale=2.0)
    if _megatron_impl is not None:
        with pytest.raises(RuntimeError):
            _megatron_impl(transposed, None, gate_clamp_scale=2.0)


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_rejects_bad_arguments():
    device = flag_train.device
    x = torch.randn((4, 8), device=device)
    for bad in (
        torch.randn((8,), device=device),
        torch.randn((1, 1, 2, 4), device=device),
    ):
        with pytest.raises(AssertionError):
            flag_train.megatron.bias_swiglu_impl(bad, None)
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_swiglu_impl(torch.randn((4, 7), device=device), None)
    with pytest.raises(AssertionError):
        flag_train.megatron.bias_swiglu_impl(
            x, None, clamp_value=1.0, gate_clamp_scale=1.0
        )
    with pytest.raises(AssertionError):
        flag_train.megatron.bias_swiglu_impl(x, None, linear_clamp_scale=1.0)
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_swiglu_impl(x, torch.randn((3,), device=device))
    # An empty input must not skip the bias checks.
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_swiglu_impl(
            torch.empty((0, 8), device=device), torch.randn((3,), device=device)
        )


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize(
    "kwargs",
    [
        {"clamp_value": 0.0},
        {"clamp_value": -1.0},
        {"clamp_value": float("nan")},
        {"clamp_value": float("inf")},
    ],
    ids=["zero", "negative", "nan", "inf"],
)
def test_bias_swiglu_impl_upstream_clamp_values(kwargs):
    """Upstream's non-positive/NaN clamp is unclamped; +inf is a clamped no-op."""
    x = torch.randn((8, 16), device=flag_train.device)
    y = flag_train.megatron.bias_swiglu_impl(x, None, **kwargs)
    expected = _expected_output(_reference(x), None, **kwargs)
    _assert_close(y, expected, x.dtype, equal_nan=True)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate_clamp_scale": -1.0},
        {"gate_clamp_scale": 0.0},
        {"gate_clamp_scale": float("nan")},
        {"gate_clamp_scale": float("inf")},
    ],
    ids=["negative", "zero", "nan", "inf"],
)
def test_bias_swiglu_impl_upstream_situ_scales(kwargs):
    """The fallback accepts the non-positive/non-finite SiTU scales upstream does."""
    x = torch.randn((8, 16), device=flag_train.device)
    y = flag_train.megatron.bias_swiglu_impl(x, None, **kwargs)
    expected = _expected_output(_reference(x), None, **kwargs)
    _assert_close(y, expected, x.dtype, equal_nan=True)


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_ignores_forward_only_flags():
    """fp8_input_store/cpu_offload_input only affect saved backward state."""
    x = torch.randn((8, 16), device=flag_train.device)
    expected = flag_train.megatron.bias_swiglu_impl(x, None)
    for name in ("fp8_input_store", "cpu_offload_input"):
        y = flag_train.megatron.bias_swiglu_impl(x, None, **{name: True})
        utils.train_assert_equal(utils.to_reference(y), utils.to_reference(expected))


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_mixed_bias_dtype(dtype):
    """A mixed-dtype bias promotes the result exactly as PyTorch and upstream do."""
    device = flag_train.device
    other = torch.float32 if dtype != torch.float32 else torch.float16
    x = torch.randn((4, 8), dtype=dtype, device=device)
    bias = torch.randn((8,), dtype=other, device=device)
    y = flag_train.megatron.bias_swiglu_impl(x, bias)

    assert y.dtype == torch.promote_types(dtype, other)
    expected = bias_swiglu_ref(_reference(x), _reference(bias))
    _assert_close(y, expected, y.dtype)
    if _megatron_impl is not None and not utils.TO_CPU:
        _assert_close(y, _megatron_impl(x, bias), y.dtype)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize(
    "dtype,device",
    [
        (torch.float64, flag_train.device),
        (torch.bfloat16, "cpu"),
        (torch.float32, "cpu"),
    ],
    ids=["fp64-device", "bf16-cpu", "fp32-cpu"],
)
def test_bias_swiglu_impl_device_and_dtype_domain(dtype, device):
    """fp64 stays on the Triton path; CPU uses a device round trip."""
    if (
        device == flag_train.device
        and dtype == torch.float64
        and not utils.fp64_is_supported
    ):
        pytest.skip("fp64 is not supported on this device")
    x = torch.randn((8, 16), dtype=dtype, device=device)
    bias = torch.randn((16,), dtype=dtype, device=device)
    _check_bias_swiglu(x, bias)
