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

import math
import warnings

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

_FULL_SHAPES = [
    (),
    (0,),
    (0, 8),
    (1,),
    (17,),
    (1024,),
    (64, 128),
    (2, 19, 7),
    (4, 8, 32, 16),
]
_QUICK_SHAPES = [(), (1,), (17,), (64, 128), (2, 19, 7)]
_SHAPES = _QUICK_SHAPES if utils.QUICK_MODE else _FULL_SHAPES
_SCALES = [1.0, 0.5, 4.0, -2.5, 1e-3, 1e3]

_EXTENDED_DTYPES = [torch.int8, torch.int16, torch.int32, torch.uint8, torch.bool]
if utils.int64_is_supported:
    _EXTENDED_DTYPES.append(torch.int64)
_WIDE_UNSIGNED_DTYPES = [torch.uint16, torch.uint32, torch.uint64]
_CPU_DTYPES = [torch.float32, torch.int32, torch.bool]

# Zero, near-zero, curve landmarks, and both tanh saturation regions.
_CURVE_VALUES = [
    -1e30,
    -50.0,
    -8.0,
    -1.0,
    -1e-3,
    -1e-6,
    0.0,
    1e-6,
    1e-3,
    1.0,
    8.0,
    50.0,
    1e30,
]
_SPECIAL_VALUES = [
    0.0,
    1.0,
    -1.0,
    1e30,
    -1e30,
    float("inf"),
    float("-inf"),
    float("nan"),
]

# --------------------------------- case tables -----------------------------

# (values, scales, equal_nan, dtype): landmarks for every float dtype, the
# non-finite set once in fp32 (its semantics do not depend on the dtype).
_VALUE_CASES = [
    pytest.param(
        _CURVE_VALUES,
        _SCALES,
        False,
        dtype,
        id=f"landmarks-{str(dtype).split('.')[-1]}",
    )
    for dtype in utils.FLOAT_DTYPES
] + [
    pytest.param(
        _SPECIAL_VALUES, [1.0, -2.0, 0.0], True, torch.float32, id="non-finite"
    ),
]

# (dtype, device, scale): the integer dtypes Triton can load and store, with
# scales that pin the cast rules (including zero and non-finite divisions).
_INTEGRAL_CASES = [
    pytest.param(
        dtype,
        flag_train.device,
        scale,
        id=f"{str(dtype).split('.')[-1]}-{scale}",
    )
    for dtype in _EXTENDED_DTYPES
    for scale in [0.5, 1e3, 0.0, -2.5, float("inf"), float("nan")]
]

# (device, dtype, layout, scale): both strided views per float dtype on the
# accelerator, representative dtypes through the CPU composition, and the two
# CPU-only layouts.
_LAYOUT_CASES = (
    [
        pytest.param(
            flag_train.device,
            dtype,
            layout,
            0.5,
            id=f"{str(dtype).split('.')[-1]}-{layout}",
        )
        for dtype in utils.FLOAT_DTYPES
        for layout in ("matrix", "cube")
    ]
    + [
        pytest.param(
            "cpu",
            dtype,
            "contiguous",
            scale,
            id=f"cpu-{str(dtype).split('.')[-1]}-{scale}",
        )
        for dtype in _CPU_DTYPES
        for scale in (0.5, 0.0, -2.5)
    ]
    + [
        pytest.param("cpu", torch.float32, "matrix", 0.5, id="cpu-matrix"),
        pytest.param("cpu", torch.float32, "empty", 0.5, id="cpu-empty"),
    ]
)

# (x_shape, scale_value, dtype): a broadcast vector and a 0-dimensional tensor.
_SCALE_FORM_CASES = [
    pytest.param((2, 1), [0.5, 1.0, 2.0], torch.float32, id="vector-float"),
    pytest.param((2, 1), [0.5, 1.0, 2.0], torch.int32, id="vector-int"),
    pytest.param((2, 3), 0.5, torch.float32, id="0-dim-tensor"),
]

# n > 16384 exercises the persistent walk; the 8-bit dtypes additionally pin
# the narrower Ascend tile (a 16384-lane int8 tile overflows UB at this size).
_LARGE_CASES = [
    pytest.param((1 << 20,), dtype, id=f"float-{str(dtype).split('.')[-1]}")
    for dtype in utils.FLOAT_DTYPES
] + [
    pytest.param((1 << 20,), dtype, id=f"narrow-{str(dtype).split('.')[-1]}")
    for dtype in (torch.int8, torch.uint8, torch.bool)
]

# Dtypes Triton cannot load and store on this stack. They are rejected with a
# clear error instead of an MLIR failure.
_UNSUPPORTED_CASES = [
    pytest.param(dtype, device, id=f"{str(dtype).split('.')[-1]}-{device}")
    for dtype in _WIDE_UNSIGNED_DTYPES
    for device in (flag_train.device, "cpu")
]


# ------------------------------ reference chain ----------------------------


def tanh_soft_clamp_ref(x, scale):
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)


def _warn(message):
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def base_op():
    try:
        import megatron.core.activations
    except ImportError:
        _warn(
            "Megatron-LM is unavailable; using the PyTorch Tanh Soft Clamp "
            "composition as the correctness reference"
        )
        return tanh_soft_clamp_ref

    compiled = getattr(megatron.core.activations, "tanh_soft_clamp", None)
    if compiled is None:
        _warn(
            "Megatron-LM predates tanh_soft_clamp (needs megatron-core >= "
            "0.20.0); using the PyTorch Tanh Soft Clamp composition as the "
            "correctness reference"
        )
        return tanh_soft_clamp_ref

    eager = getattr(compiled, "__wrapped__", compiled)
    chosen = compiled

    def reference(*args, **kwargs):
        nonlocal chosen
        if chosen is tanh_soft_clamp_ref:
            return chosen(*args, **kwargs)
        try:
            return chosen(*args, **kwargs)
        except Exception:
            if chosen is not eager:
                _warn(
                    "Megatron tanh_soft_clamp failed to compile on this "
                    "platform; falling back to its eager implementation"
                )
                chosen = eager
            else:
                _warn(
                    "Megatron tanh_soft_clamp is unusable on this platform; "
                    "falling back to the PyTorch composition"
                )
                chosen = tanh_soft_clamp_ref
            return chosen(*args, **kwargs)

    return reference


reference_tanh_soft_clamp = base_op()


# --------------------------------- helpers ---------------------------------


def _reference(tensor):
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(out, ref_out, dtype, equal_nan=False):
    utils.train_assert_close(
        utils.to_reference(out), ref_out, dtype, equal_nan=equal_nan
    )


def expected(x, scale):
    """The reference result for one call.

    Floating inputs are upcast first. Integer, bool and unsigned inputs keep
    the input dtype so the cast happens at the same point: their NaN casts
    differ between compiled and eager Megatron and between CPU and device.
    """
    if isinstance(scale, torch.Tensor):
        # ``--ref cpu`` moves the reference operands off the device together.
        scale = utils.to_reference(scale)
    if x.dtype.is_floating_point:
        return reference_tanh_soft_clamp(_reference(x), scale)
    return tanh_soft_clamp_ref(utils.to_reference(x), scale)


def check(x, scale, check_stride=False, equal_nan=False):
    """Run the operator, check its metadata, and compare with ``expected``."""
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    if check_stride:
        assert out.stride() == x.stride()
    if not x.dtype.is_floating_point:
        assert out.device == x.device
    _assert_close(out, expected(x, scale), x.dtype, equal_nan=equal_nan)
    return out


def check_tensor_scale(x, scale):
    """Compare a tensor-scale call, whose output may be broadcast wider."""
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    assert out.dtype == x.dtype
    assert out.device == x.device
    _assert_close(out, expected(x, scale), x.dtype)
    return out


def _integral_tensor(dtype, shape, device):
    n = math.prod(shape)
    values = torch.arange(n, device=device, dtype=torch.int64)
    if dtype is torch.bool:
        values = values.remainder(2)
    elif dtype not in (torch.uint8, *_WIDE_UNSIGNED_DTYPES):
        values = values - n // 2
    return values.to(dtype).view(shape)


def _layout_tensor(layout, dtype, device):
    if layout == "matrix":
        return torch.randn((32, 64), dtype=dtype, device=device).t()
    if layout == "cube":
        return torch.randn((4, 8, 16), dtype=dtype, device=device).transpose(0, 2)
    if layout == "contiguous":
        if dtype.is_floating_point:
            return torch.randn((2, 3, 7), dtype=dtype, device=device)
        return _integral_tensor(dtype, (2, 3, 7), device)
    return torch.empty((0, 8), dtype=torch.float32, device="cpu")


def _skip_nan_cast_on_cpu(scale, device=None):
    """NaN-to-integer casts differ between the reference and compute devices.

    A zero or non-finite scale turns ``x == 0`` into NaN, whose integer cast is
    platform-specific: CPU PyTorch gives ``INT_MIN`` where CUDA gives zero.
    The operator stages CPU inputs through the accelerator, and ``--ref cpu``
    moves the reference to the CPU, so either one puts the two casts on
    different platforms.
    """
    if not (scale == 0.0 or not math.isfinite(scale)):
        return
    if device is None or torch.device(device).type == "cpu" or utils.TO_CPU:
        pytest.skip("NaN-to-integer casts differ between CPU and device")


# -------------------------------- scenarios --------------------------------


@pytest.fixture
def megatron_tanh_soft_clamp():
    activations = pytest.importorskip("megatron.core.activations")
    native = getattr(activations, "tanh_soft_clamp", None)
    if native is None:
        pytest.skip("Native tanh_soft_clamp needs megatron-core >= 0.20.0")
    return native


def _check_native_forward(native, x, scale):
    expected = native(x, scale)
    actual = flag_train.megatron.tanh_soft_clamp(x, scale)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == x.dtype
    assert actual.device == expected.device == x.device
    expected = utils.to_reference(expected)
    if x.dtype == torch.int32:
        utils.train_assert_equal(actual, expected)
    else:
        utils.train_assert_close(actual, expected, x.dtype)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("scale, transpose", [(2.0, False), (2.5, True), (-2.5, False)])
def test_tanh_soft_clamp_int32_native(megatron_tanh_soft_clamp, scale, transpose):
    x = torch.tensor(
        [[-3, -1, 0], [1, 2, 3]], dtype=torch.int32, device=flag_train.device
    )
    if transpose:
        x = x.t()
        assert not x.is_contiguous()
    _check_native_forward(megatron_tanh_soft_clamp, x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize(
    "dtype, scale_values, narrow_input",
    [
        pytest.param(torch.float32, 2.0, False, id="zero-dimensional"),
        pytest.param(torch.float32, [0.5, 2.0, 3.5], False, id="broadcast"),
        pytest.param(torch.float32, [0.5, 2.0, 3.5], True, id="expands-input"),
        pytest.param(torch.int32, [0.5, 2.0, 3.5], False, id="int32-broadcast"),
    ],
)
def test_tanh_soft_clamp_tensor_scale_native(
    megatron_tanh_soft_clamp, dtype, scale_values, narrow_input
):
    x = torch.tensor([[-3, -1, 0], [1, 2, 3]], dtype=dtype, device=flag_train.device)
    if narrow_input:
        x = x[:, :1]
    scale = torch.tensor(scale_values, dtype=torch.float32, device=x.device)
    _check_native_forward(megatron_tanh_soft_clamp, x, scale)


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_int32_tensor_scale_rounding(megatron_tanh_soft_clamp):
    x = torch.tensor([5, -4764], dtype=torch.int32, device=flag_train.device)
    scale = torch.tensor(
        [30427.671875, 28793.58984375], dtype=torch.float32, device=x.device
    )
    _check_native_forward(megatron_tanh_soft_clamp, x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_tanh_soft_clamp(shape, dtype, scale):
    """The math grid: every shape, floating dtype and scale magnitude."""
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    check(x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize(("values", "scales", "equal_nan", "dtype"), _VALUE_CASES)
def test_tanh_soft_clamp_boundary_values(values, scales, equal_nan, dtype):
    """Curve landmarks and non-finite/zero-scale inputs the grid cannot draw."""
    x = torch.tensor(values, dtype=torch.float32, device=flag_train.device).to(dtype)
    for scale in scales:
        check(x, scale, equal_nan=equal_nan)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize(("dtype", "device", "scale"), _INTEGRAL_CASES)
def test_tanh_soft_clamp_integral_dtypes(dtype, device, scale):
    """Integer and bool tensors cast back to their input dtype."""
    if dtype is not torch.bool:
        _skip_nan_cast_on_cpu(scale, device)
    x = _integral_tensor(dtype, (2, 3, 7), device)
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert out.device == x.device
    ref_out = expected(x, scale)
    if dtype is torch.bool:
        _assert_close(out, ref_out, dtype)
    else:
        # The kernel's libdevice tanh and torch's tanh differ by an ulp, which
        # an integer cast turns into +/-1 only on a rounding boundary (x = -2 at
        # scale 1e3 is such a case; measured: at most 2 elements in 42).
        # A systematic truncation/rounding bug moves nearly every element whose
        # result is just below an integer, so pin both the magnitude and the
        # fraction instead of allowing a blanket +/-1.
        # ``expected`` returns the reference on its own device (the CPU under
        # ``--ref cpu``), so bring the result over before comparing.
        result = utils.to_reference(out)
        diff = (result.to(torch.int64) - ref_out.to(torch.int64)).abs()
        assert int(diff.max()) <= 1, "an integer cast is off by more than 1"
        assert (
            float((diff > 0).float().mean()) <= 0.5
        ), "more than half of the integer casts differ"


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize(("device", "dtype", "layout", "scale"), _LAYOUT_CASES)
def test_tanh_soft_clamp_layout_and_device(device, dtype, layout, scale):
    """Strided views keep their strides; empty and CPU tensors stay on the CPU."""
    if not dtype.is_floating_point and dtype is not torch.bool:
        _skip_nan_cast_on_cpu(scale, device)
    utils.init_seed(1)
    x = _layout_tensor(layout, dtype, device)
    if layout == "empty":
        out = check(x, scale)
        assert out.stride() == x.stride()
    else:
        if layout != "contiguous":
            assert not x.is_contiguous()
        out = check(x, scale, check_stride=layout != "contiguous")
    assert out.device == x.device


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("device", [flag_train.device, "cpu"])
@pytest.mark.parametrize(("x_shape", "scale_value", "dtype"), _SCALE_FORM_CASES)
def test_tanh_soft_clamp_tensor_scale(x_shape, scale_value, dtype, device):
    """A 0-dimensional or broadcast tensor scale follows upstream's expression."""
    if dtype.is_floating_point:
        x = torch.randn(x_shape, dtype=dtype, device=device)
    else:
        x = torch.arange(1, math.prod(x_shape) + 1, dtype=dtype, device=device).view(
            x_shape
        )
    scale = torch.tensor(scale_value, dtype=torch.float32, device=device)
    out = check_tensor_scale(x, scale)
    assert out.shape == torch.broadcast_shapes(x_shape, scale.shape)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize(("shape", "dtype"), _LARGE_CASES)
def test_tanh_soft_clamp_large_shapes(shape, dtype):
    """The walk kernel and the width-dependent tile (an int8 ub-overflow pin)."""
    utils.init_seed(3)
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    else:
        x = _integral_tensor(dtype, shape, flag_train.device)
    check(x, 0.5)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tanh_soft_clamp_tensor_scale_noncontiguous(dtype):
    """A same-shape scale pairs with a transposed input in logical order."""
    utils.init_seed(4)
    x = torch.randn((64, 32), dtype=dtype, device=flag_train.device).t()
    assert not x.is_contiguous()
    scale = torch.rand((32, 64), dtype=torch.float32, device=flag_train.device)
    out = check_tensor_scale(x, scale)
    assert out.shape == x.shape


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_rank_raising_scale():
    """A one-element scale can raise the output's rank over a strided input."""
    utils.init_seed(5)
    x = torch.randn((4, 3), device=flag_train.device).t()
    assert not x.is_contiguous()
    scale = torch.tensor([[[0.5]]], device=flag_train.device)
    out = check_tensor_scale(x, scale)
    assert out.shape == (1, 3, 4)


@pytest.mark.tanh_soft_clamp
@pytest.mark.skipif(not utils.fp64_is_supported, reason="device has no fp64")
def test_tanh_soft_clamp_float64():
    x = torch.randn((2, 3, 7), dtype=torch.float64, device=flag_train.device)
    check(x, 0.5)


@pytest.mark.tanh_soft_clamp
@pytest.mark.skipif(not utils.fp64_is_supported, reason="device has no fp64")
def test_tanh_soft_clamp_accepts_0dim_float64_scale():
    """A 0-dim float64 scale follows torch's wrapped-number rule and stays fp32."""
    x = torch.zeros((2, 3), dtype=torch.float32, device=flag_train.device)
    scale = torch.tensor(0.5, dtype=torch.float64)
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    assert out.dtype == torch.float32
    assert out.device == x.device


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_rejects_non_scalar_float64_scale():
    """A shaped float64 scale would promote the expression to float64."""
    x = torch.zeros((2, 3), dtype=torch.float32, device=flag_train.device)
    scale = torch.ones((1,), dtype=torch.float64)
    with pytest.raises(ValueError, match="float64"):
        flag_train.megatron.tanh_soft_clamp(x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize(("dtype", "device"), _UNSUPPORTED_CASES)
def test_tanh_soft_clamp_unsupported_dtypes(dtype, device):
    """Dtypes the kernel cannot load are rejected before any kernel runs."""
    x = torch.zeros((2, 3), dtype=torch.int32).to(dtype)
    if device != "cpu":
        x = x.to(device)
    with pytest.raises(ValueError, match="does not support dtype"):
        flag_train.megatron.tanh_soft_clamp(x, 0.5)


@pytest.mark.tanh_soft_clamp
def test_tanh_soft_clamp_api_contract():
    """The forward has no gradient, and invalid operands are rejected."""
    x = torch.randn((2, 3), device=flag_train.device, requires_grad=True)
    x_cpu = torch.randn((2, 3), requires_grad=True)
    scale_cpu = torch.tensor(0.5, requires_grad=True)
    for operands in ((x, 0.5), (x_cpu, 0.5), (x_cpu, scale_cpu)):
        assert not flag_train.megatron.tanh_soft_clamp(*operands).requires_grad
    with pytest.raises(TypeError, match="tensor"):
        flag_train.megatron.tanh_soft_clamp([1.0], 0.5)
    with pytest.raises(TypeError, match="complex"):
        flag_train.megatron.tanh_soft_clamp(
            torch.tensor([1 + 0j], device=flag_train.device), 0.5
        )
