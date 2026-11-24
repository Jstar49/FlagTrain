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

import warnings

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

_SHAPES = [
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

_SCALES = [1.0, 0.5, 4.0, -2.5, 1e-3, 1e3]


def tanh_soft_clamp_ref(x, scale):
    return (scale * torch.tanh(x.float() / scale)).to(x.dtype)


def base_op():
    try:
        import megatron.core.activations
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch Tanh Soft Clamp "
            "composition as the correctness reference",
            RuntimeWarning,
            stacklevel=2,
        )
        return tanh_soft_clamp_ref

    compiled = getattr(megatron.core.activations, "tanh_soft_clamp", None)
    if compiled is None:
        warnings.warn(
            "Megatron-LM predates tanh_soft_clamp (needs megatron-core >= "
            "0.20.0); using the PyTorch Tanh Soft Clamp composition as the "
            "correctness reference",
            RuntimeWarning,
            stacklevel=2,
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
                warnings.warn(
                    "Megatron tanh_soft_clamp failed to compile on this "
                    "platform; falling back to its eager implementation",
                    RuntimeWarning,
                    stacklevel=2,
                )
                chosen = eager
            else:
                warnings.warn(
                    "Megatron tanh_soft_clamp is unusable on this platform; "
                    "falling back to the PyTorch composition",
                    RuntimeWarning,
                    stacklevel=2,
                )
                chosen = tanh_soft_clamp_ref
            return chosen(*args, **kwargs)

    return reference


reference_tanh_soft_clamp = base_op()


def _reference(tensor):
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(out, ref_out, dtype, equal_nan=False):
    utils.train_assert_close(
        utils.to_reference(out), ref_out, dtype, equal_nan=equal_nan
    )


def _check_forward(x, scale):
    dtype = x.dtype
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    assert out.shape == x.shape
    assert out.dtype == dtype

    ref_out = reference_tanh_soft_clamp(_reference(x), scale)
    _assert_close(out, ref_out, dtype)


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
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_forward(x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scale", _SCALES)
def test_tanh_soft_clamp_values(dtype, scale):
    """Zero, near-zero, curve landmarks, and both tanh saturation regions."""
    values = [
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
    x = torch.tensor(values, dtype=torch.float32, device=flag_train.device).to(dtype)
    _check_forward(x, scale)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tanh_soft_clamp_noncontiguous(dtype):
    utils.init_seed(1)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    assert not x.is_contiguous()
    _check_forward(x, 0.5)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)
    assert not transposed.is_contiguous()
    _check_forward(transposed, 0.5)


@pytest.mark.tanh_soft_clamp
@pytest.mark.parametrize("scale", [1.0, -2.0, 0.0])
def test_tanh_soft_clamp_special_values(scale):
    """inf/nan and division by scale == 0 must match the baseline."""
    values = [
        0.0,
        1.0,
        -1.0,
        1e30,
        -1e30,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    x = torch.tensor(values, dtype=torch.float32, device=flag_train.device)
    out = flag_train.megatron.tanh_soft_clamp(x, scale)
    ref_out = reference_tanh_soft_clamp(_reference(x), scale)
    _assert_close(out, ref_out, torch.float32, equal_nan=True)
