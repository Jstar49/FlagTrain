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
"""Correctness tests for Megatron Fast GELU.

The preferred reference is Megatron-LM's official
``megatron.core.activations.fast_gelu`` interface. If Megatron-LM is not
installed, the tests log a warning and fall back to the equivalent PyTorch
composition. Under ``--ref cpu`` the reference operands are moved to the CPU
and upcast, so the comparison is against a higher-precision CPU result rather
than a second run of the device kernel. Tolerances are
``train_assert_close``'s defaults (``atol=1e-4``, dtype rtol from
``flag_train.testing.RESOLUTION``).
"""

import warnings

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils

# Scalar, empty, unaligned, and the ranks pointwise_dynamic specializes on.
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


def fast_gelu_ref(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * x * (1 + 0.044715 * x**2)))


def base_op():
    try:
        import megatron.core.activations
    except ImportError:
        warnings.warn(
            "Megatron-LM is unavailable; using the PyTorch Fast GELU "
            "composition as the correctness reference",
            RuntimeWarning,
            stacklevel=2,
        )
        return fast_gelu_ref

    compiled = megatron.core.activations.fast_gelu
    eager = getattr(compiled, "__wrapped__", compiled)
    chosen = compiled

    def reference(*args, **kwargs):
        nonlocal chosen
        if chosen is eager:
            return eager(*args, **kwargs)
        try:
            return chosen(*args, **kwargs)
        except Exception:
            warnings.warn(
                "Megatron fast_gelu failed to compile on this platform; "
                "falling back to its eager implementation",
                RuntimeWarning,
                stacklevel=2,
            )
            chosen = eager
            return eager(*args, **kwargs)

    return reference


reference_fast_gelu = base_op()


def _reference(tensor):
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


def _check_forward(x):
    dtype = x.dtype
    y = flag_train.megatron.fast_gelu(x)
    assert y.shape == x.shape
    assert y.dtype == dtype

    x_ref = _reference(x)
    y_ref = reference_fast_gelu(x_ref)
    _assert_close(y, y_ref, dtype)


@pytest.mark.fast_gelu
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fast_gelu(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_forward(x)


@pytest.mark.fast_gelu
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fast_gelu_values(dtype):
    """Zero, near-zero, curve landmarks, and both tanh saturation regions."""
    utils.init_seed(0)
    values = [
        -50.0,
        -8.0,
        -4.0,
        -2.0,
        -1.0,
        -1e-3,
        -1e-6,
        0.0,
        1e-6,
        1e-3,
        1.0,
        2.0,
        4.0,
        8.0,
        50.0,
    ]
    x = torch.tensor(values, dtype=torch.float32, device=flag_train.device).to(dtype)
    _check_forward(x)
