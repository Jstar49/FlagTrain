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
"""Correctness tests for fused bias + GeGLU (forward only).

The reference is Megatron-LM's own ``bias_geglu_impl`` when megatron-core is
installed. Otherwise, ``bias_geglu_ref`` evaluates the eager tanh-GELU formula
with the bias added by ordinary broadcasting.

Under ``--ref cpu`` the operands are moved to the CPU and upcast, so the
comparison is against a higher-precision CPU result. Tolerances are
``train_assert_close``'s defaults.

Only the forward output is checked: the backward is a separate operator.
The Triton implementation accepts contiguous active-device fp16/bf16/fp32
inputs, plus fp64 on devices with FP64 support, and optional same-dtype bias
broadcastable to the flattened input. Unsupported operands are rejected
instead of using an eager PyTorch fallback.
"""

import pytest
import torch

import flag_train

from .. import accuracy_utils as utils


def _load_megatron_impl():
    """Installed ``bias_geglu_impl``, or ``None`` when it is unavailable.

    Older Megatron installs may not provide the fusions module or this public
    function; those installations use the formula reference. Other import
    failures still raise instead of silently hiding a broken installation.
    """
    try:
        importlib_metadata_version = __import__(
            "importlib.metadata", fromlist=["version"]
        )
    except ImportError:
        return None
    try:
        importlib_metadata_version.version("megatron-core")
    except Exception:
        return None
    try:
        from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
    except ModuleNotFoundError as exc:
        if exc.name in {
            "megatron.core.fusions",
            "megatron.core.fusions.fused_bias_geglu",
        }:
            return None
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    except ImportError as exc:
        if "cannot import name 'bias_geglu_impl'" in str(exc):
            return None
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "megatron-core is installed but importing bias_geglu_impl failed: "
            f"{exc!r}"
        ) from exc
    return bias_geglu_impl


_megatron_impl = _load_megatron_impl()

# Empty, unaligned (half = 3 and 17), and MLP-like ranks. The last dim is even.
_SHAPES = [
    (0, 8),
    (0, 4, 16),
    (1, 2),
    (8, 6),
    (5, 34),
    (17, 32),
    (64, 128),
    (2, 8, 64),
    (4, 7, 32),
]

# Bias is orthogonal to the shapes above, so it is sampled on representative
# cases only.
_BIAS_SHAPES = [
    (1, 2),
    (17, 32),
    (64, 128),
    (4, 7, 32),
]


def bias_geglu_ref(x, bias=None):
    y = x if bias is None else x + bias
    gate, linear = torch.chunk(y, 2, dim=-1)
    # tanh approximation of GELU, the same constants as Megatron-LM.
    return (
        gate
        * 0.5
        * (1.0 + torch.tanh(0.79788456 * gate * (1 + 0.044715 * gate * gate)))
    ) * linear


def _reference(tensor):
    if tensor is None:
        return None
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


def _expected_output(x, bias):
    """Use Megatron-LM when available, otherwise use the eager formula.

    The Megatron side gets contiguous operands: its ``input.view(-1, hidden)``
    rejects the transposed inputs this suite deliberately feeds the operator.
    Empty inputs also use the formula: Megatron restores a rank-3 result with
    ``view(..., -1)``, which is ambiguous when the output has zero elements.
    """
    if _megatron_impl is not None and not utils.TO_CPU and x.numel() > 0:
        return _megatron_impl(
            x.contiguous(),
            None if bias is None else bias.contiguous(),
        )
    ori_shape = x.shape
    flat = x.view(-1, ori_shape[-1])
    output = bias_geglu_ref(flat, bias)
    return (
        output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
    )


def _check_bias_geglu(x, bias):
    # Restoring an empty rank-3 result with view(..., -1) is ambiguous.
    if x.ndim == 3 and x.numel() == 0:
        with pytest.raises(RuntimeError):
            flag_train.megatron.bias_geglu_impl(x, bias)
        return

    dtype = x.dtype
    y = flag_train.megatron.bias_geglu_impl(x, bias)
    assert y.shape[-1] * 2 == x.shape[-1]
    assert y.shape[:-1] == x.shape[:-1]
    assert y.dtype == dtype

    expected = _expected_output(_reference(x), _reference(bias))
    _assert_close(y, expected, dtype)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, None)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("shape", _BIAS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_bias(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    bias = torch.randn(shape[-1], dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, bias)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_bias_shapes(dtype):
    utils.init_seed(4)
    x = torch.randn((8, 32), dtype=dtype, device=flag_train.device)
    _check_bias_geglu(x, torch.randn((1, 32), dtype=dtype, device=flag_train.device))
    _check_bias_geglu(x, torch.randn((8, 32), dtype=dtype, device=flag_train.device))
    _check_bias_geglu(x, torch.tensor(0.25, dtype=dtype, device=flag_train.device))


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_large_values(dtype):
    """Extreme magnitudes: check GELU/tanh saturation."""
    values = [-1000.0, -90.0, -20.0, -1.0, 0.0, 1.0, 20.0, 90.0, 1000.0, -8.0]
    # 10 gate values and 10 linear values.
    data = values + [-1000.0, -4.0, -1.0, 0.0, 0.5, 1.0, 4.0, 20.0, 90.0, 1000.0]
    x = torch.tensor(data, dtype=torch.float32, device=flag_train.device).to(dtype)
    x = x.view(1, -1)
    _check_bias_geglu(x, None)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_noncontiguous(dtype):
    """Non-contiguous inputs are rejected without an eager fallback."""
    utils.init_seed(5)
    base = torch.randn((32, 64), dtype=dtype, device=flag_train.device)
    x = base.t()
    bias = torch.randn((64,), dtype=dtype, device=flag_train.device)[::2]

    with pytest.raises(ValueError, match="contiguous"):
        flag_train.megatron.bias_geglu_impl(x, bias)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)

    # The public view rejects this layout before operand validation.
    with pytest.raises(ValueError):
        flag_train.megatron.bias_geglu_impl(transposed, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_rank_validation():
    device = flag_train.device
    with pytest.raises(AssertionError):
        flag_train.megatron.bias_geglu_impl(torch.randn((8,), device=device), None)
    with pytest.raises(AssertionError):
        flag_train.megatron.bias_geglu_impl(
            torch.randn((1, 1, 2, 4), device=device), None
        )


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_bias_broadcast_validation():
    x = torch.randn((4, 8), device=flag_train.device)
    with pytest.raises(ValueError, match="broadcastable"):
        flag_train.megatron.bias_geglu_impl(x, torch.randn((3,), device=x.device))


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_geglu_impl_bias_dtype_validation(dtype):
    """Mixed-dtype bias is rejected instead of using eager type promotion."""
    other = torch.float32 if dtype != torch.float32 else torch.float16
    x = torch.randn((4, 8), dtype=dtype, device=flag_train.device)
    bias = torch.randn((8,), dtype=other, device=flag_train.device)
    with pytest.raises(ValueError, match="input dtype"):
        flag_train.megatron.bias_geglu_impl(x, bias)


@pytest.mark.bias_geglu_impl
@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 16)])
@pytest.mark.parametrize("bias_mode", ["none", "vector", "full"])
def test_bias_geglu_impl_fp64(shape, bias_mode):
    """FP64 inputs use FP64 arithmetic throughout the Triton kernel."""
    if not flag_train.runtime.device.support_fp64:
        pytest.skip("the active device does not support FP64")

    utils.init_seed(6)
    x = torch.randn(shape, dtype=torch.float64, device=flag_train.device)
    flat = x.view(-1, shape[-1])
    if bias_mode == "none":
        bias = None
    elif bias_mode == "vector":
        bias = torch.randn(shape[-1], dtype=torch.float64, device=flag_train.device)
    else:
        bias = torch.randn_like(flat)

    _check_bias_geglu(x, bias)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_additional_dtype_validation():
    """Unsupported dtypes are rejected rather than evaluated with PyTorch."""
    unsupported = [torch.int32, torch.bool, torch.complex64]
    if not flag_train.runtime.device.support_fp64:
        unsupported.insert(0, torch.float64)

    for dtype in unsupported:
        x = torch.zeros((4, 8), dtype=dtype, device=flag_train.device)
        with pytest.raises(ValueError, match="unsupported dtype"):
            flag_train.megatron.bias_geglu_impl(x, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_cpu_validation():
    x = torch.randn((4, 8), device="cpu")
    with pytest.raises(ValueError, match="active device"):
        flag_train.megatron.bias_geglu_impl(x, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_leading_broadcast_validation():
    x = torch.randn((4, 8), device=flag_train.device)
    bias = torch.randn((1, 1, 8), device=flag_train.device)
    with pytest.raises(ValueError, match="broadcastable"):
        flag_train.megatron.bias_geglu_impl(x, bias)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_scalar_bias_validation():
    x = torch.randn((4, 8), dtype=torch.float16, device=flag_train.device)
    with pytest.raises(TypeError, match="Tensor or None"):
        flag_train.megatron.bias_geglu_impl(x, 0.5)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_odd_last_dimension():
    x = torch.randn((4, 7), device=flag_train.device)
    with pytest.raises(ValueError, match="must be even"):
        flag_train.megatron.bias_geglu_impl(x, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_empty_3d_error():
    x = torch.empty((2, 0, 8), device=flag_train.device)
    with pytest.raises(RuntimeError):
        flag_train.megatron.bias_geglu_impl(x, None)


@pytest.mark.bias_geglu_impl
def test_bias_geglu_impl_bias_device_validation():
    if flag_train.device == "cpu":
        pytest.skip("requires distinct CPU and active devices")
    x = torch.randn((4, 8), device=flag_train.device)
    bias = torch.randn((8,), device="cpu")
    with pytest.raises(ValueError, match="input device"):
        flag_train.megatron.bias_geglu_impl(x, bias)
