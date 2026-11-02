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
signature compatibility but have nothing left to save, so passing either one is
expected to raise. ``bias`` must have the same dtype as ``input``.
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
    (0, 4, 16),
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
    gate, linear = torch.chunk(y, 2, dim=-1)
    if gate_clamp_scale is not None:
        gate_v = gate_clamp_scale * torch.tanh(gate / gate_clamp_scale)
        gate_v = gate_v * torch.sigmoid(gate)
        if linear_clamp_scale is not None:
            linear = linear_clamp_scale * torch.tanh(linear / linear_clamp_scale)
        return gate_v * linear
    if clamp_value is not None and clamp_value > 0:
        return torch.nn.functional.silu(gate.clamp(max=clamp_value)) * linear.clamp(
            min=-clamp_value, max=clamp_value
        )
    return torch.nn.functional.silu(gate) * linear


def _reference(tensor):
    if tensor is None:
        return None
    return utils.to_reference(tensor.detach(), upcast=True)


def _assert_close(actual, expected, dtype):
    utils.train_assert_close(utils.to_reference(actual), expected, dtype)


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


def _check_bias_swiglu(x, bias, **kwargs):
    dtype = x.dtype
    y = flag_train.megatron.bias_swiglu_impl(x, bias, **kwargs)
    assert y.shape[-1] * 2 == x.shape[-1]
    assert y.shape[:-1] == x.shape[:-1]
    assert y.dtype == dtype

    expected = _expected_output(_reference(x), _reference(bias), **kwargs)
    _assert_close(y, expected, dtype)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl(shape, dtype):
    utils.init_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_train.device)
    _check_bias_swiglu(x, None)


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
    _check_bias_swiglu(x, bias, clamp_value=2.0)

    cube = torch.randn((4, 8, 16), dtype=dtype, device=flag_train.device)
    transposed = cube.transpose(0, 2)
    assert not transposed.is_contiguous()
    assert transposed.shape[-1] % 2 == 0
    _check_bias_swiglu(transposed, None, gate_clamp_scale=2.0)


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_rejects_bad_arguments():
    device = flag_train.device
    x = torch.randn((4, 8), device=device)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        flag_train.megatron.bias_swiglu_impl(torch.randn((8,), device=device), None)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        flag_train.megatron.bias_swiglu_impl(
            torch.randn((1, 1, 2, 4), device=device), None
        )
    with pytest.raises(ValueError, match="even"):
        flag_train.megatron.bias_swiglu_impl(torch.randn((4, 7), device=device), None)
    with pytest.raises(ValueError, match="CPU"):
        flag_train.megatron.bias_swiglu_impl(torch.randn((4, 8)), None)
    with pytest.raises(ValueError, match="cannot both"):
        flag_train.megatron.bias_swiglu_impl(
            x, None, clamp_value=1.0, gate_clamp_scale=1.0
        )
    with pytest.raises(ValueError, match="requires gate_clamp_scale"):
        flag_train.megatron.bias_swiglu_impl(x, None, linear_clamp_scale=1.0)
    with pytest.raises(ValueError, match="greater than zero"):
        flag_train.megatron.bias_swiglu_impl(x, None, clamp_value=0.0)
    with pytest.raises(ValueError, match="greater than zero"):
        flag_train.megatron.bias_swiglu_impl(x, None, gate_clamp_scale=-1.0)
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_swiglu_impl(x, torch.randn((3,), device=device))
    # An empty input must not skip the bias checks.
    with pytest.raises(ValueError, match="broadcast"):
        flag_train.megatron.bias_swiglu_impl(
            torch.empty((0, 8), device=device), torch.randn((3,), device=device)
        )
    # Forward only: the flags that used to control the saved activations raise.
    with pytest.raises(NotImplementedError, match="forward-only"):
        flag_train.megatron.bias_swiglu_impl(x, None, fp8_input_store=True)
    with pytest.raises(NotImplementedError, match="forward-only"):
        flag_train.megatron.bias_swiglu_impl(x, None, cpu_offload_input=True)


@pytest.mark.bias_swiglu_impl
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bias_swiglu_impl_requires_matching_bias_dtype(dtype):
    """The kernel computes the add in fp32; a mixed-dtype bias would silently
    change the promoted dtype eager PyTorch produces, so it is rejected."""
    device = flag_train.device
    other = torch.float32 if dtype != torch.float32 else torch.float16
    x = torch.randn((4, 8), dtype=dtype, device=device)
    with pytest.raises(ValueError, match="input dtype"):
        flag_train.megatron.bias_swiglu_impl(
            x, torch.randn((8,), dtype=other, device=device)
        )
