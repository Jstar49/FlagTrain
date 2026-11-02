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
"""Performance benchmark for bias_swiglu_impl.

The baseline is Megatron-LM's own ``bias_swiglu_impl`` when it is installed.
That function is a PyTorch composition decorated with ``@jit_fuser``, which is
``torch.compile`` on torch >= 2.2, so the baseline is the compiled native
implementation rather than eager PyTorch. When megatron-core is older than
0.20 (the first release with the clamp / SiTU arguments), or is not installed,
the main-branch implementation vendored in the appendix at the end of this file
is used instead, so every case is still compared against a compiled baseline.
The choice is made once at import from the installed version, and the
``Baseline:`` line printed by the test records which one ran.

Timing, warmup and synchronization come from ``benchmark/base.py``
(``triton.testing.do_bench`` in kernel mode). ``--level core`` runs the shapes
below. One benchmark table covers the three activations: every row carries the
variant label and ``--dtypes`` chooses the dtypes. The operator is
forward-only, so there are no backward cases. The plain path adds a
``[hidden]`` bias. The clamped and SiTU-GLU rows stay separate so their extra
tanh/clamp work is visible instead of being folded into the plain numbers.

``torch.compile`` recompiles per shape/dtype/requires-grad pattern. The
default Dynamo recompile limit is smaller than this benchmark's case count, so
the module raises it; otherwise the baseline silently falls back to eager and
the comparison is meaningless.
"""

import importlib.metadata
import os
import re
import warnings
from typing import Optional

import pytest
import torch

import flag_train

from .. import base, consts

# Token counts and hidden widths used by a fused MLP activation. Last dim is even.
_CORE_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (16384, 4096),
    (2048, 11008),
    (16, 128, 8192),
    (256, 32768),
]


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


# Enough variants for 7 shapes x 3 activations x 3 dtypes of forward cases
# without Dynamo falling back to eager halfway through the suite.
_DYNAMO_RECOMPILE_LIMIT = 128

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


def _stabilize_megatron_compile():
    """Keep the compiled Megatron baseline working on every backend.

    ``megatron.core.jit`` compiles the baseline per shape, dtype and
    requires-grad pattern, so the default Dynamo limit is below this suite's
    case count. Vendor Megatron builds may wrap ``torch.compile`` in their own
    helper, so do not gate on an identity check against ``torch.compile``.
    Inductor's async compile also has to pickle the compiled Triton kernel back
    from its worker, which some vendor Triton builds cannot do.
    """
    try:
        from megatron.core import jit
    except ImportError:
        return

    # Inductor's async compile sends each Triton kernel to a worker process and
    # pickles the compiled kernel back. Vendor Triton builds whose JITFunction
    # cannot be pickled (it carries an RLock) fail there with
    # "TypeError: cannot pickle '_thread.RLock' object" while compiling the
    # Megatron baseline. One compile thread keeps compilation in-process; it
    # changes compile time only, not the measured kernel time.
    try:
        import torch._inductor.config

        if getattr(torch._inductor.config, "compile_threads", 1) != 1:
            torch._inductor.config.compile_threads = 1
    except ImportError:
        pass

    if jit.jit_fuser is getattr(jit, "noop_decorator", None):
        return

    try:
        import torch._dynamo
    except ImportError:
        return
    if hasattr(torch._dynamo.config, "recompile_limit"):
        current = torch._dynamo.config.recompile_limit
    elif hasattr(torch._dynamo.config, "cache_size_limit"):
        current = torch._dynamo.config.cache_size_limit
    else:
        return
    if current >= _DYNAMO_RECOMPILE_LIMIT:
        return
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = _DYNAMO_RECOMPILE_LIMIT
    else:
        torch._dynamo.config.cache_size_limit = _DYNAMO_RECOMPILE_LIMIT
    warnings.warn(
        "raised the torch._dynamo recompile limit to "
        f"{_DYNAMO_RECOMPILE_LIMIT} so the torch.compile baseline does not "
        "silently fall back to eager",
        RuntimeWarning,
        stacklevel=2,
    )


_stabilize_megatron_compile()

_CLAMPED_KWARGS = {"clamp_value": 2.0}
_SITU_KWARGS = {"gate_clamp_scale": 4.0, "linear_clamp_scale": 2.5}


def _dtypes():
    """``consts.FLOAT_DTYPES`` with bf16 dropped where the backend has none.

    Same rule as ``utils.FLOAT_DTYPES`` in the correctness suite, so both see
    the same dtypes on every backend.
    """
    dtypes = [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16]
    if flag_train.runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


class BiasSwigluImplBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _CORE_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_CORE_SHAPES)


# Variant label -> operator kwargs. The label travels with each input row, so
# one benchmark table covers all three activations.
_VARIANTS = {
    "plain": {},
    "clamped": _CLAMPED_KWARGS,
    "situ": _SITU_KWARGS,
}


def bias_swiglu_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    bias = torch.randn((shape[-1],), dtype=dtype, device=device)
    for variant in _VARIANTS:
        yield (x, bias, variant)


def torch_op(x, bias, variant):
    """Installed Megatron >= 0.20, or the vendored main-branch copy."""
    return _BASELINE_IMPL(x, bias, **_VARIANTS[variant])


def train_op(x, bias, variant):
    """The operator under test."""
    return flag_train.megatron.bias_swiglu_impl(x, bias, **_VARIANTS[variant])


@pytest.mark.bias_swiglu_impl
def test_bias_swiglu_impl_perf():
    """Plain, clamped, and SiTU-GLU rows for every shape in one table."""
    print(f"\nBaseline: {_BASELINE}")
    # On Ascend, enable parallel scheduling of all blocks to avoid excessive
    # logical cores and reduce scheduling overhead when the logical core count
    # exceeds 65,535.
    # See https://github.com/flagos-ai/FlagGems/pull/7304
    if flag_train.vendor_name == "ascend":
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

    bench = BiasSwigluImplBenchmark(
        input_fn=bias_swiglu_input_fn,
        op_name="bias_swiglu_impl",
        torch_op=torch_op,
        dtypes=_dtypes(),
        is_backward=False,
    )
    bench.set_train(train_op)
    bench.run()


# ===========================================================================
# Appendix: vendored Megatron-LM baseline
#
# Source: NVIDIA Megatron-LM, **main branch**, fetched 2026-10-28 (no release
# tag yet; these variants are expected to reach a release with 0.20.0).
#   https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_bias_swiglu.py
#
# The megatron-core releases this file runs against on older machines
# predate 0.20 and have no clamp / SiTU arguments, so those benchmark cases
# would otherwise fall back to the eager composition; vendoring the
# main-branch code keeps them compared against a compiled Megatron.
# ``_load_baseline`` below picks this copy whenever megatron-core < 0.20 or is
# not installed.
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Changes from upstream: ``jit_fuser`` is replaced by ``torch.compile`` and
# ``F.`` is expanded to ``torch.nn.functional.``. The backward functions, the
# saved-tensor bookkeeping, and the Megatron-internal decorators are dropped
# because this benchmark measures forward only. The forward math is verbatim,
# which is why it sits under ``fmt: off``: black must not reformat it, so it
# can be re-synced with upstream. Benchmark baseline only, never a correctness
# oracle.
#
# Once every supported megatron-core is >= 0.20 this appendix can be deleted
# together with ``_load_baseline`` and the ``_BASELINE_IMPL`` indirection.
# ===========================================================================

jit_fuser = torch.compile


@jit_fuser
def swiglu(y):
    """Performs SwiGLU (Swish-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SwiGLU activation: SiLU(y1) * y2, where y1, y2 are the split halves.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.nn.functional.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y, bias):
    """Performs SwiGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SwiGLU activation.
    """
    y = y + bias
    return swiglu(y)


@jit_fuser
def clamped_swiglu(y, clamp_value):
    """Perform SwiGLU after clamping both halves of the input."""
    dtype = y.dtype
    y_1, y_2 = torch.chunk(y.to(torch.float32), 2, -1)
    y_1 = y_1.clamp(min=None, max=clamp_value)
    y_2 = y_2.clamp(min=-clamp_value, max=clamp_value)
    res = torch.nn.functional.silu(y_1) * y_2
    return res.to(dtype)


@jit_fuser
def bias_clamped_swiglu(y, bias, clamp_value):
    """Perform clamped SwiGLU after bias addition."""
    return clamped_swiglu(y + bias, clamp_value)


@jit_fuser
def _tanh_clamp_and_deriv(y: torch.Tensor, clamp_scale: float):
    """Soft-clamps ``y`` and returns the clamp derivative alongside it, both in fp32."""
    t = torch.tanh(y.float() / clamp_scale)
    return clamp_scale * t, 1 - t * t


@jit_fuser
def situ_glu(y, gate_clamp_scale: float, linear_clamp_scale: Optional[float]):
    """SiTU-GLU: ``s_g * tanh(y1 / s_g) * sigmoid(y1) * y2``, with ``y2`` optionally clamped."""
    dtype = y.dtype
    y_1, y_2 = torch.chunk(y, 2, -1)
    c, _ = _tanh_clamp_and_deriv(y_1, gate_clamp_scale)
    gate = c * torch.sigmoid(y_1)
    if linear_clamp_scale is not None:
        y_2, _ = _tanh_clamp_and_deriv(y_2, linear_clamp_scale)
    return (gate * y_2).to(dtype)


@jit_fuser
def bias_situ_glu(
    y, bias, gate_clamp_scale: float, linear_clamp_scale: Optional[float]
):
    """Bias addition followed by SiTU-GLU. The clamps act on the full pre-activation."""
    return situ_glu(y + bias, gate_clamp_scale, linear_clamp_scale)


def _vendored_bias_swiglu_impl(
    input,
    bias,
    fp8_input_store=False,
    cpu_offload_input=False,
    clamp_value=None,
    gate_clamp_scale=None,
    linear_clamp_scale=None,
):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    assert gate_clamp_scale is not None or linear_clamp_scale is None
    assert gate_clamp_scale is None or clamp_value is None
    input = input.view(-1, ori_shape[-1])
    if gate_clamp_scale is not None:
        if bias is None:
            out = situ_glu(input, gate_clamp_scale, linear_clamp_scale)
        else:
            out = bias_situ_glu(input, bias, gate_clamp_scale, linear_clamp_scale)
    elif clamp_value is not None and clamp_value > 0:
        if bias is None:
            out = clamped_swiglu(input, clamp_value)
        else:
            out = bias_clamped_swiglu(input, bias, clamp_value)
    elif bias is None:
        out = swiglu(input)
    else:
        out = bias_swiglu(input, bias)
    return out if len(ori_shape) == 2 else out.view(ori_shape[0], ori_shape[1], -1)


def _load_baseline():
    """Installed Megatron when it has the clamp/SiTU arguments, else vendored."""
    version = _version_tuple(_MCORE_VERSION)
    if version is not None and version >= _MCORE_MIN_VERSION:
        try:
            from megatron.core.fusions.fused_bias_swiglu import (
                bias_swiglu_impl as installed,
            )
        except ImportError as exc:
            reason = f"megatron-core {_MCORE_VERSION} cannot import it: {exc}"
        else:
            return installed, f"megatron-core {_MCORE_VERSION}"
    elif _MCORE_VERSION == "not installed":
        reason = "Megatron-LM is not installed"
    else:
        reason = (
            f"megatron-core {_MCORE_VERSION} predates the clamp/SiTU arguments "
            f"(need >= {_MCORE_MIN_VERSION[0]}.{_MCORE_MIN_VERSION[1]})"
        )

    warnings.warn(
        f"{reason}; using the vendored main-branch implementation as the "
        "benchmark baseline",
        RuntimeWarning,
        stacklevel=2,
    )
    return _vendored_bias_swiglu_impl, "vendored main-branch implementation"


_BASELINE_IMPL, _BASELINE = _load_baseline()
