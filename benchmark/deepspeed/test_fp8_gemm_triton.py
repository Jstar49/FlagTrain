# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark FP8 GEMM against DeepSpeed's native Triton kernel.

The native kernel is the only on-machine baseline. When it cannot be loaded,
or this shape's tiles do not fit, there is no baseline to time: an eager Torch
composition is a correctness reference, not a competitor, so ``torch_op`` is
``None`` and ``--base-data`` supplies the comparison.
"""

import functools
import math
import warnings

import pytest
import torch
from triton.runtime.errors import OutOfResources

import flag_train

from .. import base

OP_NAME = "fp8_gemm_triton"
DTYPES = [torch.bfloat16]
CORE_M = (1, 2, 4, 8, 16, 17, 32, 48, 64, 96, 128, 256, 512, 1024, 2048)
CORE_K = 4096
CORE_N = 8192
CORE_GROUP_SIZE = 128


def make_portable_inputs(m, k, n, dtype, group_size, device):
    """Generate deterministic encoded weights without a quantizer."""
    generator = torch.Generator(device="cpu").manual_seed(20260922)
    x = torch.randn(m, k, generator=generator).to(dtype)
    magnitude = torch.randint(32, 88, (k, n), generator=generator, dtype=torch.int32)
    sign = torch.randint(0, 2, (k, n), generator=generator, dtype=torch.int32)
    weight = (magnitude | (sign << 7)).to(torch.uint8)
    groups = (k * n + group_size - 1) // group_size
    scale = 0.125 + 0.125 * torch.rand(groups, 1, generator=generator)
    return x.to(device), weight.to(device), scale.to(device)


@functools.lru_cache(maxsize=1)
def _load_deepspeed_fp8_gemm():
    try:
        from deepspeed.ops.fp_quantizer.fp8_gemm_triton import matmul_fp8_triton
    except Exception as exc:
        warnings.warn(
            f"DeepSpeed FP8 GEMM could not be loaded: {exc!r}; no on-machine "
            f"baseline, --base-data is the only comparison.",
            stacklevel=2,
        )
        return None
    return matmul_fp8_triton


class Fp8GemmBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPE_DESC = "M, K, N"

    def set_shapes(self, shape_file=None):
        self.shapes = [(m, CORE_K, CORE_N) for m in CORE_M]

    def set_more_shapes(self):
        return []

    def _measure_input(self, inputs, case_id=None):
        args, kwargs = self.unpack_to_args_kwargs(inputs)
        upstream = _load_deepspeed_fp8_gemm()
        if upstream is not None:
            # Probe this exact shape outside timing; large tiles may not fit.
            try:
                upstream(*args, **kwargs)
            except (OutOfResources, NotImplementedError) as exc:
                warnings.warn(
                    f"DeepSpeed FP8 GEMM unavailable for {tuple(args[0].shape)}: "
                    f"{exc!r}; no on-machine baseline for this case.",
                    stacklevel=2,
                )
                upstream = None
        # The row's on-machine baseline. The harness reads it for the
        # latency-base timing and for Method A's "no baseline at all" check;
        # ``None`` leaves --base-data as the only comparison.
        self.torch_op = upstream
        reference = "DeepSpeed" if upstream is not None else "none (--base-data only)"
        print(f"FP8 GEMM baseline {tuple(args[0].shape)}: {reference}")

        metric = super()._measure_input(inputs, case_id=case_id)
        assert metric.error_msg is None, f"{case_id}: {metric.error_msg}"
        if metric.speedup is not None:
            label = case_id or metric.shape_detail
            assert (
                math.isfinite(metric.speedup) and metric.speedup > 0.8
            ), f"{label}: speedup {metric.speedup}x must be > 0.8x"
        return metric


def fp8_input_fn(shape, dtype, device):
    tensors = make_portable_inputs(*shape, dtype, CORE_GROUP_SIZE, device)
    yield (*tensors, CORE_GROUP_SIZE)


@pytest.mark.fp8_gemm_triton
@pytest.mark.skipif(not flag_train.runtime.device.support_bf16, reason="BF16 required")
def test_fp8_gemm_triton_perf():
    bench = Fp8GemmBenchmark(
        input_fn=fp8_input_fn,
        op_name=OP_NAME,
        # None: only DeepSpeed's native kernel is a baseline. Each row swaps in
        # the native op in ``_measure_input`` when this shape fits, and leaves
        # None otherwise so --base-data supplies the comparison.
        torch_op=None,
        dtypes=DTYPES,
    )
    bench.set_train(flag_train.deepspeed.fp8_gemm_triton)
    bench.run()
