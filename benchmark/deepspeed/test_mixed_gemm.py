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
"""Performance benchmark for the mixed_gemm operator.

Run ``tests/deepspeed/test_mixed_gemm.py`` green first, then::

    pytest benchmark/deepspeed/test_mixed_gemm.py -v -s

The baseline every run gets is ``_BoundReference`` -- dequantize the weight on device,
then cuBLAS -- working from the *logical* quantized matrix, so it does no layout
arithmetic and the dequantization stays inside the timed region. The operator is the
only side that has to get the packed layout right, so the ratio between them is a
statement about this kernel. Comparing against an unquantized ``torch.mm`` would put a
different amount of work on each side and needs its own benchmark.

The reference is a plain-torch composition, so it is available on every backend. A
DeepSpeed baseline that cannot run on this GPU (its prebuilt ``dskernels`` package
does not carry every compute capability, e.g. sm_90) must therefore not remove the
comparison: it only skips ``test_mixed_gemm_vs_deepspeed``, the separate
kernel-vs-kernel check, exactly as the correctness tests treat it.

``base.py`` hands the same args/kwargs to ``torch_op`` and to ``train_op``, so the two must
have identical signatures; ``_call`` is what guarantees that here.
"""

import pytest
import torch

import flag_train
from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn

from .. import base

# ``(m, k, n, num_bits)``. The bit width travels with the shape because the
# framework fixes its dtype list before the shapes are walked. Thin and past-L2
# are the two axes: thin is where quantization should win, past L2 is where the
# saved bandwidth can first turn into saved time.
_SHAPES = [
    # Thin: few tokens, wide weight -- the decoding shape, where the weight dominates traffic.
    (1, 4096, 4096, 4),
    (64, 4096, 4096, 4),
    (1, 4096, 4096, 8),
    (64, 4096, 4096, 8),
    # Square, walking past L2: fp16 weights are 2 MiB at 1024 cubed, 8 MiB at 2048 and
    # 32 MiB at 4096, so a part with a 50 MB L2 stops fitting near 3072.
    (1024, 1024, 1024, 4),
    (2048, 2048, 2048, 4),
    (4096, 4096, 4096, 4),
    (1024, 1024, 1024, 8),
    (2048, 2048, 2048, 8),
    (4096, 4096, 4096, 8),
    # Wide and deep: many tokens, a long reduction, 16384 output columns.
    (8192, 4096, 4096, 4),
    (2048, 4096, 16384, 4),
    (8192, 4096, 4096, 8),
    (2048, 4096, 16384, 8),
]

_ACT = 6  # IDENTITY: keep activation time out of the GEMM measurement

# Both: ``mixed_gemm.cu``'s ``ACT_DTYPE_SWITCH`` instantiates exactly these two.
_DTYPES = [torch.float16, torch.bfloat16]
_DTYPE = _DTYPES[0]  # the dtype the probe call and the byte table use


# ---------------------------------------------------------------------------
# Quantization
#
# ``weight`` reaches the operator already quantized *and* already run through
# ``preprocess_weights_for_mixed_gemm``; only the quantization half is reproduced
# here, because a timing does not need the layout. Arbitrary uint8 makes the kernel
# do the same work, and whether it reads the real layout is the tests' question.
# ---------------------------------------------------------------------------


def _round_half_away(x):
    """C++ ``std::round``: ties away from zero, unlike ``torch.round``."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def quantize(w, num_bits):
    """``[k, n]`` fp16 -> ``([k, n] signed int64, scales fp16)`` per output channel.

    ``symmetric_quantize``'s rules: per-column ``max|w| / 2**(bits-1)``, rounded half
    away from zero, clamped to the signed range. Verified bit-for-bit against
    DeepSpeed.
    """
    wf = w.float()
    per_col_max = wf.abs().amax(dim=0)
    scale32 = per_col_max / float(1 << (num_bits - 1))
    scales = scale32.to(torch.float16)

    safe = torch.where(scale32 > 0, scale32, torch.ones_like(scale32))
    q = _round_half_away(wf / safe[None, :])
    q = q.clamp(-(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1)
    return q.to(torch.int64), scales


def dequantize_logical(q, scales):
    """Logical quantized matrix -> the fp16 weight, for the reference baseline.

    The baseline never touches the packed buffer, so its timing is
    dequantize-plus-GEMM.
    """
    return q.to(scales.dtype) * scales[None, :]


# ---------------------------------------------------------------------------
# Baselines
#
# Same policy as the sibling benchmarks in this directory: the torch reference is
# always timed, so a machine that cannot host DeepSpeed still gets a full table, and
# a missing upstream kernel only skips the optional kernel-vs-kernel test.
#
# DeepSpeed's own mixed_gemm is compared separately in
# ``test_mixed_gemm_vs_deepspeed``. There both sides read the same preprocessed B
# layout, so the comparison is between kernels rather than between implementations of
# the contract.
# ---------------------------------------------------------------------------

# Backends on which DeepSpeed's operator is expected to be usable as the
# kernel-vs-kernel reference. ``mixed_gemm`` ships as a CUDA op, so only a backend
# that can compile and execute one can host it.
_DEEPSPEED_BASELINE_VENDORS = {"nvidia"}

_DEEPSPEED_UNAVAILABLE_MSG = (
    "Its kernel ships prebuilt in the `dskernels` package, and "
    "`InferenceCutlassBuilder` links `-ldeepspeedft` rather than compiling it, so that "
    "package has to carry this GPU's compute capability. Install one that does, or "
    "drop the backend from _DEEPSPEED_BASELINE_VENDORS."
)


def _load_deepspeed_mixed_gemm():
    """``(op, reason)`` -- ``op`` is ``None`` when there is not a usable one.

    The kernel is prebuilt, so a library without this GPU's compute capability
    imports cleanly and fails only when called; it is therefore called once here, on
    random bytes.
    """
    if flag_train.vendor_name not in _DEEPSPEED_BASELINE_VENDORS:
        return None, None

    try:
        from deepspeed.ops.op_builder import InferenceCutlassBuilder

        op = InferenceCutlassBuilder().load().mixed_gemm
    except Exception as exc:  # noqa: BLE001
        return None, f"could not be loaded: {exc!r}"

    m = k = n = 128
    try:
        hidden = torch.randn(m, k, dtype=_DTYPE, device=runtime_device.name)
        weight = torch.randint(
            0, 256, (k, n // 2), dtype=torch.uint8, device=runtime_device.name
        )
        scales = torch.rand(n, dtype=_DTYPE, device=runtime_device.name) + 0.1
        out = torch.empty(m, n, dtype=_DTYPE, device=runtime_device.name)
        op(out, hidden, weight, scales, None, 4, _ACT)
        torch_device_fn.synchronize()
    except Exception as exc:  # noqa: BLE001
        return None, f"loaded but does not run here: {exc!r}"

    return op, None


# Resolved once, at module import time.
_DEEPSPEED_ORACLE, _DEEPSPEED_REASON = _load_deepspeed_mixed_gemm()

# The baseline every run gets, on every backend. It never depends on the upstream
# kernel being loadable here.
_BASELINE = "torch reference (dequantize on device, then cuBLAS)"

_DEEPSPEED_SKIP_REASON = (
    f"DeepSpeed's mixed_gemm {_DEEPSPEED_REASON}. {_DEEPSPEED_UNAVAILABLE_MSG}"
    if _DEEPSPEED_ORACLE is None and _DEEPSPEED_REASON is not None
    else _DEEPSPEED_UNAVAILABLE_MSG
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def mixed_gemm_input_fn(shape, dtype, device):
    """The inputs every implementation shares.

    A dict **inside a tuple**: ``base`` iterates the tuple to split args from kwargs,
    and iterating a bare dict would yield its keys instead. The shape is quantized
    once per yield and ``output`` is reused, since the operator writes it in place.
    """
    m, k, n, num_bits = shape
    hidden = torch.randn(m, k, dtype=dtype, device=device)
    w = torch.randn(k, n, dtype=dtype, device=device)
    q, scales = quantize(w.cpu(), num_bits)
    # Only the buffer's length matters to a timing: the kernel reads it through its own
    # index arithmetic, and whether it reads the real layout is the tests' question.
    packed = torch.randint(
        0, 256, (k, n // (2 if num_bits == 4 else 1)), dtype=torch.uint8, device=device
    )
    scales = scales.to(device).to(dtype)
    output = torch.empty(m, n, dtype=dtype, device=device)

    # ``q`` is the logical matrix the baseline dequantizes -- what the buffer above would
    # hold if it had been built properly. Stored at the width it needs rather than at the
    # int64 ``quantize`` returns, so the baseline's fp16 conversion does not dominate it.
    yield (
        {
            "hidden": hidden,
            "weight": packed,
            "q": q.to(torch.int8).to(device),
            "scales": scales,
            "bias": None,
            "output": output,
            "num_bits": num_bits,
        },
    )


def _call(op, **kwargs):
    """Hand the inputs to an implementation in the operator's own order.

    Every implementation goes through here, which is what keeps their signatures
    identical.
    """
    return op(
        kwargs["output"],
        kwargs["hidden"],
        kwargs["weight"],
        kwargs["scales"],
        kwargs["bias"],
        kwargs["num_bits"],
        _ACT,
    )


# ---------------------------------------------------------------------------
# Baseline and the operator under test
# ---------------------------------------------------------------------------


class _BoundReference:
    """The baseline, holding the logical weight so its signature can stay the
    operator's.

    ``base`` hands one argument tuple to both ops, so binding the matrix to the object
    keeps the shared call identical. The dequantization stays inside the timed region:
    moving it out would compare a bare GEMM against dequantize-plus-GEMM.
    """

    def __init__(self, q):
        self.q = q

    def __call__(
        self, output, hidden_states, weight, scales, bias, num_bits, activation_raw
    ):
        torch.mm(hidden_states, dequantize_logical(self.q, scales), out=output)
        if bias is not None:
            output.add_(bias)
        return output


def torch_op(**kwargs):
    """The baseline: the torch reference, on every backend and every GPU.

    It works from the logical matrix, which arrives bound to the object because
    ``base`` passes one argument tuple to both ops. Keeping the baseline independent
    of DeepSpeed's prebuilt kernels is what lets the speedup column be populated on
    parts those kernels do not support; the kernel-vs-kernel comparison is a separate
    test.
    """
    return _call(_BoundReference(kwargs["q"]), **kwargs)


def train_op(**kwargs):
    """The operator under test."""
    return _call(flag_train.deepspeed.mixed_gemm, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark body
# ---------------------------------------------------------------------------


class MixedGemmBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _SHAPES
    DEFAULT_SHAPE_DESC = "m, k, n, num_bits"

    def set_shapes(self, shape_file=None):
        self.shapes = list(_SHAPES)

    def set_more_shapes(self):
        # The base class otherwise adds 3-D and 2**28-shaped cases; ours are given.
        return []


@pytest.mark.mixed_gemm
def test_mixed_gemm():
    """Time the operator against the baseline. See the module docstring."""
    # The torch reference is always available, so this table is complete on any
    # backend; only the DeepSpeed kernel comparison below can be skipped.
    print(f"\nBaseline: {_BASELINE}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bench = MixedGemmBenchmark(
        input_fn=mixed_gemm_input_fn,
        op_name="mixed_gemm",
        torch_op=torch_op,
        dtypes=_DTYPES,
    )
    bench.set_train(train_op)
    bench.run()
