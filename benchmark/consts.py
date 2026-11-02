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

import itertools
import os
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import List, Optional, Tuple

import torch

import flag_train

# Re-export the case types alongside the existing benchmark report types.
from .cases import BenchmarkCaseList, BenchmarkCasePlan, BenchmarkCaseSpec  # noqa: F401

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]
BOOL_DTYPES = [torch.bool]
COMPLEX_DTYPES = [torch.complex64]
EXTRA_INT_DTYPES = [torch.int8, torch.uint8, torch.int64]


def get_fp8_dtype():
    if flag_train.device != "cuda" or not torch.cuda.is_available():
        return None

    major, _ = torch.cuda.get_device_capability()

    if major > 8 and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn

    if major == 8 and hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2

    return None


FP8_DTYPES = [get_fp8_dtype()]

DEFAULT_WARMUP_TIME = 1000
DEFAULT_ITER_TIME = 100

# Size of the L2 flush ``triton.testing.do_bench`` runs before each timed
# iteration.  The flush is what keeps one iteration's data out of the next one's
# cache, but it is also what hides host-side dispatch: the timed window is a
# GPU-stream interval, so dispatch that outlives the flush shows up as GPU idle
# time inside it.  The driver hardcodes 256 MB, which on some devices masks only
# ~140-200us -- less than a Triton op launched through ``LibEntry`` costs on the
# host -- and the reading then reports dispatch, not kernel time.  Readings
# converge once the flush is grown past ~384 MB.  See tools/README.md.
DEFAULT_L2_FLUSH_MB = int(os.environ.get("FLAG_TRAIN_L2_FLUSH_MB", "384"))

# LEGACY_SHAPES are maintained for legacy benchmark SIZE settings and may be removed in the future.
# Do not reference this elsewhere.
LEGACY_SHAPES = [i * 64 for i in range(1, 22, 5)]
LEGACY_NON_BLAS_SHAPES = [(1024, shape) for shape in LEGACY_SHAPES]
LEGACY_BLAS_SHAPES = [(16, shape, shape, shape) for shape in LEGACY_SHAPES]

# Default shapes settings
DEFAULT_SHAPES = [
    (1024 * 1024 * 1024,),  # from perf
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),  # from perf
]


def model_shapes():
    # batch sizes * seq lengths
    BS = [1, 2, 3, 4, 8, 98, 256, 8192]
    # attn: wqkv, wo; ffn: w13, w2
    NK = [
        # extract from llama3-8b
        (1024, 4096),
        (128256, 4096),
        (14336, 4096),
        (4096, 14336),
        (4096, 4096),
        (6144, 4096),
        (28672, 4096),
        # extract from qwen2.5-7b
        (3584, 3584),
        (18944, 3584),
        (3584, 18944),
        (152064, 3584),
        (37888, 3584),
        (512, 3584),
        (4608, 3584),
    ]

    return [(4, bs, n, k) for bs, (n, k) in itertools.product(BS, NK)]


@dataclass
class BenchmarkMetrics:
    case_id: Optional[str] = None
    # Legacy shape information for backward compatibility
    # This field corresponds to the 'size' field in the previous version's benchmark.
    legacy_shape: Optional[int] = None
    # Detailed size info
    shape_detail: Optional[Tuple[int, ...]] = None
    # Latency base in ms
    latency_base: Optional[float] = None
    # Latency in ms
    latency: Optional[float] = None
    gbps_base: Optional[float] = None
    gbps: Optional[float] = None
    # Speedup over baseline
    speedup: Optional[float] = None
    # Accuracy over baseline (not implemented yet)
    accuracy: Optional[float] = None
    # TFLOPS (not implemented yet)
    tflops: Optional[float] = None
    # Utilization (not implemented yet)
    utilization: Optional[float] = None
    # Speedup compared to base data
    compared_speedup: Optional[float] = None
    # Why the baseline comparison was skipped (op/shape/dtype missing).
    # None means "matched" or "--base-data not enabled". Excluded from metrics.
    base_miss_reason: Optional[str] = None
    # Error message
    error_msg: Optional[str] = None


ALL_AVAILABLE_METRICS = set(map(lambda x: x.name, fields(BenchmarkMetrics))) - {
    "case_id",
    "legacy_shape",
    "shape_detail",
    "base_miss_reason",
}

DEFAULT_METRICS = [
    metric
    for metric in ["latency_base", "latency", "speedup"]
    if metric in ALL_AVAILABLE_METRICS
]


def check_metric_dependencies(
    requested_metrics: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Checks if the requested metrics satisfy their dependencies.
    Returns True if the dependencies are satisfied, otherwise False.
    """
    # Predefined dependencies between metrics
    buildin_dependencies = {
        "speedup": ["latency", "latency_base"],
        "utilization": ["latency", "tflops"],
    }
    unsatisfied_metrics = []
    if requested_metrics is None:
        return unsatisfied_metrics

    satisfied_metrics = set()
    for metric in requested_metrics:
        if metric not in buildin_dependencies:
            # If the metric has no dependencies, it's automatically satisfied
            satisfied_metrics.add(metric)
        else:
            required_metrics = buildin_dependencies[metric]
            # Check if all dependencies are in the satisfied metrics list
            if not all(req in satisfied_metrics for req in required_metrics):
                unsatisfied_metrics.append(metric)
            else:
                satisfied_metrics.add(metric)
    return unsatisfied_metrics


def get_recommended_shapes(
    op_name: str, op_specified_shapes: Optional[List[Tuple[int, ...]]]
):
    def _shapes_sort(shapes):
        shapes = [shape if isinstance(shape, tuple) else (shape,) for shape in shapes]
        return sorted(shapes, key=lambda x: torch.tensor(x).prod().item())

    if op_specified_shapes:
        # TODO: handle situation that list as the basic element in shape.
        return _shapes_sort(op_specified_shapes)
    return _shapes_sort(DEFAULT_SHAPES)


class BenchMode(Enum):
    KERNEL = "kernel"
    OPERATOR = "operator"
    WRAPPER = "wrapper"
    CUDAGRAPH = "cudagraph"


class BenchLevel(Enum):
    COMPREHENSIVE = "comprehensive"
    CORE = "core"


@dataclass
class OperationAttribute:
    op_name: str
    # Recommended core benchmark shapes for the given operation
    recommended_core_shapes: List[Tuple[int, ...]]
    shape_desc: str

    def __str__(self) -> str:
        return (
            f"{'Operator name':<40} |  {self.op_name}\n"
            f"{'Recommended Core Shapes[' + self.shape_desc + ']':<40} |  {self.recommended_core_shapes}\n"
        )

    def to_dict(self) -> dict:
        return self.__dict__


def custom_json_encoder(obj):
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# ---- NVIDIA baseline lookup ("vs Base" column) --------------------------------
# Ported from FlagGems-vllm's benchmark/consts.py. The baseline JSON (produced by
# flagos-ops-baseline) records, per (op, shape, dtype), the latency measured on a
# reference NVIDIA chip plus the bottleneck resource. This block looks a record
# up and supplies the peak scaling factor used to normalize the raw time ratio:
#
#     compared_speedup = (base_ms / train_ms) / factor
#     factor           = vendor_peak / reference_peak
#
# so the result is the training implementation's achievement relative to its own
# hardware's due share.

# torch dtype string -> scaling-factor dtype key.
_DTYPE_TO_SF_KEY = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float64": "fp64",
    "torch.int8": "int8",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e5m2": "fp8",
}

# baseline bottleneck unit -> scaling-factor resource group.
_BOTTLENECK_TO_SF = {"mem": "bandwidth_gbps", "tensor": "tensor", "cuda": "vector"}


def _first_tensor_shape(shape_detail):
    """Return the first tensor shape in a record_shapes() result as a list.

    record_shapes() encodes tensors as torch.Size and may nest them inside
    lists/tuples/dicts (and wrap args+kwargs in a top-level tuple), so walk the
    structure depth-first and return the first torch.Size we hit.
    """
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            return list(item)
        if isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
    return None


def _all_tensor_shapes_from_record(shape_detail):
    """Collect every tensor shape into a sorted multiset.

    Used as a fallback signature when the baseline's key_shape is a semantic
    composite key that does not match the first tensor's shape.
    """
    shapes = []
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            shapes.append(tuple(item))
        elif isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
    return tuple(sorted(shapes))


def _all_scalars_from_record(shape_detail):
    """Collect scalar values from record_shapes() into a sorted tuple.

    Only used for the rare case of two baseline shapes sharing a tensor
    signature; bool must be tested before int (isinstance(True, int)).
    """
    scalars = []
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            continue
        if isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
        elif isinstance(item, bool):
            scalars.append(repr(item))
        elif isinstance(item, (int, float, str)):
            scalars.append(repr(item))
    return tuple(sorted(scalars))


def _base_shapes_from_config(cfg):
    """Collect tensor shapes from a baseline record's config.inputs."""
    if not isinstance(cfg, dict):
        return None
    inputs = cfg.get("inputs")
    if not isinstance(inputs, dict):
        return None
    shapes = []
    for value in inputs.values():
        if isinstance(value, dict) and isinstance(value.get("shape"), list):
            shapes.append(tuple(value["shape"]))
    return tuple(sorted(shapes))


def _base_scalars_from_config(cfg):
    """Collect scalar values from a baseline record's config.inputs."""
    scalars = []
    inputs = (cfg or {}).get("inputs", {})
    if isinstance(inputs, dict):
        for value in inputs.values():
            if isinstance(value, dict) and "scalar" in value:
                scalars.append(repr(value["scalar"]))
    return tuple(sorted(scalars))


# Per (base_data, op_name) tensor-signature -> shape-key index, keyed by the
# base_data object's id so the index is rebuilt only when the baseline changes.
_SIG_INDEX_CACHE = {}


def _resolve_op_entry(base_data, op_name, variant=None):
    """Return the baseline entry for ``op_name``.

    An op may keep one sub-entry per precision under ``variants`` (e.g. fp8 /
    int8); the benchmark selects it with ``variant``.
    """
    op_entry = base_data.get(op_name)
    if variant is not None and isinstance(op_entry, dict) and "variants" in op_entry:
        return op_entry["variants"].get(variant)
    return op_entry


def _build_signature_index(base_data, op_name, variant=None):
    """Build {dtype: {tensor_sig: [shape_key, ...]}} for one op."""
    cache_key = (id(base_data), op_name, variant)
    cached = _SIG_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index = {}
    op_entry = _resolve_op_entry(base_data, op_name, variant) or {}
    for shape_key, per_dtype in op_entry.get("shapes", {}).items():
        if not isinstance(per_dtype, dict):
            continue
        for dtype_str, rec in per_dtype.items():
            if not isinstance(rec, dict):
                continue
            sig = _base_shapes_from_config(rec.get("config"))
            if not sig:
                continue
            index.setdefault(dtype_str, {}).setdefault(sig, []).append(shape_key)
    _SIG_INDEX_CACHE[cache_key] = index
    return index


def _match_by_signature(base_data, op_name, dtype, shape_detail, variant=None):
    """Fallback match by the multiset of tensor shapes, then by scalars."""
    index = _build_signature_index(base_data, op_name, variant)
    per_dtype = index.get(str(dtype))
    if not per_dtype:
        return None, None
    sig = _all_tensor_shapes_from_record(shape_detail)
    keys = per_dtype.get(sig)
    if not keys:
        return None, None
    op_shapes = _resolve_op_entry(base_data, op_name, variant)["shapes"]
    if len(keys) == 1:
        return op_shapes[keys[0]].get(str(dtype)), keys[0]
    # Tensor-signature collision: discriminate with the scalar set.
    rec_scalars = _all_scalars_from_record(shape_detail)
    for key in keys:
        rec = op_shapes[key].get(str(dtype))
        if rec is None:
            continue
        base_scalars = _base_scalars_from_config(rec.get("config"))
        # The baseline's scalars are normally a subset of the record's (which
        # may carry extra ints such as strides).
        if base_scalars and set(base_scalars) <= set(rec_scalars):
            return rec, key
    return None, None


def lookup_base_record_ex(base_data, op_name, dtype, shape_detail, variant=None):
    """Look up the NVIDIA baseline record: op_name -> shape -> dtype.

    Returns ``(record, None)`` on success and ``(None, reason)`` when a level is
    missing, naming exactly which level failed. Never raises.
    """
    try:
        op_entry = _resolve_op_entry(base_data, op_name, variant)
        if not op_entry:
            return None, f"baseline has no operator '{op_name}'"
        shapes = op_entry.get("shapes", {})
        first_shape = _first_tensor_shape(shape_detail)
        if first_shape is None:
            return None, (
                f"operator '{op_name}' shape cannot be expressed as tensor dims "
                f"(shape_detail={shape_detail})"
            )
        # Fast path: the shape key is usually the first tensor's dimensions.
        per_dtype = shapes.get(str(first_shape))
        if per_dtype:
            rec = per_dtype.get(str(dtype))
            if rec is not None:
                return rec, None
        # Fallback: semantic key / composite shape, matched by tensor signature.
        rec, matched_key = _match_by_signature(
            base_data, op_name, dtype, shape_detail, variant
        )
        if rec is not None:
            return rec, None
        if (
            str(first_shape) in shapes
            and shapes[str(first_shape)].get(str(dtype)) is None
        ):
            return None, (
                f"operator '{op_name}' shape {first_shape} has no dtype {dtype} "
                "in the baseline"
            )
        return None, (
            f"operator '{op_name}' has no matching shape in the baseline "
            f"(first tensor={first_shape}, signature fallback tried)"
        )
    except (AttributeError, TypeError) as error:
        return None, f"baseline lookup for '{op_name}' failed: {error}"


def lookup_base_record(base_data, op_name, dtype, shape_detail, variant=None):
    """Backward-compatible wrapper returning just the record (or None)."""
    rec, _ = lookup_base_record_ex(base_data, op_name, dtype, shape_detail, variant)
    return rec


def _detect_self_chip():
    """Best-effort name of the chip this benchmark runs on.

    FlagGems-vllm uses torch.cuda.get_device_name(); FlagTrain supports several
    vendors, so read the property the runtime actually exposes (dict or object,
    with different attribute spellings).
    """
    try:
        props = flag_train.runtime.torch_device_fn.get_device_properties(0)
    except Exception:
        return ""
    if isinstance(props, dict):
        for key in ("name", "Name", "device_name"):
            if props.get(key):
                return str(props[key])
        return ""
    for attribute in ("name", "Name", "device_name"):
        value = getattr(props, attribute, None)
        if value:
            return str(value)
    return ""


_SELF_CHIP = None


def _self_chip():
    global _SELF_CHIP
    if _SELF_CHIP is None:
        _SELF_CHIP = _detect_self_chip()
    return _SELF_CHIP


def _chip_matches(chip_name, device_name):
    """True if a spec's chip name identifies this device (substring match)."""
    if not chip_name or not device_name:
        return False
    return chip_name.lower() in device_name.lower()


def lookup_scaling_factor(base_data, vendor, bottle_neck_unit, dtype, self_chip=None):
    """Return this chip's peak scaling factor (vendor_peak / reference_peak).

    The factor is selected for the resource that bottlenecks the baseline record
    and for the case's dtype. ``measured`` values are preferred over ``nominal``.
    Returns None when the vendor / chip / resource / dtype is not covered, in
    which case the caller keeps the raw latency ratio.
    """
    try:
        sf = base_data.get("_scaling_factors")
        resource = _BOTTLENECK_TO_SF.get(bottle_neck_unit)
        if not sf or resource is None:
            return None
        if self_chip is None:
            self_chip = _self_chip()
        candidates = [c for c in sf.get("chips", []) if c.get("vendor") == vendor]
        if not candidates:
            return None
        chip = next(
            (c for c in candidates if _chip_matches(c.get("chip"), self_chip)), None
        )
        if chip is None:
            # A vendor with a single chip model may be matched without the model.
            chip = candidates[0] if len(candidates) == 1 else None
        if chip is None:
            return None
        for prefer in ("measured", "nominal"):
            block = chip.get(prefer)
            if not block:
                continue
            if resource == "bandwidth_gbps":
                value = block.get("bandwidth_gbps")
            else:
                value = block.get(resource, {}).get(_DTYPE_TO_SF_KEY.get(str(dtype)))
            if value:
                return value
        return None
    except (AttributeError, TypeError):
        return None


@dataclass
class BenchmarkResult:
    """Record the benchmark result for each operator."""

    # Unique name of the operator
    op_name: str
    dtype: str
    mode: str
    level: str
    # Benchmark results
    result: List[BenchmarkMetrics]

    def __str__(self) -> str:
        header_title = (
            f"\nOperator: {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode},"
            f"level={self.level})\n"
        )
        native_baseline_skip_reason = getattr(self, "native_baseline_skip_reason", None)
        if native_baseline_skip_reason:
            header_title += f"Native baseline: N/A ({native_baseline_skip_reason})\n"
        base_data_ignored = getattr(self, "base_data_ignored", None)
        if base_data_ignored:
            header_title += f"Base data: ignored ({base_data_ignored})\n"
        show_vs_base = any(m.compared_speedup is not None for m in self.result)
        col_names = [
            f"{'Status':<10}",
            f"{'Torch Latency (ms)':>20}",
            f"{'Train Latency (ms)':>20}",
            f"{'Train Speedup':>20}",
        ]
        if show_vs_base:
            col_names.append(f"{'vs Base':>20}")
        if self.result[0].tflops and self.result[0].tflops != 0.0:
            col_names.append(f"{'TFLOPS':>20}")
        if self.result[0].gbps is not None:
            col_names.append(f"{'Torch GBPS ':>20}")
            col_names.append(f"{'Train GBPS ':>20}")
        col_names.append(f"{'Size Detail':>20}\n")
        header_col_names = " ".join(col_names)
        header_break = "-" * len(header_col_names) + "\n"
        header = header_title + header_col_names + header_break

        metrics_lines = "".join(
            self._format_metrics(ele, show_vs_base) for ele in self.result
        )
        return header + metrics_lines

    def _format_metrics(
        self, metrics: BenchmarkMetrics, show_vs_base: bool = False
    ) -> str:
        # self.gen_legacy_shape(metrics)
        # legacy_shape_str = (
        #     metrics.legacy_shape if metrics.legacy_shape is not None else "N/A"
        # )
        latency_base_str = (
            f"{metrics.latency_base:.6f}" if metrics.latency_base is not None else "N/A"
        )
        latency_str = f"{metrics.latency:.6f}" if metrics.latency is not None else "N/A"
        speedup_str = f"{metrics.speedup:.3f}" if metrics.speedup is not None else "N/A"
        compared_speedup_str = (
            f"{metrics.compared_speedup:.3f}"
            if metrics.compared_speedup is not None
            else "N/A"
        )
        torch_gbps_str = (
            f"{metrics.gbps_base:.3f}" if metrics.gbps_base is not None else "N/A"
        )
        train_gbps_str = f"{metrics.gbps:.3f}" if metrics.gbps is not None else "N/A"
        if metrics.tflops and metrics.tflops != 0.0:
            tflops_str = (
                f"{metrics.tflops:.3f}" if metrics.tflops is not None else "N/A"
            )
        shape_detail_str = (
            metrics.shape_detail if metrics.shape_detail is not None else "N/A"
        )
        status = "SUCCESS" if metrics.error_msg is None else "FAILED"
        data_line = (
            f"{status:<10}"
            f"{latency_base_str:>20}"
            f"{latency_str:>20}"
            f"{speedup_str:>20}"
        )
        if show_vs_base:
            data_line += f"{compared_speedup_str:>20}"
        if metrics.tflops and metrics.tflops != 0.0:
            data_line += f"{tflops_str:>20}"
        if metrics.gbps is not None:
            data_line += f"{torch_gbps_str:>20}{train_gbps_str:>20}"
        data_line += " " * 10
        data_line += f"{shape_detail_str}\n"
        return data_line

    def gen_legacy_shape(self, metrics: BenchmarkMetrics) -> Optional[int]:
        first_shape = (
            metrics.shape_detail[0] if isinstance(metrics.shape_detail, list) else None
        )
        to_record_shape = (
            tuple(first_shape) if isinstance(first_shape, torch.Size) else None
        )

        if to_record_shape in LEGACY_NON_BLAS_SHAPES:
            metrics.legacy_shape = to_record_shape[-1]
        elif (
            isinstance(to_record_shape, tuple)
            and len(to_record_shape) == 2
            and to_record_shape[0] == 1024
        ):
            metrics.legacy_shape = to_record_shape[-1]
        else:
            metrics.legacy_shape = None

    def to_json(self) -> str:
        import json

        # Convert to dict and handle tuple serialization for shape_detail
        result_dict = asdict(self)
        native_baseline_skip_reason = getattr(self, "native_baseline_skip_reason", None)
        if native_baseline_skip_reason:
            result_dict["native_baseline_skip_reason"] = native_baseline_skip_reason
        base_data_ignored = getattr(self, "base_data_ignored", None)
        if base_data_ignored:
            result_dict["base_data_ignored"] = base_data_ignored
        return json.dumps(result_dict, default=custom_json_encoder)

    def to_dict(self) -> dict:
        return self.__dict__


# Subset dtypes for specific operators
FP16_BF16_DTYPES = [torch.float16, torch.bfloat16]
