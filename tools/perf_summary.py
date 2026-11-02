#!/usr/bin/env python3
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
"""Summarize benchmark JSON into per-dtype and per-op speedups.

Mirrors FlagGems-vllm's tools/run_tests.py::parse_perf_data and
tools/psum_text::parse_lines:

  * per shape: ``compared_speedup`` when present (the vs-Base value), else the
    local torch-baseline ``speedup``. An operator whose benchmark declares a
    native baseline (``base_data_ignored`` in its JSON) never carries
    ``compared_speedup``, so it is summarized by the local ``speedup``;
  * per dtype: arithmetic mean over shapes that carry a numeric value
    (``None`` is excluded, ``0.0`` is kept);
  * per op: arithmetic mean over dtypes that carry a value.

Input is the file written by:

    pytest benchmark/<component>/test_<op>.py --record json \
        --output benchmark_<op>.json [--base-data=<path>]

Outputs ``summary.json``, ``summary.csv`` and ``summary.md`` under
``--output-dir``. With ``--min-speedup`` the command exits non-zero when an
operator's PerfAvg is below the threshold (no gate by default, like upstream).
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DTYPE_DISPLAY = {
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.float64": "fp64",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
    "torch.float8_e4m3fn": "fp8_e4m3",
    "torch.float8_e5m2": "fp8_e5m2",
}

DTYPE_ORDER = [
    "fp16",
    "bf16",
    "fp32",
    "fp64",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
]


def display_dtype(dtype) -> str:
    name = str(dtype)
    return DTYPE_DISPLAY.get(name, name.replace("torch.", ""))


def numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_benchmark_data(raw: dict) -> dict:
    """Return ``{op: record}`` from one benchmark JSON payload."""
    parsed = {}
    for op, bucket in (raw or {}).items():
        if op.startswith("_") or not isinstance(bucket, dict):
            continue
        status = str(bucket.get("result", "NotFound")).title()
        if status in ("Failed", "Skipped"):
            parsed[op] = {
                "status": status,
                "reason": bucket.get("reason", "Unknown"),
                "data": {},
                "perf_avg": None,
                "base_miss_reasons": [],
                "base_data_used": False,
                "base_data_ignored": False,
                "matched_shapes": None,
                "missed_shapes": None,
            }
            continue

        data = {}
        miss_reasons = []
        seen_reasons = set()
        matched = 0
        missed = 0
        base_data_used = False
        base_data_ignored = False

        for detail in bucket.get("details") or []:
            if not isinstance(detail, dict):
                continue
            dtype = display_dtype(detail.get("dtype", "unknown"))
            if detail.get("base_data_ignored"):
                base_data_ignored = True
            details = {}
            total = 0.0
            count = 0
            for row in detail.get("result") or []:
                shape = str(row.get("shape_detail", "Unknown")).replace(" ", "")
                entry = {
                    "base": row.get("latency_base", 0.0),
                    "gems": row.get("latency", 0.0),
                    # Default matches upstream: a missing key counts as 0.0,
                    # while an explicit None is excluded from the mean below.
                    "speedup": row.get("speedup", 0.0),
                }
                compared = row.get("compared_speedup")
                if compared is not None:
                    entry["speedup"] = compared
                    base_data_used = True
                    matched += 1
                miss = row.get("base_miss_reason")
                if miss:
                    entry["base_miss_reason"] = miss
                    base_data_used = True
                    missed += 1
                    if miss not in seen_reasons:
                        seen_reasons.add(miss)
                        miss_reasons.append(miss)

                details[shape] = entry
                if numeric(entry["speedup"]):
                    count += 1
                    total += entry["speedup"]

            data[dtype] = {
                "result": "OK" if count else "Unknown",
                "speedup": (total / count) if count else None,
                "details": details,
            }

        averages = [
            item["speedup"] for item in data.values() if numeric(item.get("speedup"))
        ]
        parsed[op] = {
            "status": status,
            "data": data,
            "perf_avg": (sum(averages) / len(averages)) if averages else None,
            "base_miss_reasons": miss_reasons,
            "base_data_used": base_data_used,
            "base_data_ignored": base_data_ignored,
            "matched_shapes": matched if base_data_used else None,
            "missed_shapes": missed if base_data_used else None,
        }
    return parsed


def summarize(parsed_by_file: list, inputs: list) -> dict:
    """Merge per-file results (last file wins for a duplicated operator)."""
    result = {}
    for parsed in parsed_by_file:
        for op, record in parsed.items():
            if op in result:
                print(
                    f"::warning:: operator '{op}' appears more than once; "
                    "the last input wins",
                    file=sys.stderr,
                )
            result[op] = record
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": [str(path) for path in inputs],
        "result": {
            op: {"performance": record} for op, record in sorted(result.items())
        },
    }


def dtype_columns(summary: dict) -> list:
    present = set()
    for entry in summary["result"].values():
        present.update(entry["performance"]["data"])
    ordered = [dtype for dtype in DTYPE_ORDER if dtype in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_json(path: Path, summary: dict) -> None:
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, summary: dict, columns: list) -> None:
    fields = [
        "op",
        "status",
        *columns,
        "PerfAvg",
        "matched_shapes",
        "missed_shapes",
        "base_miss_reasons",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for op, entry in summary["result"].items():
            perf = entry["performance"]
            row = {
                "op": op,
                "status": perf["status"],
                "PerfAvg": (
                    f"{perf['perf_avg']:.6f}" if perf["perf_avg"] is not None else ""
                ),
                "matched_shapes": (
                    "" if perf["matched_shapes"] is None else perf["matched_shapes"]
                ),
                "missed_shapes": (
                    "" if perf["missed_shapes"] is None else perf["missed_shapes"]
                ),
                "base_miss_reasons": " | ".join(perf["base_miss_reasons"]),
            }
            for dtype in columns:
                dtype_data = perf["data"].get(dtype)
                value = dtype_data["speedup"] if dtype_data else None
                row[dtype] = f"{value:.6f}" if value is not None else ""
            writer.writerow(row)


def write_markdown(path: Path, summary: dict, columns: list) -> None:
    header = ["op", "status", *columns, "PerfAvg", "matched", "missed"]
    lines = [
        "# Performance summary",
        "",
        f"- generated: {summary['generated_at']}",
        f"- inputs: {', '.join(summary['inputs'])}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for op, entry in summary["result"].items():
        perf = entry["performance"]
        cells = [op, perf["status"]]
        for dtype in columns:
            dtype_data = perf["data"].get(dtype)
            cells.append(fmt(dtype_data["speedup"]) if dtype_data else "-")
        cells += [
            fmt(perf["perf_avg"]),
            fmt(perf["matched_shapes"]),
            fmt(perf["missed_shapes"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    misses = [
        (op, reason)
        for op, entry in summary["result"].items()
        for reason in entry["performance"]["base_miss_reasons"]
    ]
    if misses:
        lines += ["", "## Baseline misses", ""]
        lines += [f"- {op}: {reason}" for op, reason in misses]

    ignored = [
        op
        for op, entry in summary["result"].items()
        if entry["performance"].get("base_data_ignored")
    ]
    if ignored:
        lines += ["", "## Base data ignored", ""]
        lines += [
            f"- {op}: the baseline is the native implementation, so PerfAvg is "
            "the local Train Speedup"
            for op in ignored
        ]

    lines += [
        "",
        "PerfAvg is the arithmetic mean of the per-dtype means; a shape uses",
        "`compared_speedup` when the baseline matched, otherwise the local",
        'torch-baseline `speedup`. Operators in "Base data ignored" never use',
        "`compared_speedup`: their baseline is the native implementation, so",
        "`--base-data` does not apply to them.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def gate_failures(summary: dict, min_speedup: float) -> list:
    failures = []
    for op, entry in summary["result"].items():
        perf = entry["performance"]
        value = perf["perf_avg"]
        if value is None:
            failures.append((op, None, "no numeric speedup"))
        elif value < min_speedup:
            failures.append((op, value, f"below --min-speedup {min_speedup}"))
    return failures


def collect_inputs(paths: list) -> list:
    files = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("benchmark_*.json")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"::warning:: input not found: {path}", file=sys.stderr)
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="benchmark JSON file or directory (repeatable; directories use benchmark_*.json)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help="Optional gate: exit non-zero when an operator's PerfAvg is below it",
    )
    args = parser.parse_args()
    if args.min_speedup is not None and args.min_speedup <= 0:
        parser.error("--min-speedup must be positive")
    return args


def main() -> None:
    args = parse_args()
    inputs = collect_inputs(args.input)
    if not inputs:
        raise SystemExit("no benchmark JSON inputs found")

    parsed_by_file = []
    for path in inputs:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"::warning:: cannot read {path}: {error}", file=sys.stderr)
            continue
        parsed_by_file.append(parse_benchmark_data(raw))

    if not parsed_by_file:
        raise SystemExit("no readable benchmark JSON inputs")

    summary = summarize(parsed_by_file, inputs)
    if not summary["result"]:
        raise SystemExit("no operators found in the inputs")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = dtype_columns(summary)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    write_json(json_path, summary)
    write_csv(csv_path, summary, columns)
    write_markdown(md_path, summary, columns)

    print(
        "op                         status   "
        + "  ".join(f"{d:>10}" for d in columns)
        + "   PerfAvg  matched  missed"
    )
    for op, entry in summary["result"].items():
        perf = entry["performance"]
        cells = []
        for dtype in columns:
            dtype_data = perf["data"].get(dtype)
            cells.append(fmt(dtype_data["speedup"]) if dtype_data else "-")
        print(
            f"{op:<26} {perf['status']:<8} "
            + "  ".join(f"{cell:>10}" for cell in cells)
            + f"   {fmt(perf['perf_avg']):>7}"
            + f"  {fmt(perf['matched_shapes']):>7}"
            + f"  {fmt(perf['missed_shapes']):>6}"
        )

    print(f"wrote {json_path}", file=sys.stderr)
    print(f"wrote {csv_path}", file=sys.stderr)
    print(f"wrote {md_path}", file=sys.stderr)

    if args.min_speedup is not None:
        failures = gate_failures(summary, args.min_speedup)
        for op, value, reason in failures:
            print(f"::error::{op}: {value} ({reason})")
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
