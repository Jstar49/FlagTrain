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

"""Derive which operator tests/benchmarks a PR should run on the GPU.

This is the FlagTrain analogue of FlagGems' ``tools/test-op.sh`` classification
(the step that maps a diff onto the operator tests it touches), specialised for
FlagTrain's flat operator space. FlagGems fans a changed file out three ways --
``tests/test*.py`` -> functional, ``benchmark/test*`` -> perf, an ops impl file
-> a pytest marker selected with ``-m`` -- because its operators and their test
files do not share a name. Here they do, so the mapping collapses to a 1:1 rule
by (component, filename stem):

    src/flag_train/<component>/<op>.py                        -> component/op
    src/flag_train/runtime/backend/_<v>/ops/<component>/<op>.py -> component/op
    tests/<component>/test_<op>.py                            -> component/op
    benchmark/<component>/test_<op>.py                        -> component/op

A component is an operator namespace -- ``deepspeed``, ``megatron`` and
``transformer_engine`` today -- discovered as a package under
``src/flag_train/`` that has a matching ``tests/<component>/`` directory, so
adding a component needs no change here. Infrastructure packages (runtime,
utils, testing, flagtune) have no such test directory and are therefore never
mistaken for one.

Outputs (written to GITHUB_OUTPUT when present, otherwise printed):

    run_gpu       true|false   whether the GPU jobs should run at all
    test_targets  space-separated pytest paths, or __ALL__
    bench_targets space-separated pytest paths, or __ALL__ ("" = no perf for
                  the touched operators, so the perf stage is skipped)
    changed_ops   space-separated ``component/op`` ids, or "all" for a full run
    tests_only    true|false   whether every non-doc changed file lives under
                  ``tests/``; the caller uses it to run just the nvidia
                  backend, because a test edit exercises the generic operator
                  and none of the vendor kernels

Decision matrix:

    workflow_dispatch / push / no diff computable  -> full run (__ALL__)
    docs-only change                               -> run_gpu=false
    one or more operators touched                  -> those operators only
    nothing maps to an operator                    -> run_gpu=false, so no
        backend runs: the same "nothing to run" outcome FlagGems' test-op.sh
        produces when no test file, benchmark or operator marker was selected

An ``__init__.py`` is the export surface, not an operator: on its own it maps
to no operator and selects nothing to run -- FlagGems' test-op.sh excludes
``__init__.py`` the same way and exits 0. It still takes part when it arrives
alongside a concrete operator file (the usual registration edit), because that
file maps to the operator by itself.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT / "src" / "flag_train"
TESTS_ROOT = ROOT / "tests"
BENCH_ROOT = ROOT / "benchmark"

DOC_RE = re.compile(r"\.(md|rst)$|^docs/")
GENERIC_OP_RE = re.compile(r"^src/flag_train/([^/]+)/([^/]+)\.py$")
VENDOR_OP_RE = re.compile(
    r"^src/flag_train/runtime/backend/_[^/]+/ops/([^/]+)/([^/]+)\.py$"
)
TEST_RE = re.compile(r"^tests/([^/]+)/test_(.+)\.py$")
BENCH_RE = re.compile(r"^benchmark/([^/]+)/test_(.+)\.py$")
# Any functional-test file, not just the deepspeed ones TEST_RE recognises, so
# the tests-only signal also covers tests/megatron/ and tests/transformer_engine/.
TESTS_ONLY_RE = re.compile(r"^tests/")


def components() -> list[str]:
    """Operator namespaces: packages with a matching ``tests/<name>/``."""
    return sorted(
        p.name
        for p in SRC_DIR.iterdir()
        if p.is_dir()
        and (p / "__init__.py").exists()
        and (TESTS_ROOT / p.name).is_dir()
    )


def operator_inventory() -> dict[str, list[str]]:
    """Map component -> the operators its generic implementations define.

    The generic implementations are the NVIDIA ones, and every operator there
    has a corresponding ``tests/<component>/test_<op>.py``; this stays in sync
    automatically as operators are added, so nothing here is hard-coded.
    """
    return {
        comp: sorted(
            p.stem for p in (SRC_DIR / comp).glob("*.py") if p.stem != "__init__"
        )
        for comp in components()
    }


def git_diff_files(base: str, head: str) -> list[str]:
    """Fall back to a three-dot git diff when no file list is supplied."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base}...{head}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError:
        return []


def existing_paths(ops: set[tuple[str, str]], kind: str) -> list[str]:
    """Relative pytest paths that exist for the given (component, op) pairs."""
    base = TESTS_ROOT if kind == "test" else BENCH_ROOT
    paths = []
    for comp, op in sorted(ops):
        path = base / comp / f"test_{op}.py"
        if path.exists():
            paths.append(str(path.relative_to(ROOT)))
        else:
            print(
                f"::warning::{kind} file {path.relative_to(ROOT)} for changed "
                f"operator '{comp}/{op}' does not exist; skipping it",
                file=sys.stderr,
            )
    return paths


def is_tests_only(changed: list[str]) -> bool:
    """True when every non-doc changed file lives under ``tests/``.

    A test-only edit exercises the generic (nvidia) operator -- the vendor
    overrides are untouched and each vendor's own test run adds no signal -- so
    the caller uses this to collapse the backend matrix to nvidia only.
    """
    non_doc = [p for p in changed if not DOC_RE.search(p)]
    return bool(non_doc) and all(TESTS_ONLY_RE.match(p) for p in non_doc)


def decide(
    event: str, changed: list[str], inventory: dict[str, list[str]]
) -> tuple[str, str, str, str]:
    """Return (run_gpu, test_targets, bench_targets, changed_ops)."""
    # Anything that is not a pull request has no base to scope against, and a
    # merge to main/dev must exercise every operator anyway: run everything.
    if event != "pull_request":
        return "true", "__ALL__", "__ALL__", "all"

    # No diff computed: fail open, run everything.
    if not changed:
        return "true", "__ALL__", "__ALL__", "all"

    ops: set[tuple[str, str]] = set()
    docs = 0

    for path in changed:
        if DOC_RE.search(path):
            docs += 1
            continue

        m = GENERIC_OP_RE.match(path)
        if m and m.group(1) in inventory:
            comp, stem = m.group(1), m.group(2)
        else:
            m = VENDOR_OP_RE.match(path) or TEST_RE.match(path) or BENCH_RE.match(path)
            if not m:
                # Infrastructure (workflow, docker, tools, runtime,
                # backends.yaml, pytest.ini, ...), an unrecognised path, or a
                # file in a package that is not an operator component: it maps
                # to no operator, so it selects nothing to run.
                continue
            comp, stem = m.group(1), m.group(2)

        # An __init__ is the export surface, not an operator: on its own it
        # selects nothing. When it accompanies an operator file (the usual
        # registration edit), that file already put the operator in `ops`.
        if stem == "__init__":
            continue
        ops.add((comp, stem))

    # A change whose non-doc files are all absent -> docs-only -> skip.
    if len(changed) - docs == 0:
        return "false", "", "", ""

    # Nothing maps to a concrete operator (infrastructure-only change, an
    # __init__-only edit, or the empty package scaffolding a new component
    # starts with): run nothing. FlagGems' test-op.sh does the same -- it exits
    # 0 when it selected no test, benchmark or operator marker.
    if not ops:
        return "false", "", "", ""

    test_targets = " ".join(existing_paths(ops, "test"))
    bench_targets = " ".join(existing_paths(ops, "bench"))
    if not test_targets:
        # The touched operators have no functional test file (yet), so there is
        # nothing to scope to. Running everything instead would contradict the
        # mapping; do what FlagGems does when no test matches and run nothing.
        return "false", "", "", ""

    changed_ops = " ".join(f"{comp}/{op}" for comp, op in sorted(ops))
    return "true", test_targets, bench_targets, changed_ops


def emit(name: str, value: str) -> None:
    out_file = os.environ.get("GITHUB_OUTPUT")
    if out_file:
        with open(out_file, "a") as f:
            f.write(f"{name}={value}\n")
    print(f"{name}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-name", default="pull_request", help="GitHub event name"
    )
    parser.add_argument(
        "--changed-files",
        default="",
        help="Space/newline-separated changed file paths (from tj-actions/changed-files).",
    )
    parser.add_argument(
        "--base", default="main", help="Base ref for the git-diff fallback"
    )
    parser.add_argument(
        "--head", default="HEAD", help="Head ref for the git-diff fallback"
    )
    args = parser.parse_args()

    changed = [f for f in args.changed_files.split() if f.strip()]
    if not changed:
        changed = git_diff_files(args.base, args.head)

    run_gpu, test_targets, bench_targets, changed_ops = decide(
        args.event_name, changed, operator_inventory()
    )
    tests_only = is_tests_only(changed)

    print(f"event={args.event_name}")
    print(f"changed files ({len(changed)}): {', '.join(changed[:10]) or '<none>'}")
    print(f"run_gpu={run_gpu}")
    print(f"test_targets={test_targets}")
    print(f"bench_targets={bench_targets}")
    print(f"changed_ops={changed_ops}")
    print(f"tests_only={tests_only}")

    emit("run_gpu", run_gpu)
    emit("test_targets", test_targets)
    emit("bench_targets", bench_targets)
    emit("changed_ops", changed_ops)
    emit("tests_only", "true" if tests_only else "false")


if __name__ == "__main__":
    main()
