#!/bin/bash

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

# Prepare one CI job's environment for a backend: apply the backend's
# environment, then install FlagTrain itself and the test dependencies.
#
# Usage: bash tools/setup_env.sh <backend> [--prebuilt|--no-prebuilt]
#
#   --prebuilt     (default) the vendor runtime image already ships torch and
#                  the triton stack, so only FlagTrain and the test deps are
#                  installed. Never install a vendor stack next to the one the
#                  image was built with.
#   --no-prebuilt  also install this backend's `deps:` from backends.yaml,
#                  through the vendor's package index -- for a vendor image
#                  that does not carry the FlagOS stack (thead's own pytorch
#                  image).
#
# Everything comes out of backends.yaml, the same file the nvidia runner image
# documents its pins in:
#   env:         exported through $GITHUB_ENV, so every later step sees it;
#   env_source:  vendor scripts sourced via tools/source_env_bash.sh, with only
#                the variables they changed exported (some vendor scripts
#                self-locate through ${BASH_SOURCE[0]} and misbehave when
#                sourced by a non-bash shell);
#   deps:        the --no-prebuilt install list.
# Outside CI (no $GITHUB_ENV) the same lines are printed as `export K=V`, so
# `eval "$(bash tools/setup_env.sh <backend>)"` works locally.
#
# The trailing import check is deliberate: `import flag_train` needs the
# platform's torch and triton backends, so a broken image or a wrong vendor
# shows up here rather than as a collection error inside pytest.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKENDS_YAML="${REPO_ROOT}/backends.yaml"

usage() {
    echo "usage: setup_env.sh <backend> [--prebuilt|--no-prebuilt]" >&2
    echo "known backends: $(python3 -c "import yaml,sys; print(' '.join(sorted(yaml.safe_load(open(sys.argv[1]))['backends'])))" "${BACKENDS_YAML}" 2>/dev/null || echo '<backends.yaml unreadable>')" >&2
    exit 2
}

BACKEND="${1:-}"
[ -n "${BACKEND}" ] || usage

# This script reads backends.yaml, so pyyaml must be importable by the python
# that runs it. Say so plainly instead of dying in a traceback -- on a native
# vendor runner the venv it installs into is created by the workflow, which
# installs pyyaml for this reason.
if ! python3 -c "import yaml" >/dev/null 2>&1; then
    echo "tools/setup_env.sh: pyyaml is missing from $(command -v python3); install it first (pip install pyyaml)" >&2
    exit 2
fi

PREBUILT=1
case "${2:-}" in
    "" | --prebuilt) PREBUILT=1 ;;
    --no-prebuilt) PREBUILT=0 ;;
    *) usage ;;
esac

emit_env() {
    # KEY=VALUE: applied to this shell AND handed to the steps after it.
    # Both are needed: the installs and the import check at the end of this
    # very step run here, and a vendor stack can read its environment at import
    # time -- torch_npu reads CANN's paths while torch imports it, and without
    # ASCEND_* it dies with a UnicodeDecodeError from a garbage path.
    export "$1"
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "$1" >>"${GITHUB_ENV}"
    else
        echo "export $1"
    fi
}

FLAGOS_PYPI=""
MIRROR=""
DEPS=""
COMPILER_PKG=""
ENV_SOURCES=""
sourced_env=""

while IFS=$'\t' read -r kind a b; do
    case "${kind}" in
        PYPI) FLAGOS_PYPI="${a}" ;;
        MIRROR) MIRROR="${a}" ;;
        DEP) DEPS="${DEPS} ${a}" ;;
        COMPILER) COMPILER_PKG="${a}" ;;
        SOURCE) ENV_SOURCES="${ENV_SOURCES} ${a}" ;;
        ENV)
            echo "backend env ${a}=${b}"
            emit_env "${a}=${b}"
            sourced_env="${sourced_env} ${a}=${b}"
            ;;
    esac
done < <(python3 - "${BACKENDS_YAML}" "${BACKEND}" <<'PY'
import os
import re
import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
backends = cfg["backends"]
key = sys.argv[2]
if key not in backends:
    sys.exit(
        f"tools/setup_env.sh: unknown backend {key!r}; "
        f"known: {', '.join(sorted(backends))}"
    )

backend = backends[key]
# Same rule as FlagGems: the vendor is everything before the last '-'.
vendor = key.rsplit("-", 1)[0] if "-" in key else key

pypi_base = cfg.get("pypi_base", "")
if pypi_base:
    print("PYPI\t" + pypi_base.replace("{vendor}", vendor))
if cfg.get("mirror"):
    print("MIRROR\t" + cfg["mirror"])
for dep in backend.get("deps", []):
    print("DEP\t" + dep)
# FlagTree is the preferred compiler where a backend has one; otherwise the
# backend's Triton package. Installed only with --no-prebuilt, since a
# prebuilt image already carries one of them and installing the other would
# put a second `triton` module next to it.
compiler_pkg = backend.get("flagtree") or backend.get("triton")
if compiler_pkg:
    print("COMPILER\t" + compiler_pkg)
for source in backend.get("env_source", []):
    print("SOURCE\t" + source)
# $VARs are expanded against the running environment plus whatever this same
# entry already defined (iluvatar's PATH refers to its own COREX_ROOT). A value
# that still holds a $VAR after that would be written to $GITHUB_ENV literally
# -- that file does no expansion -- so it is dropped with a warning instead of
# corrupting PATH.
def expand(value, env):
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
    return pattern.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), value)


expanded_env = dict(os.environ)
for name, value in (backend.get("env") or {}).items():
    expanded = expand(value, expanded_env)
    if "$" in expanded:
        print(
            f"tools/setup_env.sh: {name} still has an unexpanded variable "
            f"({value!r}); leaving it to the image",
            file=sys.stderr,
        )
        continue
    expanded_env[name] = expanded
    print(f"ENV\t{name}\t{expanded}")
PY
)

# env_source entries are candidates in priority order -- different installers
# put the vendor's setup script in different places (CANN ships it under both
# ascend-toolkit/ and cann/) -- so the first one that exists is used and the
# rest are ignored, rather than sourcing several and prepending PATH twice.
for script in ${ENV_SOURCES}; do
    if [ ! -f "${script}" ]; then
        continue
    fi
    echo "sourcing ${script}"
    while IFS= read -r line; do
        case "${line}" in
            "export "*)
                emit_env "${line#export }"
                sourced_env="${sourced_env} ${line#export }"
                ;;
        esac
    done < <(bash "${REPO_ROOT}/tools/source_env_bash.sh" "${script}")
    break
done

if [ -n "${ENV_SOURCES}" ] && [ -z "${sourced_env}" ]; then
    if [ "${PREBUILT}" -eq 0 ]; then
        # The stack being installed here needs that environment, and letting
        # the run continue only moves the failure into a vendor import with a
        # baffling error message.
        echo "ERROR: none of this backend's env_source scripts exist on this machine:" >&2
        for script in ${ENV_SOURCES}; do
            echo "         ${script}" >&2
        done
        echo "       Install the vendor SDK on the runner, or point env_source at where it" >&2
        echo "       actually landed (backends.yaml)." >&2
        exit 1
    fi
    # A prebuilt image is expected to carry that environment itself.
    echo "note: no env_source script found; the image is expected to provide that environment"
fi

# What the vendor environment contributed. An import failure later in this step
# (or in pytest) prints the values it failed with alongside it.
if [ -n "${sourced_env}" ]; then
    for pair in ${sourced_env}; do
        echo "  env ${pair%%=*}=${pair#*=}"
    done
fi

pip_args=(--no-cache-dir)
if [ -n "${FLAGOS_PYPI}" ]; then
    pip_args+=(--index-url "${FLAGOS_PYPI}")
fi
if [ -n "${MIRROR}" ]; then
    pip_args+=(--extra-index-url "${MIRROR}")
fi

# setuptools>=77 is the repo's build requirement (PEP 639 metadata); install it
# first so `--no-build-isolation` below has a backend to use.
python3 -m pip install "${pip_args[@]}" --upgrade "setuptools>=77"
python3 -m pip install "${pip_args[@]}" --no-build-isolation -e "${REPO_ROOT}"
# The test dependencies, plus the packages a vendor stack uses without
# declaring them -- the same set the FlagOS runtime images preinstall as their
# runtime_prereqs (build-infra's configs.yaml), at the same versions:
#   numpy     tests/accuracy_utils.py, and torch_npu's profiler, which torch
#             2.11 pulls in through its device-backend autoload -- without it
#             `import torch` fails.
#   pybind11  imported at run time by triton/backends/ascend/utils.py, so
#             `import triton` fails without it (flagtree's wheel does not
#             declare it).
#   ninja     triton JIT-builds its launcher with it.
# A native runner's fresh venv has none of them; a prebuilt image does.
python3 -m pip install "${pip_args[@]}" pytest pyyaml sqlalchemy packaging numpy \
    pybind11==3.0.3 ninja==1.13.0

# The vendor stack goes last, so a backend's own pins win over the shared list
# above (iluvatar's `numpy<2`, for one). Skipped when the image already ships
# the stack.
if [ "${PREBUILT}" -eq 0 ]; then
    if [ -n "${COMPILER_PKG}" ]; then
        echo "installing ${BACKEND} compiler (${COMPILER_PKG})"
        # shellcheck disable=SC2086
        python3 -m pip install "${pip_args[@]}" ${COMPILER_PKG}
    fi
    if [ -n "${DEPS}" ]; then
        echo "installing ${BACKEND} deps (${DEPS# })"
        # Word splitting is intended: both are space-separated package lists.
        # shellcheck disable=SC2086
        python3 -m pip install "${pip_args[@]}" ${DEPS}
    fi
fi

echo "checking that FlagTrain imports on this platform"
python3 -c "import flag_train; print('flag_train:', flag_train.vendor_name, '/', flag_train.device)"
