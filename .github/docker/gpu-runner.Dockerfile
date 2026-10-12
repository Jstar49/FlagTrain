# gpu-runner image for FlagTrain CI — the nvidia-cuda133 backend (H20, CUDA 13.3).
#
# Unlike FlagGems' CPU runner image (`.github/docker/cpu-runner.Dockerfile`,
# pure-CPU lint/registry pre-checks), the suites under tests/deepspeed/ execute
# real Triton kernels and compare them against DeepSpeed's own CUDA ops, so
# this image starts from the vendor base image and is only useful on a host
# with an NVIDIA device passed through.
#
# This is the container recipe that was verified by hand on the H20 host
# before this image existed, with the versions backends.yaml pins, baked in so
# the workflows install NOTHING at run time:
#
#   - nvcc from cuda-toolkit-13-3. The base is the NGC *runtime* tag: CUDA
#     runtime libraries, no compiler. Both DeepSpeed ops the tests import
#     (fused_lamb, ragged_device_ops) are JIT-built with nvcc on first use.
#   - /opt/venv: python3.12 with torch 2.11.0+cu130, flagtree 0.7.0 (which is
#     the `triton` module for this backend), flagcx, deepspeed 0.19.7 +
#     deepspeed-kernels, and the test tooling (pytest, pyyaml, sqlalchemy).
#     PATH puts the venv first, so `python`, `pip` and `pytest` resolve to it
#     in every shell — workflows need no `source activate`.
#
# Run-time contract:
#   - Run `pytest` from the repository root: pytest.ini sets testpaths=tests
#     and pythonpath=src, so a checkout needs no install to be importable.
#   - Run with the device passed through (`--gpus all` or the equivalent
#     device mounts) — the suites are what exercise CUDA, not this image.
#   - A job that needs `import flag_train` from outside the repo root runs
#     `pip install --no-build-isolation -e .`; setuptools>=77 is baked, so
#     that needs no network.
#
# What is deliberately absent:
#
#   - The FlagTrain package itself. CI tests the checkout of the revision
#     under review, which cannot be baked into an image.
#   - torchaudio and torchvision, although backends.yaml lists them beside
#     torch. Nothing under tests/ or benchmark/ imports either, and each is
#     another few hundred MB in an image that is pulled per job — add the two
#     pinned lines to the torch step if a component starts needing them.
#   - Pre-built DeepSpeed op caches. Build hosts have no GPU, so
#     deepspeed's jit_load() takes its build_for_cpu branch and emits a
#     different nvcc command line than the run-time build does
#     (-DBF16_AVAILABLE and friends are skipped), and ninja rebuilds whenever
#     the command line changes — a cache baked here would not be reused, only
#     add weight. The two JIT cache directories are pinned to fixed paths
#     instead (see the bottom), so a job that mounts them keeps the DeepSpeed
#     builds and Triton kernel compilations on the node instead of paying for
#     them once per job.
#
# Package indexes: pypi.org is not reliably reachable from the networks these
# images are built and run on (connections to it time out), so the aliyun
# mirror plus the FlagOS nexus are the only sources. torch's +cu130 build,
# flagtree and flagcx exist only on the nexus. The mirror is exported as ENV
# too, so pip behaves the same way at run time.
#
# Sources of truth: backends.yaml (the backend's version pins). The install
# order below — torch, drop upstream triton, flagtree, flagcx, then DeepSpeed —
# is the one that recipe settled on; keep the order if the pins move.
# Rebuild this image when the pins move.
# One known drift: backends.yaml pins deepspeed-kernels==0.0.1, which
# does not resolve — PyPI has only 0.0.1.dev* wheels — so it is installed
# unpinned here (see the DeepSpeed section).
#
# Build — there is no COPY and the context is unused, so any directory works:
#   docker build -f .github/docker/gpu-runner.Dockerfile -t <registry>/flagtrain-gpu-runner:cuda13.3.0 .

FROM harbor.baai.ac.cn/flagos-base/flagos-base-nvidia-cuda13.3:2.2.0

ENV DEBIAN_FRONTEND=noninteractive

# Relayed to apt/pip on build nodes that have no direct egress, the same
# convention build-infra's runtime Containerfile uses. ARG, not ENV: neither
# the proxy nor its credentials may persist into the image.
ARG http_proxy=
ARG https_proxy=
ARG no_proxy=

# --- package indexes -------------------------------------------------------
ARG PYPI_MIRROR="https://mirrors.aliyun.com/pypi/simple"
ARG FLAGOS_HOSTED="https://resource.flagos.net/repository/flagos-pypi-hosted/simple"
ARG FLAGOS_NVIDIA="https://resource.flagos.net/repository/flagos-pypi-nvidia/simple"

# The mirror as the default index (it carries everything that is on PyPI), the
# nexus passed per-install as the extra index for the wheels only it has.
# PIP_TRUSTED_HOST is what build-infra's runtime image does for this host:
# some build and CI nodes do not carry the nexus' CA chain.
# PIP_DISABLE_PIP_VERSION_CHECK silences pip's pypi.org version check, which
# on this network would otherwise stall every pip invocation for its timeout.
ENV PIP_INDEX_URL=${PYPI_MIRROR} \
    PIP_TRUSTED_HOST=resource.flagos.net \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- compiler: nvcc for the DeepSpeed JIT builds ---------------------------
# cuda-toolkit-13-3, not the unversioned cuda-toolkit: the latter tracks the
# newest 13.x NVIDIA publishes and would drift away from this image's 13.3
# runtime libraries and torch's cu130 build. The full toolkit is what the
# host-verified recipe used; if the image size becomes a problem, the pieces
# the JIT builds actually need are cuda-nvcc-13-3 + cuda-cudart-dev-13-3.
# python3.12-venv: the base ships the interpreter, but neither venv nor pip.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        cuda-toolkit-13-3 \
        python3.12-venv python3-pip; \
    nvcc --version; \
    apt-get clean; rm -rf /var/lib/apt/lists/*

# --- python environment ----------------------------------------------------
# A venv rather than the image's python: the vendor site-packages stays as the
# vendor shipped it, and the stack below is what the repo's pins describe.
# VIRTUAL_ENV + PATH are set in the image so that every shell — including the
# non-interactive one each CI step gets — is already "activated".
ARG VENV=/opt/venv
ENV VIRTUAL_ENV=${VENV} \
    PATH=${VENV}/bin:$PATH
RUN set -eux; \
    python3.12 -m venv "${VENV}"; \
    pip install --no-cache-dir --upgrade pip; \
    python --version; pip --version

# --- torch -----------------------------------------------------------------
# The +cu130 build (backends.yaml) is the official PyTorch CUDA-13.0 wheel,
# carried on the FlagOS nexus. Installing it drags in upstream triton==3.6.0 —
# torch declares it for Linux — which the following step removes: flagtree
# ships the same `triton` module, and the two cannot coexist in site-packages.
RUN set -eux; \
    pip install --no-cache-dir \
        --extra-index-url "${FLAGOS_NVIDIA}" \
        "torch==2.11.0+cu130"; \
    python -c "import torch; print('torch', torch.__version__, '/ cuda', torch.version.cuda)"

# --- flagtree: the triton module for this backend --------------------------
# `===`, not `==`: `==0.7.0` also matches the per-vendor local versions on the
# nexus (0.7.0+tileir3.6, 0.7.0+metax3.6, ...) and pip picks one of those
# arbitrary variants; `===0.7.0` is the exact-string match that selects the
# plain CUDA wheel.
# The removal loop repeats until nothing is left: one uninstall
# can leave a dist-info behind, and both distribution names are covered
# because upstream ships as `triton` while NVIDIA's build of it ships as
# `pytorch-triton`. Failing the build if any survives is deliberate — a
# leftover upstream triton would shadow flagtree's module at run time and the
# tests would silently exercise the wrong compiler.
RUN set -eux; \
    for _ in 1 2 3 4 5; do \
        pip uninstall -y triton pytorch-triton >/dev/null 2>&1 || true; \
        pip show triton >/dev/null 2>&1 || pip show pytorch-triton >/dev/null 2>&1 || break; \
    done; \
    if pip show triton >/dev/null 2>&1 || pip show pytorch-triton >/dev/null 2>&1; then \
        echo "ERROR: triton/pytorch-triton survived uninstall; flagtree would collide with it"; \
        exit 1; \
    fi; \
    pip install --no-cache-dir \
        --index-url "${FLAGOS_HOSTED}" \
        --extra-index-url "${PYPI_MIRROR}" \
        "flagtree===0.7.0"

# --- flagcx ----------------------------------------------------------------
# Part of the environment the host-verified recipe installed. Nothing in FlagTrain
# imports it yet (no test or source file names it), so it is here to keep this
# image equal to the verified recipe; drop the line if that stops being worth
# the size.
RUN set -eux; \
    pip install --no-cache-dir \
        --extra-index-url "${FLAGOS_NVIDIA}" \
        "flagcx==0.14.0rc2.post2+cuda13.3"

# --- DeepSpeed + test tooling ----------------------------------------------
# deepspeed 0.19.7 (backends.yaml) installs from the sdist and compiles its
# ops later, on the GPU; the sdist build itself needs no nvcc.
# deepspeed-kernels supplies `dskernels`, whose prebuilt libblockedflash the
# nvidia tier of the blocked_flash tests requires (RaggedOpsBuilder links
# -lblockedflash from it — see tests/deepspeed/test_blocked_flash.py). It is
# left unpinned because PyPI has no 0.0.1 release, only 0.0.1.dev* wheels;
# pip takes the newest. Pin it to a 0.0.1.dev<timestamp> here and in
# backends.yaml if the CI needs that determinism.
# pytest/pyyaml/sqlalchemy are what the suites import: conftest's yaml
# handling, the tune configs, and flag_train.utils.models.sql.
# All of these come from the mirror; the nexus does not carry them, so no
# extra index is passed here — a package resolving from somewhere unexpected
# would be a surprise, not a rescue.
RUN set -eux; \
    pip install --no-cache-dir \
        "deepspeed==0.19.7" \
        deepspeed-kernels \
        pytest pyyaml sqlalchemy; \
    pip list --format=freeze | grep -iE '^(deepspeed|pytest|pyyaml|sqlalchemy|dskernels)'

# --- setuptools for the run-time editable install --------------------------
# The repo's pyproject needs setuptools>=77 (PEP 639 license-files metadata);
# torch's own metadata caps it at <82, so the window is [77, 82). With this
# baked, `pip install --no-build-isolation -e .` in a checkout is offline.
RUN set -eux; \
    pip install --no-cache-dir "setuptools>=77,<82"; \
    python -c "import setuptools; print('setuptools', setuptools.__version__)"

# --- verification ----------------------------------------------------------
# Fail the build instead of the first CI job. torch.cuda.is_available() is
# False on a build host with no device and is deliberately not asserted —
# this image is not where the device path gets exercised.
# The triton assertions are flagtree's install guard: a wheel whose files did
# not land leaves an empty namespace package that imports without error, and
# the venv check makes sure the module is flagtree's and not a stray system
# triton that PATH happens to reach.
RUN set -eux; \
    python -c 'import sys, torch, triton, deepspeed, yaml, sqlalchemy, dskernels; \
assert triton.__file__ and hasattr(triton, "Config"), "triton install is incomplete"; \
assert triton.__file__.startswith(sys.prefix), "triton came from outside the venv: " + str(triton.__file__); \
assert torch.version.cuda == "13.0", torch.version.cuda; \
print("python    ", sys.version.split()[0]); \
print("torch     ", torch.__version__); \
print("triton    ", triton.__version__, triton.__file__); \
print("deepspeed ", deepspeed.__version__); \
print("dskernels ", dskernels.library_path())'; \
    python -m pip list --format=freeze \
        | grep -iE '^(torch|triton|flagtree|flagcx|deepspeed|numpy)'

# --- JIT caches on fixed paths ---------------------------------------------
# DeepSpeed's op builds (TORCH_EXTENSIONS_DIR) and Triton's compiled kernels
# (TRITON_CACHE_DIR) default under $HOME, which differs per job inside a
# runner container and dies with it. Pinning both to fixed world-writable
# paths lets a job that mounts a host directory there keep them across jobs on
# that node — the same trick as PRE_COMMIT_HOME in FlagGems' cpu-runner image.
# A mounted cache outlives the code that filled it: TORCH_EXTENSIONS_DIR is
# keyed by op name and nvcc command line, not by DeepSpeed version, so clear
# it when deepspeed is bumped. Without a mount these are ordinary
# per-container caches and nothing changes.
RUN set -eux; \
    mkdir -p /opt/torch-extensions /opt/triton-cache; \
    chmod 1777 /opt/torch-extensions /opt/triton-cache
ENV TORCH_EXTENSIONS_DIR=/opt/torch-extensions \
    TRITON_CACHE_DIR=/opt/triton-cache

# --- runtime environment ---------------------------------------------------
# The CUDA 13.3 compat directory goes ahead of the library path (the recipe
# this image reproduces did the same). If the base image carries the userspace
# driver shims there, this is what lets the image run on hosts whose driver
# predates 13.3; an absent directory in LD_LIBRARY_PATH costs nothing. The
# base image already puts /usr/local/cuda/bin on PATH.
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.3/compat:${LD_LIBRARY_PATH}
