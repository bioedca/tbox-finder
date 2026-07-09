# syntax=docker/dockerfile:1
# ─── tbox-finder CI Docker image (ADR-0002 D4, as amended by A2 C1 + A4) ──────
#
# WHAT THIS IS: the archival / portability / reviewer reproducibility artifact,
# built in CI (GHCR) on a schedule/release (CLAUDE.md §8.6), NOT per-push. It
# bakes the two committed conda-lock environments so a reviewer — or a future
# apptainer/udocker run, should a container runtime ever appear — can reproduce
# the exact ML stack.
#
# WHAT THIS IS NOT: the cluster runtime. The amlab cluster has no container
# runtime (no apptainer/singularity/docker/podman); it runs the same conda-lock
# envs directly on the user miniconda3 (ADR-0002 D1, CLAUDE.md §3.1). This image
# never runs there.
#
# Base: nvidia/cuda 12.8 -cudnn-devel. Since ADR-0002 A2 C1, mamba-ssm is NOT
# source-built (all three sm_86 kernels are prebuilt wheels pinned by URL in the
# locks), so the -devel base is retained ONLY to supply the CUDA headers/libs
# that DeepSpeed's JIT-built ZeRO ops compile against at runtime (ADR-0002 D2/D4;
# cuda-nvcc 12.8 itself ships inside both conda envs).
#
# A4 env split: the single `ml` env was split into ml-dna (Stage-1 DNA:
# transformers 4.57.5 + Caduceus) and ml-rna (Stage-2 RNA: transformers 5.13.0 +
# multimolecule 0.0.9 + RiNALMo). The image installs BOTH locks; Snakemake GPU
# rules pick the per-stage env (CLAUDE.md §3.2).

ARG CUDA_BASE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${CUDA_BASE}

LABEL org.opencontainers.image.title="tbox-finder ML CI image" \
      org.opencontainers.image.description="Archival/reproducibility artifact: CUDA 12.8 + PyTorch, installs the locked ml-dna + ml-rna conda-lock envs (ADR-0002 D4). NOT the cluster runtime." \
      org.opencontainers.image.source="https://github.com/bioedca/tbox-finder" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Minimal system deps for the Miniforge installer + conda-lock's pip URL-wheel
# fetch step. Drop the NVIDIA CUDA apt repo first so a stale repo GPG key can
# never break `apt-get update` (we install nothing from it — the CUDA toolchain
# comes from the conda envs and the base image layer).
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl bzip2 git && \
    rm -rf /var/lib/apt/lists/*

# Miniforge (conda + mamba). The base env's conda/mamba are only the installer
# backend for `conda-lock install`; they are not load-bearing for the science
# (conda-lock install fetches the exact pinned packages, no solve).
ARG MINIFORGE_VERSION=25.3.1-0
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm -f /tmp/miniforge.sh && \
    conda config --system --set channel_priority strict && \
    conda clean -afy

# conda-lock pinned to the exact version that generated the lockfiles (P0-05/
# P0-06c), so the container reproduces the same install path as the laptop.
ARG CONDA_LOCK_VERSION=4.0.2
RUN mamba install -n base -y "conda-lock=${CONDA_LOCK_VERSION}" && \
    conda clean -afy

WORKDIR /opt/tbox-finder

# Install the two locked envs. `--copy` (documented "useful for building
# containers") avoids hardlinks that break across image layers; `--mamba` uses
# the fast solver-free installer; `--validate-platform` (default) asserts the
# lockfile's linux-64 matches the build host. Separate COPY+RUN per env keeps
# the layer cache tight when only one lock changes.
COPY envs/ml-dna.conda-lock.yml envs/ml-dna.conda-lock.yml
RUN conda-lock install --name ml-dna --mamba --copy envs/ml-dna.conda-lock.yml && \
    conda clean -afy

COPY envs/ml-rna.conda-lock.yml envs/ml-rna.conda-lock.yml
RUN conda-lock install --name ml-rna --mamba --copy envs/ml-rna.conda-lock.yml && \
    conda clean -afy

# ── Build-time import smoke: fail the build if either env does not reproduce ──
# CPU-only (no GPU during build): assert torch/transformers versions + torch's
# bundled CUDA (12.8) + the three prebuilt kernel wheels are installed (metadata
# read, NOT a kernel .so import — the GPU kernel forward smoke is P1 on the
# A4000, ADR-0002 A2 C2). CUDA_HOME=<env prefix> is set for ml-rna because
# `import multimolecule` transitively imports deepspeed, which probes CUDA_HOME
# at import (ADR-0002 operational note).
RUN conda run -n ml-dna python - <<'PY'
import importlib.metadata as M
import torch, transformers
print("ml-dna:", "torch", torch.__version__, "| torch.version.cuda", torch.version.cuda,
      "| transformers", transformers.__version__)
for dist in ("mamba-ssm", "causal-conv1d", "flash-attn"):
    print("  kernel wheel", dist, M.version(dist))
assert torch.__version__.split("+")[0] == "2.7.1", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert transformers.__version__ == "4.57.5", transformers.__version__
print("ml-dna smoke OK")
PY

RUN CUDA_HOME=/opt/conda/envs/ml-rna conda run -n ml-rna python - <<'PY'
import importlib.metadata as M
import torch, transformers
import multimolecule  # deepspeed import-probe needs CUDA_HOME=$CONDA_PREFIX (ADR-0002)
print("ml-rna:", "torch", torch.__version__, "| torch.version.cuda", torch.version.cuda,
      "| transformers", transformers.__version__, "| multimolecule", M.version("multimolecule"))
for dist in ("mamba-ssm", "causal-conv1d", "flash-attn"):
    print("  kernel wheel", dist, M.version(dist))
assert torch.__version__.split("+")[0] == "2.7.1", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert transformers.__version__ == "5.13.0", transformers.__version__
print("ml-rna smoke OK")
PY

# ── Entrypoint: activate the requested env and export CUDA_HOME=$CONDA_PREFIX ──
# Select the env with `-e TBOX_ENV=ml-dna` (default) or `-e TBOX_ENV=ml-rna`.
# CUDA_HOME must equal the active env prefix so DeepSpeed's import-time probe
# succeeds (ADR-0002 operational note); the entrypoint sets it unconditionally.
COPY <<'EOF' /usr/local/bin/entrypoint.sh
#!/usr/bin/env bash
# No `set -u`: conda activation scripts (e.g. cuda-nvcc's ~cuda-nvcc_activate.sh)
# reference unset vars like NVCC_PREPEND_FLAGS, which nounset would abort on.
set -eo pipefail
ENV_NAME="${TBOX_ENV:-ml-dna}"
# shellcheck disable=SC1091
source "${CONDA_DIR:-/opt/conda}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_HOME="${CONDA_PREFIX}"
exec "$@"
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
