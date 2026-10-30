# The standalone serving image: tabctx + tabicl + tabpfn + torch, serving
# `python -m tabctx.serve` (serve/standalone.py) -- no Ray. Built and
# pushed by .github/workflows/docker_image.yml on each release tag, and
# smoke-tested there against the fake backend (no GPU in CI).
#
# Base: plain python:3.12-slim, not a pytorch/cuda image. torch's PyPI
# wheels bundle the CUDA runtime libraries; only the driver comes from
# the host, injected by the NVIDIA container toolkit that any Kubernetes
# GPU device plugin already requires. This keeps the Python version ours
# (tabctx needs >= 3.12; the pytorch images pin their own) and the torch
# version an explicit, overridable pin instead of whatever a base image
# carried. GPU validation is the deploying platform's rehearsal, by
# design: CI has no GPU.
#
# Weights are NOT baked in. Deployments mount them read-only and load
# offline, e.g.:
#   TABPFN_MODEL_CACHE_DIR=/model-weights/tabctx/tabpfn
#   HF_HUB_CACHE=/model-weights/tabctx/hf/hub  HF_HUB_OFFLINE=1
#
# Runs as a non-root user with no sudo anywhere in the image (lesson of
# an upstream deployment's 2026-09-01 incident: an in-container RCE must
# not get root for free).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

# torch is deliberately pinned here, not in pyproject (the library runs
# on whatever torch its host provides; the image must be reproducible).
ARG TORCH_VERSION=2.9.1

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

RUN useradd --create-home --uid 10001 tabctx

WORKDIR /opt/tabctx
COPY pyproject.toml README.md LICENSE.md ./
COPY src ./src

# One resolution for both, so the extras' unpinned `torch` requirement
# and the explicit pin cannot diverge.
RUN uv pip install --system --no-cache \
    "torch==${TORCH_VERSION}" \
    ".[standalone,tabicl,tabpfn]"

USER tabctx
WORKDIR /home/tabctx
EXPOSE 8000

# --host 0.0.0.0: the container is reached through a Service; the
# standalone default (127.0.0.1) is for processes on a shared host.
ENTRYPOINT ["python", "-m", "tabctx.serve", "--host", "0.0.0.0"]
