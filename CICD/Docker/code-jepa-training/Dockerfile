# syntax=docker/dockerfile:1.7

FROM --platform=$BUILDPLATFORM golang:1.23-alpine AS s5cmd-builder
ARG TARGETOS
ARG TARGETARCH
ARG S5CMD_VERSION=v2.3.0
ENV CGO_ENABLED=0 GOFLAGS="-buildvcs=false"
RUN apk add --no-cache git
RUN GOOS=${TARGETOS} GOARCH=${TARGETARCH} go install -ldflags="-s -w" github.com/peak/s5cmd/v2@${S5CMD_VERSION}

FROM --platform=linux/amd64 nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_DIR=/opt/venv \
    APP_ROOT=/proj \
    IMAGE_REPO_DIR=/opt/code-jepa-image \
    REPO_DIR=/proj/Code-JEPA \
    CODE_JEPA_DATA_ROOT=/proj/code-jepa \
    HF_HOME=/proj/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/proj/huggingface/transformers \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    gnupg \
    htop \
    iproute2 \
    less \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    nano \
    openssh-client \
    pkg-config \
    procps \
    ripgrep \
    rsync \
    tmux \
    unzip \
    vim \
    wget \
    zlib1g-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-dev" \
    "python${PYTHON_VERSION}-venv" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN "python${PYTHON_VERSION}" -m venv "$VENV_DIR" \
    && "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel uv

ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV=/opt/venv

# PyTorch wheel CUDA must match the CUDA runtime family in the base image.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch \
    torchvision \
    torchaudio

# JAX CUDA wheels use the official CUDA-12 plugin wheels.
RUN pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY CICD/Docker/code-jepa-training/requirements.txt /tmp/code-jepa-requirements.txt
RUN pip install -r /tmp/code-jepa-requirements.txt

COPY --from=s5cmd-builder /go/bin/s5cmd /usr/local/bin/s5cmd
COPY CICD/Docker/code-jepa-training/verify_gpu_stack.py /usr/local/bin/verify-gpu-stack
COPY CICD/Docker/code-jepa-training/sync-code-jepa-all.sh /usr/local/bin/sync-code-jepa-all
COPY CICD/Docker/code-jepa-training/entrypoint.sh /usr/local/bin/code-jepa-entrypoint
RUN chmod +x /usr/local/bin/verify-gpu-stack /usr/local/bin/sync-code-jepa-all /usr/local/bin/code-jepa-entrypoint \
    && s5cmd version \
    && python /usr/local/bin/verify-gpu-stack

RUN curl -fsSL https://tailscale.com/install.sh | sh

WORKDIR $IMAGE_REPO_DIR
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
RUN pip install -e ".[dev,transforms]"

RUN mkdir -p "$CODE_JEPA_DATA_ROOT" "$HF_HOME" /proj/checkpoints /proj/artifacts /var/lib/tailscale

ENTRYPOINT ["/usr/local/bin/code-jepa-entrypoint"]
CMD []
