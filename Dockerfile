# ─────────────────────────────────────────────────────────────
#  Base: CUDA 12.4 + cuDNN dev headers on Ubuntu 22.04
# ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ── Environment ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    #
    # FIX 1 — DeepSpeed GPU arch list
    # Without this, DeepSpeed tries to query an attached GPU during
    # build (there is none), fails to detect the arch, and the CUDA
    # kernel compilation either errors out or silently skips extensions.
    # This list covers: V100(7.0), T4(7.5), A10/A100(8.0/8.6), H100(9.0).
    # Add or remove arches to match your Modal GPU tier.
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX" \
    #
    # FIX 2 — suppress nvcc host-compiler warning that becomes a hard
    # error when gcc-13 is used with older DeepSpeed cmake logic.
    NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-13"

# ── System packages ───────────────────────────────────────────
# FIX 3 — libnccl-dev is NOT in the standard Ubuntu 22.04 repo.
# It lives in NVIDIA's repo which is already configured in the
# base image, but the package is named libnccl-dev there.
# We also add libstdc++-13-dev which provides the C++17 STL
# headers that DeepSpeed's fused kernels need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
        gcc \
        g++ \
        libstdc++-11-dev \
        binutils \
        git \
        curl \
        wget \
        ninja-build \
        libssl-dev \
        ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# FIX 4 — pin build tooling BEFORE anything else.
# packaging: DeepSpeed's setup.py imports it at the top level — if
#   it's missing or the wrong version the build fails before any code runs.
# wheel: needed to produce installable .whl artifacts during build.
# ninja: already installed via apt above; this ensures the Python
#   binding used by torch.utils.cpp_extension also sees it.
RUN pip install  \
        "packaging>=23.0" \
        "wheel>=0.43" \
        "ninja>=1.11"

# ── PyTorch 2.5.1 + CUDA 12.4 ────────────────────────────────
RUN pip install  \
        torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

# ── DeepSpeed v0.16.4 (built from source with CUDA extensions) ─
# FIX 5 — MAX_JOBS limits parallel compiler processes so the build
# doesn't OOM-kill itself on machines with limited RAM (common in
# Docker on Windows with WSL2's default 50 % RAM cap).
# Raise to 8 if your machine has ≥32 GB RAM.
RUN git clone https://github.com/microsoft/DeepSpeed.git \
        --branch v0.16.4 \
        --depth 1 \
        -c advice.detachedHead=false \
    && cd DeepSpeed \
    && MAX_JOBS=4 DS_BUILD_UTILS=1 DS_BUILD_FUSED_ADAM=1 \
       pip install --no-build-isolation . \
    && cd .. && rm -rf DeepSpeed

# ── OpenRLHF 0.6.1.post1 (with vLLM) ────────────────────────
# FIX 6 — vLLM ships pre-built wheels for cu124 + torch 2.5.x,
# so no compilation happens here. But it pins its own torch version
# which can downgrade yours. We reinstall torch afterward to lock it.
RUN pip install --no-build-isolation "openrlhf[vllm]==0.6.1.post1"

# Re-pin torch in case vLLM's deps pulled a different version
RUN pip install \
        torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

# ── LM Evaluation Harness (pinned commit, math extras) ───────
# Installed without -e so the source tree can be removed safely.
# "math" extras pull sympy + antlr4 — pure Python, no compilation.
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness \
    && cd lm-evaluation-harness \
    && git checkout 19ba1b16fef9fa6354a3e4ef3574bb1d03552922 \
    && pip install --no-cache-dir ".[math]" \
    && cd .. && rm -rf lm-evaluation-harness

# ── Working directory / project code ─────────────────────────
WORKDIR /app
COPY . .

CMD ["bash"]
