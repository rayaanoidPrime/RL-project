#!/usr/bin/env bash
# =============================================================================
# setup_local.sh
# Local development bootstrap using uv.
# This does NOT run the heavy GPU training — that happens on Modal.
# Use this to get a local env for editing, linting, and running modal commands.
# =============================================================================
set -euo pipefail

echo "============================================================"
echo " DR-DPO reproduction — local dev setup"
echo "============================================================"

# ---------- 1. Check prerequisites ----------
command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Installing via curl..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
}

command -v git >/dev/null 2>&1 || { echo "ERROR: git required"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 required"; exit 1; }

git submodule update --init --recursive

# ---------- 3. Create uv virtual environment ----------
echo ""
echo "Creating uv venv (.venv) with Python 3.11..."
uv venv .venv --python 3.11

# Activate
source .venv/bin/activate

# ---------- 4. Install local/dev dependencies ----------
# These are lightweight tools for local work: modal CLI, huggingface-cli,
# notebook support, and linting. The heavy GPU packages (torch, deepspeed,
# openrlhf) are installed INSIDE Modal images at runtime.
echo ""
echo "Installing local dev dependencies via uv..."

uv pip install \
  modal \
  huggingface_hub \
  transformers \
  datasets \
  accelerate \
  numpy \
  pandas \
  matplotlib \
  scikit-learn \
  tqdm \
  ipython \
  jupyter \
  black \
  ruff \
  python-dotenv

echo ""
echo "============================================================"
echo " Setup complete."
echo ""
echo " Next steps:"
echo "   1. source .venv/bin/activate"
echo "   2. cp .env.example .env   # and fill in your tokens"
echo "   3. modal setup            # authenticate Modal"
echo "   4. huggingface-cli login  # authenticate HuggingFace"
echo "   5. modal run modal_jobs.py::smoke_test  # verify infra"
echo "============================================================"
