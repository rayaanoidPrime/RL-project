#!/usr/bin/env bash
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
command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || {
  echo "ERROR: python required"; exit 1;
}

git submodule update --init --recursive

# ---------- 2. Create uv virtual environment ----------
echo ""
echo "Creating uv venv (.venv) with Python 3.11..."
uv venv .venv --python 3.11

# ---------- 3. Resolve venv python path (cross-platform) ----------
if [ -f ".venv/Scripts/python.exe" ]; then
  VENV_PY=".venv/Scripts/python.exe"   # Windows
elif [ -f ".venv/bin/python" ]; then
  VENV_PY=".venv/bin/python"           # Unix/macOS
else
  echo "ERROR: venv python not found"
  exit 1
fi

echo "Using venv python: $VENV_PY"

# ---------- 4. Install dependencies SAFELY ----------
echo ""
echo "Installing local dev dependencies via uv..."

# IMPORTANT: force uv to use THIS python
uv pip install --python "$VENV_PY" \
  modal \
  huggingface_hub \
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

# ---------- 5. Sanity check ----------
echo ""
echo "Verifying installation location..."

"$VENV_PY" -c "import sys; print('Python:', sys.executable)"
"$VENV_PY" -c "import site; print('Site-packages:', site.getsitepackages())"

echo ""
echo "============================================================"
echo " Setup complete."
echo ""
echo " To activate manually (optional):"
echo "   Windows (PowerShell): .venv\\Scripts\\Activate.ps1"
echo "   Windows (CMD):        .venv\\Scripts\\activate.bat"
echo "   Unix/macOS:           source .venv/bin/activate"
echo "============================================================"