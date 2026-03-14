#!/usr/bin/env bash
set -euo pipefail

# ── ComfyUI + Flux Setup ────────────────────────────────────────────
# One-command setup for running ComfyUI with FLUX.1 Schnell on JarvisLabs.
# Usage:
#   export HF_TOKEN=your_huggingface_token
#   bash start.sh
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR=/home/comfyui-tutorial
COMFYUI_DIR="$WORKDIR/ComfyUI"
PORT=8188

# ── Check HF_TOKEN ──────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo ""
    echo "FLUX.1 requires a Hugging Face token with access to the model."
    echo "  1. Get a token at https://huggingface.co/settings/tokens"
    echo "  2. Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-schnell"
    echo "  3. Run: export HF_TOKEN=hf_your_token_here"
    echo "  4. Then re-run: bash start.sh"
    exit 1
fi

echo "=== ComfyUI + Flux Setup ==="
echo ""

# ── Step 1: Clone ComfyUI ──────────────────────────────────────────
if [ -d "$COMFYUI_DIR" ]; then
    echo "[1/4] ComfyUI already cloned, skipping..."
else
    echo "[1/4] Cloning ComfyUI..."
    mkdir -p "$WORKDIR"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

cd "$COMFYUI_DIR"

# ── Step 2: Create venv and install dependencies ───────────────────
if [ -d .venv ]; then
    echo "[2/4] Venv exists, activating..."
else
    echo "[2/4] Creating venv and installing dependencies..."
    uv venv .venv
fi

source .venv/bin/activate
uv pip install -r requirements.txt --quiet

# ── Step 3: Download models ────────────────────────────────────────
echo "[3/4] Logging in to Hugging Face and downloading models..."
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
cp "$SCRIPT_DIR/download_models.py" /tmp/download_models.py
python /tmp/download_models.py

# ── Step 4: Start ComfyUI ──────────────────────────────────────────
echo ""
echo "=== ComfyUI is ready! ==="
echo "Access it at your instance's HTTPS endpoint (port $PORT)"
echo ""
python main.py --listen 0.0.0.0 --port "$PORT"
