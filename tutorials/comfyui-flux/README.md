# ComfyUI + Flux on JarvisLabs A100 80GB

Run ComfyUI with FLUX.1 models on a cloud A100 80GB GPU for fast AI image generation.

## What You'll Do

- Install ComfyUI on a JarvisLabs GPU instance
- Download FLUX.1 Schnell model and text encoders
- Access ComfyUI's web interface remotely
- Generate images using the Flux text-to-image workflow

## Quick Start

```bash
# On your JarvisLabs A100-80GB instance:
cd /home
mkdir comfyui-tutorial && cd comfyui-tutorial
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create isolated environment
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Download models
python download_models.py

# Start ComfyUI (accessible via your instance's HTTP endpoint)
python main.py --listen 0.0.0.0 --port 8188
```

## Requirements

- JarvisLabs A100-80GB instance (80GB VRAM)
- 100GB storage
- Hugging Face account with access to FLUX.1 models
