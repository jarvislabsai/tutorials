# ComfyUI + Flux on JarvisLabs

Run ComfyUI with FLUX.1 Schnell on a cloud A100 80GB -- one command, under 5 minutes.

## Quick Start

On your JarvisLabs A100-80GB instance (created with HTTP port `8188`):

```bash
export HF_TOKEN=your_huggingface_token
git clone https://github.com/jarvislabsai/tutorials.git
cd tutorials/comfyui-flux
bash start.sh
```

ComfyUI will be accessible at your instance's HTTPS endpoint.

## Requirements

| Requirement | Value |
|---|---|
| GPU | A100-80GB |
| Storage | 100GB |
| HTTP Ports | 8188 |
| HF Token | [Get one here](https://huggingface.co/settings/tokens) (with [FLUX.1 access](https://huggingface.co/black-forest-labs/FLUX.1-schnell)) |

## Deep Dive

For a full step-by-step walkthrough explaining what each piece does, read the [tutorial on jarvislabs.ai](https://jarvislabs.ai/tutorials/comfyui-flux-cloud-gpu).

## Files

- `start.sh` -- One-command setup (clone, install, download models, start ComfyUI)
- `download_models.py` -- Downloads all required model weights from Hugging Face
