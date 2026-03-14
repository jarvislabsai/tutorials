"""Download all required models for ComfyUI + Flux workflow.

Run this from the ComfyUI directory after activating the venv:
    cd /home/comfyui-tutorial/ComfyUI
    source .venv/bin/activate
    python download_models.py

Requires: huggingface-cli login (or HF_TOKEN env var)
"""

from huggingface_hub import hf_hub_download

MODELS = [
    {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "file": "flux1-schnell.safetensors",
        "dest": "models/diffusion_models",
        "desc": "FLUX.1 Schnell diffusion model (~24GB)",
    },
    {
        "repo": "comfyanonymous/flux_text_encoders",
        "file": "clip_l.safetensors",
        "dest": "models/text_encoders",
        "desc": "CLIP-L text encoder (~250MB)",
    },
    {
        "repo": "comfyanonymous/flux_text_encoders",
        "file": "t5xxl_fp8_e4m3fn.safetensors",
        "dest": "models/text_encoders",
        "desc": "T5-XXL text encoder FP8 (~5GB)",
    },
    {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "file": "ae.safetensors",
        "dest": "models/vae",
        "desc": "Flux VAE (~335MB)",
    },
]


def main():
    for m in MODELS:
        print(f"\nDownloading {m['desc']}...")
        hf_hub_download(
            repo_id=m["repo"],
            filename=m["file"],
            local_dir=m["dest"],
        )
        print(f"  -> {m['dest']}/{m['file']}")

    print("\nAll models downloaded!")


if __name__ == "__main__":
    main()
