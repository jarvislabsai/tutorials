---
title: "How to Run ComfyUI with Flux on a Cloud GPU (A100 80GB)"
slug: comfyui-flux-cloud-gpu
description: "Step-by-step guide to installing and running ComfyUI with FLUX.1 on a JarvisLabs A100 80GB GPU. Generate stunning AI images in under 2 seconds with a node-based workflow you can access from any browser."
authors: [vishnu_subramanian]
date: '2026-03-14'
tags: [vision, ComfyUI, Flux]
keywords: [comfyui, flux, cloud gpu, ai image generation, a100, comfyui tutorial, flux tutorial, text to image]
---

![ComfyUI running on JarvisLabs A100](./images/comfyui_flux/comfyui_templates.webp)

ComfyUI is a powerful, node-based interface for running AI image and video generation workflows. Unlike simple prompt-in-image-out tools, ComfyUI gives you full control over every step of the generation pipeline — model loading, text encoding, sampling, and post-processing are all visible and customizable nodes you can rewire.

In this tutorial, we'll set up ComfyUI with FLUX.1 Schnell on a JarvisLabs A100 80GB instance. You'll have a fully functional ComfyUI environment accessible from your browser, generating 1024x1024 images in under 2 seconds.

<!--truncate-->

## Why Run ComfyUI on a Cloud GPU?

If you've watched NVIDIA's [ComfyUI tutorial](https://www.youtube.com/watch?v=kqVlmscvuNc) with digital artist Max Novak, you'll notice it focuses on running locally with an RTX GPU. That's great if you have one — but there are strong reasons to run on a cloud GPU instead:

- **80GB VRAM** — The A100-80GB lets you load Flux at full precision and still have room for additional models (ControlNet, IP-Adapter, etc.) without running out of memory.
- **No local hardware needed** — Access ComfyUI from any device with a browser, including a laptop or even a tablet.
- **Faster iteration** — With models cached in VRAM, subsequent generations take under 2 seconds on A100.
- **Easy to scale** — Need more power? Spin up a second instance. Done for the day? Destroy it and stop paying.

## Prerequisites

To follow along, you'll need a JarvisLabs.ai GPU instance:

| Requirement | Value |
|---|---|
| GPU | A100-80GB (80GB VRAM) |
| Storage | 100GB |
| Template | PyTorch |
| HTTP Ports | 8188 |
| Estimated time | ~30 minutes |
| Estimated cost | ~$0.75 |

You'll also need a [Hugging Face account](https://huggingface.co/join) with access to [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) (free, but requires accepting the license).

**New to JarvisLabs?**
1. [Create an account](https://accounts.jarvislabs.ai/sign-up)
2. [Add funds](https://jarvislabs.ai/settings#recharge) to your wallet
3. [Launch an A100-80GB instance](https://jarvislabs.ai/templates/pytorch) with 100GB storage

**Important:** When creating your instance, set **HTTP Ports** to `8188`. This exposes ComfyUI's web interface through a secure HTTPS endpoint that you can access from any browser.

## Step 1: Install ComfyUI

Once your instance is running, open a terminal (via JupyterLab or SSH) and run:

```bash
cd /home
mkdir comfyui-tutorial && cd comfyui-tutorial
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

Create an isolated Python environment with `uv` (pre-installed on JarvisLabs instances):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

This installs PyTorch 2.10 with CUDA 12.8 support, along with all ComfyUI dependencies. The install takes about 1-2 minutes.

## Step 2: Download the Models

FLUX.1 requires three components:
- **The diffusion model** — the core image generation weights (~24GB)
- **Text encoders** — CLIP-L and T5-XXL that convert your prompts into embeddings the model understands
- **VAE** — decodes the latent representation into actual pixels

First, log in to Hugging Face:

```bash
python -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"
```

Replace `YOUR_HF_TOKEN` with your [Hugging Face access token](https://huggingface.co/settings/tokens).

Now download all models:

```python
# Save as download_models.py and run with: python download_models.py
from huggingface_hub import hf_hub_download

# FLUX.1 Schnell diffusion model (~24GB)
hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-schnell",
    filename="flux1-schnell.safetensors",
    local_dir="models/diffusion_models"
)

# CLIP-L text encoder (~250MB)
hf_hub_download(
    repo_id="comfyanonymous/flux_text_encoders",
    filename="clip_l.safetensors",
    local_dir="models/text_encoders"
)

# T5-XXL text encoder in FP8 (~5GB)
hf_hub_download(
    repo_id="comfyanonymous/flux_text_encoders",
    filename="t5xxl_fp8_e4m3fn.safetensors",
    local_dir="models/text_encoders"
)

# Flux VAE (~335MB)
hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-schnell",
    filename="ae.safetensors",
    local_dir="models/vae"
)

print("All models downloaded!")
```

The total download is about 30GB. On JarvisLabs' network this takes about 2-3 minutes.

### Why T5-XXL in FP8?

We use the FP8 (8-bit) quantized version of T5-XXL from `comfyanonymous/flux_text_encoders`. The full T5-XXL is ~10GB; the FP8 version is ~5GB with virtually no quality difference for text encoding. This saves VRAM for the diffusion model where precision matters more.

## Step 3: Start ComfyUI

```bash
python main.py --listen 0.0.0.0 --port 8188
```

The key flags:
- `--listen 0.0.0.0` — makes ComfyUI accessible from outside the container (required for cloud access)
- `--port 8188` — the port we exposed during instance creation

You'll see output like:

```
To see the GUI go to: http://0.0.0.0:8188
```

**Accessing the UI:** Open the HTTP endpoint URL from your JarvisLabs instance dashboard. It will be something like `https://<instance-id>.notebooksn.jarvislabs.net`. This routes to port 8188 inside your instance.

![ComfyUI Templates Page](./images/comfyui_flux/comfyui_templates.webp)

When ComfyUI loads, you'll see the Templates gallery — a collection of pre-built workflows for common tasks like text-to-image, image-to-video, and 3D model generation.

## Step 4: Generate Your First Image

### Using the Built-in Templates

The easiest way to start is with ComfyUI's built-in templates. Click **Templates** in the left sidebar and select **"1.1 Starter - Text to Image"**. This loads a pre-configured Flux workflow.

But for our tutorial, let's understand what's happening by looking at the nodes:

1. **UNETLoader** — loads `flux1-schnell.safetensors` into GPU memory
2. **DualCLIPLoader** — loads both CLIP-L and T5-XXL text encoders
3. **CLIPTextEncode** — converts your text prompt into model-readable embeddings
4. **EmptySD3LatentImage** — creates a blank latent canvas at your desired resolution
5. **KSampler** — the core sampling node that iteratively denoises the latent image
6. **VAEDecode** — converts the final latent into viewable pixels
7. **SaveImage** — saves the result

### Generation via the API

ComfyUI also exposes a REST API, which is useful for automation or batch generation. Here's a Python script that queues a generation:

```python
import json
import urllib.request

workflow = {
    "prompt": {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-schnell.safetensors",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux"
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "A majestic snow leopard on a mountain peak at sunset, golden hour, photorealistic",
                "clip": ["2", 0]
            }
        },
        "4": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["6", 0],
                "latent_image": ["4", 0]
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["2", 0]}
        },
        "7": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"}
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["7", 0]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "flux_output", "images": ["8", 0]}
        }
    }
}

data = json.dumps(workflow).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:8188/prompt",
    data=data,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req)
print(json.loads(resp.read()))
```

## Step 5: Results

### Performance on A100 80GB

Here's what we measured:

| Metric | Value |
|---|---|
| First generation (cold start) | **6.6 seconds** |
| Subsequent generations | **1.75 seconds** |
| Resolution | 1024 x 1024 |
| Sampling steps | 4 (Flux Schnell) |
| VRAM usage | ~28GB / 80GB |

The first generation takes ~6.6 seconds because the model needs to be loaded into GPU memory. After that, the model stays cached in VRAM and subsequent images generate in under 2 seconds.

With only 28GB of the 80GB VRAM used, there's plenty of headroom to:
- Load additional models (ControlNet, IP-Adapter)
- Run higher resolutions (2048x2048)
- Use the full-precision Flux Dev model instead of Schnell

### Sample Outputs

Here are two images generated with Flux Schnell on our A100 instance:

**Prompt:** *"A majestic snow leopard sitting on a rocky mountain peak at sunset, golden hour lighting, photorealistic, 8k detail"*

![Snow leopard on mountain peak](./images/comfyui_flux/flux_test.webp)

**Prompt:** *"A futuristic cyberpunk cityscape at night with neon signs reflecting in rain-soaked streets, volumetric fog, cinematic lighting, ultra detailed"*

![Cyberpunk cityscape](./images/comfyui_flux/flux_cyberpunk.webp)

## Tips for Cloud ComfyUI

### Persist Your Work

All tutorial code lives under `/home/` — this is the only directory that persists across pause/resume cycles on JarvisLabs. Your models and ComfyUI installation will survive a pause.

### Install Custom Nodes

ComfyUI Manager comes pre-installed. Click the **Manager** button in the top toolbar to browse and install custom nodes like:
- **ComfyUI-Impact-Pack** — face detection, segmentation
- **ComfyUI-AnimateDiff** — video generation
- **ComfyUI-WAN** — WAN 2.2 Animate for reference-based video

### Pause vs Destroy

- **Pause** your instance if you plan to come back later — your models and setup are preserved, and you only pay for storage.
- **Destroy** when you're done — no ongoing costs, but you'll need to reinstall next time.

### Use the FP8 Flux Model for Lower VRAM

If you want to save VRAM (e.g., to fit additional models), ComfyUI ships with `flux1-schnell-fp8.safetensors` (~17GB) which uses half the VRAM of the full model with minimal quality impact.

## Going Further

Now that you have ComfyUI running, here are some workflows to try:

- **Image-to-Image with Flux** — Use a reference image as a starting point, like the 3D texture workflow from Max Novak's tutorial
- **WAN 2.2 Animate** — Turn still images into short videos with pose-driven animation
- **ControlNet + Flux** — Guide generation with edge maps, depth maps, or pose references
- **Batch Generation** — Use the API to generate hundreds of variations automatically

The full tutorial code is available at [jarvislabsai/tutorials/comfyui-flux](https://github.com/jarvislabsai/tutorials/tree/main/comfyui-flux).

## Conclusion

ComfyUI on a cloud A100-80GB gives you the best of both worlds: the flexibility and control of a node-based interface with the raw power of a datacenter GPU. With Flux Schnell generating 1024x1024 images in under 2 seconds, you can iterate rapidly on creative ideas without being bottlenecked by hardware.

The total cost for this tutorial session was under $1 — about 30 minutes of A100 time including setup, model downloads, and generation.
