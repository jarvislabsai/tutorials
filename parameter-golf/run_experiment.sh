#!/bin/bash
set -e

# Parameter Golf on JarvisLabs
# =============================
# This script runs OpenAI's Parameter Golf challenge on JarvisLabs GPUs.
# It clones the official repo, installs dependencies, downloads the dataset,
# and trains a GPT model evaluated by bits-per-byte (BPB) on FineWeb.
#
# The training code is entirely from https://github.com/openai/parameter-golf
# This script just sets up the environment and runs it on JarvisLabs infra.

cd /home

# Clone the official parameter-golf repo
[ -d parameter-golf ] || git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Install deps into the jl-managed venv.
# jl run creates a venv at /home/.venv with --system-site-packages before running this script,
# so torch and numpy from the PyTorch template are already available.
# We only install the extras the template doesn't have.
# You can also provide a custom torch version to override the template one,
# or use requirements.txt / pyproject.toml with directory mode (see below).
uv pip install -q huggingface-hub kernels setuptools "typing-extensions==4.15.0" datasets tiktoken sentencepiece

# Download FineWeb data (10 shards = 1B tokens, enough for dev runs)
# Use --train-shards 80 for the full 8B token dataset
# For more details, see the original repo: https://github.com/openai/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Train
# --nproc_per_node should match your GPU count (1 for dev, 8 for submission)
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=50 \
python3 -m torch.distributed.run --standalone --nproc_per_node=1 --local-addr 127.0.0.1 train_gpt.py

# ---
# How to run
# ==========
#
# Single file mode (this script):
#
#   jl run run_experiment.sh --gpu H100 --keep --yes
#
#   jl run uploads run_experiment.sh to /home/run_experiment.sh on the instance, creates a venv at
#   /home/.venv with --system-site-packages (so template packages like torch
#   are visible), activates it, and runs the script from /home/.
#   Logs are tracked. Use jl run logs <run_id> to follow.
#
# Directory mode (for your own project):
#
#   jl run . --script train.py --gpu H100 --keep --yes
#
#   Rsyncs your project directory to the instance. If it finds a
#   requirements.txt or pyproject.toml, deps are installed automatically.
#   torch from the template is inherited, so you don't need it in your deps.
#
# Monitor:
#
#   jl run logs <run_id> --tail 50     # check progress
#
# Download the trained model:
#
#   jl download <machine_id> /home/parameter-golf/final_model.int8.ptz
#
#   This downloads the compressed int8+zlib model artifact to your local machine.
#   The artifact is what gets submitted. It contains the quantized weights and
#   must be under 16MB total (code + model) per competition rules.
#   A raw unquantized checkpoint is also saved at /home/parameter-golf/final_model.pt
#
# Cleanup:
#
#   jl pause <machine_id> --yes        # stop billing, keep data
#   jl destroy <machine_id> --yes      # delete everything
#
# Full 8-GPU submission run:
#
#   Edit --nproc_per_node=8 above, set MAX_WALLCLOCK_SECONDS=600,
#   and launch with --num-gpus 8:
#
#   jl run run_experiment.sh --gpu H100 --num-gpus 8 --keep --yes
#
# For more on the jl CLI: https://docs.jarvislabs.ai/cli/
