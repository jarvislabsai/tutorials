# Parameter Golf Autoresearch

You are running autonomous experiments on the OpenAI Parameter Golf challenge. The goal is to get the lowest `val_bpb` (validation bits per byte) on the FineWeb dataset. Lower is better. The full 8xH100 baseline is 1.2244. Dev runs (1 GPU, 2 min, 10 data shards) will produce higher values around 1.5-1.6. Use your dev baseline for relative comparisons.

The experiments run on a JarvisLabs GPU instance via the `jl` CLI.

## Setup

Do this once before starting the loop:

1. Install the JarvisLabs skill if not already installed. Ask the human to run:
  ```
   jl setup --yes
  ```
   This authenticates and installs the skill to `~/.agents/skills/jarvislabs/SKILL.md`. Read the skill file before proceeding. It covers the full `jl` CLI: instance lifecycle, managed runs, monitoring, file transfer, and anti-patterns.
2. Create an H100 instance and run `run_experiment.sh` on it to clone the parameter-golf repo, install deps, and download data. Use the patterns from the JarvisLabs skill (create instance, start a run, monitor logs, check for success).
3. The first run also serves as your baseline. Note the `val_bpb` from the final output line:
  ```
   final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
  ```
   Record this as your starting point.
4. Read `train_gpt.py` on the instance to understand the baseline architecture, optimizer, hyperparameters, and training loop before proposing changes.

After setup, the parameter-golf repo is at `/home/parameter-golf/` on the instance, data is downloaded, and deps are installed in `/home/.venv`.

## What you CAN modify

- `train_gpt.py` in the parameter-golf repo on the instance. This contains the full model architecture, optimizer, hyperparameters, and training loop. Everything is fair game: layer count, model width, attention configuration, MLP expansion, quantization scheme, learning rates, training schedule, tokenizer, evaluation strategy.

## What you CANNOT modify

- `data/cached_challenge_fineweb.py` and anything in `data/` (the dataset and tokenizer are fixed)
- The evaluation metric (val_bpb computed by the training script itself)
- The constraint: code + compressed model must be under 16,000,000 bytes

## Competition constraints

- Artifact size: code bytes + compressed int8+zlib model bytes < 16,000,000 (decimal, not MiB)
- Training time: under 10 minutes on 8xH100 for leaderboard submissions
- No network calls during evaluation
- You can use any Python package, any architecture, any training technique

## The experiment loop

LOOP FOREVER:

1. Read the current `train_gpt.py`.
2. Decide on a change. Ideas to explore:
  - Learning rate sweeps (MATRIX_LR, EMBED_LR, SCALAR_LR)
  - More layers vs wider layers (NUM_LAYERS, MODEL_DIM)
  - MLP expansion factor (MLP_MULT=3 for 3x MLP)
  - Quantization-aware training (int6, int5, mixed precision)
  - Sliding window evaluation for longer effective context
  - Different warmdown schedules (WARMDOWN_ITERS)
  - Weight decay with Muon optimizer
  - EMA or SWA (stochastic weight averaging)
  - Architectural changes: cross-sequence attention, skip connections, BigramHash embeddings
3. Upload the modified `train_gpt.py` to `/home/parameter-golf/train_gpt.py` on the instance.
4. Run training via `jl run` command mode (see the JarvisLabs skill). The venv from the initial `run_experiment.sh` is auto-detected. Command mode runs from `~`, so `cd` into the repo first:
   ```
   cd /home/parameter-golf && MAX_WALLCLOCK_SECONDS=120 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=50 python3 -m torch.distributed.run --standalone --nproc_per_node=1 --local-addr 127.0.0.1 train_gpt.py
   ```
5. Monitor the run using the patterns from the JarvisLabs skill (early check, steady-state polling, footer parsing).
6. Parse results from the log tail. The line that matters:
  ```
   final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
  ```
   Also check `Total submission size int8+zlib` stays under 16,000,000 bytes.
7. Compare with previous best val_bpb:
  - **Improved** (lower val_bpb): keep the change. This is the new baseline.
  - **Same or worse**: discard. Restore the previous `train_gpt.py`.
  - **Crashed**: check the log tail for the error. Fix if trivial, discard if fundamental.
8. Log each experiment in a running table:
  ```
   | # | Change | val_bpb | artifact_bytes | status | notes |
   |---|--------|---------|----------------|--------|-------|
   | 1 | baseline | 1.6040 | 9079238 | keep | starting point |
   | 2 | MATRIX_LR=0.02 | 1.5890 | 9031552 | keep | lower LR helps |
   | 3 | NUM_LAYERS=10 | crash | - | discard | OOM on 1 GPU |
  ```
9. Go to step 1.

Do NOT stop to ask the human if you should continue. The human may be away. You are autonomous. If you run out of ideas, re-read the leaderboard at the top of the parameter-golf README for inspiration, try combining previous near-misses, or try more radical changes. The loop runs until the human interrupts you. Repo link: [https://github.com/openai/parameter-golf](https://github.com/openai/parameter-golf)

## Output format

The training script prints these lines you need to parse:

Training progress:

```
step:100/20000 train_loss:3.5453 train_time:32516ms step_avg:325.16ms
```

Validation (periodic and final):

```
step:200/20000 val_loss:2.8038 val_bpb:1.6606 train_time:66588ms step_avg:332.94ms
```

Early stopping (normal when using wallclock cap):

```
stopping_early: wallclock_cap train_time:120029ms step:369/20000
```

Final results after quantization roundtrip:

```
Serialized model int8+zlib: 9100181 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 9147867 bytes
final_int8_zlib_roundtrip_exact val_loss:2.46424750 val_bpb:1.45946618
```

The number that matters is `val_bpb` from the `final_int8_zlib_roundtrip_exact` line. This is the post-quantization score, which is what gets submitted.

## Notes

- Use `python3 -m torch.distributed.run` instead of `torchrun` so it picks up the venv python. Always pass `--local-addr 127.0.0.1` to avoid IPv6 resolution failures in containers.
- You can edit environment variables (NUM_LAYERS, MODEL_DIM, MATRIX_LR, etc.) directly in `run_experiment.sh` before the training command. See the env vars table in the README for the full list.
- For 8-GPU submission runs, use `--nproc_per_node=8`, set `MAX_WALLCLOCK_SECONDS=600`, and edit `run_experiment.sh` to download the full dataset with `--train-shards 80` (8B tokens). Create an 8-GPU instance with `--num-gpus 8`.

