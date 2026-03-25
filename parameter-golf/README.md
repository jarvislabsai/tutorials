# Parameter Golf on JarvisLabs

Run [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf) challenge on JarvisLabs GPUs. One script, one command.

Parameter Golf is a competition to train the best language model that fits in a 16MB artifact. Leaderboard submissions must train in under 10 minutes on 8xH100s, though non-record submissions with unlimited compute are also accepted. The metric is bits per byte (BPB) on the FineWeb validation set, a tokenizer-agnostic compression score where lower is better (like golf). The competition runs from March 18 to April 30, 2026.

The baseline scores 1.2244 BPB.

You can also use this with an [autoresearch](https://github.com/karpathy/autoresearch)-style loop using Claude Code, Codex, or any coding agent. A [program.md](program.md) is included that you can modify to fit your needs. The `[jl` CLI](https://docs.jarvislabs.ai/cli/) is designed to be agent-first: agents can create GPU instances, run experiments, monitor logs, and manage billing on your behalf. Run `jl setup` to install the agent skill.

## Quick start

Install the JarvisLabs CLI and authenticate:

```bash
uv tool install jarvislabs   # or: pip install jarvislabs
jl setup
```

Run the baseline on a single H100:

```bash
jl run run_experiment.sh --gpu H100
```

That's it. This creates an H100 instance, uploads `run_experiment.sh`, clones the parameter-golf repo, installs dependencies, downloads the FineWeb dataset, and starts training. The whole setup takes about 2 minutes before training begins.

Monitor your run:

```bash
jl run logs <run_id> --tail 50
```

You'll see output like:

```
step:100/20000 train_loss:3.5637 train_time:33317ms step_avg:333.17ms
step:200/20000 train_loss:2.8703 train_time:66661ms step_avg:333.31ms
...
final_int8_zlib_roundtrip_exact val_loss:2.70827121 val_bpb:1.60399076
```

Download your model artifact:

```bash
jl download <machine_id> /home/parameter-golf/final_model.int8.ptz
```

This is the compressed int8+zlib model file you'd submit. The competition requires code + model under 16,000,000 bytes (decimal, not MiB). If you need the raw unquantized checkpoint for debugging, it's at `/home/parameter-golf/final_model.pt`.

When you're done, stop billing:

```bash
jl pause <machine_id> --yes    # keeps your data, stops compute charges
jl destroy <machine_id> --yes  # deletes everything
```

## What run_experiment.sh does

The script clones the official [openai/parameter-golf](https://github.com/openai/parameter-golf) repo and runs the baseline `train_gpt.py` on JarvisLabs infra. The full training setup (model architecture, optimizer, data loading, quantization, evaluation) comes from OpenAI's repo. Before your script runs, `jl run` creates a Python venv at `/home/.venv` with `--system-site-packages`, so torch and numpy from the PyTorch template are already available without reinstalling. The script only installs the extras the template doesn't have. For more on how `jl run` manages environments, see the [CLI docs](https://docs.jarvislabs.ai/cli/).

The default config trains a 9-layer, 512-dim GPT with GQA (8 heads, 4 KV heads), tied embeddings, 1024-token SentencePiece vocabulary, and the Muon optimizer. Every hyperparameter is configurable via environment variables. See the [parameter-golf README](https://github.com/openai/parameter-golf#readme) for the full list.

## Full 8-GPU submission run

OpenAI recommends iterating on a single H100 first to test ideas cheaply, then scaling up for the real submission. A single 8xH100 run costs roughly $3-5 (check `jl gpus` for current pricing).

For leaderboard submissions, edit `run_experiment.sh` to set:

- `--nproc_per_node=8` (match your GPU count)
- `MAX_WALLCLOCK_SECONDS=600` (the competition's 10-minute cap)
- `--train-shards 80` for the full 8B token dataset

Then launch:

```bash
jl run run_experiment.sh --gpu H100 --num-gpus 8 --keep --yes
```

## Running your own code

If you have your own training script or a fork of `train_gpt.py`, you have two options.

**Single file**: Drop your modified `train_gpt.py` and a `run_experiment.sh` that sets it up. Same pattern as the baseline.

**Directory mode**: If you have a project folder with `requirements.txt` or `pyproject.toml`:

```bash
jl run <project_folder> --script <entry_script> --gpu H100 --keep --yes

# example:
jl run ./my-golf --script train.py --gpu H100 --keep --yes
```

`<project_folder>` is your local directory, `<entry_script>` is the Python or bash file to run inside it. This syncs the folder to `/home/<project_folder>/` on the instance, creates a venv inside it, auto-detects and installs dependencies from `requirements.txt` or `pyproject.toml`, then runs your script.

## Using with coding agents

`jl setup` installs the JarvisLabs agent skill to `~/.agents/skills/jarvislabs/SKILL.md` (works with Codex, Cursor, Gemini CLI, and most agents) and optionally to `~/.claude/skills/jarvislabs/SKILL.md` for Claude Code. Once installed, you can tell your agent something like:

> "Read program.md. Create an H100 instance on JarvisLabs. Run the autoresearch loop. Keep improving val_bpb. Commit each improvement."

See [program.md](program.md) for the full autoresearch spec. It's a starting point; modify it to fit your workflow.

## Useful environment variables

All hyperparameters in `train_gpt.py` are configurable via env vars. Common ones:


| Variable                | Default | What it does                           |
| ----------------------- | ------- | -------------------------------------- |
| `NUM_LAYERS`            | 9       | Transformer depth                      |
| `MODEL_DIM`             | 512     | Hidden dimension                       |
| `NUM_HEADS`             | 8       | Attention heads                        |
| `NUM_KV_HEADS`          | 4       | KV heads (GQA)                         |
| `MLP_MULT`              | 2       | MLP expansion factor                   |
| `MATRIX_LR`             | 0.04    | Muon learning rate for matrix params   |
| `EMBED_LR`              | 0.6     | Adam learning rate for embeddings      |
| `TRAIN_SEQ_LEN`         | 1024    | Training sequence length               |
| `MAX_WALLCLOCK_SECONDS` | 600     | Training time cap in seconds           |
| `VAL_LOSS_EVERY`        | 1000    | Validation frequency (0 = only at end) |
| `TRAIN_LOG_EVERY`       | 200     | Training log frequency                 |
| `SEED`                  | 1337    | Random seed                            |


Set these in your `run_experiment.sh` before the training command. For example, to try 10 layers with a lower learning rate:

```bash
NUM_LAYERS=10 \
MATRIX_LR=0.02 \
python3 -m torch.distributed.run --standalone --nproc_per_node=1 --local-addr 127.0.0.1 train_gpt.py
```

