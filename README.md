# Math Reasoning with Reinforcement Learning

This project trains and evaluates small language models for mathematical
reasoning with supervised fine-tuning (SFT) and Group-Relative Policy
Optimization (GRPO). The experiments use `Qwen/Qwen2.5-Math-1.5B` as the base
model, evaluate mathematical correctness on MATH-12K, warm-start the policy on
Prime Intellect math reasoning traces, and refine it with reward-based training
on Countdown tasks.

The repository includes from-scratch implementations of the SFT data masking
pipeline, GRPO reward normalization and policy-gradient losses, single- and
multi-GPU training loops, vLLM-backed evaluation, a math-answer reward/grader,
Slurm launch scripts, and a small Streamlit dashboard for inspecting the
measured results.

Scaffolding and tests are adapted from [Stanford CS336](https://github.com/stanford-cs336/); credit to the Stanford course staff for the original materials.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.

Run project commands through `uv`:

```sh
uv run <python_file_path>
```

### pyproject-mac.toml

The `pyproject-mac.toml` and `uv-mac.lock` variants omit `vllm`, which is
CUDA-only. They are useful for local CPU-side development and unit tests; full
training and fast evaluation are intended for CUDA machines.

### Run Unit Tests

```bash
uv run pytest
```

### Train And Evaluate

```bash
uv run python -m reasoning_rl.sft_train --max-train-samples 128 --no-wandb
uv run python -m reasoning_rl.grpo_train --n-grpo-steps 200 --no-wandb
uv run python -m reasoning_rl.evaluate --model Qwen/Qwen2.5-Math-1.5B --max-examples 500
```

### Dashboard

```bash
uv run --with streamlit streamlit run dashboard/app.py
```
