#!/bin/bash
#SBATCH --job-name=math_baseline
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# Log paths must be absolute: ./logs depends on cwd when you run sbatch (e.g. this folder vs repo root).
# Keep prefix in sync with REPO below.
#SBATCH --output=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/math_baseline_%j.out
#SBATCH --error=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/math_baseline_%j.err

# MATH zero-shot baseline: vLLM + Qwen2.5-Math-1.5B + JSONL + reward buckets.
#
# Edit SCRATCH / SIF / OVERLAY / REPO and the two #SBATCH log lines above.
# hf: huggingface-cli login in this env if math12k is gated.
#
# You can sbatch from anywhere, e.g.:
#   sbatch sbatch_math_baseline.sh
# from this directory, or from repo root with the path to this file.

set -euo pipefail

# --- edit these ---
SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/a3/nyu-llm-reasoners-a3"
# ------------------

mkdir -p "${REPO}/logs"

singularity exec --bind "${SCRATCH}" --nv \
  --overlay "${OVERLAY}" \
  "${SIF}" \
  /bin/bash -c "
set -euo pipefail

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
export PATH=${SCRATCH}/tools/bin:\$PATH
export UV_CACHE_DIR=${SCRATCH}/.uv_cache

cd \"${REPO}\"

echo \"=== math_baseline | job \${SLURM_JOB_ID:-local} ===\"
echo \"Repo: \$(pwd)\"
echo \"CUDA: \${CUDA_VISIBLE_DEVICES:-unset}\"

OUT=\"outputs/math_part_a_\${SLURM_JOB_ID:-local}.jsonl\"
mkdir -p outputs

uv run python -m student.evaluate \\
  --math-baseline \\
  --log-jsonl \"\${OUT}\" \\
  --max-examples 500 \\
  --model Qwen/Qwen2.5-Math-1.5B \\
  --gpu-memory-utilization 0.85

echo \"Wrote: \${OUT}\"
"
