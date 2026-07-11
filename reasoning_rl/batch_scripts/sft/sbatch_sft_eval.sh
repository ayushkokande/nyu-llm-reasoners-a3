#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=sft_eval_%j.out
#SBATCH --error=sft_eval_%j.err

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-outputs/sft_run}"

SCRATCH="${SCRATCH:-/scratch/${USER}}"
SIF="${SIF:-${SCRATCH}/ubuntu-20.04.3.sif}"
OVERLAY="${OVERLAY:-${SCRATCH}/overlay-25GB-500K.ext3:ro}"
REPO="${REPO:-${SCRATCH}/math-reasoning-rl}"

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

echo \"=== SFT Eval | job \${SLURM_JOB_ID:-local} | model=${MODEL_PATH} ===\"
echo \"Repo: \$(pwd)\"
echo \"CUDA: \${CUDA_VISIBLE_DEVICES:-unset}\"

uv run python -m reasoning_rl.evaluate \\
  --model \"${MODEL_PATH}\" \\
  --max-examples 500 \\
  --gpu-memory-utilization 0.85

echo \"=== Done ===\"
"
