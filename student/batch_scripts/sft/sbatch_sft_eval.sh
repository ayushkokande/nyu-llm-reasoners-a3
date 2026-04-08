#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_eval_%j.out
#SBATCH --error=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_eval_%j.err

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-outputs/sft_run}"

SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/a3/nyu-llm-reasoners-a3"

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

uv run python -m student.evaluate \\
  --model \"${MODEL_PATH}\" \\
  --max-examples 500 \\
  --gpu-memory-utilization 0.85

echo \"=== Done ===\"
"
