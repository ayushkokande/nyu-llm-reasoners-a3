#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_%j.out
#SBATCH --error=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_%j.err

set -euo pipefail

SAMPLES="${SAMPLES:-512}"
LR="${LR:-2e-5}"
BS="${BS:-1}"
GA="${GA:-8}"

SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/a3/nyu-llm-reasoners-a3"

LR_SAFE=$(printf '%s' "${LR}" | sed 's/[^A-Za-z0-9]/_/g')

if [ "${SAMPLES}" = "full" ]; then
    SAMPLE_ARG=""
    RUN_NAME="sft_full_bs${BS}_ga${GA}_lr${LR}"
    OUT_DIR="outputs/sft_full_bs${BS}_ga${GA}_${LR_SAFE}"
else
    SAMPLE_ARG="--max-train-samples ${SAMPLES}"
    RUN_NAME="sft_n${SAMPLES}_bs${BS}_ga${GA}_lr${LR}"
    OUT_DIR="outputs/sft_n${SAMPLES}_bs${BS}_ga${GA}_${LR_SAFE}"
fi

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

echo \"=== SFT | job \${SLURM_JOB_ID:-local} | samples=${SAMPLES} lr=${LR} bs=${BS} ga=${GA} ===\"
echo \"Repo: \$(pwd)\"
echo \"CUDA: \${CUDA_VISIBLE_DEVICES:-unset}\"

mkdir -p outputs

uv run python -m student.sft_train \\
  ${SAMPLE_ARG} \\
  --learning-rate ${LR} \\
  --per-device-batch-size ${BS} \\
  --gradient-accumulation-steps ${GA} \\
  --max-grad-norm 1.0 \\
  --bf16 \\
  --device cuda:0 \\
  --vllm-device cuda:1 \\
  --eval-every 50 \\
  --math-eval-n 128 \\
  --eval-max-new-tokens 2048 \\
  --output-dir \"${OUT_DIR}\" \\
  --wandb-project nyu-llm-reasoners-a3-sft \\
  --wandb-run-name \"${RUN_NAME}\"

echo \"=== Done: checkpoint at ${OUT_DIR} ===\"
"
