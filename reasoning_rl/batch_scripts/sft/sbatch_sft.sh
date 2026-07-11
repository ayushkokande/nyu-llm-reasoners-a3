#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --partition=g4-standard-48
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=sft_%j.out
#SBATCH --error=sft_%j.err
set -euo pipefail

SAMPLES="${SAMPLES:-512}"
LR="${LR:-2e-5}"
BS="${BS:-1}"
GA="${GA:-8}"
EVAL_EVERY="${EVAL_EVERY:-50}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
VLLM_DEVICE="${VLLM_DEVICE:-}"

SCRATCH="${SCRATCH:-/scratch/${USER}}"
SIF="${SIF:-${SCRATCH}/ubuntu-20.04.3.sif}"
OVERLAY="${OVERLAY:-${SCRATCH}/overlay-25GB-500K.ext3:ro}"
REPO="${REPO:-${SCRATCH}/math-reasoning-rl}"

LR_SAFE=$(printf '%s' "${LR}" | sed 's/[^A-Za-z0-9]/_/g')

if [ -n "${VLLM_DEVICE}" ]; then
    VLLM_ARG="--vllm-device ${VLLM_DEVICE}"
else
    VLLM_ARG=""
fi

EPOCH_SUFFIX=""
if [ "${NUM_EPOCHS}" != "1" ]; then
    EPOCH_SUFFIX="_e${NUM_EPOCHS}"
fi

if [ "${SAMPLES}" = "full" ]; then
    SAMPLE_ARG=""
    RUN_NAME="sft_full_bs${BS}_ga${GA}${EPOCH_SUFFIX}_lr${LR}"
    OUT_DIR="outputs/sft_full_bs${BS}_ga${GA}${EPOCH_SUFFIX}_${LR_SAFE}"
else
    SAMPLE_ARG="--max-train-samples ${SAMPLES}"
    RUN_NAME="sft_n${SAMPLES}_bs${BS}_ga${GA}${EPOCH_SUFFIX}_lr${LR}"
    OUT_DIR="outputs/sft_n${SAMPLES}_bs${BS}_ga${GA}${EPOCH_SUFFIX}_${LR_SAFE}"
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

echo \"=== SFT | job \${SLURM_JOB_ID:-local} | samples=${SAMPLES} lr=${LR} bs=${BS} ga=${GA} epochs=${NUM_EPOCHS} eval_every=${EVAL_EVERY} ===\"
echo \"Repo: \$(pwd)\"
echo \"CUDA: \${CUDA_VISIBLE_DEVICES:-unset}\"
echo \"vLLM device: ${VLLM_DEVICE:-none (single-GPU generate eval)}\"

mkdir -p outputs

uv sync --extra sft

uv run python -m reasoning_rl.sft_train \\
  ${SAMPLE_ARG} \\
  --num-epochs ${NUM_EPOCHS} \\
  --learning-rate ${LR} \\
  --per-device-batch-size ${BS} \\
  --gradient-accumulation-steps ${GA} \\
  --max-grad-norm 1.0 \\
  --bf16 \\
  --device cuda:0 \\
  ${VLLM_ARG} \\
  --eval-every ${EVAL_EVERY} \\
  --math-eval-n 128 \\
  --eval-max-new-tokens 2048 \\
  --output-dir \"${OUT_DIR}\" \\
  --wandb-project math-reasoning-rl-sft \\
  --wandb-run-name \"${RUN_NAME}\"

echo \"=== Done: checkpoint at ${OUT_DIR} ===\"
"
