#!/bin/bash
#SBATCH --job-name=sft_submit_n512
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_submit_n512_%j.out
#SBATCH --error=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_submit_n512_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_sft.sh"

SAMPLES=512
LR_LIST=(1e-5 2e-5 5e-5 1e-4 2e-4)
BS_GA_PAIRS=(
  "1 2"
  "1 4"
  "1 8"
  "2 8"
  "2 16"
)

submitted=0
for lr in "${LR_LIST[@]}"; do
  for pair in "${BS_GA_PAIRS[@]}"; do
    bs=$(echo "$pair" | awk '{print $1}')
    ga=$(echo "$pair" | awk '{print $2}')
    eff=$((bs * ga))
    echo "Submitting: samples=${SAMPLES} lr=${lr} bs=${bs} ga=${ga} (eff_bs=${eff})"
    sbatch --export="SAMPLES=${SAMPLES},LR=${lr},BS=${bs},GA=${ga}" "${SBATCH_SCRIPT}"
    submitted=$((submitted + 1))
  done
done

echo "Submitted ${submitted} jobs."
