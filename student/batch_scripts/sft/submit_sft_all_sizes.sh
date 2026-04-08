#!/bin/bash
#SBATCH --job-name=sft_submit_all
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_submit_all_%j.out
#SBATCH --error=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_submit_all_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_sft.sh"

BEST_LR="2e-5"
BEST_BS=1
BEST_GA=8

SIZES=(128 256 512 1024 full)

for size in "${SIZES[@]}"; do
  echo "Submitting: samples=${size} lr=${BEST_LR} bs=${BEST_BS} ga=${BEST_GA}"
  sbatch --export="SAMPLES=${size},LR=${BEST_LR},BS=${BEST_BS},GA=${BEST_GA}" "${SBATCH_SCRIPT}"
done

echo "Submitted ${#SIZES[@]} jobs."
