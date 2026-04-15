#!/bin/bash
#SBATCH --job-name=sft_submit_all
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_submit_all_%j.out
#SBATCH --error=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_submit_all_%j.err

set -euo pipefail

# Stage 2 of the SFT workflow: run this only after the n=512 sweep and
# reuse the selected hyperparameters on the full size sweep.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_sft.sh"

BEST_LR="${BEST_LR:-}"
BEST_BS="${BEST_BS:-}"
BEST_GA="${BEST_GA:-}"

if [ -z "${BEST_LR}" ] || [ -z "${BEST_BS}" ] || [ -z "${BEST_GA}" ]; then
  echo "Run submit_sft_grid_n512.sh first, then resubmit this script with"
  echo "the exact BEST_LR/BEST_BS/BEST_GA chosen from the n=512 sweep."
  echo
  echo "Example:"
  echo "  sbatch --export=BEST_LR=2e-5,BEST_BS=1,BEST_GA=8 submit_sft_all_sizes.sh"
  exit 1
fi

SIZES=(128 256 512 1024 full)

echo "Using selected n=512 hyperparameters: lr=${BEST_LR} bs=${BEST_BS} ga=${BEST_GA}"
for size in "${SIZES[@]}"; do
  echo "Submitting: samples=${size} lr=${BEST_LR} bs=${BEST_BS} ga=${BEST_GA}"
  sbatch --export="SAMPLES=${size},LR=${BEST_LR},BS=${BEST_BS},GA=${BEST_GA}" "${SBATCH_SCRIPT}"
done

echo "Submitted ${#SIZES[@]} jobs."
