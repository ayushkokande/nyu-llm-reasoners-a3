#!/usr/bin/env bash
#SBATCH --job-name=sft_submit_n128_n256
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=g4-standard-48
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_submit_n128_n256_%j.out
#SBATCH --error=/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs/sft_submit_n128_n256_%j.err
#SBATCH --mail-user=ak13124@nyu.edu
#SBATCH --mail-type=END
#
# Submit two SFT jobs: n=128 and n=256 with lr=5e-5, bs=1, ga=2 (eff_bs=2).
# NUM_EPOCHS>1 repeats the same training subset so train loss can decay over more steps.
# (n=128: 64 steps/epoch; n=256: 128 steps/epoch — see logs for total optimizer steps.)
# Uses student/batch_scripts/sft/sbatch_sft.sh.
# Run: sbatch submit_sft_n128_n256_lr5e5_bs1_ga2.sh   OR   bash submit_sft_n128_n256_lr5e5_bs1_ga2.sh
set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_sft.sh"
LOG_DIR="/scratch/ak13124/a3/nyu-llm-reasoners-a3/logs"

LR=5e-5
BS=1
GA=2
NUM_EPOCHS=4

mkdir -p "${LOG_DIR}"

SAMPLES_LIST=(128 256 512 1024 full)
JOB_IDS=()

for samples in "${SAMPLES_LIST[@]}"; do
  case "${samples}" in
    128) eval_every=10 ;;
    256) eval_every=20 ;;
    512) eval_every=40 ;;
    1024) eval_every=80 ;;
    full) eval_every=200 ;;
    *) eval_every=50 ;;
  esac

  echo "Submitting: samples=${samples} lr=${LR} bs=${BS} ga=${GA} (eff_bs=$((BS * GA))) epochs=${NUM_EPOCHS} eval_every=${eval_every}"
  job_id=$(sbatch --parsable --export="SAMPLES=${samples},LR=${LR},BS=${BS},GA=${GA},NUM_EPOCHS=${NUM_EPOCHS},EVAL_EVERY=${eval_every}" "${SBATCH_SCRIPT}")
  JOB_IDS+=("${job_id}")
  echo "Submitted training job: ${job_id}"
  echo "  logs: ${LOG_DIR}/sft_${job_id}.out and ${LOG_DIR}/sft_${job_id}.err"
done

echo "Submitted ${#JOB_IDS[@]} jobs."
echo "Watch progress with:"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo "  tail -f ${LOG_DIR}/sft_${JOB_IDS[0]}.out"
