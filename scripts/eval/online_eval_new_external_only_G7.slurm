#!/usr/bin/env bash
# Compatibility entry point for the former condition-specific G7 launcher.
# Historical result folders are untouched. New runs use the declarative plan
# and canonical output layout; invoke online_eval_plan.slurm directly for new
# campaigns.

#SBATCH --job-name=g7-ext-cal
#SBATCH --output=/home/harshilb/3dfa_unified/logs/eval/g7-ext-cal-%A_%a.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/eval/g7-ext-cal-%A_%a.err
#SBATCH --partition=all
#SBATCH --array=0-103
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --exclude=grogu-3-30,grogu-4-23

set -euo pipefail
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-/home/harshilb/3dfa_unified}}"
exec "$REPO_ROOT/scripts/eval/online_eval_plan.slurm" \
  "${1:-instructions/eval_plans/g7_external_calibration.json}"
