#!/usr/bin/env bash
# Submit once alongside the paired 100k→200k continuations.  The Python watcher
# immediately submits an unthrottled 12-cell array for each new 10k checkpoint.

#SBATCH --job-name=sbrs-ckpt-watch
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=23:59:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/eval/sbrs-ckpt-watch-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/eval/sbrs-ckpt-watch-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-/home/harshilb/3dfa_unified}}"
cd "$REPO_ROOT"
python scripts/eval/watch_checkpoint_ladder.py \
  --run base train_logs/PerAct2/peract2_orbital_new_external_only_miscal_base_200k \
  --run view_align train_logs/PerAct2/peract2_orbital_new_external_only_miscal_deltam_external_200k \
  --stop-step 200000
