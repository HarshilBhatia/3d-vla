#!/usr/bin/env bash
# Watch the completed 100k Video-DeltaM run plus its 100k->150k continuation.
# Historical 50k--100k intermediate checkpoints are submitted immediately;
# later 10k checkpoints are submitted atomically as training writes them.

#SBATCH --job-name=vdm-ckpt-watch
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=16:00:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/eval/vdm-ckpt-watch-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/eval/vdm-ckpt-watch-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT_OVERRIDE:-${SLURM_SUBMIT_DIR:-/home/harshilb/3dfa_unified}}"
cd "$REPO_ROOT"
python scripts/eval/watch_checkpoint_ladder.py \
  --config instructions/eval_plans/video_deltam_checkpoint_ladder_v01.json \
  --run video_deltam train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume \
  --run video_deltam train_logs/PerAct2/peract2_orbital_video_deltam_external_last100k_resume50k \
  --stop-step 150000 \
  --poll-seconds 120
