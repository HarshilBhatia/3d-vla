#!/bin/bash
#SBATCH --job-name=todo8
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/todo8-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18
# TODO 8: PART=dm (delta_M per layer + trunk) or attn (camera attention); non-history models at K=1 as in online eval, history models at their saved K.
cd /home/harshilb/3dfa_unified
PY=/home/harshilb/miniconda3/envs/3dfa/bin/python
L=train_logs/PerAct2
DM=$L/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth
BASE=$L/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth
if [ "$PART" = dm ]; then
  M=(--model deltam140k=$DM:raw:1
     --model exp3_200k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth:ema:ckpt
     --model exp3_110k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_110000.pth:ema:ckpt
     --model videopooled84k=$L/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth:raw:ckpt
     --model randhist_warm50k=$L/peract2_orbital_vid_deltam_pooled_randhist_notime/interm_step_50000.pth:raw:ckpt)
  $PY scripts/analysis/delta_m_per_layer.py --faithful "${M[@]}" --out-dir results/analysis/todo8/delta_m_per_layer "$@"
  $PY scripts/analysis/delta_m_trunk_activity.py --faithful "${M[@]}" --out results/analysis/todo8/delta_m_per_layer/trunk_activity.json
else
  $PY scripts/analysis/camera_attention.py --faithful \
    --model deltam140k=$DM:raw:1 --model base160k=$BASE:raw:1 \
    --model exp3_200k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth:ema:ckpt \
    --model adaln1b_200k=$L/peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch/interm_step_200000.pth:ema:ckpt \
    --model controlA_200k=$L/peract2_orbital_vid_pooled_noalign_randhist_scratch/interm_step_200000.pth:raw:ckpt \
    --model ema20k_randhist_15k=$L/peract2_orbital_ema20k_randhist/interm_step_15000.pth:ema:ckpt \
    --out-dir results/analysis/todo8/camera_attention "$@"
fi
