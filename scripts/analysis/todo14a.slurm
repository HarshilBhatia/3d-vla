#!/bin/bash
#SBATCH --job-name=todo14a
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/todo14a-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-1-20,grogu-4-23
# TODO 14a: PART=attn | ctok on exp 4a (pre-LN extractor) vs exp 3 at STEP, both EMA at their saved K=5; extra args pass through.
cd /home/harshilb/3dfa_unified
PY=/home/harshilb/miniconda3/envs/3dfa/bin/python
L=train_logs/PerAct2
M=(--model exp4a_${STEP}=$L/peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch/interm_step_${STEP}.pth:ema:ckpt
   --model exp3_${STEP}=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_${STEP}.pth:ema:ckpt)
O=${OUT:-results/analysis/todo14a}
case "$PART" in
  attn) $PY scripts/analysis/camera_attention.py --faithful "${M[@]}" --out-dir $O/camera_attention "$@" ;;
  ctok) $PY scripts/analysis/ctoken_behaviour.py "${M[@]}" --out-dir $O/ctoken_behaviour "$@" ;;
esac
