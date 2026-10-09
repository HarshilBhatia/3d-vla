#!/bin/bash
#SBATCH --job-name=exp4_dm
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/exp4_dm-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/analysis/exp4_dm-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-1-20,grogu-4-23
# TODO 1-style per-layer delta_M and C-token activity on real data for exp 4a/4b vs exp 3 at matching steps.
set -e
cd /home/harshilb/3dfa_unified
export PYTHONPATH=$PWD
L=train_logs/PerAct2
M=(--model exp4a_prenorm_50k=$L/peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch/interm_step_50000.pth:ema:ckpt
   --model exp4b_inputskip_30k=$L/peract2_orbital_vid_pooled_randhist_deltam_inputskip_scratch/interm_step_30000.pth:ema:ckpt
   --model exp3_50k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_50000.pth:ema:ckpt
   --model exp3_30k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_30000.pth:ema:ckpt)
PY=/home/harshilb/miniconda3/envs/3dfa/bin/python
$PY scripts/analysis/delta_m_per_layer.py --faithful "${M[@]}" --out-dir results/analysis/exp4_delta_m
$PY scripts/analysis/delta_m_trunk_activity.py --faithful "${M[@]}" --out results/analysis/exp4_delta_m/trunk_activity.json
