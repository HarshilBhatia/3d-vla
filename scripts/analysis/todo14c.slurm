#!/bin/bash
#SBATCH --job-name=todo14c
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/todo14c-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-4-23,grogu-1-20
# TODO 14c: PART=follow|patch, STEP=120000|200000; exp 4a (pre-LN extractor) vs exp 3, both EMA, loaded through load_model_for_evaluation.
cd /home/harshilb/3dfa_unified
L=train_logs/PerAct2
M=(--model exp4a_${STEP}=$L/peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch/interm_step_${STEP}.pth:ema:ckpt
   --model exp3_${STEP}=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_${STEP}.pth:ema:ckpt)
if [ "$PART" = follow ]; then
  /home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/follow_miscal.py --eval-loader "${M[@]}" --out-dir results/analysis/todo14c/follow_miscal "$@"
else
  /home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/activation_patching.py --eval-loader "${M[@]}" --out-dir results/analysis/todo14c/activation_patching "$@"
fi
