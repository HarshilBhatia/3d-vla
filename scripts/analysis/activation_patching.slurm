#!/bin/bash
#SBATCH --job-name=act_patch
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/act_patch-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-4-23,grogu-1-20
# Clean<->heavy60 activation patching for exp 3, 1b and control A (TODO 7); extra args pass through.
cd /home/harshilb/3dfa_unified
L=train_logs/PerAct2
/home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/activation_patching.py \
  --model exp3_200k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth:ema \
  --model adaln1b_200k=$L/peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch/interm_step_200000.pth:ema \
  --model controlA_200k=$L/peract2_orbital_vid_pooled_noalign_randhist_scratch/interm_step_200000.pth:raw \
  "$@"
