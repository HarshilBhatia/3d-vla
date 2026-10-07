#!/bin/bash
#SBATCH --job-name=act_patch9
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/act_patch9-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-4-23,grogu-1-20
# TODO 9: activation patching on non-history Base 160k and DeltaM 140k (K=1, as online eval) and history models (saved K) with the TODO 9 combinations.
cd /home/harshilb/3dfa_unified
L=train_logs/PerAct2
/home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/activation_patching.py --faithful \
  --model deltam140k=$L/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth:raw:1 \
  --model base160k=$L/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth:raw:1 \
  --model exp3_200k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth:ema:ckpt \
  --model adaln1b_200k=$L/peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch/interm_step_200000.pth:ema:ckpt \
  --model controlA_200k=$L/peract2_orbital_vid_pooled_noalign_randhist_scratch/interm_step_200000.pth:raw:ckpt \
  --out-dir results/analysis/todo9/activation_patching "$@"
