#!/bin/bash
#SBATCH --job-name=ctoken
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/ctoken-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18,grogu-1-20,grogu-4-23
# TODO 10: C-token behaviour for Base 160k, DeltaM 140k (K=1, as online eval), 1b and exp 3 at their saved K=5; extra args pass through.
cd /home/harshilb/3dfa_unified
L=train_logs/PerAct2
/home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/ctoken_behaviour.py \
  --model base160k=$L/peract2_orbital_new_external_only_miscal_base_200k/interm_step_160000.pth:raw:1 \
  --model deltam140k=$L/peract2_orbital_new_external_only_miscal_deltam_external_200k/interm_step_140000.pth:raw:1 \
  --model adaln1b_200k=$L/peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch/interm_step_200000.pth:ema:ckpt \
  --model exp3_200k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth:ema:ckpt \
  "$@"
