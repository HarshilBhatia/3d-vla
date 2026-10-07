#!/bin/bash
#SBATCH --job-name=dm_per_layer
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/dm_per_layer-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18
# Records per-layer delta_M for exp 3 (110k, 200k EMA), VideoPooled 84k and warm random history 50k; extra args pass through.
cd /home/harshilb/3dfa_unified
L=train_logs/PerAct2
/home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/delta_m_per_layer.py \
  --model exp3_200k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_200000.pth:ema \
  --model exp3_110k=$L/peract2_orbital_vid_pooled_randhist_deltam_scratch/interm_step_110000.pth:ema \
  --model videopooled84k=$L/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth:raw \
  --model randhist_warm50k=$L/peract2_orbital_vid_deltam_pooled_randhist_notime/interm_step_50000.pth:raw \
  "$@"
