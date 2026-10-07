#!/bin/bash
#SBATCH --job-name=probe_feats
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/probe_feats-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18
# Extracts probe features for one model per call (TODO 5); pass --model label=ckpt:raw|ema|none and any extra args.
cd /home/harshilb/3dfa_unified
/home/harshilb/miniconda3/envs/3dfa/bin/python scripts/analysis/probe_features.py "$@"
