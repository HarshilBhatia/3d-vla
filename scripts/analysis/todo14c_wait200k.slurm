#!/bin/bash
#SBATCH --job-name=todo14c_wait
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/todo14c_wait-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=16:00:00
#SBATCH --partition=all
# CPU-only watcher: once exp 4a's 200k checkpoint exists, submit TODO 14c's patching and following jobs at STEP=200000.
cd /home/harshilb/3dfa_unified
F=train_logs/PerAct2/peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch/interm_step_200000.pth
until [ -f "$F" ]; do sleep 300; done
sleep 300
sbatch --export=ALL,PART=patch,STEP=200000 scripts/analysis/todo14c.slurm
sbatch --export=ALL,PART=follow,STEP=200000 scripts/analysis/todo14c.slurm
