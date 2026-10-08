#!/bin/bash
#SBATCH --job-name=fit_probes
#SBATCH --output=/home/harshilb/3dfa_unified/logs/analysis/fit_probes-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-5,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-4-18
# CPU-only probe fitting for one feature file (TODOs 5, 11): fit_probes.slurm <label> [features dir, default results/analysis/probes].
cd /home/harshilb/3dfa_unified
D=${2:-results/analysis/probes}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/harshilb/miniconda3/envs/3dfa/bin/python -W ignore scripts/analysis/fit_probes.py --features $D/$1.pt --out $D/fits/probe_results_$1.json --workers 8
