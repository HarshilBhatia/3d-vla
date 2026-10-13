#!/usr/bin/env bash
# Builds a dense lift_ball zarr (args pass to peract2_dense_to_zarr.py, e.g. --target chunk --horizon 8 --out ...).
#SBATCH --job-name=p2dense_convert
#SBATCH --partition=shubhamlong,all
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-3-10,grogu-3-15,grogu-3-20,grogu-3-25,grogu-3-30,grogu-3-5,grogu-4-18,grogu-1-20,grogu-4-23
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/data/p2dense_convert-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/data/p2dense_convert-%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT_OVERRIDE:-/home/harshilb/3dfa_unified}"
RAW_ROOT="${RAW_ROOT:-/grogu/datasets/hbhatia/3dfa_dense/peract2_lift_ball/raw}"
# Runs inside the sim container so low_dim_obs.pkl unpickles into real RLBench Observations.
apptainer exec --fakeroot \
    --bind "$REPO_ROOT:/workspace/3dfa_unified" \
    --bind "/home/harshilb/peract2_eval_deps/PyRep:/workspace/PerAct2PyRep" \
    --bind "/home/harshilb/peract2_eval_deps/RLBench:/workspace/PerAct2RLBench" \
    --bind "/home/harshilb/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04:/opt/CoppeliaSim" \
    --bind "/grogu:/grogu" "${APPTAINER_IMAGE:-/home/harshilb/my_eval_env}" bash -c \
    'export PYTHONPATH=/workspace/3dfa_unified:/workspace/PerAct2PyRep:/workspace/PerAct2RLBench; \
     export COPPELIASIM_ROOT=/opt/CoppeliaSim LD_LIBRARY_PATH=/opt/CoppeliaSim PYTHONUNBUFFERED=1 BLOSC_NTHREADS=1; \
     cd /workspace/3dfa_unified; \
     /root/miniconda3/envs/3dfa/bin/python data/processing/convert_to_zarr/peract2_dense_to_zarr.py "$@"' _ \
    --root "$RAW_ROOT" --tasks bimanual_lift_ball --workers 8 "$@"
