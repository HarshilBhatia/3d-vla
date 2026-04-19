#!/bin/bash
#SBATCH --job-name=strip_mesh_points
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/strip_mesh_points_%j.out
#SBATCH --error=logs/strip_mesh_points_%j.err

set -euo pipefail
ulimit -c 0

REPO_DIR="/ocean/projects/cis240058p/hbhatia1/3d-vla"
COPPELIASIM_ROOT="${REPO_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/peract_raw_ood}"

echo "Stripping mesh_points from: ${DATA_ROOT}"

mkdir -p "${REPO_DIR}/logs"

apptainer exec \
    --env "COPPELIASIM_ROOT=${COPPELIASIM_ROOT}" \
    --env "LD_LIBRARY_PATH=${COPPELIASIM_ROOT}" \
    --env "PYTHONPATH=${REPO_DIR}/RLBench:${REPO_DIR}" \
    --bind "${REPO_DIR}:${REPO_DIR}" \
    /ocean/projects/cis240058p/hbhatia1/containers/3dfa-sandbox.sif python3 - <<EOF
import pickle, os, glob

root = "${DATA_ROOT}"
pkls = sorted(glob.glob(f"{root}/**/low_dim_obs.pkl", recursive=True))
print(f"Found {len(pkls)} pkl files")

for path in pkls:
    before = os.path.getsize(path)
    with open(path, 'rb') as f:
        demo = pickle.load(f)
    changed = False
    for obs in demo:
        if hasattr(obs, 'mesh_points'):
            del obs.mesh_points
            changed = True
    if changed:
        with open(path, 'wb') as f:
            pickle.dump(demo, f)
        after = os.path.getsize(path)
        print(f"  {os.path.relpath(path, root):70s}  {before//1024:>6} KB -> {after//1024:>6} KB")
    else:
        print(f"  SKIP (no mesh_points): {os.path.relpath(path, root)}")

print("Done.")
EOF
