#!/bin/bash
# Set up the LIBERO simulation venv on the B200 devbox (uv, no docker).
# One venv serves LIBERO / LIBERO-plus / LIBERO-PRO; the variant is chosen
# per run via PYTHONPATH because all three ship a conflicting `libero` package.
set -euo pipefail

ROOT=/k8s-nfs/harsvbha/3dfa
VENV=$ROOT/libero/.venv
UV=/root/.local/bin/uv
export UV_CACHE_DIR=/k8s-nfs/harsvbha/uv-cache

$UV venv --python 3.11 "$VENV"

# Sim stack. The repos pin ancient versions (numpy 1.22, transformers 4.21)
# that modern forks ignore; robosuite 1.4.1 + numpy<2 is the working combo
# used by OpenVLA-style LIBERO evals.
$UV pip install --python "$VENV/bin/python" \
    "numpy<2" scipy "robosuite==1.4.1" "bddl==1.0.1" \
    "mujoco>=2.3,<3.2" easydict einops "gym==0.25.2" cloudpickle future \
    "hydra-core>=1.2" opencv-python-headless matplotlib pillow imageio \
    h5py "zarr>=2.16,<3" "numcodecs>=0.12,<0.16" tqdm pyyaml

# robomimic 0.2.0 needed only for LIBERO's lifelong-learning training code,
# which we don't use; skip it to avoid its torch pin.

echo "OK venv=$VENV"
"$VENV/bin/python" -c "import robosuite, mujoco, numpy; print('robosuite', robosuite.__version__, 'mujoco', mujoco.__version__, 'numpy', numpy.__version__)"
