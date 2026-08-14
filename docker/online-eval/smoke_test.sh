#!/usr/bin/env bash
# Verify the online-eval image: env vars, PyRep/RLBench/CoppeliaSim, python deps.
# Expected to be run under xvfb-run (see run.sh --smoke-test).
set -euo pipefail

echo "COPPELIASIM_ROOT = ${COPPELIASIM_ROOT}"
echo "QT_QPA_PLATFORM  = ${QT_QPA_PLATFORM:-<unset, correct for headless>}"
echo "DISPLAY          = ${DISPLAY:-<none>}"
test -f "${COPPELIASIM_ROOT}/libcoppeliaSim.so"
grep -q allowOldEduRelease "${COPPELIASIM_ROOT}/system/usrset.txt"
echo "coppeliasim      OK"

python - <<'PY'
import sys
print("python          ", sys.version.split()[0])

from pyrep import PyRep
print("pyrep           OK")

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode
print("rlbench         OK")

import torch
print("torch           ", torch.__version__, "cuda_available=", torch.cuda.is_available())

import transformers
assert transformers.is_torch_available(), "transformers has no torch backend (version skew)"
print("transformers    ", transformers.__version__, "torch backend OK")

import open3d, zarr, kornia, diffusers, hydra, scipy, cv2
assert zarr.__version__.startswith("2."), f"zarr must be 2.x, got {zarr.__version__}"
from zarr.storage import DirectoryStore
from zarr import LRUStoreCache
print("deps            OK (open3d/zarr%s/kornia/diffusers/hydra/scipy/cv2)" % zarr.__version__)

import numpy as np
assert np.__version__.startswith("1."), f"numpy must be 1.x, got {np.__version__}"
print("numpy           ", np.__version__)
PY

echo
echo "SMOKE TEST PASSED"
