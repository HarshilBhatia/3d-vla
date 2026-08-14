#!/usr/bin/env bash
# Entrypoint for the 3DFA online-eval image.
#
# Establishes the CoppeliaSim / PyRep environment described in docs/commands.md
# and then execs whatever command was passed. PyRep and RLBench are already
# installed into /opt/venv (baked at build time) — nothing is installed here.
set -euo pipefail

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

# Must be unset for headless CoppeliaSim under Xvfb: the "offscreen" plugin
# cannot create the GL context CoppeliaSim's renderer needs.
unset QT_QPA_PLATFORM || true

export REPO_DIR="${REPO_DIR:-/workspace/3d_flowmatch_actor}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# CoppeliaSim/Qt want a writable XDG_RUNTIME_DIR.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

if [[ ! -d "${REPO_DIR}/online_evaluation_rlbench" ]]; then
    echo "[entrypoint] WARNING: ${REPO_DIR} does not look like the 3dfa repo." >&2
    echo "[entrypoint]          Mount it with -v /path/to/3d_flowmatch_actor:${REPO_DIR}" >&2
fi

exec "$@"
