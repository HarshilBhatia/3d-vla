#!/usr/bin/env bash
# Run the 3DFA online-evaluation image with the repo bind-mounted.
#
# Usage:
#   # interactive shell
#   bash docker/online-eval/run.sh
#
#   # verify PyRep / RLBench / CoppeliaSim under Xvfb
#   bash docker/online-eval/run.sh --smoke-test
#
#   # an actual eval (note: xvfb-run wraps the python process)
#   bash docker/online-eval/run.sh -- \
#       xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
#       python online_evaluation_rlbench/evaluate_policy.py \
#           dataset=OrbitalWrist bimanual=false task=open_drawer \
#           checkpoint=/data/train_logs/Orbital/my_run/last.pth \
#           data_dir=/data/orbital_rollouts \
#           val_instructions=instructions/peract/instructions.json \
#           cameras_file=instructions/orbital_cameras_grouped.json \
#           task_group_mapping_file=instructions/task_group_mapping_subset.json \
#           output_file=eval_logs/my_run/open_drawer.json \
#           headless=true max_tries=1
#
#   # or via the repo's wrapper script
#   bash docker/online-eval/run.sh -- \
#       xvfb-run -a bash scripts/eval/eval_orbital.sh \
#           task=open_drawer checkpoint=/data/.../last.pth
#
# Env knobs:
#   IMAGE, TAG      image to run                  (3dfa-online-eval:latest)
#   GPUS            docker --gpus value           (all; set GPUS=none for CPU)
#   DATA_DIRS       colon-separated host paths bind-mounted read-only at the
#                   same path inside the container (datasets, checkpoints)
#   SHM_SIZE        /dev/shm size                 (16g)
#   EXTRA_DOCKER    extra args appended to docker run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

IMAGE="${IMAGE:-3dfa-online-eval}"
TAG="${TAG:-latest}"
GPUS="${GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-16g}"
CONTAINER_REPO=/workspace/3d_flowmatch_actor

SMOKE=0
if [[ "${1:-}" == "--smoke-test" ]]; then
    SMOKE=1
    shift
elif [[ "${1:-}" == "--" ]]; then
    shift
fi

docker_args=(
    --rm
    # The image already uses tini as ENTRYPOINT; --init is belt-and-braces so a
    # stray CoppeliaSim child can never be left unreaped.
    --init
    --shm-size "${SHM_SIZE}"
    -v "${REPO_ROOT}:${CONTAINER_REPO}"
    -w "${CONTAINER_REPO}"
)

# Only request a TTY when we actually have one. Passing -t from a non-tty caller
# (CI, nohup, an agent shell) leaves the container waiting on a terminal it will
# never get, which looks exactly like a hung eval.
if [[ -t 0 && -t 1 ]]; then
    docker_args+=(-it)
fi

if [[ "${GPUS}" != "none" ]]; then
    docker_args+=(--gpus "${GPUS}")
fi

# Extra read-only mounts for datasets / checkpoints, same path in and out.
if [[ -n "${DATA_DIRS:-}" ]]; then
    IFS=':' read -r -a _dirs <<< "${DATA_DIRS}"
    for d in "${_dirs[@]}"; do
        [[ -n "${d}" ]] && docker_args+=(-v "${d}:${d}:ro")
    done
fi

if [[ -n "${EXTRA_DOCKER:-}" ]]; then
    # shellcheck disable=SC2206
    docker_args+=(${EXTRA_DOCKER})
fi

if [[ "${SMOKE}" == "1" ]]; then
    exec docker run "${docker_args[@]}" "${IMAGE}:${TAG}" \
        xvfb-run -a --server-args="-screen 0 1280x1024x24 +extension GLX +render" \
        3dfa-smoke-test
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run "${docker_args[@]}" "${IMAGE}:${TAG}" "$@"
