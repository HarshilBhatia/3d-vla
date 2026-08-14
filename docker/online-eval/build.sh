#!/usr/bin/env bash
# Build the 3DFA online-evaluation image.
#
#   bash docker/online-eval/build.sh                  # -> 3dfa-online-eval:latest
#   IMAGE=myrepo/3dfa-eval TAG=v1 bash docker/online-eval/build.sh
#   bash docker/online-eval/build.sh --no-cache       # extra args go to docker build
#
# The build context is the repo root; docker/online-eval/Dockerfile.dockerignore
# trims it down to PyRep/, RLBench/ and this directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

IMAGE="${IMAGE:-3dfa-online-eval}"
TAG="${TAG:-latest}"

for d in PyRep RLBench; do
    if [[ ! -d "${REPO_ROOT}/${d}" ]]; then
        echo "ERROR: ${REPO_ROOT}/${d} is missing. The image installs the repo's" >&2
        echo "       local ${d} fork, not upstream. Check it out first." >&2
        exit 1
    fi
done

echo "Building ${IMAGE}:${TAG} (context: ${REPO_ROOT})"
DOCKER_BUILDKIT=1 docker build \
    -f "${HERE}/Dockerfile" \
    -t "${IMAGE}:${TAG}" \
    "$@" \
    "${REPO_ROOT}"

echo
echo "Built ${IMAGE}:${TAG}"
echo "Smoke test:  bash docker/online-eval/run.sh --smoke-test"
