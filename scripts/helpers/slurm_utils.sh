#!/bin/bash
# Shared helpers for SLURM training scripts.

_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_UTILS_DIR/../.." && pwd)"

# ── Cluster profile ───────────────────────────────────────────────────────────

# Load a named cluster profile and set up the environment.
# Sources config/clusters/<name>.env, cd's to CLUSTER_REPO, activates conda,
# exports PYTHONPATH, and (for tmpdir staging) creates $CLUSTER_LOCAL_DATA.
# Usage: load_cluster grogu|delta|deltaai
load_cluster() {
    local name=$1
    local env_file="$_REPO_ROOT/config/clusters/${name}.env"
    if [[ ! -f "$env_file" ]]; then
        echo "[ERROR] Unknown cluster: '$name' (no file at $env_file)"
        echo "        Available: $(ls "$_REPO_ROOT/config/clusters/"*.env 2>/dev/null | xargs -n1 basename | sed 's/\.env//' | tr '\n' ' ')"
        exit 1
    fi
    source "$env_file"

    cd "$CLUSTER_REPO"
    export PYTHONPATH="$CLUSTER_REPO:${PYTHONPATH:-}"
    export BLOSC_NTHREADS=1

    # shellcheck disable=SC1090
    source "$CLUSTER_CONDA_INIT"
    eval "$CLUSTER_CONDA_CMD"

    if [[ "$CLUSTER_STAGE_METHOD" == "tmpdir" ]]; then
        export CLUSTER_LOCAL_DATA="/tmp/${SLURM_JOB_ID:-$$}"
        mkdir -p "$CLUSTER_LOCAL_DATA"
    fi

    if [[ -n "${CLUSTER_LOG_DIR:-}" ]]; then
        mkdir -p "$CLUSTER_LOG_DIR"
    fi

    # Cluster-specific env vars (e.g. HF_HOME, TRITON_CACHE_DIR on Babel)
    if [[ -n "${CLUSTER_EXTRA_ENV:-}" ]]; then
        eval "$CLUSTER_EXTRA_ENV"
    fi

    echo "[cluster] $CLUSTER_NAME | repo=$CLUSTER_REPO | data=$CLUSTER_DATA_ROOT | stage=$CLUSTER_STAGE_METHOD"
}

# ── Data staging ──────────────────────────────────────────────────────────────

# Remap and rsync a path if the destination doesn't already exist.
# Usage: rsync_if_missing <src> <dst>
rsync_if_missing() {
    local src=$1 dst=$2
    if [ ! -d "$dst" ]; then
        echo "Copying $(basename "$src") → $dst"
        mkdir -p "$(dirname "$dst")"
        rsync -a --info=progress2 "$src" "$(dirname "$dst")/"
    else
        echo "Scratch copy exists, skipping: $dst"
    fi
}

# Remap a /grogu/user/harshilb path to /scratch/harshilb and rsync if missing.
# Sets the variable named by $1 to the scratch path.
# Usage: to_scratch train_data_dir
to_scratch() {
    local varname=$1
    local src=${!varname}
    local dst=${src/\/grogu\/user\/harshilb/${CLUSTER_SCRATCH_ROOT}}
    rsync_if_missing "$src" "$dst"
    printf -v "$varname" '%s' "$dst"
}

# Stage data for the current cluster (call load_cluster first).
#   grogu  → rsync CLUSTER_DATA_ROOT → CLUSTER_SCRATCH_ROOT (to_scratch)
#   delta/deltaai → rsync to /tmp/$SLURM_JOB_ID
# Sets the variable named by $1 to the staged path.
# Usage: stage_data train_data_dir
stage_data() {
    local varname=$1
    local src=${!varname}

    if [[ "$CLUSTER_STAGE_METHOD" == "scratch" ]]; then
        to_scratch "$varname"
    elif [[ "$CLUSTER_STAGE_METHOD" == "tmpdir" ]]; then
        local dst="$CLUSTER_LOCAL_DATA/$(basename "$src")"
        rsync_if_missing "$src" "$dst"
        printf -v "$varname" '%s' "$dst"
    else
        echo "[stage_data] CLUSTER_STAGE_METHOD not set — using data in-place: $src"
    fi
}
