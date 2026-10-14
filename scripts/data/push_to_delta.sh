#!/usr/bin/env bash
# Push dense datasets from grogu to NCSA Delta. Usage: DELTA_USER=<you> [CKPTS=1] scripts/data/push_to_delta.sh [dataset ...]
# Datasets are names under /grogu/datasets/hbhatia/3dfa_dense; default is handover_chunk8 (the exp 8d / exp 10 data).
set -euo pipefail

DELTA_USER="${DELTA_USER:?set DELTA_USER to your Delta username}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
DEST="${DEST:-/projects/bgkz/3dfa_dense}"
SRC_ROOT="${SRC_ROOT:-/grogu/datasets/hbhatia/3dfa_dense}"
DATASETS=("${@:-handover_chunk8}")

# One Duo-authenticated master connection, reused by every ssh/rsync below.
CTL="$HOME/.ssh/delta-ctl-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=4h)
"${SSH[@]}" -fN "$DELTA_USER@$DELTA_HOST"
trap '"${SSH[@]}" -O exit "$DELTA_USER@$DELTA_HOST" 2>/dev/null || true' EXIT

for name in "${DATASETS[@]}"; do
    src="$SRC_ROOT/$name"
    [[ -d "$src" ]] || { echo "missing $src" >&2; exit 2; }
    echo "== $name ($(du -sh "$src" | cut -f1)) -> $DELTA_HOST:$DEST/$name"
    "${SSH[@]}" "$DELTA_USER@$DELTA_HOST" "mkdir -p '$DEST/$name'"
    # Zarr stores are many small chunk files: --partial resumes, -W skips delta-diffing, no compression for already-compressed chunks.
    rsync -a --partial --info=progress2 -W --no-compress -e "${SSH[*]}" "$src/" "$DELTA_USER@$DELTA_HOST:$DEST/$name/"
    # Verify: same file count on both sides.
    here=$(find "$src" -type f | wc -l)
    there=$("${SSH[@]}" "$DELTA_USER@$DELTA_HOST" "find '$DEST/$name' -type f | wc -l")
    echo "   files: grogu $here, delta $there $([[ $here == "$there" ]] && echo OK || echo MISMATCH)"
done

# CKPTS=1 also copies each exp 10 run's last.pth so a Delta launch resumes instead of restarting.
if [[ "${CKPTS:-0}" == 1 ]]; then
    REPO="${REPO:-/home/harshilb/3dfa_unified}"
    CKPT_DEST="${CKPT_DEST:-/projects/bgkz/3dfa_train_logs/PerAct2}"
    for c in deltam rel fullpatch_refine fullpatch_refine_x0 fullpatch_refine_rel pooled_refine; do
        run=peract2_orbital_handover_dense_chunk8_$c
        "${SSH[@]}" "$DELTA_USER@$DELTA_HOST" "mkdir -p '$CKPT_DEST/$run'"
        rsync -a --partial -e "${SSH[*]}" "$REPO/train_logs/PerAct2/$run/last.pth" "$DELTA_USER@$DELTA_HOST:$CKPT_DEST/$run/"
        echo "== $run/last.pth -> $CKPT_DEST/$run/"
    done
fi
