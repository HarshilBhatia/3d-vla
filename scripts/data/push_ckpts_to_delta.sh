#!/usr/bin/env bash
# Copy exp 10 checkpoints (last.pth + newest interm_step) to Delta. Usage: DELTA_USER=<you> scripts/data/push_ckpts_to_delta.sh [run_suffix ...]
# Run suffixes are the part after peract2_orbital_handover_dense_chunk8_; default is all six exp 10 runs.
set -euo pipefail

DELTA_USER="${DELTA_USER:?set DELTA_USER to your Delta username}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
REPO="${REPO:-/home/harshilb/3dfa_unified}"
CKPT_DEST="${CKPT_DEST:-/projects/bgkz/3dfa_train_logs/PerAct2}"
RUNS=("$@")
[[ $# -eq 0 ]] && RUNS=(deltam rel fullpatch_refine fullpatch_refine_x0 fullpatch_refine_rel pooled_refine)

# One Duo-authenticated master connection, reused by every ssh/rsync below.
CTL="$HOME/.ssh/delta-ctl-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=4h)
"${SSH[@]}" -fN "$DELTA_USER@$DELTA_HOST"
trap '"${SSH[@]}" -O exit "$DELTA_USER@$DELTA_HOST" 2>/dev/null || true' EXIT

for c in "${RUNS[@]}"; do
    run=peract2_orbital_handover_dense_chunk8_$c
    src="$REPO/train_logs/PerAct2/$run"
    [[ -f "$src/last.pth" ]] || { echo "no last.pth in $src, skipping" >&2; continue; }
    files=("$src/last.pth")
    newest=$(ls "$src" | grep -E '^interm_step_[0-9]+\.pth$' | sort -V | tail -1)
    [[ -n "$newest" ]] && files+=("$src/$newest")
    "${SSH[@]}" "$DELTA_USER@$DELTA_HOST" "mkdir -p '$CKPT_DEST/$run'"
    rsync -a --partial -e "${SSH[*]}" "${files[@]}" "$DELTA_USER@$DELTA_HOST:$CKPT_DEST/$run/"
    iter=$(/home/harshilb/miniconda3/envs/3dfa/bin/python -c "import torch;print(torch.load('$src/last.pth',map_location='cpu',weights_only=False).get('iter'))" 2>/dev/null || echo ?)
    echo "== $run: last.pth (iter $iter)${newest:+ + $newest} -> $CKPT_DEST/$run/"
done
