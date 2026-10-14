#!/usr/bin/env bash
# Sync this checkout with the shared grogu<->Delta comm branch: merge the other side's commits, refresh our status file, push.
# Usage: COMM_BRANCH=comm-<timestamp> SITE=grogu|delta scripts/comm/sync_comm.sh   (run in a loop to keep it current)
set -uo pipefail

COMM_BRANCH="${COMM_BRANCH:?set COMM_BRANCH, e.g. comm-20261014-1310}"
SITE="${SITE:?set SITE to grogu or delta}"
cd "$(git rev-parse --show-toplevel)"
STATUS="docs/comm/${SITE}_status.md"

git fetch -q origin "$COMM_BRANCH" || { echo "$(date +%H:%M) fetch failed"; exit 1; }
# Bring the other site's commits in; never resolve conflicts automatically.
if ! git merge -q --no-edit "origin/$COMM_BRANCH"; then
    git merge --abort; echo "$(date +%H:%M) CONFLICT merging origin/$COMM_BRANCH; left untouched, resolve by hand"; exit 1
fi

# Status file: what this site is running and how far each exp 10 run has got.
mkdir -p docs/comm
{
    echo "# ${SITE} status"
    echo
    echo "Updated $(date '+%Y-%m-%d %H:%M %Z') at $(git rev-parse --short HEAD)."
    echo
    echo "| job | run | state | node | GPU | progress (this launch) | speed |"
    echo "|---|---|---|---|---|---|---|"
    squeue -h -u "$USER" -o "%i %j %T %N" 2>/dev/null | grep -E " exp10[a-f]_" | sort -k2 | while read j n st node; do
        gpu=$( [[ -n $node && $node != "(null)" ]] && sinfo -h -N -n "${node%%,*}" -o %f 2>/dev/null | head -1 )
        p=$({ tr '\r' '\n' < "logs/train/slurm-fixedbias-randall-$j.err"; } 2>/dev/null | grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)
        echo "| $j | $n | $st | ${node:--} | ${gpu:--} | ${p%% \[*} | $(echo "$p" | grep -oE '[0-9.]+it/s' || echo -) |"
    done
    echo
    echo "GPU speed: A5000, A6000 and RTX 3090 train at about the same rate (full-patch history ~1.15-1.2 it/s at 32/GPU); Turing nodes (rtx6000, rtx2080ti) are ~4x slower and are excluded."
    echo
    echo "| run | newest interm_step | last.pth iter |"
    echo "|---|---|---|"
    for d in train_logs/PerAct2/peract2_orbital_handover_dense_chunk8_*; do
        [[ -d $d ]] || continue
        newest=$(ls "$d" | grep -E '^interm_step_[0-9]+\.pth$' | sort -V | tail -1)
        it=$(python -c "import torch;print(torch.load('$d/last.pth',map_location='cpu',weights_only=False).get('iter'))" 2>/dev/null || echo -)
        echo "| ${d##*/} | ${newest:--} | $it |"
    done
} > "$STATUS"
git add "$STATUS"
git diff --cached --quiet || git commit -q -m "comm: ${SITE} status $(date +%H:%M)"

git push -q origin "HEAD:$COMM_BRANCH" && echo "$(date +%H:%M) synced $(git rev-parse --short HEAD) -> $COMM_BRANCH"
