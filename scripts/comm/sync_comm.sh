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
    echo '```'
    squeue -h -u "$USER" -o "%i %j %T %N %M %R" 2>/dev/null | grep -E " exp10" | sort -k2 || true
    echo '```'
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
