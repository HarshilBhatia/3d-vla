#!/usr/bin/env bash
# Once any copy of a run owns its directory, cancel the run's other PENDING single-node copies and its multinode twin if that lost.
set -uo pipefail
NAME="${1:?usage: race_many.sh <job name of the single-node copies> <run dir> <multinode supervisor log> <supervisor pgid>}"
RUN_DIR="$2"; SUP_LOG="$3"; SUP_PGID="$4"
state() { squeue -h -j "$1" -o %T 2>/dev/null | head -1; }
while true; do
    owner=$(cat "$RUN_DIR/.owner_job" 2>/dev/null || true)
    if [[ -n "$owner" && "$(state "$owner")" == RUNNING ]]; then
        mn=$(grep -oE 'job [0-9]+ submitted' "$SUP_LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
        echo "[$NAME] $(date +%T) job $owner owns $RUN_DIR"
        for j in $(squeue -h -u "$USER" -n "$NAME" -t PENDING -o %i); do
            [[ "$j" != "$owner" ]] && scancel "$j" && echo "[$NAME] cancelled pending copy $j"
        done
        if [[ "$owner" != "$mn" ]]; then
            kill -TERM -"$SUP_PGID" 2>/dev/null && echo "[$NAME] stopped multinode supervisor"
            sleep 5
            [[ -n "$mn" && "$(state "$mn")" == PENDING ]] && scancel "$mn" && echo "[$NAME] cancelled pending multinode $mn"
        fi
        break
    fi
    sleep 30
done
echo "[$NAME] watcher done"
