#!/usr/bin/env bash
# Keep whichever twin of a run starts first (single-node chain vs multinode supervisor); cancel only the loser's PENDING jobs.
set -uo pipefail
NAME="${1:?usage: race_twins.sh <name> <supervisor log> <supervisor pgid> <single-node job ids, comma-separated>}"
SUP_LOG="$2"; SUP_PGID="$3"; IFS=, read -ra SINGLE <<< "$4"
state() { squeue -h -j "$1" -o %T 2>/dev/null | head -1; }
cancel_if_pending() { local s; s=$(state "$1"); if [[ "$s" == PENDING ]]; then scancel "$1" && echo "[$NAME] cancelled pending $1"; elif [[ -n "$s" ]]; then echo "[$NAME] left $1 alone (state=$s)"; fi; }
while true; do
    mn=$(grep -oE 'job [0-9]+ submitted' "$SUP_LOG" | tail -1 | grep -oE '[0-9]+')
    single_running=""; for j in "${SINGLE[@]}"; do [[ "$(state "$j")" == RUNNING ]] && single_running=$j; done
    mn_running=""; [[ -n "$mn" && "$(state "$mn")" == RUNNING ]] && mn_running=$mn
    if [[ -n "$single_running" && -n "$mn_running" ]]; then
        echo "[$NAME] $(date +%T) BOTH running (single $single_running, multinode $mn_running); claim_run.sh stops the later one, not cancelling anything"
        break
    elif [[ -n "$single_running" ]]; then
        echo "[$NAME] $(date +%T) single-node $single_running started first; stopping multinode supervisor"
        kill -TERM -"$SUP_PGID" 2>/dev/null
        sleep 5; [[ -n "$mn" ]] && cancel_if_pending "$mn"
        break
    elif [[ -n "$mn_running" ]]; then
        echo "[$NAME] $(date +%T) multinode $mn_running started first; cancelling pending single-node chain"
        for j in "${SINGLE[@]}"; do cancel_if_pending "$j"; done
        break
    fi
    sleep 30
done
echo "[$NAME] watcher done"
