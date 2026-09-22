#!/usr/bin/env bash
# Hold a training submit until shubhamlong has a node that can actually start it.
# We may only submit to shubhamlong and all. Other tier-5 partitions belong to
# other groups -- Slurm accepts them, but do not use them.
# shubhamlong is PriorityTier 5; all is tier 1 and gets preempted.
set -euo pipefail

PARTITION="${PARTITION:-shubhamlong}"   # shubhamlong or all only
WANT_GPUS="${WANT_GPUS:-4}"
WANT_CPUS="${WANT_CPUS:-32}"
WANT_MEM_GB="${WANT_MEM_GB:-400}"
GPU_MODELS="${GPU_MODELS:-A6000Ada A6000 A5000}"
BAD_NODES="${BAD_NODES:-grogu-2-35,grogu-3-20,grogu-4-13}"
POLL_SEC="${POLL_SEC:-120}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-86400}"
[[ $# -ge 1 ]] || { echo "usage: [env...] $0 <sbatch script> [sbatch args...]" >&2; exit 2; }

# %N is a bracketed hostlist (grogu-2-[25,30]); only scontrol can expand it.
nodes_in_partition() { sinfo -h -p "$PARTITION" -o "%N" | while read -r hl; do scontrol show hostnames "$hl" 2>/dev/null; done | sort -u; }

# Echo "<node> <model>" for every node that can start the job right now.
find_ready() {
  local n d model used free cpualloc cputot memtot memalloc
  for n in $(nodes_in_partition); do
    [[ ",$BAD_NODES," == *",$n,"* ]] && continue
    d=$(scontrol show node -d "$n" 2>/dev/null | tr '\n' ' ') || continue
    [[ "$d" == *"State=DOWN"* || "$d" == *"State=DRAIN"* ]] && continue
    model=$(grep -oE "AvailableFeatures=[^ ]+" <<<"$d" | cut -d= -f2)
    [[ " $GPU_MODELS " == *" $model "* ]] || continue
    used=$(grep -oE "GresUsed=gpu:[0-9]+" <<<"$d" | cut -d: -f2); used=${used:-99}
    local total; total=$(grep -oE "Gres=gpu:[0-9]+" <<<"$d" | cut -d: -f2); total=${total:-0}
    free=$((total - used)); (( free >= WANT_GPUS )) || continue
    cpualloc=$(grep -oE "CPUAlloc=[0-9]+" <<<"$d" | cut -d= -f2)
    cputot=$(grep -oE "CPUTot=[0-9]+" <<<"$d" | cut -d= -f2)
    (( cputot - cpualloc >= WANT_CPUS )) || continue
    memtot=$(grep -oE "RealMemory=[0-9]+" <<<"$d" | cut -d= -f2)
    memalloc=$(grep -oE "AllocMem=[0-9]+" <<<"$d" | cut -d= -f2)
    (( (memtot - memalloc) / 1024 >= WANT_MEM_GB )) || continue
    echo "$n $model"
  done
}

start=$(date +%s)
while true; do
  ready=$(find_ready | head -1 || true)
  if [[ -n "$ready" ]]; then
    node=${ready%% *}; model=${ready##* }
    echo "READY $node ($model) — submitting to $PARTITION"
    jid=$(sbatch --parsable --partition="$PARTITION" --constraint="$model" --exclude="$BAD_NODES" "$@")
    echo "SUBMITTED job=$jid node_hint=$node gpu=$model partition=$PARTITION"
    exit 0
  fi
  (( $(date +%s) - start > MAX_WAIT_SEC )) && { echo "TIMEOUT: no $PARTITION node free after ${MAX_WAIT_SEC}s"; exit 1; }
  echo "WAITING no $PARTITION node with ${WANT_GPUS}x{$GPU_MODELS}, ${WANT_CPUS} cpu, ${WANT_MEM_GB}G — retry in ${POLL_SEC}s"
  sleep "$POLL_SEC"
done
