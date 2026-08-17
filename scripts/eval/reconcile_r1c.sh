#!/usr/bin/env bash
# Reconcile the R1c eval grid: submit every (cond, task) cell that is neither
# already in the sky managed-job queue nor already finished in S3.
#
# Submissions are SERIAL with retry. Running several `sky jobs launch` loops in
# parallel overwhelms the RestfulAdminPolicy sidecar (`admin-policy:80` refuses
# the connection / times out) and silently drops most of the wave — 25 of 91 on
# the first attempt. One at a time, retried, is the reliable path.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

CKPT=s3://far-research-internal/harsvbha/3dfa/eval/ckpt/orbital_miscal_deltam_eeaux.pth
OUT_BASE=s3://far-research-internal/harsvbha/3dfa/eval/results/orbital_miscal_deltam_eeaux
MAP=instructions/peract2_orbital_task_group_mapping.json
OOD_FILE=instructions/orbital_miscalibration_noise_ood.json
CONDS=(level0 clean0 ood_miscal noise_2deg2cm noise_5deg5cm noise_10deg10cm noise_15deg15cm)
TASKS=(
  bimanual_push_box bimanual_lift_ball bimanual_dual_push_buttons
  bimanual_pick_plate bimanual_put_item_in_drawer bimanual_put_bottle_in_fridge
  bimanual_handover_item bimanual_pick_laptop bimanual_straighten_rope
  bimanual_sweep_to_dustpan bimanual_lift_tray bimanual_handover_item_easy
  bimanual_take_tray_out_of_oven
)
MAX_TRIES="${MAX_TRIES:-4}"

# What already exists: live/finished managed jobs, and completed result JSONs.
# Job names contain hyphens, so pick the NAME column rather than a character
# class — an [A-Za-z0-9_]+ pattern truncates at the first hyphen and makes every
# cell look unsubmitted.
sky jobs queue 2>/dev/null | awk '$3 ~ /^hb-3dfa-r1c-/ {print $3}' | sort -u > /tmp/r1c_inqueue.txt
AWS_PROFILE=far-compute aws s3 ls "$OUT_BASE/" --recursive 2>/dev/null \
  | awk '{print $NF}' \
  | sed -n 's#.*/orbital_miscal_deltam_eeaux/\([^/]*\)/\(bimanual_[a-z_]*\)\.json$#\1|\2#p' \
  | sort -u > /tmp/r1c_ins3.txt
echo "already in queue: $(wc -l < /tmp/r1c_inqueue.txt), already in S3: $(wc -l < /tmp/r1c_ins3.txt)"

launch_one() {
  local cond="$1" t="$2" g="$3" lvl="$4" rot="$5" trans="$6" mfile="$7"
  local try=1
  while [ "$try" -le "$MAX_TRIES" ]; do
    if sky jobs launch -n "hb-3dfa-r1c-${cond}-${t}" -d -y --infra k8s/sky-us-east-1 \
        --env PREEMPTIBLE=1 --env TASK="$t" --env SPAWN_GROUP="$g" \
        --env CKPT_S3="$CKPT" --env ORBITAL_MISCAL_LEVEL="$lvl" \
        --env MISCAL_ROT="$rot" --env MISCAL_TRANS="$trans" --env MISCAL_FILE="$mfile" \
        --env OUT_S3="$OUT_BASE/$cond" \
        scripts/sky/peract2_orbital_online_eval.yaml 2>&1 | grep -q "Managed Job ID"; then
      echo "OK   $cond/$t (try $try)"
      return 0
    fi
    echo "RETRY $cond/$t (try $try failed)"
    try=$((try + 1)); sleep 20
  done
  echo "FAIL $cond/$t after $MAX_TRIES tries"
  return 1
}

n_ok=0; n_skip=0; n_fail=0
for cond in "${CONDS[@]}"; do
  lvl=""; rot=""; trans=""; mfile=""
  case "$cond" in
    clean0)     ;;
    level0)     lvl=medium ;;
    ood_miscal) lvl=medium; mfile="$OOD_FILE" ;;
    noise_*)    lvl=medium; pair="${cond#noise_}"; rot="${pair%%deg*}deg"; trans="${pair#*deg}" ;;
  esac
  for t in "${TASKS[@]}"; do
    if grep -qx "hb-3dfa-r1c-${cond}-${t}" /tmp/r1c_inqueue.txt \
       || grep -qx "${cond}|${t}" /tmp/r1c_ins3.txt; then
      n_skip=$((n_skip + 1)); continue
    fi
    g=$(python -c "import json,sys; print(json.load(open('$MAP'))['tasks'][sys.argv[1]]['eval_group'])" "$t")
    if launch_one "$cond" "$t" "$g" "$lvl" "$rot" "$trans" "$mfile"; then
      n_ok=$((n_ok + 1))
    else
      n_fail=$((n_fail + 1))
    fi
  done
done
echo "RECONCILE_DONE launched=$n_ok already=$n_skip failed=$n_fail"
