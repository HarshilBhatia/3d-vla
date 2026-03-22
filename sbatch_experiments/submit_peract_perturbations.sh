#!/bin/bash
# Submit 90 parallel collection jobs (18 tasks × 5 batches of 25 episodes each),
# then a single zarr-build job that runs after all complete.
#
# Each task gets its assigned camera perturbation config.
# The 5 batches per task run in parallel and write non-overlapping episode numbers
# (ep0-24, ep25-49, ep50-74, ep75-99, ep100-124) via --ep_offset.
#
# Estimated wall time: ~2.5 hours (25 eps ≈ 30 min, all 90 jobs run concurrently).
#
# Usage:
#   cd /home/lzaceria/mscv/3dvla/3d-vla
#   bash sbatch_experiments/submit_peract_perturbations.sh
#
# Overrides:
#   EPISODES_PER_BATCH=5 bash submit_peract_perturbations.sh   # smoke test (5 batches × 5 eps = 25 total)
#   BATCHES_PER_TASK=1   bash submit_peract_perturbations.sh   # single batch (25 eps total per task)
#   DRY_RUN=1            bash submit_peract_perturbations.sh   # print commands only

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/lzaceria/mscv/3dvla/3d-vla}"
EPISODES_PER_BATCH="${EPISODES_PER_BATCH:-5}"    # episodes per job
BATCHES_PER_TASK="${BATCHES_PER_TASK:-25}"        # jobs per task  → total = 25×5 = 125 eps/task
DRY_RUN="${DRY_RUN:-0}"

# ── Camera perturbation configs ────────────────────────────────────────────────
# Format: "CAM_ROT_DEG:CAM_TRANS_X:CAM_TRANS_Y:CAM_TRANS_Z"
declare -A CONFIGS
CONFIGS["rot_plus5"]="5.0:0.0:0.0:0.0"
CONFIGS["rot_minus5"]="-5.0:0.0:0.0:0.0"
CONFIGS["rot_plus10"]="10.0:0.0:0.0:0.0"
CONFIGS["trans_x8cm"]="0.0:0.08:0.0:0.0"
CONFIGS["trans_y8cm"]="0.0:0.0:0.08:0.0"
CONFIGS["trans_z5cm"]="0.0:0.0:0.0:0.05"

# ── Task → config assignment (3 tasks per config) ──────────────────────────────
declare -A TASK_CONFIG
TASK_CONFIG["open_drawer"]="rot_plus5"
TASK_CONFIG["push_buttons"]="rot_plus5"
TASK_CONFIG["turn_tap"]="rot_plus5"

TASK_CONFIG["reach_and_drag"]="rot_minus5"
TASK_CONFIG["slide_block_to_color_target"]="rot_minus5"
TASK_CONFIG["sweep_to_dustpan_of_size"]="rot_minus5"

TASK_CONFIG["close_jar"]="rot_plus10"
TASK_CONFIG["place_cups"]="rot_plus10"
TASK_CONFIG["place_wine_at_rack_location"]="rot_plus10"

TASK_CONFIG["put_item_in_drawer"]="trans_x8cm"
TASK_CONFIG["put_money_in_safe"]="trans_x8cm"
TASK_CONFIG["meat_off_grill"]="trans_x8cm"

TASK_CONFIG["stack_blocks"]="trans_y8cm"
TASK_CONFIG["stack_cups"]="trans_y8cm"
TASK_CONFIG["place_shape_in_shape_sorter"]="trans_y8cm"

TASK_CONFIG["insert_onto_square_peg"]="trans_z5cm"
TASK_CONFIG["light_bulb_in"]="trans_z5cm"
TASK_CONFIG["put_groceries_in_cupboard"]="trans_z5cm"

# ── Submit collection jobs ─────────────────────────────────────────────────────
total_jobs=$(( ${#TASK_CONFIG[@]} * BATCHES_PER_TASK ))
total_eps=$(( total_jobs * EPISODES_PER_BATCH ))
echo "Submitting ${total_jobs} collection jobs  (${#TASK_CONFIG[@]} tasks × ${BATCHES_PER_TASK} batches × ${EPISODES_PER_BATCH} eps = ${total_eps} total episodes)"
echo "EPISODES_PER_TASK = $(( BATCHES_PER_TASK * EPISODES_PER_BATCH ))  |  DRY_RUN=$DRY_RUN"
echo ""

JOB_IDS=()
for task in $(echo "${!TASK_CONFIG[@]}" | tr ' ' '\n' | sort); do
    config="${TASK_CONFIG[$task]}"
    IFS=':' read -r rot tx ty tz <<< "${CONFIGS[$config]}"

    for (( batch=0; batch<BATCHES_PER_TASK; batch++ )); do
        ep_offset=$(( batch * EPISODES_PER_BATCH ))

        printf "  %-40s  config=%-14s  batch=%d  ep_offset=%3d\n" \
            "$task" "$config" "$batch" "$ep_offset"

        cmd=(sbatch
            --job-name="co_${task:0:10}_b${batch}"
            --export="ALL,PERACT_ROT10_SKIP_PYREP_REBUILD=1,\
TASK=$task,\
CONFIG_NAME=$config,\
CAM_ROT_DEG=$rot,\
CAM_TRANS_X=$tx,\
CAM_TRANS_Y=$ty,\
CAM_TRANS_Z=$tz,\
EPISODES_PER_BATCH=$EPISODES_PER_BATCH,\
BATCH_IDX=$batch,\
EP_OFFSET=$ep_offset,\
REPO_DIR=$REPO_DIR"
            "$REPO_DIR/sbatch_experiments/peract_collect.sbatch"
        )

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "    [DRY_RUN] ${cmd[*]}"
        else
            jid=$("${cmd[@]}" | awk '{print $NF}')
            JOB_IDS+=("$jid")
            echo "    → job $jid"
        fi
    done
    echo ""
done

# ── Submit zarr job with afterok dependency ────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] Would submit zarr job after all ${total_jobs} collection jobs complete."
else
    if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
        dep="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
        zarr_jid=$(sbatch \
            --dependency="$dep" \
            --export="ALL,REPO_DIR=$REPO_DIR" \
            "$REPO_DIR/sbatch_experiments/peract_zarr_perturbed.sbatch" \
            | awk '{print $NF}')
        echo "Zarr build job $zarr_jid submitted (runs after all ${#JOB_IDS[@]} collection jobs)"
    fi
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     $REPO_DIR/logs/"
echo "Output:   $REPO_DIR/Peract_zarr_perturbed/"
