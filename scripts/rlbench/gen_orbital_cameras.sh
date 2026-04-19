#!/usr/bin/env bash
# Run rerun visualisation for all 18 PerAct tasks using a shared orbital
# cameras file.
#
# Usage:
#   xvfb-run -a bash scripts/rlbench/gen_orbital_cameras.sh
#
# Output: orbital_viz/<task>.rrd  (one file per task)

set -euo pipefail

SCRIPT="scripts/rlbench/visualize_cameras_rerun.py"
CAMERAS_FILE="instructions/orbital_cameras.json"
OUT_DIR="orbital_viz"
IMAGE_SIZE=256

TASKS=(
    place_cups
    close_jar
    insert_onto_square_peg
    light_bulb_in
    meat_off_grill
    open_drawer
    place_shape_in_shape_sorter
    place_wine_at_rack_location
    push_buttons
    put_groceries_in_cupboard
    put_item_in_drawer
    put_money_in_safe
    reach_and_drag
    slide_block_to_color_target
    stack_blocks
    stack_cups
    sweep_to_dustpan_of_size
    turn_tap
)

if [[ ! -f "${CAMERAS_FILE}" ]]; then
    echo "[ERROR] ${CAMERAS_FILE} not found. Generate it first with --cameras-file." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

total=${#TASKS[@]}
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    out_rrd="${OUT_DIR}/${task}.rrd"
    echo "──────────────────────────────────────────"
    echo "[$(( i + 1 ))/${total}] ${task}"

    if [[ -f "${out_rrd}" ]]; then
        echo "  Already exists, skipping."
        continue
    fi

    python "${SCRIPT}" \
        --task "${task}" \
        --image_size "${IMAGE_SIZE}" \
        --cameras-file "${CAMERAS_FILE}" \
        --out "${out_rrd}" || {
        echo "[WARN] Failed for task: ${task}" >&2
    }
done

echo "──────────────────────────────────────────"
echo "[DONE] Saved to ${OUT_DIR}/"
ls -1 "${OUT_DIR}/"
