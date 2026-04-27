#!/usr/bin/env bash
# Evaluate grogu_train_logs/best.pth (deltaM_full + large miscalibration) on all tasks.
# Usage: bash scripts/rlbench/eval_orbital_grogu_best.sh

main_dir=Orbital
run_log_dir="grogu_best_miscal_large_deltaM"
checkpoint=/home/harshilb/work/3d-vla/grogu_train_logs/best.pth

tasks=(
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

mkdir -p eval_logs/${main_dir}/${run_log_dir}/logs

run_task() {
    local task=$1
    local gpu=$2
    local output_file=eval_logs/${main_dir}/${run_log_dir}/${task}.json
    local log_file=eval_logs/${main_dir}/${run_log_dir}/logs/${task}.log

    echo "[GPU $gpu] Starting $task ..."
    CUDA_VISIBLE_DEVICES=$gpu xvfb-run -a \
        bash scripts/eval/eval_orbital.sh \
            miscalibration_noise_level=large \
            checkpoint=$checkpoint \
            output_file=$output_file \
            task=$task \
        > "$log_file" 2>&1
    echo "[GPU $gpu] Done $task — exit $?"
}

num_gpus=2
pids=()
gpu_slots=(0 1)

for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    gpu=${gpu_slots[$((i % num_gpus))]}

    if [ $i -ge $num_gpus ]; then
        wait ${pids[$((i - num_gpus))]}
    fi

    run_task "$task" "$gpu" &
    pids[$i]=$!
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo "All tasks done. Results in eval_logs/${main_dir}/${run_log_dir}/"
