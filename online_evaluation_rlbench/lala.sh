exp=peract2
tasks=(
    bimanual_push_box
)



checkpoint=/home/harshilb/work/3d-vla/grogu_train_logs/exp/final_default_full/best.pth
                 

checkpoint_dir=$(dirname "$checkpoint")

experiment=default


# Shared wandb run for all tasks in this eval (optional)
wandb_run_name="peract2_online_eval_seed${seed}"
wandb_run_id="peract2-online-eval-seed${seed}-$(date +%s)"

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    task_name=${tasks[$i]}
    log_dir="$checkpoint_dir/seed$seed/$task_name"
    mkdir -p "$log_dir"

    CUDA_VISIBLE_DEVICES=1 xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
        checkpoint="$checkpoint" \
        task="$task_name" \
        experiment=$experiment \
        use_wandb=true \
        wandb_project=3d_flowmatch_actor \
        wandb_run_name="$wandb_run_name" \
        wandb_run_id="$wandb_run_id" \
        2>&1 | tee "$log_dir/eval.log"
done

python online_evaluation_rlbench/collect_results.py \
    --folder "$checkpoint_dir/seed$seed/"