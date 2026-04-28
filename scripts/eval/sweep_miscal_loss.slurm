#!/bin/bash
#SBATCH --job-name=miscal_sweep
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=shubhamlong

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/harshilb/3d_flowmatch_actor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 3dfa

CHECKPOINT_PATH='/home/harshilb/3d_flowmatch_actor/train_logs/Orbital/open_drawer_test/interm160000.pth'

CHECKPOINT=${CHECKPOINT:-$CHECKPOINT_PATH}
# Derive output dir from checkpoint path: train_logs/Orbital/my_run/best.pth -> results/miscal_sweep/my_run/
EXP_NAME=$(basename $(dirname $CHECKPOINT))
OUTPUT_CSV=results/miscal_sweep/${EXP_NAME}/sweep.csv
mkdir -p results/miscal_sweep/${EXP_NAME}

echo "Checkpoint: $CHECKPOINT"
echo "Output CSV: $OUTPUT_CSV"

data=orbital


python scripts/eval/sweep_miscal_loss.py \
    eval_data_dir=/grogu/user/harshilb/orbital_rollouts/4task_new.zarr \
    val_instructions=instructions/peract/instructions.json \
    dataset=OrbitalWrist \
    bimanual=false \
    checkpoint=$CHECKPOINT \
    output_csv=$OUTPUT_CSV \
    num_samples=1000 \
    "$@"
