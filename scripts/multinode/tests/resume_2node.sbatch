#!/usr/bin/env bash
# One leg of the real multi-node resume test: 2 nodes x 1 GPU on the actual
# PerAct2 zarr. Driven by TOTAL / RUN / CKPT from the submitting environment.
#SBATCH --partition=all
#SBATCH --constraint=A5000
#SBATCH --exclude=grogu-4-13,grogu-3-20
#SBATCH --nodes=2 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --cpus-per-task=8 --mem-per-cpu=10G
#SBATCH --time=00:30:00
#SBATCH --output=/home/harshilb/3dfa_unified/logs/multinode/r2n-%x-%j.out
#SBATCH --error=/home/harshilb/3dfa_unified/logs/multinode/r2n-%x-%j.err
set -uo pipefail
ulimit -l unlimited
REPO=/home/harshilb/3dfa_unified
cd "$REPO"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT=$((26000 + SLURM_JOB_ID % 3000))
export WORLD_SIZE="$SLURM_NTASKS"
export NCCL_SOCKET_IFNAME=ib0 NCCL_IB_HCA=mlx5_0 NCCL_IB_DISABLE=0
export RESILIENCE_INDEX_LOG="$REPO/train_logs/MultiNode/$RUN/indices"
mkdir -p "$(dirname "$RESILIENCE_INDEX_LOG")"
echo "### run=$RUN total=$TOTAL ckpt=${CKPT:-none} nodes=$SLURM_JOB_NODELIST"
E=/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820
srun --kill-on-bad-exit=1 bash -c '
  ulimit -l unlimited
  source /home/harshilb/miniconda3/etc/profile.d/conda.sh && conda activate 3dfa
  cd '"$REPO"'
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-} BLOSC_NTHREADS=1
  export RANK=$SLURM_PROCID LOCAL_RANK=0
  exec python main.py \
    data=orbital_peract2_nfs experiment=peract2_orbital_2d dataset=OrbitalPeract2 \
    train_data_dir='"$E"'/datasets/orbital_peract2/zarr/train.zarr \
    eval_data_dir='"$E"'/datasets/orbital_peract2/zarr/val.zarr \
    exp_log_dir=MultiNode run_log_dir='"$RUN"' seed=1234 \
    batch_size=16 num_workers=4 prefetch_factor=2 \
    train_iters='"$TOTAL"' ckpt_interval_steps=50 val_interval_steps=1000000 \
    interm_ckpt_interval_steps=1000000 '"${CKPT:+checkpoint=$CKPT}"'
'
echo "### exit=$?"
