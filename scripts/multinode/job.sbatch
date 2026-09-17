#!/usr/bin/env bash
# Generic multi-node job body. Contains no #SBATCH resource directives on
# purpose -- submit.py supplies them all so nothing is hardcoded here.
#
# Order matters: preflight runs as its own srun step and must pass before
# train.py is launched. A run that starts and then discovers it is on the
# socket path has already wasted the allocation.

set -uo pipefail

: "${RESILIENCE_TRAIN_CMD:?submit.py must export RESILIENCE_TRAIN_CMD}"
: "${RESILIENCE_CKPT_DIR:?submit.py must export RESILIENCE_CKPT_DIR}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# ── limits ────────────────────────────────────────────────────────────────────
# RDMA needs pinned memory. The login node's memlock hard limit is 64 KB and
# PropagateResourceLimits=ALL copies it in; the compute nodes' own hard limit is
# unlimited, so raising the soft limit here always works. Without this NCCL dies
# in ibv_create_cq with "Cannot allocate memory" -- it does not fall back to
# sockets, it crashes.
ulimit -l unlimited || echo "[job] WARNING: could not raise memlock; preflight will fail"

# ── distributed env ───────────────────────────────────────────────────────────
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE="${SLURM_NTASKS}"

# NCCL_SOCKET_IFNAME scopes only the bootstrap/out-of-band socket. The data
# path is selected separately by NCCL_IB_HCA -- pinning mlx5_0 both forces
# verbs RDMA and avoids the Intel irdma* RoCE devices that some nodes expose
# alongside it, which would otherwise make bandwidth depend on which nodes the
# scheduler happened to give us.
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # NCCL_-prefixed name is deprecated
export TOKENIZERS_PARALLELISM=false

# Include the restart count: a requeued job keeps the SAME SLURM_JOB_ID, so a
# plain job-id path still holds the previous attempt's preflight files. That
# made the GPU-homogeneity check count hosts from the old allocation ("across
# 3 node(s)" for a 2-node job) and could just as easily mask a real mismatch.
RUN_DIR="${RESILIENCE_CKPT_DIR}/_run/${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
mkdir -p "$RUN_DIR"
export RESILIENCE_SENTINEL="${RUN_DIR}/preempt.sentinel"
export PREFLIGHT_SHARE="${RUN_DIR}/preflight"
export RESILIENCE_DONE_MARKER="${RUN_DIR}/checkpoint.done"

echo "[job] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST ntasks=$SLURM_NTASKS"
echo "[job] constraint=${RESILIENCE_CONSTRAINT:-<none>} master=$MASTER_ADDR:$MASTER_PORT"
echo "[job] restart count=${SLURM_RESTART_COUNT:-0}"

# ── preflight (fails closed) ──────────────────────────────────────────────────
PREFLIGHT_ARGS=(
  --ckpt-dir "$RESILIENCE_CKPT_DIR"
  --min-busbw "${RESILIENCE_MIN_BUSBW:-4.0}"
  --min-free-gb "${RESILIENCE_MIN_FREE_GB:-50}"
  --runlog "${RESILIENCE_RUNLOG:-logs/multinode/runs.jsonl}"
  --bad-nodes-out "${RESILIENCE_CKPT_DIR}/_bad_nodes.txt"
  --max-node-load "${RESILIENCE_MAX_NODE_LOAD:-1.0}"
)
[[ "${RESILIENCE_ALLOW_HET:-0}" == "1" ]] && PREFLIGHT_ARGS+=(--allow-heterogeneous)

echo "[job] --- preflight ---"
if ! srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
        python scripts/multinode/preflight.py "${PREFLIGHT_ARGS[@]}"; then
    echo "[job] preflight FAILED — not starting training. Exiting 78."
    # 78 (EX_CONFIG) distinguishes "this allocation is unusable" from a
    # training crash, so the run log can tell them apart.
    exit 78
fi
echo "[job] --- preflight passed ---"

# ── preemption trap ───────────────────────────────────────────────────────────
# Only the batch-script shell reliably receives Slurm's signal; srun children
# may not. So the trap touches a sentinel that every rank polls, waits for the
# trainer to checkpoint at a step boundary, then requeues.
#
# GraceTime on this cluster is 120 s and is set per-partition by the admins.
# --signal=...@N does NOT extend it: N only moves the *time-limit* warning
# earlier. So the checkpoint must fit in ~120 s on the preemption path, and the
# periodic timer in Checkpointer is the real safety net. The trap is best-effort.
on_preempt() {
    echo "[job] caught USR1/TERM at $(date -Is) — signalling ranks to checkpoint"
    touch "$RESILIENCE_SENTINEL"
    # Give the trainer a bounded window, then requeue regardless. Requeueing
    # keeps the original --constraint, so the job returns to the same pool;
    # submit.py counts requeues and rotates the constraint if a pool keeps
    # evicting us.
    for _ in $(seq 1 100); do
        [[ -f "$RESILIENCE_DONE_MARKER" ]] && break
        sleep 1
    done
    if [[ -f "$RESILIENCE_DONE_MARKER" ]]; then
        echo "[job] trainer confirmed checkpoint; requeueing"
    else
        echo "[job] WARNING: no checkpoint confirmation after 100s; requeueing anyway "
        echo "[job]          (the periodic checkpoint is the fallback)"
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap on_preempt USR1 TERM

# ── train ─────────────────────────────────────────────────────────────────────
echo "[job] --- training ---"
# train_cmd is run VERBATIM. Configuration reaches the trainer through the
# environment (RESILIENCE_CKPT_DIR, RESILIENCE_CKPT_INTERVAL, RESILIENCE_SENTINEL),
# never by appending argv. Appending would break the moment train_cmd is a
# wrapper like `bash -lc '...'` or `torchrun ... script.py`, because the flags
# would land on the wrapper instead of the trainer.
srun --kill-on-bad-exit=1 bash -c '
    ulimit -l unlimited
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=0   # one GPU per task; cgroup makes it device 0 everywhere
    exec '"$RESILIENCE_TRAIN_CMD"'
' &
SRUN_PID=$!
# `wait` is what makes the trap fire promptly; a foreground srun would not
# return control to the shell until it died.
wait $SRUN_PID
RC=$?
echo "[job] training exited rc=$RC"
exit $RC
