#!/usr/bin/env bash
# Resume-correctness matrix for the REAL trainer (main.py), run in parallel.
#
# The pytest suite covers the resilience layer in isolation. This covers the
# thing that actually matters: that `training/base.py` resumes onto the exact
# step it would have reached uninterrupted. The assertion is bit-identity of
# the weights, which only holds if the data order AND every RNG stream were
# restored -- a weaker check would pass while the run silently diverged.
#
# Four comparisons, each an independent pair of jobs so they run concurrently:
#   fresh run pre-patch vs patched   -- did the change alter existing runs?
#   1 GPU  uninterrupted vs resumed  -- the plain DistributedSampler branch
#   2 GPU  uninterrupted vs resumed  -- per-rank RNG gather/restore
#   diverse sampler unint. vs resumed-- the batch_sampler branch
#
#   ./run_resume_matrix.sh            # submit
#   ./run_resume_matrix.sh --compare  # once the jobs finish
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
SB="scripts/multinode/tests/resume_matrix.sbatch"
L="train_logs/ResumeTest"
STEPS="${STEPS:-40}"
HALF=$((STEPS / 2))
CONSTRAINT="${CONSTRAINT:-A6000}"

if [[ "${1:-}" == "--compare" ]]; then
    exec python scripts/multinode/tests/compare_indices.py
fi

# The pre-patch tree is a hardlink copy with training/base.py replaced, so it
# costs no real disk and shares everything else with the working tree.
OLD="${OLD_TREE:-$HOME/3dfa_bitcheck}"
if [[ ! -f "$OLD/training/base.py" ]]; then
    echo "note: no pre-patch tree at $OLD — skipping the bit-identity check."
    echo "      create it with: cp -al $REPO $OLD && cp <old base.py> $OLD/training/base.py"
    SKIP_OLD=1
fi

sub() {  # sub <name> <ngpu> <total> <run> [ckpt] [extra] [dep]
    local name=$1 ngpu=$2 total=$3 run=$4 ckpt=${5:-} extra=${6:-} dep=${7:-} tree=${8:-$REPO}
    local args=(--parsable -J "$name" --constraint="$CONSTRAINT" --gres=gpu:"$ngpu")
    [[ -n "$dep" ]] && args+=(--dependency=afterok:"$dep")
    local env="ALL,TOTAL=$total,RUN=$run,NGPU=$ngpu,TREE=$tree"
    [[ -n "$ckpt" ]] && env="$env,CKPT=$REPO/$L/$run/last.pth"
    [[ -n "$extra" ]] && env="$env,EXTRA=$extra"
    sbatch "${args[@]}" --export="$env" "$SB"
}

D=video_deltam_cache_batches=2
[[ -z "${SKIP_OLD:-}" ]] && OLDA=$(sub oldA 1 "$STEPS" oldA "" "" "" "$OLD")
A=$(sub A40 1 "$STEPS" phaseA)
B1=$(sub B20 1 "$HALF" phaseB); B2=$(sub B40 1 "$STEPS" phaseB ckpt "" "$B1")
C=$(sub C40 2 "$STEPS" phaseC)
D1=$(sub D20 2 "$HALF" phaseD); D2=$(sub D40 2 "$STEPS" phaseD ckpt "" "$D1")
E=$(sub E40 1 "$STEPS" phaseE "" "$D")
F1=$(sub F20 1 "$HALF" phaseF "" "$D"); F2=$(sub F40 1 "$STEPS" phaseF ckpt "$D" "$F1")

echo "submitted: oldA=${OLDA:-skipped} A=$A B=$B1,$B2 C=$C D=$D1,$D2 E=$E F=$F1,$F2"
echo "when they finish:  $0 --compare"
