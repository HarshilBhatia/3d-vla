#!/usr/bin/env bash
# reprocess_all.sh
#
# Orchestrates the full multilab data-processing pipeline by submitting SLURM
# jobs with correct dependency chains.
#
# Usage:
#   bash scripts/reprocess_all.sh --all           # submit full pipeline
#   bash scripts/reprocess_all.sh --dry-run --all # echo commands only
#   bash scripts/reprocess_all.sh --step N        # run only step N (1-6)
#   bash scripts/reprocess_all.sh --step 0        # run only step 0
#
# Pipeline:
#   Step 0: select_episodes          (CPU, single job)
#   Step 1: cache_backbone_features  (GPU array 0-3)   ─┐ parallel
#   Step 2: download_multilab_depths (CPU, single job)  ─┘ both after step 0
#   Step 3: extract_svo_depth        (GPU array 0-31)  after step 2
#   Step 4: build_episode_frame_index (manual srun)    after step 1+3
#   Step 5: check_depth_completeness  (manual srun)    after step 4
#   Step 6: cache_depth_features     (CPU array 0-31)  after step 1+3
#
# Steps 4 and 5 require an interactive session; their commands are printed
# instead of submitted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_DIR="$SCRIPT_DIR/slurm"

DRY_RUN=false
RUN_ALL=false
RUN_STEP=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --all)       RUN_ALL=true; shift ;;
        --step)      RUN_STEP="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--dry-run] [--all | --step N]" >&2
            exit 1
            ;;
    esac
done

if [[ "$RUN_ALL" == "false" && -z "$RUN_STEP" ]]; then
    echo "Error: specify --all or --step N" >&2
    echo "Usage: $0 [--dry-run] [--all | --step N]" >&2
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
sbatch_cmd() {
    # sbatch_cmd [--dependency=...] <slurm_file> [extra_args...]
    # Returns job ID (or echoes the command in dry-run mode).
    local dep=""
    if [[ "$1" == --dependency=* ]]; then
        dep="$1"
        shift
    fi
    local slurm_file="$1"
    shift
    local extra_args=("$@")

    local cmd=(sbatch --parsable)
    [[ -n "$dep" ]] && cmd+=("$dep")
    cmd+=("$slurm_file" "${extra_args[@]}")

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] ${cmd[*]}"
        echo "DRY_RUN_JOB_ID"
    else
        "${cmd[@]}"
    fi
}

banner() {
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  $*"
    echo "══════════════════════════════════════════════"
}

# ── Step runners ──────────────────────────────────────────────────────────────

run_step0() {
    banner "Step 0: select_episodes"
    JOB0=$(sbatch_cmd "$SLURM_DIR/select_episodes_multilab.slurm")
    echo "  Submitted job $JOB0"
    echo "$JOB0"
}

run_step1() {
    local dep_arg="${1:-}"
    banner "Step 1: cache_backbone_features (array 0-3)"
    [[ -n "$dep_arg" ]] && echo "  Dependency: $dep_arg"
    JOB1=$(sbatch_cmd ${dep_arg:+"$dep_arg"} --array=0-3 "$SLURM_DIR/cache_features_multilab.slurm")
    echo "  Submitted job $JOB1"
    echo "$JOB1"
}

run_step2() {
    local dep_arg="${1:-}"
    banner "Step 2: download_multilab_depths"
    [[ -n "$dep_arg" ]] && echo "  Dependency: $dep_arg"
    JOB2=$(sbatch_cmd ${dep_arg:+"$dep_arg"} "$SLURM_DIR/download_depths_multilab.slurm")
    echo "  Submitted job $JOB2"
    echo "$JOB2"
}

run_step3() {
    local dep_arg="${1:-}"
    banner "Step 3: extract_svo_depth (array 0-31)"
    [[ -n "$dep_arg" ]] && echo "  Dependency: $dep_arg"
    JOB3=$(sbatch_cmd ${dep_arg:+"$dep_arg"} "$SLURM_DIR/extract_depths_multilab.slurm")
    echo "  Submitted job $JOB3"
    echo "$JOB3"
}

run_step4() {
    local dep_arg="${1:-}"
    banner "Step 4: build_episode_frame_index"
    [[ -n "$dep_arg" ]] && echo "  Dependency: $dep_arg"
    JOB4=$(sbatch_cmd ${dep_arg:+"$dep_arg"} "$SLURM_DIR/build_frame_index_multilab.slurm")
    echo "  Submitted job $JOB4"
    echo "$JOB4"
}

run_step5() {
    local dep_arg="${1:-}"
    banner "Step 5: check_depth_completeness"
    [[ -n "$dep_arg" ]] && echo "  Dependency: $dep_arg"
    JOB5=$(sbatch_cmd ${dep_arg:+"$dep_arg"} "$SLURM_DIR/check_depth_completeness_multilab.slurm")
    echo "  Submitted job $JOB5"
    echo "$JOB5"
}

run_step6() {
    local dep_arg="${1:-}"
    banner "Step 6: cache_depth_features (array 0-31)"
    [[ -n "$dep_arg" ]] && echo "  Dependency: $dep_arg"
    JOB6=$(sbatch_cmd ${dep_arg:+"$dep_arg"} "$SLURM_DIR/cache_depth_features_multilab.slurm")
    echo "  Submitted job $JOB6"
    echo "$JOB6"
}

# ── Single-step mode ──────────────────────────────────────────────────────────
if [[ -n "$RUN_STEP" ]]; then
    case "$RUN_STEP" in
        0) run_step0 > /dev/null ;;
        1) run_step1 > /dev/null ;;
        2) run_step2 > /dev/null ;;
        3) run_step3 > /dev/null ;;
        4) run_step4 > /dev/null ;;
        5) run_step5 > /dev/null ;;
        6) run_step6 > /dev/null ;;
        *)
            echo "Unknown step: $RUN_STEP  (valid: 0-6)" >&2
            exit 1
            ;;
    esac
    exit 0
fi

# ── Full pipeline mode ────────────────────────────────────────────────────────
banner "Submitting full multilab pipeline"

# Step 0
JOB0=$(run_step0)
JOB0=$(echo "$JOB0" | tail -1)  # last line = job ID

# Steps 1 and 2 in parallel, both after step 0
JOB1=$(run_step1 "--dependency=afterok:${JOB0}")
JOB1=$(echo "$JOB1" | tail -1)

JOB2=$(run_step2 "--dependency=afterok:${JOB0}")
JOB2=$(echo "$JOB2" | tail -1)

# Step 3 after step 2
JOB3=$(run_step3 "--dependency=afterok:${JOB2}")
JOB3=$(echo "$JOB3" | tail -1)

# Step 4 after steps 1 and 3 (needs backbone shards for index + raw dir for serial map)
JOB4=$(run_step4 "--dependency=afterok:${JOB1}:${JOB3}")
JOB4=$(echo "$JOB4" | tail -1)

# Step 5 after step 4
JOB5=$(run_step5 "--dependency=afterok:${JOB4}")
JOB5=$(echo "$JOB5" | tail -1)

# Step 6 after steps 4 and 5
JOB6=$(run_step6 "--dependency=afterok:${JOB4}:${JOB5}")
JOB6=$(echo "$JOB6" | tail -1)

banner "All jobs submitted"
echo ""
echo "  Job 0 (select_episodes):          $JOB0"
echo "  Job 1 (backbone cache):           $JOB1"
echo "  Job 2 (download depths):          $JOB2"
echo "  Job 3 (extract depths):           $JOB3"
echo "  Job 4 (frame index):              $JOB4"
echo "  Job 5 (check depth):              $JOB5"
echo "  Job 6 (depth features cache):     $JOB6"
echo ""
echo "  Monitor with:  squeue -u $USER"
echo "  Logs in:       $REPO_DIR/logs/"
