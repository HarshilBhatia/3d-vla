#!/bin/bash
#SBATCH --job-name=clean_proprio_gate
#SBATCH --output=/home/harshilb/3dfa_unified/logs/eval/clean_proprio_gate-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --partition=all
# TODO 16 gate: submit the 78-cell clean-proprio IDLT campaign only if the offline check passed and both dry-run cells behaved.
set -uo pipefail
cd /home/harshilb/3dfa_unified
CHECK="logs/analysis/check_clean_proprio-${CHECK_JOB}.out"
CAL="logs/eval/online_eval_plan-${CAL_JOB}_0.out"
HEAVY="logs/eval/online_eval_plan-${HEAVY_JOB}_0.out"
fail() { echo "GATE FAILED: $1"; exit 1; }
[ "$(grep -c ' PASS$' "$CHECK")" -eq 3 ] || fail "offline check did not PASS for all 3 models"
grep -q "\[eval_clean_proprio\] max |true - used| pcd per camera (m): \[0.0, 0.0, 0.0, 0.0\]" "$CAL" || fail "calibrated dry run: true and used clouds differ"
L=$(grep -m1 "\[eval_clean_proprio\] max |true - used|" "$HEAVY") || fail "60/60 dry run printed no oracle line"
python3 - "$L" <<'PY' || fail "60/60 dry run: expected cameras 0,1 to differ and 2,3 to match"
import ast, sys
d = ast.literal_eval(sys.argv[1].split(":", 1)[1].strip())
sys.exit(0 if d[0] > 0.05 and d[1] > 0.05 and d[2] == 0 and d[3] == 0 else 1)
PY
ls /grogu/datasets/hbhatia/3dfa_online_eval_dryrun/*/exp1b_adaln_pooled_s200k_ema_clean_proprio/*/*/results_bimanual_push_box.json | wc -l | grep -q '^2$' || fail "dry-run result files missing"
echo "GATE PASSED; submitting"
sbatch --array=0-38 scripts/eval/online_eval_plan.slurm instructions/eval_plans/idlt_task_heldout_group_v01_clean_proprio.json
sbatch --array=0-38 scripts/eval/online_eval_plan.slurm instructions/eval_plans/idlt_scaled_extreme_task_heldout_group_v01_clean_proprio.json
