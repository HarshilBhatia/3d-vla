#!/usr/bin/env bash
# Install RLBench for PerAct (unimanual, 18 tasks).
# Run from repo root. Uses a separate clone; set RLBENCH_PERACT_DIR to override.
# After install, modify the close_jar success condition - see README_RLBench.md.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RLBENCH_DIR="${RLBENCH_PERACT_DIR:-$REPO_ROOT/RLBench_PerAct}"

cd "$REPO_ROOT"
if [[ -d "$RLBENCH_DIR" ]]; then
  echo "Using existing RLBench at $RLBENCH_DIR"
  cd "$RLBENCH_DIR"
  git fetch origin
  git checkout -b peract --track origin/peract 2>/dev/null || git checkout peract
else
  echo "Cloning RLBench for PerAct (MohitShridhar/RLBench, branch peract)..."
  git clone https://github.com/MohitShridhar/RLBench.git "$RLBENCH_DIR"
  cd "$RLBENCH_DIR"
  git checkout -b peract --track origin/peract
fi
pip install -r requirements.txt
pip install -e .
echo "PerAct RLBench installed at $RLBENCH_DIR"
echo "Remember to fix the close_jar success condition - see scripts/rlbench/README_RLBench.md"
