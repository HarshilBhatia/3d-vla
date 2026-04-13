#!/usr/bin/env bash
# Install RLBench for PerAct2 (bimanual, 13 tasks).
# Run from repo root. Uses a sibling RLBench dir; adjust RLBENCH_DIR if needed.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RLBENCH_DIR="${RLBENCH_DIR:-$REPO_ROOT/RLBench}"

cd "$REPO_ROOT"
if [[ -d "$RLBENCH_DIR" ]]; then
  echo "Using existing RLBench at $RLBENCH_DIR"
else
  echo "Cloning RLBench for PerAct2 (markusgrotz/RLBench)..."
  git clone https://github.com/markusgrotz/RLBench.git "$RLBENCH_DIR"
fi
cd "$RLBENCH_DIR"
pip install -r requirements.txt
pip install -e .
echo "PerAct2 RLBench installed at $RLBENCH_DIR"
