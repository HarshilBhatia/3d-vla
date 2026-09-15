#!/usr/bin/env python3
"""List or submit missing cells for a declarative online-evaluation plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.planning.run_plan import cells, command_for_cell, load_plan  # noqa: E402


def is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload)


def compact_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    ranges: list[str] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--submit", action="store_true", help="Submit exactly the missing cells as a Slurm array.")
    parser.add_argument("--max-parallel", type=int, default=24)
    parser.add_argument("--max-requeues", type=int, default=3)
    args = parser.parse_args()
    if args.max_parallel <= 0 or args.max_requeues < 0:
        raise ValueError("max-parallel must be positive and max-requeues non-negative")
    plan, plan_path = load_plan(args.plan)
    missing = []
    for index, cell in enumerate(cells(plan)):
        _, output = command_for_cell(plan, cell)
        if not is_complete(output):
            missing.append(index)
    print(f"{plan['campaign_id']}: complete={len(cells(plan)) - len(missing)} missing={len(missing)} total={len(cells(plan))}")
    if not missing or not args.submit:
        print(compact_indices(missing))
        return
    array = f"{compact_indices(missing)}%{args.max_parallel}"
    command = [
        "sbatch", f"--job-name={plan['campaign_id']}", f"--array={array}",
        f"--export=ALL,EVAL_MAX_REQUEUES={args.max_requeues}",
        str(REPO_ROOT / "scripts/eval/online_eval_plan.slurm"), str(plan_path),
    ]
    print("Submitting:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
