#!/usr/bin/env python3
"""Resolve and run one reproducible online-evaluation campaign cell.

The campaign plan is declarative.  A Slurm array index selects a stable
method × viewpoint × calibration-realization × task cell; no evaluator
semantics are encoded in shell case statements.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_plan(path: str | Path) -> tuple[dict, Path]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as handle:
        plan = json.load(handle)
    required = ("campaign_id", "calibration_registry", "output_root", "methods", "calibrations", "tasks", "runtime")
    missing = [key for key in required if not plan.get(key)]
    has_viewpoints = bool(plan.get("viewpoints")) or bool(plan.get("task_viewpoints"))
    if plan.get("schema_version") != 1 or missing or not has_viewpoints:
        raise ValueError(f"Invalid plan {path}: schema_version=1 and {required} are required; missing={missing}")
    task_calibrations = plan.get("task_calibrations")
    if task_calibrations is not None:
        missing_tasks = set(plan["tasks"]) - set(task_calibrations)
        extra_tasks = set(task_calibrations) - set(plan["tasks"])
        if missing_tasks or extra_tasks:
            raise ValueError(
                "task_calibrations must cover exactly plan.tasks; "
                f"missing={sorted(missing_tasks)}, extra={sorted(extra_tasks)}"
            )
        empty = [task for task, calibration_ids in task_calibrations.items() if not calibration_ids]
        if empty:
            raise ValueError(f"task_calibrations has no calibration IDs for: {empty}")
    return plan, path


def cells(plan: dict) -> list[dict]:
    """Return cells in the documented stable order: method, view, calibration, task."""
    # A task-heldout campaign has one distinct G1--G6 viewpoint per task.  It
    # cannot be represented by the ordinary Cartesian product without
    # incorrectly running every task against every camera group.  Keep the
    # original schema unchanged for shared-viewpoint campaigns.
    task_views = plan.get("task_viewpoints")
    task_calibrations = plan.get("task_calibrations")
    if task_calibrations is not None:
        # Some protocols pair a task/viewpoint with a *different* materialized
        # delta for each camera group.  A Cartesian product over the registry
        # would apply G1's trained base while spawning G2, which is invalid.
        # This explicit mapping keeps only the matched cells.
        if task_views is None:
            raise ValueError("task_calibrations requires task_viewpoints")
        missing = set(plan["tasks"]) - set(task_views)
        if missing:
            raise ValueError(f"task_viewpoints is missing tasks: {sorted(missing)}")
        return [
            {"method": method, "viewpoint": task_views[task], "calibration_id": calibration, "task": task}
            for method, task in itertools.product(plan["methods"], plan["tasks"])
            for calibration in task_calibrations[task]
        ]
    if task_views is not None:
        missing = set(plan["tasks"]) - set(task_views)
        if missing:
            raise ValueError(f"task_viewpoints is missing tasks: {sorted(missing)}")
        return [
            {"method": method, "viewpoint": task_views[task], "calibration_id": calibration, "task": task}
            for method, calibration, task in itertools.product(
                plan["methods"], plan["calibrations"], plan["tasks"]
            )
        ]
    return [
        {"method": method, "viewpoint": viewpoint, "calibration_id": calibration, "task": task}
        for method, viewpoint, calibration, task in itertools.product(
            plan["methods"], plan["viewpoints"], plan["calibrations"], plan["tasks"]
        )
    ]


def resolve_cell(plan: dict, index: int) -> dict:
    resolved = cells(plan)
    if not 0 <= index < len(resolved):
        raise IndexError(f"cell index {index} outside [0, {len(resolved)})")
    return resolved[index]


def _repo_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def command_for_cell(plan: dict, cell: dict, output_root: str | None = None) -> tuple[list[str], Path]:
    method, view = cell["method"], cell["viewpoint"]
    root = Path(output_root or plan["output_root"])
    output = root / plan["campaign_id"] / method["id"] / view["id"] / cell["calibration_id"] / f"results_{cell['task']}.json"
    runtime = plan["runtime"]
    args = [
        sys.executable, "-m", "evaluation.cli",
        f"data={runtime['data']}", f"dataset={runtime['dataset']}", f"bimanual={str(runtime['bimanual']).lower()}",
        f"data_dir={runtime['data_dir']}", f"eval_data_dir={runtime['data_dir']}",
        f"checkpoint={_repo_path(method['checkpoint'])}", f"task={cell['task']}",
        f"headless={str(runtime['headless']).lower()}", f"max_tries={runtime['max_tries']}",
        f"eval_use_depth2cloud={str(runtime['eval_use_depth2cloud']).lower()}", f"output_file={output}",
        f"num_demos_total={runtime['num_demos_total']}",
        f"eval_protocol={plan['campaign_id']}", f"eval_viewpoint_regime={view['regime']}",
        f"eval_calibration_registry={_repo_path(plan['calibration_registry'])}",
        f"eval_calibration_id={cell['calibration_id']}",
    ]
    # Orbital viewpoints require an explicit camera rig and spawn group.  The
    # published PerAct2 rig is native to RLBench, so a clean standard-camera
    # protocol deliberately omits both instead of carrying fake orbital IDs.
    if "cameras_file" in view:
        args.append(f"cameras_file={_repo_path(view['cameras_file'])}")
    if "spawn_camera_group" in view:
        args.append(f"spawn_camera_group={view['spawn_camera_group']}")
    # Most methods share the campaign runtime, but an architecture can require
    # an evaluator-side compatibility override (e.g. Video-DeltaM's history
    # ordering).  Keep that fact attached to the method so a mixed-method plan
    # does not accidentally apply it to every arm.
    overrides = {**runtime.get("overrides", {}), **method.get("overrides", {})}
    for key, value in overrides.items():
        args.append(f"{key}={value}")
    return args, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--output-root", help="Replace plan output_root without changing condition semantics.")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    plan, plan_path = load_plan(args.plan)
    cell = resolve_cell(plan, args.cell_index)
    command, output = command_for_cell(plan, cell, args.output_root)
    print(json.dumps({"plan": str(plan_path), "cell_index": args.cell_index, "cell": cell, "output": str(output)}, indent=2), flush=True)
    if output.exists():
        print(f"[skip] output exists: {output}", flush=True)
        return
    if args.print_command:
        print(" ".join(command), flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
