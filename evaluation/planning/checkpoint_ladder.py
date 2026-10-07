#!/usr/bin/env python3
"""Submit a 20-rollout external-only SBRS evaluation for every 10k checkpoint.

The script materializes one ordinary evaluation plan per intermediate checkpoint
and submits its six-task-by-two-condition cells as an unthrottled Slurm array. Its submission
ledger makes it safe to run repeatedly: historical checkpoints and checkpoints
noticed by the watcher are submitted exactly once unless --force is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from evaluation.planning.run_plan import cells


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERM_RE = re.compile(r"interm_step_(\d+)\.pth$")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with repo_path(path).open() as handle:
        config = json.load(handle)
    if "source_plans" in config:
        return _check_source_config(config)
    required = {"protocol_id", "calibration_registry", "output_root", "tasks", "runtime"}
    missing = required - set(config)
    if config.get("schema_version") != 1 or missing:
        raise ValueError(f"Invalid checkpoint-ladder config: missing={sorted(missing)}")
    if not config["tasks"] or not config["residual_conditions"]:
        raise ValueError("tasks and residual_conditions must be non-empty")
    return config


def _check_source_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = {"protocol_id", "output_root", "source_plans", "tasks", "runtime", "min_step", "step_interval"} - set(config)
    if config.get("schema_version") != 1 or missing or not config["source_plans"]:
        raise ValueError(f"Invalid source-plan checkpoint-ladder config: missing={sorted(missing)}")
    names = [source.get("name") for source in config["source_plans"]]
    if None in names or len(set(names)) != len(names):
        raise ValueError("Each source_plans entry needs a unique name")
    if "eval_weights" in config["runtime"]:
        raise ValueError("eval_weights must go in runtime.overrides; the evaluator ignores it at runtime level")
    for source in config["source_plans"]:
        if not source.get("plan") or not source.get("calibrations"):
            raise ValueError(f"source_plans entry {source['name']} needs plan and calibrations")
    return config


def _keeps(calibration_id: str, levels: list[str]) -> bool:
    return any(calibration_id == level or f"-{level}-" in calibration_id for level in levels)


def build_source_plan(config: dict[str, Any], source: dict[str, Any], checkpoint: Path, method_id: str, step: int) -> dict[str, Any]:
    with repo_path(source["plan"]).open() as handle:
        base = json.load(handle)
    if base["tasks"] != config["tasks"]:
        raise ValueError(f"{source['plan']} tasks differ from the recipe's tasks")
    levels = source["calibrations"]
    unknown = set(levels) - set(base["calibrations"])
    if unknown:
        raise ValueError(f"{source['plan']} has no calibration levels {sorted(unknown)}")
    task_calibrations = {task: [c for c in ids if _keeps(c, levels)] for task, ids in base["task_calibrations"].items()}
    empty = [task for task, ids in task_calibrations.items() if not ids]
    if empty:
        raise ValueError(f"{source['plan']}: no kept calibration for tasks {empty}")
    return {
        "schema_version": 1,
        "campaign_id": f"{config['protocol_id']}_{method_id}_s{step:06d}",
        "description": f"{config['description']} Source {source['name']} ({source['plan']}, levels {levels}). Checkpoint step {step}: {checkpoint}.",
        "calibration_registry": base["calibration_registry"],
        "output_root": config["output_root"],
        "methods": [{"id": method_id, "checkpoint": str(checkpoint.resolve())}],
        "task_viewpoints": base["task_viewpoints"],
        "task_calibrations": task_calibrations,
        "calibrations": [c for c in base["calibrations"] if c in levels],
        "tasks": base["tasks"],
        "runtime": config["runtime"],
    }


def build_plans(config: dict[str, Any], checkpoint: Path, method_id: str, step: int) -> list[tuple[str, str, dict[str, Any]]]:
    key = f"{method_id}:s{step:06d}"
    if "source_plans" not in config:
        plan = build_plan(config, checkpoint, method_id, step)
        return [(key, plan["campaign_id"], plan)]
    built = []
    for source in config["source_plans"]:
        plan = build_source_plan(config, source, checkpoint, method_id, step)
        built.append((f"{key}:{source['name']}", f"{plan['campaign_id']}_{source['name']}", plan))
    return built


def checkpoint_steps(checkpoint_dir: Path, min_step: int, interval: int, max_step: int | None = None) -> list[tuple[int, Path]]:
    found = []
    for checkpoint in checkpoint_dir.glob("interm_step_*.pth"):
        match = INTERM_RE.fullmatch(checkpoint.name)
        if not match:
            continue
        step = int(match.group(1))
        if step < min_step or step % interval or (max_step is not None and step > max_step):
            continue
        found.append((step, checkpoint))
    return sorted(found)


def _task_view(task: dict[str, str]) -> dict[str, str]:
    group = task["spawn_camera_group"]
    return {
        "id": f"{group.lower()}_seen_train_group",
        "spawn_camera_group": group,
        "regime": "seen_train_group",
        "cameras_file": "instructions/orbital_cameras_grouped.json",
    }


def build_plan(config: dict[str, Any], checkpoint: Path, method_id: str, step: int) -> dict[str, Any]:
    task_views = {entry["task"]: _task_view(entry) for entry in config["tasks"]}
    task_calibrations = {}
    for entry in config["tasks"]:
        group = entry["spawn_camera_group"].lower()
        task_calibrations[entry["task"]] = [
            f"seen-{group}-{config['base_level']}-{residual}-{config['realization_version']}"
            for residual in config["residual_conditions"]
        ]
    suffix = f"{method_id}_s{step:06d}"
    return {
        "schema_version": 1,
        "campaign_id": f"{config['protocol_id']}_{suffix}",
        "description": f"{config['description']} Checkpoint step {step}: {checkpoint}.",
        "calibration_registry": config["calibration_registry"],
        "output_root": config["output_root"],
        "methods": [{"id": method_id, "checkpoint": str(checkpoint.resolve())}],
        "task_viewpoints": task_views,
        "task_calibrations": task_calibrations,
        "calibrations": config["residual_conditions"],
        "tasks": [entry["task"] for entry in config["tasks"]],
        "runtime": config["runtime"],
    }


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "submissions": {}}
    with path.open() as handle:
        return json.load(handle)


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(ledger, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")


def submit_plan(plan_path: Path, plan: dict[str, Any]) -> str:
    cell_count = len(cells(plan))
    # The generated plan always contains one method.  Do not append %N here:
    # every task cell should be schedulable immediately.
    job_name = plan_path.stem[:120]
    command = [
        "sbatch", "--partition=all", f"--job-name={job_name}",
        f"--array=0-{cell_count - 1}",
        f"--output={REPO_ROOT}/logs/eval/{job_name}_%A_%a.out",
        f"--error={REPO_ROOT}/logs/eval/{job_name}_%A_%a.err",
        str(REPO_ROOT / "scripts/eval/online_eval_plan.slurm"), str(plan_path),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout.strip(), flush=True)
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse Slurm submission: {result.stdout!r}")
    return match.group(1)


def process_directory(
    config: dict[str, Any], checkpoint_dir: Path, method_id: str, *, min_step: int,
    max_step: int | None, submit: bool, force: bool, ledger_path: Path,
) -> list[int]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    ledger = load_ledger(ledger_path)
    submitted = []
    for step, checkpoint in checkpoint_steps(checkpoint_dir, min_step, config["step_interval"], max_step):
        for key, plan_name, plan in build_plans(config, checkpoint, method_id, step):
            plan_path = checkpoint_dir / "checkpoint_ladder_plans" / f"{plan_name}.json"
            write_plan(plan_path, plan)
            if key in ledger["submissions"] and not force:
                print(f"[skip] {key}: already submitted as job {ledger['submissions'][key]['job_id']}", flush=True)
                continue
            if not submit:
                print(f"[plan] {key}: {plan_path}", flush=True)
            else:
                job_id = submit_plan(plan_path, plan)
                ledger["submissions"][key] = {
                    "job_id": job_id, "checkpoint": str(checkpoint.resolve()), "plan": str(plan_path.resolve()),
                }
                save_ledger(ledger_path, ledger)
            if not submitted or submitted[-1] != step:
                submitted.append(step)
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--method-id", required=True, help="Stable arm label, e.g. base or view_align.")
    parser.add_argument("--config", default="instructions/eval_plans/sbrs_checkpoint_ladder_v01.json")
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--submit", action="store_true", help="Submit generated plans to Slurm; omitted means dry-run plan generation.")
    parser.add_argument("--force", action="store_true", help="Submit even if the ledger already records this checkpoint.")
    parser.add_argument("--ledger", help="Defaults to <checkpoint-dir>/checkpoint_ladder_submissions.json")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_dir = repo_path(args.checkpoint_dir)
    ledger_path = repo_path(args.ledger) if args.ledger else checkpoint_dir / "checkpoint_ladder_submissions.json"
    process_directory(
        config, checkpoint_dir, args.method_id,
        min_step=args.min_step if args.min_step is not None else config["min_step"],
        max_step=args.max_step, submit=args.submit, force=args.force, ledger_path=ledger_path,
    )


if __name__ == "__main__":
    main()
