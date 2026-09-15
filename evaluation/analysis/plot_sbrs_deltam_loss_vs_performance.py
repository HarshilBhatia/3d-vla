#!/usr/bin/env python3
"""Plot DeltaM task losses against completed SBRS performance checkpoints.

Uses only canonical artifacts: rank-0 train logs for task-level train/val
metrics and SBRS result JSONs for rollout success.  A checkpoint is paired with
the most recent validation event at or before it (never future loss data).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = (
    "bimanual_lift_ball",
    "bimanual_push_box",
    "bimanual_handover_item_easy",
    "bimanual_pick_laptop",
    "bimanual_pick_plate",
    "bimanual_straighten_rope",
)
CONDITIONS = ("e0", "e3deg-t1cm")
STEP_RE = re.compile(r"^Step (\d+):$")
METRIC_RE = re.compile(
    r"^(train|val)-loss/(bimanual_[^/]+)/(traj_pos_l2|traj_rot_l1): ([0-9.eE+-]+)$"
)
CKPT_RE = re.compile(r"sbrs_checkpoint_ladder_v01_view_align_s(\d+)")
COND_RE = re.compile(r"/seen-[^/]+-(e0|e3deg-t1cm)-v01/")


def parse_loss_events(log_path: Path) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    """Return split -> validation/training step -> task -> loss component."""
    events: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        "train": defaultdict(lambda: defaultdict(dict)),
        "val": defaultdict(lambda: defaultdict(dict)),
    }
    current_step: int | None = None
    for raw_line in log_path.read_text(errors="replace").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        step_match = STEP_RE.match(line)
        if step_match:
            current_step = int(step_match.group(1))
            continue
        metric_match = METRIC_RE.match(line)
        if metric_match and current_step is not None:
            split, task, metric, value = metric_match.groups()
            if task in TASKS:
                events[split][current_step][task][metric] = float(value)
    return events


def parse_sbrs_results(results_root: Path) -> dict[int, dict[str, dict[str, float]]]:
    results: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in results_root.glob("sbrs_checkpoint_ladder_v01_view_align_s*/view_align/**/results_*.json"):
        if path.name.endswith(".manifest.json"):
            continue
        checkpoint_match = CKPT_RE.search(str(path))
        condition_match = COND_RE.search(str(path))
        if not checkpoint_match or not condition_match:
            continue
        checkpoint, condition = int(checkpoint_match.group(1)), condition_match.group(1)
        with path.open() as handle:
            payload = json.load(handle)
        if len(payload) != 1:
            raise ValueError(f"Expected one task in {path}")
        task, task_payload = next(iter(payload.items()))
        if task not in TASKS:
            continue
        if task in results[checkpoint][condition]:
            raise ValueError(f"Duplicate SBRS result for step={checkpoint}, condition={condition}, task={task}")
        results[checkpoint][condition][task] = float(task_payload["mean"])
    return results


def aligned_rows(events, results):
    rows = []
    val_steps = sorted(events["val"])
    for checkpoint in sorted(results):
        eligible = [step for step in val_steps if step + 1 <= checkpoint]
        if not eligible:
            raise ValueError(f"No causal validation loss event for SBRS checkpoint {checkpoint}")
        loss_step = eligible[-1]
        for condition in CONDITIONS:
            task_results = results[checkpoint].get(condition, {})
            missing = set(TASKS) - set(task_results)
            if missing:
                raise ValueError(f"SBRS checkpoint {checkpoint}, {condition} missing tasks: {sorted(missing)}")
            for task in TASKS:
                for split in ("train", "val"):
                    losses = events[split].get(loss_step, {}).get(task, {})
                    missing_metrics = {"traj_pos_l2", "traj_rot_l1"} - set(losses)
                    if missing_metrics:
                        raise ValueError(f"{split} step {loss_step} missing {task}: {sorted(missing_metrics)}")
                rows.append({
                    "checkpoint": checkpoint,
                    "loss_step": loss_step + 1,
                    "condition": condition,
                    "task": task,
                    "success": task_results[task],
                    "train_pos_l2": events["train"][loss_step][task]["traj_pos_l2"],
                    "val_pos_l2": events["val"][loss_step][task]["traj_pos_l2"],
                    "train_rot_l1": events["train"][loss_step][task]["traj_rot_l1"],
                    "val_rot_l1": events["val"][loss_step][task]["traj_rot_l1"],
                })
    return rows


def plot_aggregate(rows, output: Path):
    checkpoints = sorted({row["checkpoint"] for row in rows})
    summaries = {}
    for checkpoint in checkpoints:
        selected = [row for row in rows if row["checkpoint"] == checkpoint]
        summaries[checkpoint] = {key: float(np.mean([row[key] for row in selected])) for key in (
            "train_pos_l2", "val_pos_l2", "train_rot_l1", "val_rot_l1"
        )}
        for condition in CONDITIONS:
            condition_rows = [row for row in selected if row["condition"] == condition]
            summaries[checkpoint][condition] = float(np.mean([row["success"] for row in condition_rows]))

    x = np.asarray(checkpoints) / 1000
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True, constrained_layout=True)
    colors = {"e0": "#0072B2", "e3deg-t1cm": "#D55E00"}
    labels = {"e0": "e0 (seen base only)", "e3deg-t1cm": "e3 (seen base + 3°/1 cm residual)"}
    for condition in CONDITIONS:
        y = [summaries[checkpoint][condition] for checkpoint in checkpoints]
        axes[0].plot(x, y, marker="o", linewidth=2.2, color=colors[condition], label=labels[condition])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("SBRS success")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("DeltaM SBRS: six tasks, 20 rollouts/task; external camera base + residual")

    for key, label, style in (("train_pos_l2", "train", "--"), ("val_pos_l2", "validation", "-")):
        axes[1].plot(x, [summaries[c][key] for c in checkpoints], style, marker="o", linewidth=2, label=label)
    axes[1].set_ylabel("Mean position L2")
    axes[1].legend(fontsize=8)
    for key, label, style in (("train_rot_l1", "train", "--"), ("val_rot_l1", "validation", "-")):
        axes[2].plot(x, [summaries[c][key] for c in checkpoints], style, marker="o", linewidth=2, label=label)
    axes[2].set_ylabel("Mean rotation L1")
    axes[2].set_xlabel("Checkpoint (thousands of steps)")
    axes[2].legend(fontsize=8)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    return summaries


def plot_per_task(rows, output: Path):
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(10, 15), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    checkpoints = sorted({row["checkpoint"] for row in rows})
    checkpoint_color = {step: cmap(i / max(1, len(checkpoints) - 1)) for i, step in enumerate(checkpoints)}
    for task_index, task in enumerate(TASKS):
        task_rows = [row for row in rows if row["task"] == task]
        for axis, metric, label in zip(axes[task_index], ("pos_l2", "rot_l1"), ("Position L2", "Rotation L1")):
            for split, face in (("train", "none"), ("val", None)):
                for condition, marker in (("e0", "o"), ("e3deg-t1cm", "^")):
                    selected = [row for row in task_rows if row["condition"] == condition]
                    colors = [checkpoint_color[row["checkpoint"]] for row in selected]
                    axis.scatter(
                        [row[f"{split}_{metric}"] for row in selected], [row["success"] for row in selected],
                        c=colors if face is None else "none", marker=marker,
                        edgecolors=colors, linewidths=1.2, s=48,
                    )
            axis.set_ylim(0, 1.05)
            axis.grid(alpha=0.22)
            axis.set_xlabel(f"Task {label} (hollow train, filled validation)")
            axis.set_ylabel("SBRS success")
            axis.set_title(task.replace("bimanual_", "").replace("_", " "), fontsize=10)
    fig.suptitle("DeltaM task success vs task-level loss (○ e0, △ e3; hollow train, filled validation; color checkpoint)", fontsize=12)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, default=REPO_ROOT / "logs/train/slurm-fixedbias-randall-3946147.out")
    parser.add_argument("--results-root", type=Path, default=Path("/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "eval/interp/sbrs_deltam_loss_vs_performance")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = aligned_rows(parse_loss_events(args.train_log), parse_sbrs_results(args.results_root))
    csv_path = args.output_dir / "aligned_task_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = plot_aggregate(rows, args.output_dir / "aggregate_loss_vs_performance")
    plot_per_task(rows, args.output_dir / "per_task_loss_vs_performance")
    print(f"Wrote {csv_path}")
    for name in ("aggregate_loss_vs_performance", "per_task_loss_vs_performance"):
        print(f"Wrote {args.output_dir / (name + '.png')}")
        print(f"Wrote {args.output_dir / (name + '.pdf')}")
    for checkpoint, values in summary.items():
        print(f"{checkpoint}: e0={values['e0']:.4f}, e3={values['e3deg-t1cm']:.4f}, val_pos_l2={values['val_pos_l2']:.5f}")


if __name__ == "__main__":
    main()
