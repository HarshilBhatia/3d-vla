#!/usr/bin/env python
"""Plot the 50-rollout external-vs-wrist extrinsic-noise sweep."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/grogu/datasets/hbhatia/3dfa_online_eval_50rollouts")
OUT = Path(__file__).resolve().parents[1] / "docs/results/figures"
LEVELS = [2, 5, 10]
TASKS = [
    "push_box", "lift_ball", "dual_push_buttons", "pick_plate",
    "put_item_in_drawer", "put_bottle_in_fridge", "handover_item",
    "pick_laptop", "straighten_rope", "sweep_to_dustpan", "lift_tray",
    "handover_item_easy", "take_tray_out_of_oven",
]


def read_values(camera, level):
    path = ROOT / f"{camera}_{level}deg_{level}cm" / camera
    values = {}
    for file in path.glob("results_*.json"):
        data = json.loads(file.read_text())
        task, result = next(iter(data.items()))
        values[task.removeprefix("bimanual_")] = float(result["mean"])
    if set(values) != set(TASKS):
        raise RuntimeError(f"Expected 13 tasks in {path}, found {len(values)}")
    return np.array([values[t] for t in TASKS])


def read_condition_values(condition):
    path = ROOT / condition
    values = {}
    for file in path.glob("results_*.json"):
        data = json.loads(file.read_text())
        task, result = next(iter(data.items()))
        values[task.removeprefix("bimanual_")] = float(result["mean"])
    if set(values) != set(TASKS):
        raise RuntimeError(f"Expected 13 tasks in {path}, found {len(values)}")
    return np.array([values[t] for t in TASKS])


external = np.stack([read_values("external", x) for x in LEVELS])
wrist = np.stack([read_values("wrist", x) for x in LEVELS])
means = {"External cameras": external.mean(1), "Shoulder cameras": wrist.mean(1)}
clean_baseline = read_condition_values("clean/all").mean()

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(13, 9), dpi=180)
grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.55], hspace=0.42, wspace=0.18)

ax = fig.add_subplot(grid[0, :])
colors = {"External cameras": "#1769aa", "Shoulder cameras": "#d95f02"}
for label, vals in means.items():
    ax.plot(LEVELS, vals, marker="o", linewidth=3, markersize=8, label=label, color=colors[label])
    for x, y in zip(LEVELS, vals):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
ax.axhline(clean_baseline, color="#555", linestyle=(0, (4, 3)), linewidth=1.7,
           label=f"All-camera clean baseline ({clean_baseline:.3f})")
ax.set_title("PerAct2 Online Evaluation: Camera-Subset Extrinsic Noise", fontsize=15, weight="bold", pad=12)
ax.set_xlabel("Random test-time rotation / translation noise")
ax.set_ylabel("Mean success rate (13 tasks)")
ax.set_xticks(LEVELS, [f"{x}° + {x} cm" for x in LEVELS])
ax.set_ylim(0, 0.82)
ax.legend(loc="lower left", ncol=3, frameon=True)
ax.text(0.99, 0.04, "50 rollouts/task · fixed noise direction per level", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color="#555")

for col, (name, data) in enumerate([("External-only", external), ("Shoulder-only", wrist)]):
    ax = fig.add_subplot(grid[1, col])
    ax.set_title(name, fontsize=12, weight="bold")
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{value:.3f}" for value in row] for row in data.T],
        rowLabels=TASKS,
        colLabels=[f"{x}° + {x} cm" for x in LEVELS],
        cellLoc="center",
        rowLoc="center",
        loc="center",
        colWidths=[0.22, 0.22, 0.22],
        edges="closed",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)
    for (row, cell_col), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
        if cell_col == -1:
            cell.set_text_props(ha="right")

fig.suptitle("Clean B200 PerAct2 checkpoint · external vs shoulder-camera sensitivity", fontsize=11, y=0.995)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "peract2_external_wrist_miscal_50rollouts.png", bbox_inches="tight")
print(OUT / "peract2_external_wrist_miscal_50rollouts.png")
print(f"all-camera clean baseline: {clean_baseline:.4f}")
for label, vals in means.items():
    print(label + ": " + ", ".join(f"{level}={value:.4f}" for level, value in zip(LEVELS, vals)))
