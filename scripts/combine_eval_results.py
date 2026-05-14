#!/usr/bin/env python3
"""Combine eval_logs/test/*/ JSON results into a single CSV."""

import csv
import json
import sys
from pathlib import Path

root = Path("eval_logs/test")
rows = []

for json_path in sorted(root.glob("*/*/new_results_0.json")):
    experiment = json_path.parts[-3]
    dataset = json_path.parts[-2]

    with open(json_path) as f:
        data = json.load(f)

    for task, result in data.items():
        row = {
            "experiment": experiment,
            "dataset": dataset,
            "task": task,
            "mean": result.get("mean", ""),
        }
        gt = result.get("GT", {})
        for seed, score in sorted(gt.items(), key=lambda x: int(x[0])):
            row[f"seed_{seed}"] = score
        rows.append(row)

if not rows:
    print("No results found.", file=sys.stderr)
    sys.exit(1)

# Pivot: experiment x dataset -> mean (only G1 / G3)
target_datasets = ["G1", "G3"]
pivot: dict[str, dict[str, float]] = {}
for row in rows:
    if row["dataset"] not in target_datasets:
        continue
    exp = row["experiment"]
    pivot.setdefault(exp, {})
    pivot[exp][row["dataset"]] = row["mean"]

fieldnames = ["experiment"] + target_datasets
out_path = Path("eval_logs/results.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for exp in sorted(pivot):
        writer.writerow({"experiment": exp, **pivot[exp]})

print(f"Wrote {len(pivot)} rows to {out_path}")
