"""Plot online eval success rate vs training step.

Each run directory should have the layout:
    <run_dir>/step_<N>/<scenario>/results.json

results.json format:
    {task: {group: {variation: rate, ...}, "mean": float}}

Usage:
    # one run
    python scripts/eval/plot_online_eval.py eval_logs/Orbital/siglip_multi_group_od_interm

    # compare multiple runs (legend shows run+scenario)
    python scripts/eval/plot_online_eval.py \\
        eval_logs/Orbital/siglip_multi_group_od_interm \\
        eval_logs/Orbital/siglip_single_cam \\
        --out plots/online_eval.png
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def load_run(run_dir: Path) -> list[dict]:
    """Return list of {step, scenario, rate} dicts for a run directory."""
    records = []
    for result_path in sorted(run_dir.glob("step_*/*/results.json")):
        step_match = re.search(r"step_(\d+)", result_path.parts[-3])
        if not step_match:
            continue
        step = int(step_match.group(1))
        scenario = result_path.parts[-2]
        data = json.loads(result_path.read_text())
        # data = {task: {..., "mean": float}}
        means = [v["mean"] for v in data.values() if isinstance(v, dict) and "mean" in v]
        if not means:
            continue
        rate = sum(means) / len(means)
        records.append({"step": step, "scenario": scenario, "rate": rate})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="One or more eval_logs run directories")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: first run_dir/plots/online_eval.png)")
    parser.add_argument("--title", default="Online Eval — Success Rate vs Step")
    args = parser.parse_args()

    out_path = args.out or (args.run_dirs[0] / "plots" / "online_eval.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    color_idx = 0

    for run_dir in args.run_dirs:
        run_name = run_dir.name
        records = load_run(run_dir)
        if not records:
            print(f"  No results found in {run_dir}")
            continue

        # Group by scenario
        scenarios: dict[str, list] = {}
        for r in records:
            scenarios.setdefault(r["scenario"], []).append((r["step"], r["rate"]))

        for scenario, points in sorted(scenarios.items()):
            points.sort()
            steps, rates = zip(*points)
            label = f"{run_name} / {scenario}" if len(args.run_dirs) > 1 else scenario
            ax.plot(steps, rates,
                    marker="o", linewidth=2, markersize=5,
                    color=COLORS[color_idx % len(COLORS)],
                    label=label)
            color_idx += 1
            print(f"  {label}: {dict(zip(steps, [f'{r:.3f}' for r in rates]))}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean success rate")
    ax.set_title(args.title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
