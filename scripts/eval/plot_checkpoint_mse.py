"""Aggregate per-checkpoint CSVs and plot metrics vs training step.

Usage:
    python scripts/eval/plot_checkpoint_mse.py results/checkpoint_mse/siglip_single_cam/
    python scripts/eval/plot_checkpoint_mse.py results/checkpoint_mse/siglip_single_cam/ --out plots/siglip_single_cam/

Each CSV in the directory should have columns:
    step, dataset, n_samples, traj_pos_l2, traj_rot_l1, traj_pos_acc_001, traj_rot_acc_0025, traj_gripper

Outputs one PNG per metric into --out (default: <csv_dir>/plots/).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "traj_pos_l2":       "Position L2 (m)",
    "traj_rot_l1":       "Rotation L1 (quat)",
    "traj_pos_acc_001":  "Position Acc @ 1cm",
    "traj_rot_acc_0025": "Rotation Acc @ 0.025",
    "traj_gripper":      "Gripper Accuracy",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_csvs(csv_dir: Path) -> pd.DataFrame:
    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    df = df.sort_values("step").reset_index(drop=True)
    print(f"Loaded {len(csvs)} CSV(s) → {len(df)} rows")
    print(f"  Steps:    {sorted(df['step'].unique())}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    return df


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path):
    datasets = sorted(df["dataset"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset].sort_values("step")
        ax.plot(
            subset["step"],
            subset[metric],
            marker="o",
            linewidth=2,
            markersize=5,
            color=COLORS[i % len(COLORS)],
            label=dataset,
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", type=Path, help="Directory containing per-checkpoint CSV files")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for PNGs (default: csv_dir/plots/)")
    args = parser.parse_args()

    out_dir = args.out or (args.csv_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(args.csv_dir)

    print(f"\nPlotting {len(METRIC_LABELS)} metrics → {out_dir}/")
    for metric in METRIC_LABELS:
        if metric not in df.columns:
            print(f"  Skipping {metric} (not in CSV)")
            continue
        plot_metric(df, metric, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
