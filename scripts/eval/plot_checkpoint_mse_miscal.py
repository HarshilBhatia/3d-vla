"""Aggregate per-checkpoint miscal CSVs and plot metrics vs training step.

Usage:
    python scripts/eval/plot_checkpoint_mse_miscal.py results/checkpoint_mse_miscal/siglip_multi_group_od/
    python scripts/eval/plot_checkpoint_mse_miscal.py results/checkpoint_mse_miscal/siglip_multi_deltaM/ --out plots/

Each CSV must have columns:
    step, miscal_level, dataset, n_samples, traj_pos_l2, traj_rot_l1, traj_pos_acc_001, traj_rot_acc_0025, traj_gripper

Outputs one PNG per metric into --out (default: <csv_dir>/plots/).
Each plot has one line per (dataset, miscal_level) combination.
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

NOISE_ORDER = ["none", "small", "medium", "large"]
NOISE_STYLE = {
    "none":   dict(linestyle="-",             marker="o", linewidth=2.5, markersize=7,  alpha=1.0),
    "small":  dict(linestyle="--",            marker="s", linewidth=1.8, markersize=5,  alpha=0.85),
    "medium": dict(linestyle=(0, (4, 2, 1, 2)), marker="^", linewidth=1.8, markersize=6,  alpha=0.85),
    "large":  dict(linestyle=":",             marker="D", linewidth=1.8, markersize=5,  alpha=0.85),
}
_FALLBACK_STYLE = dict(linestyle=(0, (3, 1, 1, 1, 1, 1)), marker="P", linewidth=1.8, markersize=6, alpha=0.85)
# Distinct, colorblind-friendly palette
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]


def load_csvs(csv_dir: Path, miscal_level_fill: str | None = None) -> pd.DataFrame:
    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    if "miscal_level" not in df.columns:
        df["miscal_level"] = miscal_level_fill or "none"
    return df


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path):
    datasets = sorted(df["dataset"].unique())
    known = set(NOISE_ORDER)
    noise_levels = [n for n in NOISE_ORDER if n in df["miscal_level"].unique()] + \
                   [n for n in sorted(df["miscal_level"].unique()) if n not in known]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, dataset in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        for noise in noise_levels:
            subset = df[(df["dataset"] == dataset) & (df["miscal_level"] == noise)].sort_values("step")
            if subset.empty:
                continue
            ax.plot(
                subset["step"],
                subset[metric],
                color=color,
                label=f"{dataset} / {noise}",
                **NOISE_STYLE.get(noise, _FALLBACK_STYLE),
            )

    ax.set_xlabel("Training step")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric))
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", type=Path)
    parser.add_argument("--no-miscal-dir", type=Path, default=None,
                        help="Directory of no-miscal CSVs (from eval_checkpoints_mse.py) to overlay as 'none'")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (args.csv_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(args.csv_dir)
    if args.no_miscal_dir:
        df_base = load_csvs(args.no_miscal_dir, miscal_level_fill="none")
        df = pd.concat([df_base, df], ignore_index=True)

    df = df.sort_values(["step", "miscal_level", "dataset"]).reset_index(drop=True)
    print(f"Total rows: {len(df)}")
    print(f"  Steps:        {sorted(df['step'].unique())}")
    print(f"  Noise levels: {sorted(df['miscal_level'].unique())}")
    print(f"  Datasets:     {sorted(df['dataset'].unique())}")

    print(f"\nPlotting {len(METRIC_LABELS)} metrics → {out_dir}/")
    for metric in METRIC_LABELS:
        if metric not in df.columns:
            print(f"  Skipping {metric} (not in CSV)")
            continue
        plot_metric(df, metric, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
