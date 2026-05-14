"""Plot checkpoint MSE metrics vs training step, grouped by CFG scale.

Results directory layout expected:
    <results_dir>/cfg_<N>/step_<S>.csv   (one CSV per checkpoint per cfg)

Each CSV has columns:
    step, cfg_scale, dataset, n_samples, traj_pos_l2, traj_rot_l1,
    traj_pos_acc_001, traj_rot_acc_0025, traj_gripper

Output layout:
    <out_dir>/cfg_<N>/<metric>.png   — one PNG per metric, 1 curve per dataset

Usage:
    python scripts/eval/plot_checkpoint_mse_cfg.py results/checkpoint_mse_cfg/siglip_single_dense/
    python scripts/eval/plot_checkpoint_mse_cfg.py results/checkpoint_mse_cfg/siglip_single_dense/ --out plots/siglip_single_dense/
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

CFG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_all(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("cfg_*/step_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No cfg_*/step_*.csv found in {results_dir}")
    frames = []
    for f in csvs:
        df = pd.read_csv(f)
        if "cfg_scale" not in df.columns:
            df["cfg_scale"] = f.parent.name.replace("cfg_", "")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["cfg_scale", "step"]).reset_index(drop=True)
    print(f"Loaded {len(csvs)} CSV(s) → {len(df)} rows")
    print(f"  CFG scales: {sorted(df['cfg_scale'].unique())}")
    print(f"  Steps:      {sorted(df['step'].unique())}")
    print(f"  Datasets:   {sorted(df['dataset'].unique())}")
    return df


def plot_dataset(df_dataset, dataset: str, metric: str, cfg_scales, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cfg in enumerate(cfg_scales):
        subset = df_dataset[df_dataset["cfg_scale"] == cfg].sort_values("step")
        ax.plot(
            subset["step"],
            subset[metric],
            marker="o",
            linewidth=2,
            markersize=5,
            color=CFG_COLORS[i % len(CFG_COLORS)],
            label=f"cfg={cfg}",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)}  [{dataset}]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df = load_all(args.results_dir)
    base_out = args.out or (args.results_dir / "plots")

    cfg_scales = sorted(df["cfg_scale"].unique(), key=lambda x: float(x))

    for dataset in sorted(df["dataset"].unique()):
        df_ds = df[df["dataset"] == dataset]
        out_dir = base_out / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\ndataset={dataset}  → {out_dir}/")
        for metric in METRIC_LABELS:
            if metric not in df_ds.columns:
                continue
            plot_dataset(df_ds, dataset, metric, cfg_scales, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
