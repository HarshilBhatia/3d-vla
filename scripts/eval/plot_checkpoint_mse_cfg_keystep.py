"""Plot keystep MSE metrics vs training step, broken down by keypose step (start/mid/end).

Results directory layout expected:
    <results_dir>/cfg_<N>/step_<S>.csv

Each CSV has columns:
    step (0=start, 1=mid, 2=end), cfg_scale, dataset, n_samples,
    traj_pos_l2, traj_rot_l1, traj_pos_acc_001, traj_rot_acc_0025, traj_gripper

For each (metric, dataset): one figure with side-by-side subplots per cfg_scale,
each showing 3 lines (start / mid / end) vs training step.

Usage:
    python scripts/eval/plot_checkpoint_mse_cfg_keystep.py results/keystep/siglip_single_group/
    python scripts/eval/plot_checkpoint_mse_cfg_keystep.py results/keystep/siglip_multi_group/ --out plots/keystep/mg/
"""
import argparse
import re
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

STEP_LABELS = {0: "start", 1: "mid", 2: "end"}
STEP_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}


def load_all(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("cfg_*/step_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No cfg_*/step_*.csv found in {results_dir}")
    frames = []
    for f in csvs:
        df = pd.read_csv(f)
        m = re.search(r"step_(\d+)\.csv$", f.name)
        df["train_step"] = int(m.group(1)) if m else 0
        cfg_raw = f.parent.name.replace("cfg_", "")
        df["cfg_scale"] = cfg_raw
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["cfg_scale", "train_step", "step"]).reset_index(drop=True)
    print(f"Loaded {len(csvs)} CSV(s) → {len(df)} rows")
    print(f"  CFG scales:    {sorted(df['cfg_scale'].unique(), key=float)}")
    print(f"  Train steps:   {sorted(df['train_step'].unique())}")
    print(f"  Keypose steps: {sorted(df['step'].unique())}  (0=start 1=mid 2=end)")
    print(f"  Datasets:      {sorted(df['dataset'].unique())}")
    return df


def plot_metric_dataset(df_ds, dataset, metric, cfg_scales, out_dir: Path):
    ncfg = len(cfg_scales)
    fig, axes = plt.subplots(1, ncfg, figsize=(5 * ncfg, 4), sharey=True)
    if ncfg == 1:
        axes = [axes]

    for ax, cfg in zip(axes, cfg_scales):
        df_cfg = df_ds[df_ds["cfg_scale"] == cfg]
        for kstep, label in STEP_LABELS.items():
            subset = df_cfg[df_cfg["step"] == kstep].sort_values("train_step")
            if subset.empty:
                continue
            ax.plot(
                subset["train_step"],
                subset[metric],
                marker="o",
                linewidth=2,
                markersize=5,
                color=STEP_COLORS[kstep],
                label=label,
            )
        ax.set_title(f"cfg={cfg}")
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric))
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)}  [{dataset}]", fontsize=11)
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

    cfg_scales = sorted(df["cfg_scale"].unique(), key=float)

    for dataset in sorted(df["dataset"].unique()):
        df_ds = df[df["dataset"] == dataset]
        out_dir = base_out / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\ndataset={dataset}  → {out_dir}/")
        for metric in METRIC_LABELS:
            if metric not in df_ds.columns:
                continue
            plot_metric_dataset(df_ds, dataset, metric, cfg_scales, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
