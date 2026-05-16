"""Plot miscal-sweep eval results: metric vs noise magnitude, per sweep axis.

Reads all *.csv under <results_dir> (output of eval_checkpoints_mse_miscal_sweep.py).
Each CSV row is (ckpt, sweep_axis, rot_deg, trans_cm, camera_group, n_samples, metrics...).

Produces one figure per camera_group + an aggregate (n_samples-weighted) figure.
Each figure is a (n_metrics x 3) grid:
  rows = metrics (pos L2, rot L1, pos acc @1cm, rot acc @0.025)
  cols = sweep_axis (diagonal, rot, trans)
  lines = checkpoints

X-axis:
  diagonal:  rot_deg (== trans_cm)
  rot:       rot_deg     (trans_cm = 0)
  trans:     trans_cm    (rot_deg  = 0)

Usage:
    python scripts/eval/plot_miscal_sweep.py results/miscal_sweep/G1G3/
    python scripts/eval/plot_miscal_sweep.py results/miscal_sweep/G1G3/ --out-dir plots/miscal_sweep_G1G3/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "traj_pos_l2":       "Position L2 (m)",
    "traj_rot_l1":       "Rotation L1",
    "traj_pos_acc_001":  "Position Acc @ 1cm",
    "traj_rot_acc_0025": "Rotation Acc @ 0.025",
}
METRICS = list(METRIC_LABELS.keys())

AXES = ["diagonal", "rot", "trans"]
AXIS_X = {
    "diagonal": ("rot_deg",  "rot=trans noise (deg / cm)"),
    "rot":      ("rot_deg",  "rotation noise (deg)"),
    "trans":    ("trans_cm", "translation noise (cm)"),
}


def aggregate_by_cell(df):
    """n_samples-weighted mean across camera_groups, per (ckpt, axis, rot_deg, trans_cm)."""
    keys = ["ckpt_name", "sweep_axis", "rot_deg", "trans_cm"]
    out_rows = []
    for key, g in df.groupby(keys):
        n_total = g["n_samples"].sum()
        row = dict(zip(keys, key))
        row["camera_group"] = "all"
        row["n_samples"] = int(n_total)
        for m in METRICS:
            row[m] = float((g[m] * g["n_samples"]).sum() / n_total)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def render_grid(df_view, ckpts, title, out_path):
    """Render the 4 x 3 metric x axis grid for one view of the data."""
    n_rows, n_cols = len(METRICS), len(AXES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows),
                             sharex="col")
    if n_rows == 1:
        axes = axes[None, :]

    cmap = plt.get_cmap("tab10")
    color = {c: cmap(i) for i, c in enumerate(ckpts)}

    for j, axis in enumerate(AXES):
        x_col, x_label = AXIS_X[axis]
        sub = df_view[df_view["sweep_axis"] == axis]
        for i, metric in enumerate(METRICS):
            ax = axes[i, j]
            for ckpt in ckpts:
                g = sub[sub["ckpt_name"] == ckpt].sort_values(x_col)
                if g.empty:
                    continue
                ax.plot(g[x_col], g[metric], marker="o", linewidth=1.5,
                        color=color[ckpt], label=ckpt)
            if i == 0:
                ax.set_title(axis, fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(ckpts),
                   bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.suptitle(title, y=1.05, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=Path,
                   help="Directory containing the per-task sweep CSVs")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <results_dir>)")
    args = p.parse_args()

    csv_files = sorted(args.results_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSVs found in {args.results_dir}")
    print(f"Reading {len(csv_files)} CSVs from {args.results_dir}")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    ckpts = sorted(df["ckpt_name"].unique())
    groups = sorted(df["camera_group"].unique()) if "camera_group" in df.columns else []
    print(f"Checkpoints: {ckpts}")
    print(f"Sweep axes:  {sorted(df['sweep_axis'].unique())}")
    print(f"Groups:      {groups}")
    print(f"Rows:        {len(df)}")

    out_dir = args.out_dir or args.results_dir

    if "camera_group" in df.columns:
        for g in groups:
            view = df[df["camera_group"] == g]
            render_grid(view, ckpts,
                        title=f"Miscal sweep — {g} samples only",
                        out_path=out_dir / f"plot_{g}.png")

        agg = aggregate_by_cell(df)
        groups_label = "+".join(groups)
        render_grid(agg, ckpts,
                    title=f"Miscal sweep — {groups_label} (n_samples-weighted mean)",
                    out_path=out_dir / "plot_all.png")
    else:
        # Backward-compat: pre-bucketing CSVs (no camera_group column).
        render_grid(df, ckpts,
                    title="Miscal sweep",
                    out_path=out_dir / "plot.png")


if __name__ == "__main__":
    main()
