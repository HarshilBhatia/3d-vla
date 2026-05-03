import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "3dfa_run": "3DFA Run",
    "open_drawer_default_G1": "Default G1",
    "open_drawer_default_G1_miscal": "Default G1 (miscal)",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

dfs = {}
for run_dir, label in RUNS.items():
    path = os.path.join(DATA_DIR, run_dir, "sweep.csv")
    dfs[label] = pd.read_csv(path)

NOISE_CONFIGS = [
    ("R_only",  "angle_deg",  "Rotation noise (deg)",       "R Variation"),
    ("T_only",  "trans_m",    "Translation noise (m)",       "T Variation"),
    ("RT",      None,          "Noise level",                 "R & T Variation"),
]

METRICS = [
    ("traj_pos_l2", "Position L2"),
    ("traj_rot_l1", "Rotation L1"),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 11))
fig.suptitle("Miscalibration Sweep — G1", fontsize=14, fontweight="bold", y=1.01)

for row, (noise_type, x_col, x_label, row_title) in enumerate(NOISE_CONFIGS):
    for col, (metric, metric_label) in enumerate(METRICS):
        ax = axes[row, col]

        for (label, df), color in zip(dfs.items(), COLORS):
            sub = df[df["noise_type"] == noise_type].copy()

            if noise_type == "RT":
                # Build combined string labels; use index as x
                x_vals = range(len(sub))
                x_ticks = list(x_vals)
                x_ticklabels = [
                    f"{int(r.angle_deg)}°\n{r.trans_m}m"
                    for _, r in sub.iterrows()
                ]
            else:
                x_vals = sub[x_col].values
                x_ticks = None

            ax.plot(x_vals, sub[metric].values, marker="o", label=label,
                    color=color, linewidth=1.8, markersize=5)

        if noise_type == "RT":
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, fontsize=7.5)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(f"{row_title}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

        if row == 0 and col == 1:
            ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
out_path = os.path.join(DATA_DIR, "sweep_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
