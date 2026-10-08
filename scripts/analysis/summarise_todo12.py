#!/usr/bin/env python3
"""TODO 12: delta_M stats for the Direct full-patch model (one extractor delta_M) and a compact print of the full-patch suite."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from plot_delta_m_per_layer import EYE, fro, log_angle, task_share

ROOT = Path("results/analysis/todo12")


def direct_stats(path):
    rec = torch.load(path, weights_only=False)
    c, h = rec["clean"].double()[:, :, 0], rec["heavy60"].double()[:, :, 0]  # (N, steps, ncam, 6, 6)
    out = {"n": c.shape[0], "steps_identical_max": float((c - c[:, :1]).abs().max())}
    for k in range(c.shape[2]):
        d = fro(c[:, 0, k] - EYE)
        lf, ang = log_angle(c[:, 0, k] @ h[:, 0, k].transpose(-1, -2))
        share, rms = task_share(c[:, 0, k].reshape(len(c), -1), rec["tasks"])
        out[f"cam{k}"] = {"dist_from_I": [float(d.mean()), float(d.std())], "spread_rms": rms, "task_eta2": share,
                          "clean_vs_heavy_fro": float(fro(c[:, 0, k] - h[:, 0, k]).mean()),
                          "clean_vs_heavy_deg_mean_p99": [float(ang.mean()), float(ang.quantile(0.99))]}
    lf, ang = log_angle(c[:, 0, 0] @ c[:, 0, 1].transpose(-1, -2))
    out["cam0_vs_cam1"] = {"log_fro": float(lf.mean()), "max_angle_deg": float(ang.mean())}
    out["pos_l2"] = {k: float(rec[f"pos_l2_{k}"].mean()) for k in ("clean", "heavy60")}
    return out


def main():
    res = {"direct": direct_stats(ROOT / "delta_m_per_layer" / "fp_r3d_direct.pt")}
    (ROOT / "delta_m_per_layer" / "direct_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
