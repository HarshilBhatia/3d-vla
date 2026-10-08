#!/usr/bin/env python3
"""C-token and first-layer delta_M variation across inputs, from a checkpoint's extractor and head predictor (CPU)."""
import re, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.policy.video_deltam import HistoryFeatureExtractor
from utils.config_migrations import migrate_config

torch.manual_seed(0)
X = torch.randn(64, 5, 4, 120) * 0.05


def measure(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = migrate_config(ck["config"])
    w = ck["ema_weight" if cfg.get("use_ema") and ck.get("ema_weight") is not None else "weight"]
    pre = "module.prediction_head.video_deltam."
    ext = HistoryFeatureExtractor(120, 6, cfg.get("video_deltam_depth", 4), max_history=32,
                                  max_cameras=cfg.get("video_deltam_max_cameras", 8), dropout=0.0,
                                  use_time_embedding=cfg.get("video_deltam_time_embedding", True),
                                  use_register=pre + "camera_readout.0.attn.in_proj_weight" in w,
                                  prenorm=cfg.get("video_deltam_prenorm", False),
                                  input_skip=cfg.get("video_deltam_input_skip", False))
    ext.load_state_dict({k[len(pre):]: v for k, v in w.items() if k.startswith(pre)})
    ext.eval()
    h = "module.prediction_head."
    with torch.no_grad():
        c = ext(X, torch.zeros(64, 1, 120))[0][:, -1]
        z = torch.nn.functional.linear(c, w[h + "camera_proj.weight"], w[h + "camera_proj.bias"])
        z = torch.relu(torch.nn.functional.linear(z, w[h + "camera_trunk.0.weight"], w[h + "camera_trunk.0.bias"]))
        live = (z > 0).any(0).float().sum(-1)[:2].mean().item()
        a = torch.nn.functional.linear(z, w[h + "camera_predictor.weight"], w[h + "camera_predictor.bias"]).reshape(64, 4, 6, 6)
        a = a - a.transpose(-1, -2)
        n = torch.linalg.norm(a, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        dm = torch.linalg.matrix_exp(a * (n.clamp(max=3.0) / n))[:, :2]
    rel = lambda t: (t.std(0).norm(dim=-1).mean() / t.norm(dim=-1).mean()).item()
    return dict(c_spread=round(c[:, :2].std(0).mean().item(), 4), c_rel=round(rel(c[:, :2]), 4),
                dm_size=round((dm - torch.eye(6)).flatten(-2).norm(dim=-1).mean().item(), 3),
                dm_spread=round(dm.flatten(-2).std(0).norm(dim=-1).mean().item(), 4),
                live_units=round(live, 1))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        step = re.findall(r"(\d+)\.pth", p)
        print(f"{p.split('/')[-2][16:]:52s} {step[0] if step else 'best':>7}  {measure(p)}")
