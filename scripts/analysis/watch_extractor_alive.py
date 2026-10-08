#!/usr/bin/env python3
"""Log whether each new checkpoint's history extractor still depends on its input (collapse watcher for exp 4a/4b)."""
import glob, json, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.policy.video_deltam import HistoryFeatureExtractor
from utils.config_migrations import migrate_config

RUNS = sys.argv[1:]
LOG = Path("logs/analysis/extractor_alive.log")
DONE = Path("logs/analysis/extractor_alive.done.json")
seen = set(json.loads(DONE.read_text())) if DONE.exists() else set()
torch.manual_seed(0)
X = torch.randn(16, 5, 4, 120) * 0.05


def measure(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = migrate_config(ck["config"])
    key = "ema_weight" if cfg.get("use_ema") and ck.get("ema_weight") is not None else "weight"
    pre = "module.prediction_head.video_deltam."
    w = {k[len(pre):]: v for k, v in ck[key].items() if k.startswith(pre)}
    ext = HistoryFeatureExtractor(120, 6, cfg.get("video_deltam_depth", 4), max_history=32,
                                  max_cameras=cfg.get("video_deltam_max_cameras", 8), dropout=0.0,
                                  use_time_embedding=cfg.get("video_deltam_time_embedding", True),
                                  use_register="camera_readout.0.attn.in_proj_weight" in w,
                                  prenorm=cfg.get("video_deltam_prenorm", False),
                                  input_skip=cfg.get("video_deltam_input_skip", False))
    ext.load_state_dict(w)
    ext.eval()
    out = {}
    with torch.no_grad():
        for label, skip in (("output", ext.input_skip), ("stack_only", False)):
            ext.input_skip = skip
            c = ext(X, torch.zeros(16, 1, 120))[0][:, -1]
            out[label] = dict(spread=round(c.std(0).mean().item(), 5),
                              cam01=round((c[:, 0] - c[:, 1]).norm(dim=-1).mean().item(), 4),
                              norm=round(c.norm(dim=-1).mean().item(), 2))
    return out


for run in RUNS:
    for f in sorted(glob.glob(f"train_logs/PerAct2/{run}/interm_step_*.pth"), key=lambda p: int(re.findall(r"(\d+)\.pth", p)[0])):
        if f in seen:
            continue
        try:
            m = measure(f)
        except Exception as e:
            m = {"error": str(e)[:200]}
        step = int(re.findall(r"(\d+)\.pth", f)[0])
        line = f"{time.strftime('%m-%d %H:%M')} {run} {step // 1000}k {json.dumps(m)}"
        with LOG.open("a") as fh:
            fh.write(line + "\n")
        seen.add(f)
DONE.write_text(json.dumps(sorted(seen)))
