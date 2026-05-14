"""
Backfill config + iter into intermediate checkpoints that were saved without them.

Usage:
    python scripts/patch_interm_ckpts.py train_logs/Orbital/siglip_multi_group_od
"""
import sys
import re
import glob
import torch
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
last_pth = ckpt_dir / "last.pth"

print(f"Loading config from {last_pth}")
last = torch.load(last_pth, map_location="cpu", weights_only=False)
if "config" not in last:
    raise ValueError("last.pth is also missing config — can't proceed")
config = last["config"]

interm_paths = sorted(ckpt_dir.glob("interm_step_*.pth"))
if not interm_paths:
    print("No interm_step_*.pth files found.")
    sys.exit(0)

for path in interm_paths:
    match = re.search(r"interm_step_(\d+)\.pth$", path.name)
    if not match:
        continue
    step = int(match.group(1))

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "config" in ckpt and "iter" in ckpt:
        print(f"  {path.name} — already has config+iter, skipping")
        continue

    ckpt["config"] = config
    ckpt.setdefault("iter", step)
    ckpt.setdefault("best_loss", None)

    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.rename(path)
    print(f"  {path.name} — patched (iter={step})")

print("Done.")
