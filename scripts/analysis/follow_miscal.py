#!/usr/bin/env python3
"""Record predicted keyposes, ground truth and the injected miscalibration N per camera under clean, single- and dual-camera fixed-magnitude miscalibration (TODO 6)."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.batch import base_collate_fn
from data.geometry import fetch_depth2cloud
from data.preprocessing.rlbench import RLBenchDataPreprocessor
from datasets import fetch_dataset_class
from diagnose_delta_m import VAL, load
from modeling.encoder.text import fetch_tokenizers

MAGS = (10, 20, 40, 60)
CAMS = {"both": [0, 1], "cam0": [0], "cam1": [1]}


def conditions():
    """name -> preprocessor kwargs; fixed magnitude m deg / m cm, random direction."""
    out = {"clean": dict(miscal_mode="none")}
    for m in MAGS:
        for c, ids in CAMS.items():
            out[f"{c}_{m}"] = dict(miscal_mode="none", perturbation_noise_fixed_rot_deg=float(m),
                                   perturbation_noise_fixed_trans_m=m / 100.0, miscal_cameras=ids)
        # Every camera, wrists included, gets camera 0's draw: a rigid move of the whole scene relative to proprio.
        out[f"rigid_{m}"] = dict(miscal_mode="none", perturbation_noise_fixed_rot_deg=float(m),
                                 perturbation_noise_fixed_trans_m=m / 100.0, rigid=True)
    return out


def make_pre(args, kw):
    kw = dict(kw)
    rigid = kw.pop("rigid", False)
    pre = RLBenchDataPreprocessor(
        args.keypose_only, args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **kw)
    if rigid:
        orig = pre._get_miscal_noise
        pre._get_miscal_noise = lambda *a, **k: orig(*a, **k)[:, :1].expand(-1, a[1], -1, -1).contiguous()
    return pre


def capture_noise(pre):
    """Wrap _get_miscal_noise so the last returned (B, ncam, 4, 4) transform is kept."""
    box = {}
    orig = pre._get_miscal_noise

    def wrapped(*a, **kw):
        box["T"] = orig(*a, **kw)
        return box["T"]
    pre._get_miscal_noise = wrapped
    return box


def check_direction(args, loader, conds):
    """On the first batch, check heavy pcd of camera j equals N_j applied to the clean pcd."""
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    sample = next(iter(loader))
    res = {}
    for name in ("clean", "both_60"):
        pre = make_pre(args, conds[name])
        box = capture_noise(pre)
        torch.manual_seed(1000)
        _, pcds = pre.process_obs(sample["rgb"], sample.get("rgb2d"), sample["depth"], sample["extrinsics"],
                                  sample["intrinsics"], augment=False, task=sample["task"],
                                  camera_group=sample.get("camera_group"))
        res[name] = ((pcds[:, -1] if pcds.ndim == 6 else pcds).double(), box.get("T"))
    clean, _ = res["clean"]
    heavy, T = res["both_60"]
    T = T.cuda().double()
    out = {}
    for j in range(clean.shape[1]):
        x = clean[:, j].flatten(2)
        nx = T[:, j, :3, :3] @ x + T[:, j, :3, 3:]
        ninvx = torch.linalg.inv(T[:, j])[:, :3, :3] @ x + torch.linalg.inv(T[:, j])[:, :3, 3:]
        y = heavy[:, j].flatten(2)
        out[j] = dict(err_Nx=(y - nx).norm(dim=1).median().item(), err_Ninvx=(y - ninvx).norm(dim=1).median().item(),
                      err_x=(y - x).norm(dim=1).median().item())
    print("[direction check] median |heavy - N x|, |heavy - N^-1 x|, |heavy - x| per camera (m):", out, flush=True)
    return out


@torch.inference_mode()
def run(model, args, kw, loader, batches, tokenizer):
    pre = make_pre(args, kw)
    box = capture_noise(pre)
    preds, gts, Ns, tasks, props = [], [], [], [], []
    for i, sample in enumerate(loader):
        if i >= batches:
            break
        torch.manual_seed(1000 + i)
        action = pre.process_actions(sample["action"])
        proprio = pre.process_proprio(sample["proprioception"])
        box["T"] = None
        rgbs, pcds = pre.process_obs(sample["rgb"], sample.get("rgb2d"), sample["depth"], sample["extrinsics"],
                                     sample["intrinsics"], augment=False, task=sample["task"],
                                     camera_group=sample.get("camera_group"))
        instr = tokenizer(sample["instr"]).cuda(non_blocking=True)
        mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")
        # Reseed after preprocessing so every condition shares the flow noise (same as TODO 1).
        torch.manual_seed(5000 + i)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
        B, ncam = action.shape[0], rgbs.shape[-4]
        T = box["T"] if box["T"] is not None else torch.eye(4).expand(B, ncam, 4, 4)
        preds.append(pred.float().cpu()); gts.append(action.float().cpu()); Ns.append(T.float().cpu())
        props.append(proprio[:, -1].float().cpu())
        tasks += [t[0] if isinstance(t, (list, tuple)) else t for t in sample["task"]]
    return dict(pred=torch.cat(preds), gt=torch.cat(gts), N=torch.cat(Ns), proprio=torch.cat(props), tasks=tasks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=checkpoint:raw|ema")
    p.add_argument("--batches", type=int, default=1000)
    p.add_argument("--out-dir", default="results/analysis/follow_miscal")
    p.add_argument("--conditions", nargs="+", default=None, help="subset of condition names; default all")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conds = conditions()
    run_conds = a.conditions or list(conds)
    for k, spec in enumerate(a.model):
        label, rest = spec.split("=", 1)
        path, weights = rest.rsplit(":", 1)
        model, args = load(path, weights)
        tokenizer = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)
        ds = fetch_dataset_class(args.dataset)(
            root=VAL, instructions=args.val_instructions, copies=1, relative_action=args.relative_action,
            mem_limit=0.1, chunk_size=args.chunk_size, visual_num_history=args.visual_num_history,
            proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
            visual_history_sampling=getattr(args, "visual_history_sampling", "consecutive"))
        loader = DataLoader(ds, batch_size=max(1, 16 // args.chunk_size), shuffle=False, num_workers=0,
                            collate_fn=base_collate_fn)
        rec = {"checkpoint": path, "weights": weights, "relative_action": args.relative_action,
               "rotation_format": args.rotation_format}
        if k == 0 and "both_60" in run_conds:
            rec["direction_check"] = check_direction(args, loader, conds)
        for name in run_conds:
            kw = conds[name]
            torch.manual_seed(0); random.seed(0); np.random.seed(0)
            rec[name] = run(model, args, kw, loader, a.batches, tokenizer)
            e = (rec[name]["pred"][..., :3] - rec[name]["gt"][..., :3]).norm(dim=-1).mean()
            print(label, name, tuple(rec[name]["pred"].shape), f"pos_l2={e:.4f}", flush=True)
        torch.save(rec, out / f"{label}{a.tag}.pt")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
