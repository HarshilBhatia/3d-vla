#!/usr/bin/env python3
"""Per head call: spread of the C-tokens, live ReLU units in camera_trunk, and distance of delta_M from the bias-only exp(skew(b))."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.batch import base_collate_fn
from data.geometry import fetch_depth2cloud
from data.preprocessing.rlbench import RLBenchDataPreprocessor
from datasets import fetch_dataset_class
from diagnose_delta_m import CONDITIONS, VAL, load
from eval_faithful_load import load as faithful_load, parse_spec
from modeling.encoder.text import fetch_tokenizers

CALLS = ["init", "CA", "SA0", "SA1", "SA2", "SA3"]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=checkpoint:raw|ema")
    p.add_argument("--batches", type=int, default=5)
    p.add_argument("--out", default="results/analysis/delta_m_per_layer/trunk_activity.json")
    p.add_argument("--faithful", action="store_true", help="load as online eval does; spec may end in :K or :ckpt for visual history")
    a = p.parse_args()
    res = {}
    for spec in a.model:
        if a.faithful:
            label, path, weights, k = parse_spec(spec)
            model, args = faithful_load(path, weights, k)
        else:
            label, rest = spec.split("=", 1)
            path, weights = rest.rsplit(":", 1)
            model, args = load(path, weights)
        head = model.prediction_head
        log = []

        def pre_hook(mod, inp):
            log.append({"x": inp[0].detach().float()})

        def trunk_hook(mod, inp, out):
            log[-1]["t"] = out.detach().float()
        direct = not hasattr(head, "camera_proj")  # predict_delta_m role: the extractor's delta_m_head owns delta_M
        if direct:
            dmh = head.video_deltam.delta_m_head
            h1, h2, predictor, calls = dmh.register_forward_pre_hook(pre_hook), dmh[1].register_forward_hook(trunk_hook), dmh[2], ["direct"]
        else:
            h1 = head.camera_proj.register_forward_pre_hook(pre_hook)
            h2 = head.camera_trunk.register_forward_hook(trunk_hook)
            predictor, calls = head.camera_predictor, CALLS
        vd, hist_len = [], []

        def vd_hook(mod, inp, out):
            x = inp[0][:, -1, :2].detach().float()
            vd.append((x.mean(2) if x.ndim == 4 else x, out[0][:, -1, :2].detach().float()))  # full-image input is per patch: pool it
            hist_len.append(inp[0].shape[1])
        h3 = head.video_deltam.register_forward_hook(vd_hook) if getattr(head, "video_deltam", None) is not None else None
        b = predictor.bias.detach().float().reshape(6, 6)
        a0 = b - b.T
        a0 = a0 * (a0.norm().clamp(max=3.0) / a0.norm().clamp(min=1e-8))
        dm_bias = torch.linalg.matrix_exp(a0)
        tokenizer = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)
        ds = fetch_dataset_class(args.dataset)(
            root=VAL, instructions=args.val_instructions, copies=1, relative_action=args.relative_action,
            mem_limit=0.1, chunk_size=args.chunk_size, visual_num_history=args.visual_num_history,
            proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
            visual_history_sampling=getattr(args, "visual_history_sampling", "consecutive"))
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=base_collate_fn)
        pre = RLBenchDataPreprocessor(
            args.keypose_only, args.visual_num_history,
            proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
            custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **CONDITIONS["clean"])
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
        spread = lambda x: float(x.flatten(0, 1).std(0).mean())
        acc_vd = {"in_spread": [], "in_cam0_vs_cam1": [], "out_spread": [], "out_cam0_vs_cam1": []}
        acc = {c: {"x_spread": [], "x_norm": [], "live_frac": [], "t_norm": [], "dm_minus_bias_dm": []} for c in calls}
        for i, sample in enumerate(loader):
            if i >= a.batches:
                break
            torch.manual_seed(1000 + i)
            action = pre.process_actions(sample["action"])
            proprio = pre.process_proprio(sample["proprioception"])
            rgbs, pcds = pre.process_obs(sample["rgb"], sample.get("rgb2d"), sample["depth"], sample["extrinsics"],
                                         sample["intrinsics"], augment=False, task=sample["task"],
                                         camera_group=sample.get("camera_group"))
            instr = tokenizer(sample["instr"]).cuda(non_blocking=True)
            mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")
            log.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
            for x, y in vd:
                acc_vd["in_spread"].append(spread(x)); acc_vd["out_spread"].append(spread(y))
                acc_vd["in_cam0_vs_cam1"].append(float((x[:, 0] - x[:, 1]).norm(dim=-1).mean()))
                acc_vd["out_cam0_vs_cam1"].append(float((y[:, 0] - y[:, 1]).norm(dim=-1).mean()))
            vd.clear()
            for j, e in enumerate(log):
                c = calls[j % len(calls)]
                x, t = e["x"][:, :2], e["t"][:, :2]  # external cameras only
                acc[c]["x_spread"].append(float(x.flatten(0, 1).std(0).mean()))
                acc[c]["x_norm"].append(float(x.norm(dim=-1).mean()))
                acc[c]["live_frac"].append(float((t > 0).any(1).any(0).float().mean()))
                acc[c]["t_norm"].append(float(t.norm(dim=-1).mean()))
                at = predictor(e["t"][:, :2].to(predictor.weight.dtype)).float().reshape(-1, 2, 6, 6)
                aa = at - at.transpose(-1, -2)
                n = aa.flatten(-2).norm(dim=-1)[..., None, None]
                dm = torch.linalg.matrix_exp(aa * (n.clamp(max=3.0) / n.clamp(min=1e-8)))
                acc[c]["dm_minus_bias_dm"].append(float((dm - dm_bias.to(dm.device)).flatten(-2).norm(dim=-1).max()))
        res[label] = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in acc.items()}
        res[label]["video_deltam_current_frame"] = {k: float(np.mean(v)) for k, v in acc_vd.items()}
        res[label]["history_frames_seen"] = sorted(set(hist_len)) if h3 is not None else None
        res[label]["bias_dm_dist_from_I"] = float((dm_bias - torch.eye(6, device=dm_bias.device)).norm())
        print(label, json.dumps(res[label], indent=1), flush=True)
        h1.remove(); h2.remove()
        if h3 is not None:
            h3.remove()
        del model
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
