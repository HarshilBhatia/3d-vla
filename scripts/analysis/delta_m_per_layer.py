#!/usr/bin/env python3
"""Record every delta_M the head predicts, tagged (sample, denoising step, call, camera), under clean and heavy60 cameras."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.metrics import compute_metrics
from data.batch import base_collate_fn
from data.geometry import fetch_depth2cloud
from data.preprocessing.rlbench import RLBenchDataPreprocessor
from datasets import fetch_dataset_class
from diagnose_delta_m import CONDITIONS, VAL, load
from eval_faithful_load import load as faithful_load, parse_spec
from modeling.encoder.text import fetch_tokenizers


def hook(model):
    """Append every _predict_from_cam_feat output (B, ncam, 6, 6) to a list, in call order."""
    head, seen = model.prediction_head, []
    orig = head._predict_from_cam_feat

    def wrapped(cam_feat):
        rt, dm = orig(cam_feat)
        seen.append(dm.detach().float().cpu())
        return rt, dm
    head._predict_from_cam_feat = wrapped
    if getattr(head, "video_deltam", None) is not None and getattr(head.video_deltam, "predict_delta_m", False):
        def direct(mod, args, kwargs):  # Direct role: the extractor's delta_M reaches the head as history_view_align
            seen.append(kwargs["history_view_align"].detach().float().cpu())
        head.register_forward_pre_hook(direct, with_kwargs=True)
    return seen


@torch.inference_mode()
def run(model, args, cond, loader, batches, tokenizer, seen):
    pre = RLBenchDataPreprocessor(
        args.keypose_only, args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **CONDITIONS[cond])
    dms, tasks, pos_l2 = [], [], []
    for i, sample in enumerate(loader):
        if i >= batches:
            break
        torch.manual_seed(1000 + i)
        action = pre.process_actions(sample["action"])
        proprio = pre.process_proprio(sample["proprioception"])
        rgbs, pcds = pre.process_obs(sample["rgb"], sample.get("rgb2d"), sample["depth"], sample["extrinsics"],
                                     sample["intrinsics"], augment=False, task=sample["task"],
                                     camera_group=sample.get("camera_group"))
        instr = tokenizer(sample["instr"]).cuda(non_blocking=True)
        mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")
        # Reseed after preprocessing: heavy60 draws its rotation axes from the torch RNG, which would otherwise shift the flow noise.
        torch.manual_seed(5000 + i)
        seen.clear()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
        n_steps = model.n_steps
        assert len(seen) % n_steps == 0, (len(seen), n_steps)
        calls = len(seen) // n_steps
        d = torch.stack(seen).reshape(n_steps, calls, *seen[0].shape)
        dms.append(d.permute(2, 0, 1, 3, 4, 5).contiguous())
        tasks += [t[0] if isinstance(t, (list, tuple)) else t for t in sample["task"]]
        pos_l2.append(compute_metrics(pred, action)[0]["traj_pos_l2"].item())
    return torch.cat(dms), tasks, torch.tensor(pos_l2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=checkpoint:raw|ema")
    p.add_argument("--batches", type=int, default=1000)
    p.add_argument("--conditions", nargs="+", default=["clean", "heavy60"])
    p.add_argument("--out-dir", default="results/analysis/delta_m_per_layer")
    p.add_argument("--faithful", action="store_true", help="load as online eval does; spec may end in :K or :ckpt for visual history")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.model:
        if a.faithful:
            label, path, weights, k = parse_spec(spec)
            model, args = faithful_load(path, weights, k)
        else:
            label, rest = spec.split("=", 1)
            path, weights = rest.rsplit(":", 1)
            model, args = load(path, weights)
        seen = hook(model)
        tokenizer = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)
        ds = fetch_dataset_class(args.dataset)(
            root=VAL, instructions=args.val_instructions, copies=1, relative_action=args.relative_action,
            mem_limit=0.1, chunk_size=args.chunk_size, visual_num_history=args.visual_num_history,
            proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
            visual_history_sampling=getattr(args, "visual_history_sampling", "consecutive"))
        loader = DataLoader(ds, batch_size=max(1, 16 // args.chunk_size), shuffle=False, num_workers=0,
                            collate_fn=base_collate_fn)
        rec = {"checkpoint": path, "weights": weights, "n_steps": model.n_steps,
               "n_sa_layers": model.prediction_head.self_attn.num_layers,
               "n_ca_layers": model.prediction_head.cross_attn.num_layers}
        for cond in a.conditions:
            torch.manual_seed(0); random.seed(0); np.random.seed(0)
            dm, tasks, err = run(model, args, cond, loader, a.batches, tokenizer, seen)
            rec[cond] = dm
            rec["tasks"] = tasks
            rec[f"pos_l2_{cond}"] = err
            print(label, cond, tuple(dm.shape), f"pos_l2={err.mean():.4f}", flush=True)
        torch.save(rec, out / f"{label}.pt")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
