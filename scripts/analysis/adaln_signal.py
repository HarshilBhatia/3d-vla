#!/usr/bin/env python3
"""Record the camera-adaLN signal terms and every AdaLN's scale/shift with and without the camera term, per (sample, step, layer)."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.metrics import compute_metrics
from data.batch import base_collate_fn
from data.geometry import fetch_depth2cloud
from data.preprocessing.rlbench import RLBenchDataPreprocessor
from datasets import fetch_dataset_class
from diagnose_delta_m import CONDITIONS, VAL, load
from modeling.encoder.text import fetch_tokenizers
from modeling.utils.layers import AdaLN


def hook(model):
    """Per forward: signal terms in call order, and per-AdaLN norms of scale/shift (full, time-only, difference)."""
    head = model.prediction_head
    st = {"sig": [], "ada": {}, "mismatch": 0}
    orig = head._camera_adaln_signal

    def sig(time_embs, camera_summaries, fps_cam_ids, traj_seq_len):
        out = orig(time_embs, camera_summaries, fps_cam_ids, traj_seq_len)
        # fp32 recomputation: full - time in bf16 leaves rounding noise larger than the camera term's variation.
        with torch.autocast(device_type="cuda", enabled=False):
            mc = camera_summaries.mean(1).detach().float()
            cam = head.camera_adaln_proj(mc)
            st["sig"].append({"time": time_embs.detach().float(), "cam": cam, "full": time_embs.detach().float() + cam,
                              "full_used": out.detach().float(), "meanC": mc})
        return out
    head._camera_adaln_signal = sig
    names = []
    for name, mod in head.named_modules():
        if isinstance(mod, AdaLN):
            names.append(name)

            def pre(m, inp, name=name):
                t = inp[1]
                cur = st["sig"][-1]
                if t.shape != cur["full_used"].shape or not torch.equal(t.float(), cur["full_used"]):
                    st["mismatch"] += 1
                with torch.autocast(device_type="cuda", enabled=False):
                    sf, hf = m.modulation(cur["full"]).chunk(2, -1)
                    stt, ht = m.modulation(cur["time"]).chunk(2, -1)
                st["ada"][name] = torch.stack([(sf - stt).norm(dim=-1), stt.norm(dim=-1), sf.norm(dim=-1),
                                               (hf - ht).norm(dim=-1), ht.norm(dim=-1), hf.norm(dim=-1),
                                               torch.full_like(sf[:, 0], len(st["sig"]) - 1)], -1).cpu()
            mod.register_forward_pre_hook(pre)
    return st, names


@torch.inference_mode()
def run(model, args, cond, loader, batches, tokenizer, st, names):
    pre = RLBenchDataPreprocessor(
        args.keypose_only, args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **CONDITIONS[cond])
    cam, tim, meanc, ada, tasks, pos_l2 = [], [], [], [], [], []
    head = model.prediction_head
    orig_fwd = head.forward
    steps = []

    def fwd(*a, **kw):
        st["sig"].clear(); st["ada"].clear()
        out = orig_fwd(*a, **kw)
        steps.append((torch.stack([s["cam"] for s in st["sig"]], 1).cpu(),
                      torch.stack([s["time"] for s in st["sig"]], 1).cpu(),
                      torch.stack([s["meanC"] for s in st["sig"]], 1).cpu(),
                      torch.stack([st["ada"][n] for n in names], 1)))
        return out
    head.forward = fwd
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
        # Reseed after preprocessing so the flow noise matches across conditions.
        torch.manual_seed(5000 + i)
        steps.clear()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
        for lst, k in ((cam, 0), (tim, 1), (meanc, 2), (ada, 3)):
            lst.append(torch.stack([s[k] for s in steps], 1))  # (B, steps, ...)
        tasks += [t[0] if isinstance(t, (list, tuple)) else t for t in sample["task"]]
        pos_l2.append(compute_metrics(pred, action)[0]["traj_pos_l2"].item())
    head.forward = orig_fwd
    return {"cam": torch.cat(cam), "time": torch.cat(tim), "meanC": torch.cat(meanc), "ada": torch.cat(ada),
            "tasks": tasks, "pos_l2": torch.tensor(pos_l2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="label=checkpoint:raw|ema")
    p.add_argument("--batches", type=int, default=1000)
    p.add_argument("--conditions", nargs="+", default=["clean", "train", "heavy60"])
    p.add_argument("--out-dir", default="results/analysis/adaln_signal")
    a = p.parse_args()
    label, rest = a.model.split("=", 1)
    path, weights = rest.rsplit(":", 1)
    model, args = load(path, weights)
    st, names = hook(model)
    tokenizer = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)
    ds = fetch_dataset_class(args.dataset)(
        root=VAL, instructions=args.val_instructions, copies=1, relative_action=args.relative_action,
        mem_limit=0.1, chunk_size=args.chunk_size, visual_num_history=args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        visual_history_sampling=getattr(args, "visual_history_sampling", "consecutive"))
    loader = DataLoader(ds, batch_size=max(1, 16 // args.chunk_size), shuffle=False, num_workers=0,
                        collate_fn=base_collate_fn)
    rec = {"checkpoint": path, "weights": weights, "ada_names": names,
           "ada_fields": ["d_scale", "scale_time", "scale_full", "d_shift", "shift_time", "shift_full", "signal_idx"]}
    for cond in a.conditions:
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
        rec[cond] = run(model, args, cond, loader, a.batches, tokenizer, st, names)
        print(label, cond, tuple(rec[cond]["cam"].shape), tuple(rec[cond]["ada"].shape),
              f"pos_l2={rec[cond]['pos_l2'].mean():.4f}", "mismatch", st["mismatch"], flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(rec, out / f"{label}.pt")


if __name__ == "__main__":
    main()
