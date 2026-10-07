#!/usr/bin/env python3
"""Measure each checkpoint's predicted delta_M and offline action error under clean, train-style and heavy external miscalibration."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.metrics import compute_metrics
from data.batch import base_collate_fn
from data.geometry import fetch_depth2cloud
from data.preprocessing.rlbench import RLBenchDataPreprocessor
from datasets import fetch_dataset_class
from evaluation.checkpoints import overlay_checkpoint_config
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from modeling.policy.construction import build_model_kwargs
from utils.config_migrations import migrate_config
from utils.hydra_utils import get_config, get_config_path, normalize_public_vocabulary_args

VAL = "/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820/datasets/orbital_peract2/zarr/val.zarr"
GROUP_FILE = "instructions/orbital_miscalibration_noise_bimanual_external_only.json"
# name -> preprocessor miscalibration kwargs; external cameras [0, 1] only, as in training and IDLT.
CONDITIONS = {
    "clean": dict(miscal_mode="none"),
    "train": dict(miscal_mode="group", miscal_group_level="medium", miscal_group_file=GROUP_FILE,
                  perturbation_noise_rot_deg=3.0, perturbation_noise_trans_m=0.01, miscal_cameras=[0, 1]),
    "heavy60": dict(miscal_mode="none", perturbation_noise_fixed_rot_deg=60.0,
                    perturbation_noise_fixed_trans_m=0.6, miscal_cameras=[0, 1]),
}


def load(path, weights):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = get_config(overrides=["data=orbital_peract2_nfs"], config_name="config", config_path=get_config_path())
    cfg = ckpt["config"]
    legacy_cameras = "dynamic_rope_from_camtoken" in cfg and "delta_m_camera_ids" in cfg
    overlay_checkpoint_config(args, migrate_config(cfg))
    normalize_public_vocabulary_args(args)
    # visual_num_history is eval-owned, so the overlay skips it; online plans pass the trained K explicitly.
    args.visual_num_history = cfg.get("visual_num_history", cfg.get("num_history", args.visual_num_history))
    # Same legacy rule as load_model_for_evaluation: old-schema checkpoints corrected every camera.
    if legacy_cameras and getattr(args, "view_align_mode", "none") != "none":
        args.view_align_cameras = None
    print(f"[load] {path}: visual_num_history={args.visual_num_history} view_align_cameras={args.view_align_cameras}", flush=True)
    cls = fetch_model_class(args.model_type)
    model = cls(**build_model_kwargs(args, cls))
    key = {"raw": "weight", "ema": "ema_weight"}[weights]
    model.load_state_dict({k[7:]: v for k, v in ckpt[key].items()}, strict=False)
    return model.cuda().eval(), args


def record_delta_m(model):
    """Wrap the head so every delta_M it predicts or receives is appended to a list."""
    head, seen = model.prediction_head, []
    if hasattr(head, "camera_predictor") and head.predict_extrinsics:
        orig = head._predict_from_cam_feat

        def wrapped(cam_feat):
            rt, dm = orig(cam_feat)
            if dm is not None:
                seen.append(dm.detach().float())
            return rt, dm
        head._predict_from_cam_feat = wrapped
    orig_fwd = head.forward

    def fwd(*a, **kw):
        if kw.get("history_view_align") is not None:
            seen.append(kw["history_view_align"].detach().float())
        return orig_fwd(*a, **kw)
    head.forward = fwd
    return seen


def stats(dms):
    """Mean ||delta_M - I||_F per camera, over every call and sample."""
    if not dms:
        return None
    d = torch.cat([x.reshape(-1, *x.shape[-3:]) for x in dms])
    eye = torch.eye(d.shape[-1], device=d.device)
    return (d - eye).flatten(-2).norm(dim=-1).mean(0).tolist()


@torch.inference_mode()
def run(model, args, cond, loader, batches, tokenizer):
    pre = RLBenchDataPreprocessor(
        args.keypose_only, args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **CONDITIONS[cond])
    seen = record_delta_m(model)
    metrics, firsts, n = {}, [], 0
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
        before = len(seen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
        if len(seen) > before:
            firsts.append(seen[before])
        agg, _ = compute_metrics(pred, action)
        b = action.shape[0]
        for k, v in agg.items():
            metrics[k] = metrics.get(k, 0.0) + v.item() * b
        n += b
    return {k: v / n for k, v in metrics.items()}, stats(seen), firsts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=checkpoint:raw|ema")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    results = {}
    for spec in a.model:
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
        results[label] = {}
        firsts = {}
        for cond in CONDITIONS:
            # Same samples, history draws and flow noise in every condition, so delta_M differences come from the cameras alone.
            torch.manual_seed(0); random.seed(0); np.random.seed(0)
            m, dm, firsts[cond] = run(model, args, cond, loader, a.batches, tokenizer)
            results[label][cond] = {"pos_l2": m["traj_pos_l2"], "pos_acc_001": m["traj_pos_acc_001"],
                                    "rot_l1": m["traj_rot_l1"], "delta_m_dist_per_cam": dm}
            print(label, cond, json.dumps(results[label][cond]), flush=True)
        if firsts["clean"] and firsts["heavy60"]:
            c, h = torch.cat(firsts["clean"]), torch.cat(firsts["heavy60"])
            results[label]["clean_vs_heavy_first_delta_m_change_per_cam"] = (c - h).flatten(-2).norm(dim=-1).mean(0).tolist()
            print(label, "delta_M change clean->heavy", results[label]["clean_vs_heavy_first_delta_m_change_per_cam"], flush=True)
        del model
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
