#!/usr/bin/env python3
"""Verify eval_clean_proprio on real data: bit-identical when calibrated, and under heavy60 only proprio_feats changes (to exactly its clean value)."""
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

from activation_patching import FIELDS
from data.batch import base_collate_fn
from data.geometry import fetch_depth2cloud
from data.preprocessing.rlbench import RLBenchDataPreprocessor
from datasets import fetch_dataset_class
from diagnose_delta_m import CONDITIONS, VAL
from eval_faithful_load import parse_spec
from evaluation.checkpoints import load_model_for_evaluation
from modeling.encoder.text import fetch_tokenizers
from utils.config_migrations import migrate_config
from utils.hydra_utils import get_config, get_config_path


def load(path, weights):
    args = get_config(overrides=["data=orbital_peract2_nfs"], config_name="config", config_path=get_config_path())
    args.checkpoint, args.eval_weights, args.eval_clean_proprio = path, weights, True
    model = load_model_for_evaluation(args)
    args.visual_num_history = migrate_config(torch.load(path, map_location="cpu", weights_only=False)["config"])["visual_num_history"]
    return model.eval(), args


@torch.inference_mode()
def check(model, args, batches):
    tok = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)
    ds = fetch_dataset_class(args.dataset)(
        root=VAL, instructions=args.val_instructions, copies=1, relative_action=args.relative_action, mem_limit=0.1,
        chunk_size=args.chunk_size, visual_num_history=args.visual_num_history,
        proprio_num_history=args.proprio_num_history, visual_history_sampling=args.visual_history_sampling)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=base_collate_fn)
    pre = {c: RLBenchDataPreprocessor(args.keypose_only, args.visual_num_history, proprio_num_history=args.proprio_num_history,
                                      custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset),
                                      **CONDITIONS[c]) for c in ("clean", "heavy60")}
    worst = {"calibrated_pred_on_vs_off": 0.0, "heavy_proprio_vs_clean_proprio": 0.0, "heavy_proprio_changed_by": 0.0}
    other = {f: 0.0 for f in FIELDS}
    for i, s in enumerate(loader):
        if i >= batches:
            break
        action = pre["clean"].process_actions(s["action"])
        proprio = pre["clean"].process_proprio(s["proprioception"])
        instr = tok(s["instr"]).cuda()
        mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")
        obs = {}
        for c in ("clean", "heavy60"):
            torch.manual_seed(1000 + i)
            obs[c] = pre[c].process_obs(s["rgb"], s.get("rgb2d"), s["depth"], s["extrinsics"], s["intrinsics"],
                                        augment=False, task=s["task"], camera_group=s.get("camera_group"))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rgb, pcd = obs["clean"]
            preds = []
            for on in (False, True):
                model.eval_clean_proprio = on
                torch.manual_seed(5000 + i)
                preds.append(model(action, mask, rgb, None, pcd, instr, proprio, run_inference=True,
                                   **({"proprio_pcd": pcd} if on else {})))
            worst["calibrated_pred_on_vs_off"] = max(worst["calibrated_pred_on_vs_off"], (preds[0] - preds[1]).abs().max().item())
            torch.manual_seed(3000 + i)
            fc = model.encode_inputs(rgb, None, pcd, instr, proprio)
            torch.manual_seed(3000 + i)
            fh = model.encode_inputs(obs["heavy60"][0], None, obs["heavy60"][1], instr, proprio)
            torch.manual_seed(3000 + i)
            fo = model.encode_inputs(obs["heavy60"][0], None, obs["heavy60"][1], instr, proprio, proprio_pcd=pcd)
        k = FIELDS.index("proprio_feats")
        worst["heavy_proprio_vs_clean_proprio"] = max(worst["heavy_proprio_vs_clean_proprio"], (fo[k] - fc[k]).abs().max().item())
        worst["heavy_proprio_changed_by"] = max(worst["heavy_proprio_changed_by"], (fo[k] - fh[k]).abs().max().item())
        for j, f in enumerate(FIELDS):
            if j != k and torch.is_tensor(fo[j]):
                other[f] = max(other[f], (fo[j].float() - fh[j].float()).abs().max().item())
    return worst, other


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=ckpt:raw|ema[:ckpt]")
    p.add_argument("--batches", type=int, default=5)
    a = p.parse_args()
    for spec in a.model:
        label, path, weights, _ = parse_spec(spec)
        model, args = load(path, weights)
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
        worst, other = check(model, args, a.batches)
        print(label, "max |diff|:", worst, flush=True)
        print(label, "oracle vs plain heavy60, every other fixed input (must be 0):", other, flush=True)
        ok = worst["calibrated_pred_on_vs_off"] == 0 and worst["heavy_proprio_vs_clean_proprio"] == 0 \
            and worst["heavy_proprio_changed_by"] > 0 and all(v == 0 for v in other.values())
        print(label, "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
