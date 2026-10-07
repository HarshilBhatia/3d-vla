#!/usr/bin/env python3
"""Extract per-camera C-token features (encoder, history extractor, head SA layers, delta_M, adaLN term) and camera/miscalibration labels for linear probes (TODO 5, wk 2026-10-06)."""
import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

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

ZARR = "/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820/datasets/orbital_peract2/zarr"
GROUP_FILE = "instructions/orbital_miscalibration_noise_bimanual_external_only.json"
SWEEP_MAX_ROT_DEG, SWEEP_MAX_TRANS_M = 60.0, 0.6
# Same three-way split as diagnose_delta_m.py, except 'sweep' replaces heavy60 with continuous magnitudes.
CONDITIONS = {
    "clean": dict(miscal_mode="none"),
    "train": dict(miscal_mode="group", miscal_group_level="medium", miscal_group_file=GROUP_FILE,
                  perturbation_noise_rot_deg=3.0, perturbation_noise_trans_m=0.01, miscal_cameras=[0, 1]),
    "sweep": dict(miscal_mode="none", perturbation_noise_fixed_rot_deg=1.0, perturbation_noise_fixed_trans_m=0.01,
                  miscal_cameras=[0, 1]),
}


def load(path, weights):
    """Build the model from the checkpoint's saved config; weights is raw | ema | none (random init, seed 0)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = get_config(overrides=["data=orbital_peract2_nfs"], config_name="config", config_path=get_config_path())
    cfg = ckpt["config"]
    legacy_cameras = "dynamic_rope_from_camtoken" in cfg and "delta_m_camera_ids" in cfg
    overlay_checkpoint_config(args, migrate_config(cfg))
    normalize_public_vocabulary_args(args)
    # visual_num_history is eval-owned, so the overlay skips it; same fixes as diagnose_delta_m.load (5ef5cf3).
    args.visual_num_history = cfg.get("visual_num_history", cfg.get("num_history", args.visual_num_history))
    if legacy_cameras and getattr(args, "view_align_mode", "none") != "none":
        args.view_align_cameras = None
    print(f"[load] {path}: visual_num_history={args.visual_num_history} view_align_cameras={args.view_align_cameras}", flush=True)
    cls = fetch_model_class(args.model_type)
    torch.manual_seed(0)
    model = cls(**build_model_kwargs(args, cls))
    if weights != "none":
        key = {"raw": "weight", "ema": "ema_weight"}[weights]
        missing, unexpected = model.load_state_dict({k[7:]: v for k, v in ckpt[key].items()}, strict=False)
        print("missing", len(missing), "non-backbone missing", [k for k in missing if "backbone" not in k][:10], "unexpected", len(unexpected), flush=True)
    return model.cuda().eval(), args


def sweep_perturbation(B, ncam, device, dtype):
    """Random-direction rotation with angle ~ U[0, 60 deg] and translation with length ~ U[0, 60 cm], per (sample, camera)."""
    axes = torch.randn(B, ncam, 3, device=device)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ang = torch.rand(B, ncam, device=device) * math.radians(SWEEP_MAX_ROT_DEG)
    K = torch.zeros(B, ncam, 3, 3, device=device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -axes[..., 2], axes[..., 1], -axes[..., 0]
    K = K - K.transpose(-1, -2)
    I = torch.eye(3, device=device).expand(B, ncam, 3, 3)
    R = I + ang.sin()[..., None, None] * K + (1 - ang.cos())[..., None, None] * (K @ K)
    tdir = torch.randn(B, ncam, 3, device=device)
    t = tdir / tdir.norm(dim=-1, keepdim=True).clamp(min=1e-8) * (torch.rand(B, ncam, 1, device=device) * SWEEP_MAX_TRANS_M)
    T = torch.eye(4, device=device).repeat(B, ncam, 1, 1)
    T[..., :3, :3], T[..., :3, 3] = R, t
    return T.to(dtype)


def make_preprocessor(args, cond, record):
    pre = RLBenchDataPreprocessor(
        args.keypose_only, args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **CONDITIONS[cond])
    if cond == "sweep":
        pre._sample_random_perturbation = sweep_perturbation
    orig = pre._get_miscal_noise

    def wrapped(B, ncam, device, dtype, camera_group=None, task=None):
        T = orig(B, ncam, device, dtype, camera_group, task)
        record.append(torch.eye(4).repeat(B, ncam, 1, 1) if T is None else T.detach().float().cpu())
        return T
    pre._get_miscal_noise = wrapped
    return pre


class Recorder:
    """Hooks the encoder, history extractor, head SA layers, delta_M predictor and adaLN projection."""

    def __init__(self, model):
        self.model, head = model, model.prediction_head
        self.head, self.last = head, model.n_steps - 1
        self.nreg = head.register_tokens.shape[0]
        self.nsa = head.self_attn.num_layers
        self.reset()
        model.encoder.register_forward_hook(self._enc)
        if getattr(head, "video_deltam", None) is not None:
            head.video_deltam.register_forward_hook(self._hist)
        for i, layer in enumerate(head.self_attn.ffw_layers):
            layer.register_forward_hook(self._sa(i))
        orig_fwd = head.forward

        def fwd(*a, **kw):
            self.step += 1
            return orig_fwd(*a, **kw)
        head.forward = fwd
        self.has_dm = bool(getattr(head, "predict_extrinsics", False)) and hasattr(head, "camera_predictor")
        if self.has_dm:
            orig_dm = head._predict_from_cam_feat

            def dm(cam_feat):
                rt, d = orig_dm(cam_feat)
                if d is not None and self._slot() is not None:
                    self.dm[self._slot()].append(d.detach().float().cpu())
                return rt, d
            head._predict_from_cam_feat = dm
        self.has_adaln = getattr(head, "camera_adaln", "none") != "none"
        if self.has_adaln:
            head.camera_adaln_proj.register_forward_hook(self._adaln)

    def reset(self):
        self.step, self.enc, self.hist = -1, None, None
        self.sa = [[None] * 4 for _ in range(2)]
        self.dm, self.ad = [[], []], [[], []]

    def _slot(self):
        return 0 if self.step == 0 else (1 if self.step == self.last else None)

    def _enc(self, mod, inp, out):
        self.enc = out[11].detach().float().cpu()

    def _hist(self, mod, inp, out):
        self.hist = out[0][:, -1].detach().float().cpu()

    def _sa(self, i):
        def f(mod, inp, out):
            s = self._slot()
            if s is not None:
                ncam = self.enc.shape[1]
                end = out.shape[1] - self.nreg - 1
                self.sa[s][i] = out[:, end - ncam:end].detach().float().cpu()
        return f

    def _adaln(self, mod, inp, out):
        s = self._slot()
        if s is not None:
            self.ad[s].append(out.detach().float().cpu())

    def collect(self):
        f = {"enc": self.enc, "hist": self.hist if self.hist is not None else self.enc,
             "sa": torch.stack([torch.stack(self.sa[s], 1) for s in range(2)], 1)}
        if self.has_dm and self.dm[0]:
            f["dm"] = torch.stack([torch.stack(self.dm[s], 1) for s in range(2)], 1)
        if self.has_adaln and self.ad[0]:
            f["adaln"] = torch.stack([torch.stack(self.ad[s], 1) for s in range(2)], 1)
        return f


class Indexed(Dataset):
    """Wraps the dataset so each sample also returns the zarr row it came from."""

    def __init__(self, ds, rows):
        self.ds, self.rows = ds, rows
        orig = ds._get_task

        def get_task(idx):
            ds._last_idx = idx
            return orig(idx)
        ds._get_task = get_task

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        s = self.ds[self.rows[i]]
        s["row"] = torch.tensor([self.ds._last_idx])
        return s


@torch.inference_mode()
def extract(model, rec, args, split, rows, tokenizer, batch_size, fp32=False):
    ds = fetch_dataset_class(args.dataset)(
        root=f"{ZARR}/{split}.zarr", instructions=args.val_instructions, copies=1,
        relative_action=args.relative_action, mem_limit=1, chunk_size=1, visual_num_history=args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        visual_history_sampling=getattr(args, "visual_history_sampling", "consecutive"))
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    loader = DataLoader(Indexed(ds, rows), batch_size=batch_size, shuffle=False, num_workers=6,
                        collate_fn=base_collate_fn, worker_init_fn=lambda w: random.seed(1234 + w))
    noises = {c: [] for c in CONDITIONS}
    pres = {c: make_preprocessor(args, c, noises[c]) for c in CONDITIONS}
    out = {c: {} for c in CONDITIONS}
    meta = {"row": [], "task": []}
    for i, sample in enumerate(loader):
        meta["row"].append(sample["row"])
        meta["task"].extend(sample["task"])
        for cond, pre in pres.items():
            torch.manual_seed(1000 + i)
            action = pre.process_actions(sample["action"])
            proprio = pre.process_proprio(sample["proprioception"])
            rgbs, pcds = pre.process_obs(sample["rgb"], sample.get("rgb2d"), sample["depth"], sample["extrinsics"],
                                         sample["intrinsics"], augment=False, task=sample["task"],
                                         camera_group=sample.get("camera_group"))
            instr = tokenizer(sample["instr"]).cuda(non_blocking=True)
            mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")
            torch.manual_seed(2000 + i)
            rec.reset()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
            for k, v in rec.collect().items():
                out[cond].setdefault(k, []).append(v if fp32 else v.half())
        if i % 20 == 0:
            print(split, i, len(loader), flush=True)
    feats = {c: {k: torch.cat(v) for k, v in d.items()} for c, d in out.items()}
    noise = {c: torch.cat(v) for c, v in noises.items()}
    row = torch.cat(meta["row"]).numpy()
    z = zarr.open(f"{ZARR}/{split}.zarr", "r")
    labels = {"row": row, "task": meta["task"], "task_id": z["task_id"][:][row], "demo_id": z["demo_id"][:][row],
              "camera_group": z["camera_group"][:][row], "variation": z["variation"][:][row],
              "extrinsics": z["extrinsics"][:][row].astype(np.float32)}
    return {"feats": feats, "noise": noise, "labels": labels}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="label=checkpoint:raw|ema|none")
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_val", type=int, default=0, help="0 = whole validation split")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_dir", default="results/analysis/probes")
    p.add_argument("--fp32", action="store_true", help="store features in float32 instead of float16")
    a = p.parse_args()
    label, rest = a.model.split("=", 1)
    path, weights = rest.rsplit(":", 1)
    model, args = load(path, weights)
    tokenizer = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)
    rng = np.random.default_rng(0)
    n_tr = len(zarr.open(f"{ZARR}/train.zarr", "r")["task_id"])
    n_va = len(zarr.open(f"{ZARR}/val.zarr", "r")["task_id"])
    tr_rows = np.sort(rng.choice(n_tr, size=min(a.n_train, n_tr), replace=False)).tolist()
    va_rows = list(range(n_va)) if a.n_val == 0 else np.sort(rng.choice(n_va, size=a.n_val, replace=False)).tolist()
    rec = Recorder(model)
    res = {"model": label, "checkpoint": path, "weights": weights, "n_steps": model.n_steps,
           "visual_num_history": args.visual_num_history, "view_align_cameras": args.view_align_cameras,
           "train": extract(model, rec, args, "train", tr_rows, tokenizer, a.batch_size, a.fp32),
           "val": extract(model, rec, args, "val", va_rows, tokenizer, a.batch_size, a.fp32)}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(res, f"{a.out_dir}/{label}.pt")
    print("saved", f"{a.out_dir}/{label}.pt", {k: tuple(v.shape) for k, v in res["val"]["feats"]["clean"].items()}, flush=True)


if __name__ == "__main__":
    main()
