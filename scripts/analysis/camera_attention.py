#!/usr/bin/env python3
"""Record where the head's trajectory queries put their attention mass (per camera, C-tokens, registers) under clean, train and heavy60 cameras."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import modeling.utils.multihead_custom_attention as mca
from common.metrics import compute_metrics
from data.batch import base_collate_fn
from data.geometry import fetch_depth2cloud
from data.preprocessing.rlbench import RLBenchDataPreprocessor
from datasets import fetch_dataset_class
from evaluation.checkpoints import overlay_checkpoint_config
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from modeling.policy.construction import build_model_kwargs
from modeling.utils.position_encodings import RotaryPositionEncoding
from utils.config_migrations import migrate_config
from utils.hydra_utils import get_config, get_config_path, normalize_public_vocabulary_args

# Copied from scripts/eval/diagnose_delta_m.py (not imported: that file is being edited concurrently).
VAL = "/grogu/datasets/hbhatia/3dfa_egress_extracted/3dfa_egress_20260820/datasets/orbital_peract2/zarr/val.zarr"
GROUP_FILE = "instructions/orbital_miscalibration_noise_bimanual_external_only.json"
CONDITIONS = {
    "clean": dict(miscal_mode="none"),
    "train": dict(miscal_mode="group", miscal_group_level="medium", miscal_group_file=GROUP_FILE,
                  perturbation_noise_rot_deg=3.0, perturbation_noise_trans_m=0.01, miscal_cameras=[0, 1]),
    "heavy60": dict(miscal_mode="none", perturbation_noise_fixed_rot_deg=60.0,
                    perturbation_noise_fixed_trans_m=0.6, miscal_cameras=[0, 1]),
}
# Key groups: scene tokens per camera 0-3, C-tokens per camera 4-7, registers 8, camera token 9, trajectory tokens 10.
GROUPS = ["scene0", "scene1", "scene2", "scene3", "ctok0", "ctok1", "ctok2", "ctok3", "registers", "camtoken", "traj"]
NG = len(GROUPS)


def load(path, weights):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = get_config(overrides=["data=orbital_peract2_nfs"], config_name="config", config_path=get_config_path())
    overlay_checkpoint_config(args, migrate_config(ckpt["config"]))
    normalize_public_vocabulary_args(args)
    cls = fetch_model_class(args.model_type)
    model = cls(**build_model_kwargs(args, cls))
    key = {"raw": "weight", "ema": "ema_weight"}[weights]
    missing, unexpected = model.load_state_dict({k[7:]: v for k, v in ckpt[key].items()}, strict=False)
    print("missing", len(missing), missing[:5], "unexpected", len(unexpected), unexpected[:5], flush=True)
    return model.cuda().eval(), args


class Recorder:
    """Patches the attention function so named head layers also report trajectory-query attention mass per key group."""

    def __init__(self, model):
        self.head = model.prediction_head
        self.name, self.ctx, self.verify = None, None, False
        self.calls, self.errs = [], {}
        self.step = 0
        for prefix, mod in [("ca", "cross_attn"), ("sa", "self_attn"), ("pos", "position_self_attn"),
                            ("rot", "rotation_self_attn")]:
            for i, layer in enumerate(getattr(self.head, mod).attn_layers):
                layer.attention.register_forward_pre_hook(self._enter(f"{prefix}{i}"))
                layer.attention.register_forward_hook(self._exit)
        self.head.register_forward_pre_hook(self._head_ctx, with_kwargs=True)
        self.orig = mca.multi_head_attention_forward
        mca.multi_head_attention_forward = self._patched

    def _enter(self, name):
        def f(mod, inp):
            self.name = name
        return f

    def _exit(self, mod, inp, out):
        self.name = None

    def _head_ctx(self, mod, args, kwargs):
        traj = args[1]
        fps_cam_ids, rgb3d = kwargs["fps_cam_ids"], kwargs["rgb3d_feats"]
        B, M = fps_cam_ids.shape
        ncam = kwargs["fps_scene_feats"].shape[1] - M
        T = traj.shape[1] * traj.shape[2]
        Np = rgb3d.shape[1]
        dense = torch.arange(ncam, device=rgb3d.device).repeat_interleave(Np // ncam).expand(B, -1)
        sa = torch.cat([torch.full((B, T), 10, device=rgb3d.device), fps_cam_ids.long(),
                        (4 + torch.arange(ncam, device=rgb3d.device)).expand(B, -1),
                        torch.full((B, self.head.register_tokens.shape[0]), 8, device=rgb3d.device),
                        torch.full((B, 1), 9, device=rgb3d.device)], 1)
        self.ctx = dict(T=T, dense=dense, sa=sa, step=self.step)
        self.step += 1

    def _patched(self, query, key, value, embed_dim, num_heads, in_w, in_b, dropout_p, out_w, out_b,
                 training=True, attn_mask=None, force_math=False, rotary_pe=None, rotary_pe_module=None,
                 rope_lambda_reg=0.0, rotary_pe_mode=None):
        out = self.orig(query, key, value, embed_dim, num_heads, in_w, in_b, dropout_p, out_w, out_b,
                        training=training, attn_mask=attn_mask, force_math=force_math, rotary_pe=rotary_pe,
                        rotary_pe_module=rotary_pe_module, rope_lambda_reg=rope_lambda_reg,
                        rotary_pe_mode=rotary_pe_mode)
        if self.name is not None:
            assert attn_mask is None and rotary_pe_module is None
            with torch.autocast(device_type="cuda", enabled=False):
                self._record(query, key, value, num_heads, in_w, in_b, out_w, out_b, rotary_pe, out[0])
        return out

    def _record(self, query, key, value, H, in_w, in_b, out_w, out_b, rotary_pe, out):
        name, ctx = self.name, self.ctx
        is_ca = name.startswith("ca")
        q, k, v = F._in_projection_packed(query.float(), key.float(), value.float(), in_w.float(), in_b.float())
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if rotary_pe is not None:
            qp, kp = rotary_pe
            q = RotaryPositionEncoding.embed_rotary(q, qp[..., 0].float(), qp[..., 1].float())
            k = RotaryPositionEncoding.embed_rotary(k, kp[..., 0].float(), kp[..., 1].float())
        B, S, E = q.shape
        d = E // H
        q, k, v = (t.reshape(B, t.shape[1], H, d).transpose(1, 2) for t in (q, k, v))
        rows = S if is_ca else ctx["T"]
        p = torch.softmax(q[:, :, :rows] @ k.transpose(-1, -2) * d ** -0.5, dim=-1)
        if self.verify:
            pf = torch.softmax(q @ k.transpose(-1, -2) * d ** -0.5, dim=-1)
            o = F.linear((pf @ v).transpose(1, 2).reshape(B, S, E), out_w.float(), out_b.float())
            ref = out.float().transpose(0, 1)
            ref32 = self.orig(query.float(), key.float(), value.float(), E, H, in_w.float(), in_b.float(), 0.0,
                              out_w.float(), out_b.float(), training=False,
                              rotary_pe=None if rotary_pe is None else tuple(r.float() for r in rotary_pe))[0]
            ref32 = ref32.transpose(0, 1)
            e16, e32 = ((o - ref).norm() / ref.norm()).item(), ((o - ref32).norm() / ref32.norm()).item()
            old = self.errs.get(name, (0.0, 0.0))
            self.errs[name] = (max(old[0], e16), max(old[1], e32))
        gid = ctx["dense"] if is_ca else ctx["sa"]
        assert gid.shape[1] == p.shape[-1], (name, gid.shape, p.shape)
        mass = torch.zeros(B, H, rows, NG, device=p.device).scatter_add_(
            -1, gid[:, None, None].expand(-1, H, rows, -1), p)
        ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
        counts = torch.zeros(B, NG, device=p.device).scatter_add_(1, gid, torch.ones_like(gid, dtype=torch.float))
        self.calls.append((name, ctx["step"], mass.cpu(), ent.cpu(), counts.cpu()))


@torch.inference_mode()
def run(model, args, rec, cond, loader, batches, tokenizer):
    pre = RLBenchDataPreprocessor(
        args.keypose_only, args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **CONDITIONS[cond])
    out = {"mass": {}, "ent": {}, "counts": {}}
    tasks, pos_l2 = [], []
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
        # Same reseed as TODO 1 (scripts/analysis/delta_m_per_layer.py) so FPS draws and flow noise match across conditions.
        torch.manual_seed(5000 + i)
        rec.calls.clear()
        rec.step = 0
        rec.verify = i == 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
        per = {}
        for name, step, mass, ent, counts in rec.calls:
            per.setdefault(name, {})[step] = (mass, ent, counts)
        for name, d in per.items():
            steps = sorted(d)
            out["mass"].setdefault(name, []).append(torch.stack([d[s][0] for s in steps], 1))
            out["ent"].setdefault(name, []).append(torch.stack([d[s][1] for s in steps], 1))
            out["counts"].setdefault(name, []).append(d[steps[0]][2])
        tasks += [t[0] if isinstance(t, (list, tuple)) else t for t in sample["task"]]
        pos_l2.append(compute_metrics(pred, action)[0]["traj_pos_l2"].item())
    res = {k: {n: torch.cat(v) for n, v in d.items()} for k, d in out.items()}
    res["tasks"], res["pos_l2"] = tasks, torch.tensor(pos_l2)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=checkpoint:raw|ema")
    p.add_argument("--batches", type=int, default=1000)
    p.add_argument("--conditions", nargs="+", default=["clean", "train", "heavy60"])
    p.add_argument("--out-dir", default="results/analysis/camera_attention")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.model:
        label, rest = spec.split("=", 1)
        path, weights = rest.rsplit(":", 1)
        model, args = load(path, weights)
        rec = Recorder(model)
        tokenizer = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)
        ds = fetch_dataset_class(args.dataset)(
            root=VAL, instructions=args.val_instructions, copies=1, relative_action=args.relative_action,
            mem_limit=0.1, chunk_size=args.chunk_size, visual_num_history=args.visual_num_history,
            proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
            visual_history_sampling=getattr(args, "visual_history_sampling", "consecutive"))
        loader = DataLoader(ds, batch_size=max(1, 16 // args.chunk_size), shuffle=False, num_workers=0,
                            collate_fn=base_collate_fn)
        meta = {"checkpoint": path, "weights": weights, "groups": GROUPS, "n_steps": model.n_steps,
                "camera_adaln": getattr(model.prediction_head, "camera_adaln", None),
                "layerwise_view_align": getattr(model.prediction_head, "layerwise_view_align", None)}
        saved = {"meta": meta}
        for cond in a.conditions:
            torch.manual_seed(0); random.seed(0); np.random.seed(0)
            rec.errs = {}
            saved[cond] = run(model, args, rec, cond, loader, a.batches, tokenizer)
            meta[f"verify_rel_err_{cond}"] = dict(rec.errs)
            print(label, cond, "verify rel err", json.dumps(rec.errs), flush=True)
            ca = saved[cond]["mass"]["ca0"]
            print(label, cond, "n", ca.shape[0], "pos_l2", f"{saved[cond]['pos_l2'].mean():.4f}",
                  "ca0 ext share", f"{ca[..., :2].sum(-1).mean():.4f}", flush=True)
        torch.save(saved, out / f"{label}.pt")
        mca.multi_head_attention_forward = rec.orig
        del model, rec
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
