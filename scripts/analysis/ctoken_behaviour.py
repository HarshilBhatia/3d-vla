#!/usr/bin/env python3
"""TODO 10: C-token content, position shift, per-layer change under miscalibration, attention to/from C-tokens, and C-token ablations."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import modeling.utils.multihead_custom_attention as mca
from camera_attention import CONDITIONS, NG, VAL
from common.metrics import compute_metrics
from data.batch import base_collate_fn
from data.geometry import fetch_depth2cloud
from data.preprocessing.rlbench import RLBenchDataPreprocessor
from datasets import fetch_dataset_class
from eval_faithful_load import load as faithful_load, parse_spec
from modeling.encoder.text import fetch_tokenizers
from modeling.utils.position_encodings import RotaryPositionEncoding

ABLATIONS = [("clean", "zero_all"), ("clean", "mean_all"), ("clean", "mask_all"), ("clean", "mask_ext"),
             ("heavy60", "mean_all"), ("heavy60", "mask_all"), ("heavy60", "mask_ext")]


class Probe:
    """Hooks the head: input context, per-layer hidden states, attention rows of trajectory and C-tokens, and optional C-token ablation."""

    def __init__(self, model):
        self.head = model.prediction_head
        self.name, self.ctx, self.record, self.ablate, self.mean_ctok = None, None, True, None, None
        self.states, self.attn, self.inputs = {}, {}, []
        self.step = 0
        self.layer_names = []
        for prefix, mod in [("ca", "cross_attn"), ("sa", "self_attn"), ("pos", "position_self_attn"),
                            ("rot", "rotation_self_attn")]:
            m = getattr(self.head, mod)
            for i, layer in enumerate(m.attn_layers):
                n = f"{prefix}{i}"
                self.layer_names.append(n)
                layer.attention.register_forward_pre_hook(self._enter(n))
                layer.attention.register_forward_hook(self._exit)
                m.ffw_layers[i].register_forward_hook(self._state(n))
        self.head.self_attn.attn_layers[0].register_forward_pre_hook(self._sa_input)
        self.head.register_forward_pre_hook(self._head_ctx, with_kwargs=True)
        self.orig = mca.multi_head_attention_forward
        mca.multi_head_attention_forward = self._patched

    def _enter(self, name):
        def f(mod, inp):
            self.name = name
        return f

    def _exit(self, mod, inp, out):
        self.name = None

    def _state(self, name):
        def f(mod, inp, out):
            if self.record:
                self.states[(name, self.ctx["step"])] = out.detach().float()
        return f

    def _sa_input(self, mod, args):
        if self.record:
            self.states[("in", self.ctx["step"])] = args[0].detach().float()

    def _head_ctx(self, mod, args, kwargs):
        traj, fps_cam_ids, rgb3d = args[1], kwargs["fps_cam_ids"], kwargs["rgb3d_feats"]
        B, M = fps_cam_ids.shape
        ncam = kwargs["fps_scene_feats"].shape[1] - M
        T = traj.shape[1] * traj.shape[2]
        dev = rgb3d.device
        dense = torch.arange(ncam, device=dev).repeat_interleave(rgb3d.shape[1] // ncam).expand(B, -1)
        sa = torch.cat([torch.full((B, T), 10, device=dev), fps_cam_ids.long(), (4 + torch.arange(ncam, device=dev)).expand(B, -1),
                        torch.full((B, self.head.register_tokens.shape[0]), 8, device=dev), torch.full((B, 1), 9, device=dev)], 1)
        self.ctx = dict(T=T, M=M, ncam=ncam, dense=dense, sa=sa, step=self.step, fps_cam_ids=fps_cam_ids)
        if self.step == 0 and self.record:
            self.inputs.append((kwargs["fps_scene_feats"][:, M:].detach().float(), kwargs["fps_scene_pos"][:, M:].detach().float()))
        self.step += 1
        if self.ablate in ("zero_all", "mean_all"):
            f = kwargs["fps_scene_feats"].clone()
            f[:, M:] = 0 if self.ablate == "zero_all" else self.mean_ctok.to(f.dtype)
            kwargs = dict(kwargs, fps_scene_feats=f)
            return args, kwargs

    def _patched(self, query, key, value, embed_dim, num_heads, in_w, in_b, dropout_p, out_w, out_b,
                 training=True, attn_mask=None, force_math=False, rotary_pe=None, rotary_pe_module=None,
                 rope_lambda_reg=0.0, rotary_pe_mode=None):
        sa_like = self.name is not None and not self.name.startswith("ca")
        if sa_like and self.ablate in ("mask_all", "mask_ext"):
            c = self.ctx
            keep = torch.ones(c["sa"].shape, dtype=torch.bool, device=c["sa"].device)
            hi = c["ncam"] if self.ablate == "mask_all" else 2
            keep[:, c["T"] + c["M"]:c["T"] + c["M"] + hi] = False
            attn_mask = keep[:, None, None, :]
        out = self.orig(query, key, value, embed_dim, num_heads, in_w, in_b, dropout_p, out_w, out_b,
                        training=training, attn_mask=attn_mask, force_math=force_math, rotary_pe=rotary_pe,
                        rotary_pe_module=rotary_pe_module, rope_lambda_reg=rope_lambda_reg, rotary_pe_mode=rotary_pe_mode)
        if self.record and sa_like:
            with torch.autocast(device_type="cuda", enabled=False):
                self._record(query, key, num_heads, in_w, in_b, rotary_pe)
        return out

    def _record(self, query, key, H, in_w, in_b, rotary_pe):
        c = self.ctx
        E = query.shape[-1]
        w_q, w_k = in_w[:E].float(), in_w[E:2 * E].float()
        b_q, b_k = in_b[:E].float(), in_b[E:2 * E].float()
        q = F.linear(query.float(), w_q, b_q).transpose(0, 1)
        k = F.linear(key.float(), w_k, b_k).transpose(0, 1)
        if rotary_pe is not None:
            qp, kp = rotary_pe
            q = RotaryPositionEncoding.embed_rotary(q, qp[..., 0].float(), qp[..., 1].float())
            k = RotaryPositionEncoding.embed_rotary(k, kp[..., 0].float(), kp[..., 1].float())
        B, S, _ = q.shape
        d = E // H
        rows = list(range(c["T"])) + list(range(c["T"] + c["M"], c["T"] + c["M"] + c["ncam"]))
        q = q[:, rows].reshape(B, len(rows), H, d).transpose(1, 2)
        k = k.reshape(B, S, H, d).transpose(1, 2)
        p = torch.softmax(q @ k.transpose(-1, -2) * d ** -0.5, dim=-1)
        gid = c["sa"]
        mass = torch.zeros(B, H, len(rows), NG, device=p.device).scatter_add_(
            -1, gid[:, None, None].expand(-1, H, len(rows), -1), p)
        self.attn[(self.name, c["step"])] = mass.mean(1)


def rel(h, c):
    return (h - c).norm(dim=-1) / c.norm(dim=-1).clamp_min(1e-8)


def compare(states_c, states_h, ctx, names, n_steps):
    """Per-sample relative change clean -> other per layer: C-tokens per camera, FPS ext/wrist, traj; averaged over steps."""
    T, M, ncam, cam = ctx["T"], ctx["M"], ctx["ncam"], ctx["fps_cam_ids"]
    out = {}
    for n in names:
        acc = []
        for s in range(n_steps):
            if (n, s) not in states_c:
                continue
            c, h = states_c[(n, s)], states_h[(n, s)]
            if n.startswith("ca"):
                acc.append(torch.cat([torch.full((c.shape[0], ncam + 2), float("nan"), device=c.device), rel(h, c).mean(1, keepdim=True)], 1))
                continue
            r = rel(h, c)
            ctok = r[:, T + M:T + M + ncam]
            fps = r[:, T:T + M]
            ext = (cam < 2).float()
            fe = (fps * ext).sum(1) / ext.sum(1).clamp_min(1)
            fw = (fps * (1 - ext)).sum(1) / (1 - ext).sum(1).clamp_min(1)
            acc.append(torch.cat([ctok, fe[:, None], fw[:, None], r[:, :T].mean(1, keepdim=True)], 1))
        out[n] = torch.stack(acc, 1).cpu()
    return out


@torch.inference_mode()
def run_model(model, args, loader, batches, tokenizer, conds):
    probe = Probe(model)
    pres = {c: RLBenchDataPreprocessor(
        args.keypose_only, args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **CONDITIONS[c]) for c in conds}
    names = ["in"] + [n for n in probe.layer_names]
    res = {"tasks": [], "groups": [], "ctok": {c: [] for c in conds}, "cpos": {c: [] for c in conds},
           "rel": {c: {n: [] for n in names} for c in conds if c != "clean"},
           "attn": {c: {} for c in conds}, "pos_l2": {}, "ctok_layers": {c: {n: [] for n in names if not n.startswith("ca")} for c in conds}}
    # Pre-pass: clean C-tokens at head input over the whole set, for the mean ablation; same seeds, so the main pass sees identical batches.
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    probe.record, probe.ablate, sums, cnt = True, None, 0.0, 0
    for i, sample in enumerate(loader):
        if i >= batches:
            break
        torch.manual_seed(1000 + i)
        action = pres["clean"].process_actions(sample["action"])
        proprio = pres["clean"].process_proprio(sample["proprioception"])
        rgbs, pcds = pres["clean"].process_obs(sample["rgb"], sample.get("rgb2d"), sample["depth"], sample["extrinsics"],
                                               sample["intrinsics"], augment=False, task=sample["task"],
                                               camera_group=sample.get("camera_group"))
        instr = tokenizer(sample["instr"]).cuda(non_blocking=True)
        mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")
        probe.states, probe.attn, probe.inputs, probe.step = {}, {}, [], 0
        torch.manual_seed(5000 + i)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
        sums = sums + probe.inputs[0][0].sum(0)
        cnt += probe.inputs[0][0].shape[0]
    probe.mean_ctok = sums / cnt
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    for i, sample in enumerate(loader):
        if i >= batches:
            break
        prepared = {}
        for c in conds:
            torch.manual_seed(1000 + i)
            action = pres[c].process_actions(sample["action"])
            proprio = pres[c].process_proprio(sample["proprioception"])
            rgbs, pcds = pres[c].process_obs(sample["rgb"], sample.get("rgb2d"), sample["depth"], sample["extrinsics"],
                                             sample["intrinsics"], augment=False, task=sample["task"],
                                             camera_group=sample.get("camera_group"))
            prepared[c] = (action, proprio, rgbs, pcds)
        instr = tokenizer(sample["instr"]).cuda(non_blocking=True)
        states, ctxs = {}, {}
        runs = [(c, None) for c in conds] + [r for r in ABLATIONS if r[0] in conds]
        for c, ab in runs:
            action, proprio, rgbs, pcds = prepared[c]
            mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")
            probe.record, probe.ablate = ab is None, ab
            probe.states, probe.attn, probe.inputs, probe.step = {}, {}, [], 0
            torch.manual_seed(5000 + i)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(action, mask, rgbs, None, pcds, instr, proprio, run_inference=True)
            key = c if ab is None else f"{c}+{ab}"
            res["pos_l2"].setdefault(key, []).append(compute_metrics(pred, action)[0]["traj_pos_l2"].item() * action.shape[0])
            if ab is not None:
                continue
            states[c], ctxs[c] = probe.states, probe.ctx
            res["ctok"][c].append(probe.inputs[0][0].cpu())
            res["cpos"][c].append(probe.inputs[0][1].cpu())
            T, M, ncam = probe.ctx["T"], probe.ctx["M"], probe.ctx["ncam"]
            for n in res["ctok_layers"][c]:
                res["ctok_layers"][c][n].append(torch.stack([probe.states[(n, s)][:, T + M:T + M + ncam] for s in range(model.n_steps)], 1).cpu())
            for (n, s), m in probe.attn.items():
                res["attn"][c].setdefault(n, {}).setdefault(s, []).append(m.cpu())
        assert all(torch.equal(ctxs[c]["fps_cam_ids"], ctxs["clean"]["fps_cam_ids"]) for c in conds), "FPS selection differs across conditions"
        for c in conds:
            if c == "clean":
                continue
            for n, v in compare(states["clean"], states[c], ctxs["clean"], names, model.n_steps).items():
                res["rel"][c][n].append(v)
        res["tasks"] += [t[0] if isinstance(t, (list, tuple)) else t for t in sample["task"]]
        res["groups"] += [int(g) for g in sample["camera_group"]]
    out = {"tasks": res["tasks"], "groups": res["groups"], "T": ctxs["clean"]["T"], "M": ctxs["clean"]["M"],
           "ncam": ctxs["clean"]["ncam"], "n_steps": model.n_steps}
    n = len(res["tasks"])
    out["pos_l2"] = {k: sum(v) / n for k, v in res["pos_l2"].items()}
    out["ctok"] = {c: torch.cat(v) for c, v in res["ctok"].items()}
    out["cpos"] = {c: torch.cat(v) for c, v in res["cpos"].items()}
    out["ctok_layers"] = {c: {nm: torch.cat(v) for nm, v in d.items()} for c, d in res["ctok_layers"].items()}
    out["rel"] = {c: {nm: torch.cat(v) for nm, v in d.items()} for c, d in res["rel"].items()}
    out["attn"] = {c: {nm: torch.stack([torch.cat(d[s]) for s in sorted(d)], 1) for nm, d in dd.items()} for c, dd in res["attn"].items()}
    mca.multi_head_attention_forward = probe.orig
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=checkpoint:raw|ema[:K|ckpt]")
    p.add_argument("--batches", type=int, default=1000)
    p.add_argument("--conditions", nargs="+", default=["clean", "train", "heavy60"])
    p.add_argument("--out-dir", default="results/analysis/ctoken_behaviour")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.model:
        label, path, weights, k = parse_spec(spec)
        model, args = faithful_load(path, weights, k)
        tokenizer = fetch_tokenizers(getattr(args, "text_backbone", None) or args.backbone)
        ds = fetch_dataset_class(args.dataset)(
            root=VAL, instructions=args.val_instructions, copies=1, relative_action=args.relative_action,
            mem_limit=0.1, chunk_size=args.chunk_size, visual_num_history=args.visual_num_history,
            proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
            visual_history_sampling=getattr(args, "visual_history_sampling", "consecutive"))
        loader = DataLoader(ds, batch_size=max(1, 16 // args.chunk_size), shuffle=False, num_workers=0,
                            collate_fn=base_collate_fn)
        res = run_model(model, args, loader, a.batches, tokenizer, a.conditions)
        res["meta"] = {"checkpoint": path, "weights": weights, "visual_num_history": args.visual_num_history,
                       "camera_adaln": getattr(model.prediction_head, "camera_adaln", None)}
        torch.save(res, out / f"{label}.pt")
        print(label, "n", len(res["tasks"]), "pos_l2", {k: round(100 * v, 3) for k, v in res["pos_l2"].items()}, flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
