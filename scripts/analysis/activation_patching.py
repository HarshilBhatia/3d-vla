#!/usr/bin/env python3
"""Patch encoded inputs and mechanism outputs between clean and heavy60 runs of the same sample with identical flow noise, and record per-sample action error (TODO 7)."""
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
from utils.hydra_utils import get_config, get_config_path

FIELDS = ("query_trajectory", "rgb3d_feats", "pcd", "rgb2d_feats", "rgb2d_pos", "instr_feats", "instr_pos",
          "proprio_feats", "fps_scene_feats", "fps_scene_pos", "fps_cam_ids", "history_register",
          "history_view_align")
# Component -> list of (field, slice) where slice is "all", "scene" (first M FPS tokens) or "ctok" (C-tokens).
PARTS = {
    "pos_dense": [("pcd", "all")],
    "pos_fps": [("fps_scene_pos", "scene")],
    "pos_ctok": [("fps_scene_pos", "ctok")],
    "pos_all": [("pcd", "all"), ("fps_scene_pos", "all")],
    "feats_scene": [("rgb3d_feats", "all"), ("fps_scene_feats", "scene")],
    "ctok_feats": [("fps_scene_feats", "ctok")],
    "proprio_feats": [("proprio_feats", "all")],
    "lang_query": [("instr_feats", "all"), ("instr_pos", "all"), ("query_trajectory", "all")],
    "every_input": [(f, "all") for f in FIELDS],
    "ext_out": [("fps_scene_feats", "ctok"), ("history_register", "all"), ("history_view_align", "all")],
    "direct_dm": [("history_view_align", "all")],
}
# Extractor outputs taken from a run whose extractor alone saw the other condition's patch xyz (frame_pcd).
EXT_XYZ = PARTS["ext_out"]
FORWARD = ["none", "pos_dense", "pos_fps", "pos_ctok", "pos_all", "feats_scene", "ctok_feats", "proprio_feats",
           "lang_query", "pos_all+proprio", "every_input", "mech", "pos_all+mech", "proprio+mech", "pos_dense+pos_fps",
           "pos_ctok+proprio", "pos_ctok+mech"]
REVERSE = ["pos_dense", "pos_fps", "pos_ctok", "pos_all", "proprio_feats", "pos_all+proprio", "mech",
           "pos_dense+pos_fps", "pos_ctok+proprio"]
EXT_FORWARD = ["ext_xyz", "ext_out", "pos_all+ext_xyz", "ext_xyz+proprio", "pos_all+ext_xyz+proprio", "ctok_feats+proprio"]
EXT_REVERSE = ["ext_xyz", "pos_all+ext_xyz", "ext_xyz+proprio", "pos_all+ext_xyz+proprio"]
DIRECT = ["direct_dm", "pos_all+direct_dm", "proprio+direct_dm"]


def eval_load(path, weights, k="ckpt"):
    """Load through evaluation.checkpoints.load_model_for_evaluation itself (legacy camera rule and full-patch readout compat), with the saved visual history."""
    import torch as _t
    from evaluation.checkpoints import load_model_for_evaluation
    from utils.config_migrations import migrate_config
    args = get_config(overrides=["data=orbital_peract2_nfs"], config_name="config", config_path=get_config_path())
    args.checkpoint, args.eval_weights = path, weights
    model = load_model_for_evaluation(args)
    cfg = migrate_config(_t.load(path, map_location="cpu", weights_only=False)["config"])
    args.visual_num_history = cfg.get("visual_num_history", args.visual_num_history) if k == "ckpt" else int(k)
    vd = getattr(model.prediction_head, "video_deltam", None)
    print(f"[load] {path} {weights} visual_num_history={args.visual_num_history} view_align_cameras={getattr(args, 'view_align_cameras', None)} "
          f"full_image={getattr(args, 'video_deltam_full_image', False)} patch_rope3d={getattr(args, 'video_deltam_patch_rope3d', False)} "
          f"role={getattr(args, 'video_deltam_role', None)} legacy_readout={getattr(vd, '_legacy_checkpoint_compat', False)} "
          f"history_sampling={getattr(args, 'visual_history_sampling', None)}", flush=True)
    return model.eval(), args


class FramePcd:
    """Records the extractor's frame_pcd per encode, or overrides it with a stored one."""

    def __init__(self, model):
        self.vd = getattr(model.prediction_head, "video_deltam", None)
        self.active = self.vd is not None and getattr(self.vd, "patch_rope3d", False)
        self.last, self.override = None, None
        if self.active:
            orig = self.vd.forward

            def fwd(*a, frame_pcd=None, **kw):
                if self.override is not None:
                    frame_pcd = self.override
                self.last = frame_pcd
                return orig(*a, frame_pcd=frame_pcd, **kw)
            self.vd.forward = fwd


class Mech:
    """Records or replays the mechanism output (exp 3: every delta_M call; 1b: every adaLN camera term) in call order."""

    def __init__(self, model):
        head = model.prediction_head
        self.mode, self.buf, self.rec, self.i = None, None, [], 0
        if getattr(head, "camera_adaln", "none") not in ("none", None):
            self.kind = "adaln"
            head.camera_adaln_proj.register_forward_hook(self._hook)
        elif getattr(head, "predict_extrinsics", False) and hasattr(head, "camera_predictor"):
            self.kind = "delta_m"
            orig = head._predict_from_cam_feat

            def wrapped(cam_feat):
                rt, dm = orig(cam_feat)
                return rt, self._io(dm)
            head._predict_from_cam_feat = wrapped
        else:
            self.kind = None

    def _io(self, x):
        if self.mode == "record":
            self.rec.append(x.detach().clone())
        elif self.mode == "replay":
            x = self.buf[self.i].to(x.dtype)
            self.i += 1
        return x

    def _hook(self, module, inp, out):
        return self._io(out)

    def start(self, mode, buf=None):
        self.mode, self.buf, self.rec, self.i = mode, buf, [], 0

    def stop(self):
        if self.mode == "replay":
            assert self.i == len(self.buf), (self.i, len(self.buf))
        rec, self.mode = self.rec, None
        return rec


def mix(dst, src, parts, M):
    """Copy the listed parts of src (fixed inputs) into a copy of dst."""
    out = list(dst)
    for name, sl in parts:
        k = FIELDS.index(name)
        if src[k] is None:
            continue
        if sl == "all":
            out[k] = src[k]
        else:
            t = out[k].clone()
            if sl == "scene":
                t[:, :M] = src[k][:, :M]
            else:
                t[:, M:] = src[k][:, M:]
            out[k] = t
    return tuple(out)


def sample_traj(model, noise, fixed):
    traj = model.conditional_sample(noise.clone(), device=noise.device, fixed_inputs=fixed)
    _, tl, nh, _ = traj.shape
    traj = model.unconvert_rot(traj.flatten(1, 2)).unflatten(1, (tl, nh))
    traj = model.unnormalize_pos(traj)
    traj[..., -1] = traj[..., -1].sigmoid()
    return traj


def per_sample(pred, action):
    _, ps = compute_metrics(pred.float(), action.float())
    return {k: ps[f"traj_{k}"].cpu() for k in ("pos_l2", "pos_acc_001", "rot_l1")}


@torch.inference_mode()
def run(model, args, loader, batches, tokenizer):
    mk = lambda c: RLBenchDataPreprocessor(
        args.keypose_only, args.visual_num_history,
        proprio_num_history=getattr(args, "proprio_num_history", args.visual_num_history),
        custom_imsize=args.custom_img_size, depth2cloud=fetch_depth2cloud(args.dataset), **CONDITIONS[c])
    pre = {c: mk(c) for c in ("clean", "heavy60")}
    mech = Mech(model)
    fp = FramePcd(model)
    out, diffs, exact, tasks = {}, {}, {"none_vs_heavy": 0.0, "every_input_vs_clean": 0.0, "ext_xyz_vs_ext_out": 0.0}, []
    add = lambda key, m: [out.setdefault(key, {}).setdefault(k, []).append(v) for k, v in m.items()]
    for i, sample in enumerate(loader):
        if i >= batches:
            break
        action = pre["clean"].process_actions(sample["action"])
        proprio = pre["clean"].process_proprio(sample["proprioception"])
        instr = tokenizer(sample["instr"]).cuda(non_blocking=True)
        fixed, obs, fpcd = {}, {}, {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for c in ("clean", "heavy60"):
                torch.manual_seed(1000 + i)
                obs[c] = pre[c].process_obs(sample["rgb"], sample.get("rgb2d"), sample["depth"],
                                            sample["extrinsics"], sample["intrinsics"], augment=False,
                                            task=sample["task"], camera_group=sample.get("camera_group"))
                torch.manual_seed(3000 + i)
                fixed[c] = model.encode_inputs(obs[c][0], None, obs[c][1], instr, proprio)
                fpcd[c] = fp.last
            if fp.active:
                # ext[c]: condition c's inputs everywhere, except the extractor sees the other condition's patch xyz.
                ext = {}
                for c, o in (("clean", "heavy60"), ("heavy60", "clean")):
                    fp.override = fpcd[o]
                    torch.manual_seed(3000 + i)
                    ext[c] = model.encode_inputs(obs[c][0], None, obs[c][1], instr, proprio)
                    fp.override = None
            M = fixed["clean"][FIELDS.index("fps_cam_ids")].shape[1]
            for k, name in enumerate(FIELDS):
                a, b = fixed["clean"][k], fixed["heavy60"][k]
                if torch.is_tensor(a):
                    d = (a.float() - b.float()).abs()
                    rec = diffs.setdefault(name, {"max_abs": 0.0, "rel": []})
                    rec["max_abs"] = max(rec["max_abs"], d.max().item())
                    rec["rel"].append((d.norm() / a.float().norm().clamp(min=1e-12)).item())
            g = torch.Generator(device="cuda").manual_seed(5000 + i)
            out_dim = 6 if model._rotation_format == "euler" else 9
            noise = torch.randn(tuple(action.shape[:-1]) + (out_dim,), device="cuda", generator=g)
            mech.start("record")
            pred_c = sample_traj(model, noise, fixed["clean"])
            rec_c = mech.stop()
            mech.start("record")
            pred_h = sample_traj(model, noise, fixed["heavy60"])
            rec_h = mech.stop()
            add("clean", per_sample(pred_c, action))
            add("heavy", per_sample(pred_h, action))
            has_direct = fixed["clean"][FIELDS.index("history_view_align")] is not None
            fwd_names = FORWARD + (EXT_FORWARD if fp.active else []) + (DIRECT if has_direct else [])
            rev_names = REVERSE + (EXT_REVERSE if fp.active else []) + (DIRECT[:1] if has_direct else [])
            preds = {}
            for direction, names, dst, src, rec_src in (("fwd", fwd_names, "heavy60", "clean", rec_c),
                                                        ("rev", rev_names, "clean", "heavy60", rec_h)):
                for name in names:
                    pieces = [{"proprio": "proprio_feats"}.get(x, x) for x in name.split("+")]
                    parts = [q for x in pieces if x in PARTS for q in PARTS[x]]
                    use_mech = "mech" in pieces
                    if use_mech and mech.kind is None:
                        continue
                    f = mix(fixed[dst], fixed[src], parts, M)
                    if "ext_xyz" in pieces:
                        f = mix(f, ext[dst], EXT_XYZ, M)
                    if use_mech:
                        mech.start("replay", rec_src)
                    pred = sample_traj(model, noise, f)
                    if use_mech:
                        mech.stop()
                    add(f"{direction}:{name}", per_sample(pred, action))
                    preds[f"{direction}:{name}"] = pred
                    if direction == "fwd" and name == "none":
                        exact["none_vs_heavy"] = max(exact["none_vs_heavy"], (pred - pred_h).abs().max().item())
                    if direction == "fwd" and name == "every_input":
                        exact["every_input_vs_clean"] = max(exact["every_input_vs_clean"],
                                                            (pred - pred_c).abs().max().item())
            if fp.active:
                exact["ext_xyz_vs_ext_out"] = max(exact["ext_xyz_vs_ext_out"],
                                                  (preds["fwd:ext_xyz"] - preds["fwd:ext_out"]).abs().max().item())
        tasks += [t[0] if isinstance(t, (list, tuple)) else t for t in sample["task"]]
    res = {k: {m: torch.cat(v) for m, v in d.items()} for k, d in out.items()}
    for v in diffs.values():
        v["rel"] = float(np.mean(v["rel"]))
    return res, diffs, exact, tasks, mech.kind


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="label=checkpoint:raw|ema")
    p.add_argument("--batches", type=int, default=1000)
    p.add_argument("--out-dir", default="results/analysis/activation_patching")
    p.add_argument("--faithful", action="store_true", help="load as online eval does; spec may end in :K or :ckpt for visual history")
    p.add_argument("--eval-loader", action="store_true", help="load through load_model_for_evaluation itself; spec label=ckpt:raw|ema[:K|ckpt]")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.model:
        if a.eval_loader:
            label, path, weights, k = parse_spec(spec)
            model, args = eval_load(path, weights, k or "ckpt")
        elif a.faithful:
            label, path, weights, k = parse_spec(spec)
            model, args = faithful_load(path, weights, k)
        else:
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
        torch.manual_seed(0); random.seed(0); np.random.seed(0)
        res, diffs, exact, tasks, kind = run(model, args, loader, a.batches, tokenizer)
        print(label, "mechanism:", kind, "| exactness:", exact, flush=True)
        print(label, "clean vs heavy encoded-input diffs:",
              {k: (round(v["max_abs"], 5), round(v["rel"], 5)) for k, v in diffs.items()}, flush=True)
        for k, v in res.items():
            print(label, k, {m: round(x.mean().item(), 4) for m, x in v.items()}, flush=True)
        torch.save({"checkpoint": path, "weights": weights, "mechanism": kind, "metrics": res,
                    "input_diffs": diffs, "exactness": exact, "tasks": tasks}, out / f"{label}.pt")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
