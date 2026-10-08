#!/usr/bin/env python3
"""Fit linear probes for true camera pose (A) and injected miscalibration (B) on features saved by probe_features.py (TODO 5, wk 2026-10-06)."""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CENTRE = np.array([0.248, 0.0, 0.954])  # least-squares intersection of the G1-G6 optical axes
ALPHAS = np.logspace(-2, 5, 8)
CS = np.logspace(-3, 2, 6)
EXT = [0, 1]


def episodes(lab):
    return lab["task_id"].astype(np.int64) * 100000 + lab["demo_id"].astype(np.int64)


def cv_splits(groups, n=5):
    return list(GroupKFold(n).split(np.zeros(len(groups)), groups=groups))


def ridge(Xtr, Ytr, Xte, groups, shuffle=False):
    """Standardised ridge; alpha picked by episode-grouped 5-fold CV (closed form via SVD)."""
    sc = StandardScaler().fit(Xtr)
    if shuffle:
        Ytr = Ytr[np.random.default_rng(0).permutation(len(Ytr))]
    ys = StandardScaler().fit(Ytr)
    X, Y = sc.transform(Xtr), ys.transform(Ytr)

    def fit(Xa, Ya, alphas):
        mx, my = Xa.mean(0), Ya.mean(0)
        U, s, Vt = np.linalg.svd(Xa - mx, full_matrices=False)
        UtY = U.T @ (Ya - my)
        return [(Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY), mx, my) for a in alphas]

    err = np.zeros(len(ALPHAS))
    for tr, va in cv_splits(groups):
        for j, (W, mx, my) in enumerate(fit(X[tr], Y[tr], ALPHAS)):
            err[j] += (((X[va] - mx) @ W + my - Y[va]) ** 2).sum()
    W, mx, my = fit(X, Y, [ALPHAS[err.argmin()]])[0]
    return ys.inverse_transform(((sc.transform(Xte) - mx) @ W + my).reshape(len(Xte), -1))


def logistic(Xtr, ytr, Xte, groups, shuffle=False):
    sc = StandardScaler().fit(Xtr)
    if shuffle:
        ytr = ytr[np.random.default_rng(0).permutation(len(ytr))]
    m = LogisticRegressionCV(Cs=CS, cv=cv_splits(groups), max_iter=3000).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def r2(y, p):
    keep = y.reshape(len(y), -1).std(0) > 1e-6
    return float(r2_score(y.reshape(len(y), -1)[:, keep], p.reshape(len(p), -1)[:, keep]))


def angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-8)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-8)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def pose_targets(E):
    """Per camera: position (m), unit forward axis, azimuth/elevation (deg) and distance (m) about CENTRE."""
    pos, fwd = E[..., :3, 3], E[..., :3, 2]
    v = pos - CENTRE
    dist = np.linalg.norm(v, axis=-1)
    return {"pos": pos, "fwd": fwd,
            "azel": np.stack([np.degrees(np.arctan2(v[..., 1], v[..., 0])), np.degrees(np.arcsin(v[..., 2] / dist))], -1),
            "dist": dist}


def miscal_targets(T):
    """Per camera: world-frame rotation vector (deg), its angle (deg), translation (cm) and its length (cm)."""
    rv = np.degrees(Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3).astype(np.float64)).as_rotvec()).reshape(*T.shape[:-2], 3)
    t = T[..., :3, 3] * 100
    return {"rotvec": rv, "rot_deg": np.linalg.norm(rv, axis=-1), "trans": t, "trans_cm": np.linalg.norm(t, axis=-1)}


def feature_sets(F, last_slot=1):
    """name -> (depth index, step, array (N, ncam, D) or (N, D) for global). Depth: 0 enc, 1 hist, 2..5 after SA layer 0..3."""
    out = {"enc": (0, None, F["enc"]), "hist": (1, None, F["hist"])}
    for s, sn in ((0, "first"), (last_slot, "last")):
        for l in range(F["sa"].shape[2]):
            out[f"sa{l}@{sn}"] = (2 + l, sn, F["sa"][:, s, l])
        if "dm" in F:
            calls = F["dm"].shape[2]
            for l in range(4):
                d = F["dm"][:, s, calls - 4 + l]
                out[f"dm{l}@{sn}"] = (1 + l, sn, (d - np.eye(d.shape[-1])).reshape(*d.shape[:2], -1))
        if "adaln" in F:
            for k in range(F["adaln"].shape[2]):
                out[f"ad{k}@{sn}"] = (1 + k, sn, F["adaln"][:, s, k])
    for k in sorted((k for k in F if k.startswith("x") and k != "xdm"), key=lambda k: (int(k[1:k.index("s" if "st" in k else "t")]), k)):
        i = int(k[1:k.index("st" if "st" in k else "tp")])
        out[k] = (100 + 2 * i + (0 if "st" in k else 1), None, F[k])
    if "xdm" in F:
        out["xdm"] = (99, None, (F["xdm"] - np.eye(6)).reshape(*F["xdm"].shape[:2], -1))
    return out


def cam_feature(arr, cam):
    """Per-camera slice, wrist-mean control, or the global feature itself."""
    if arr.ndim == 2:
        return arr
    if cam == "wrist":
        return arr[:, 2:4].mean(1)
    return arr[:, cam]


_CACHE = {}


def feature_names(path):
    r = torch.load(path, weights_only=False)
    return list(feature_sets({k: v.float().numpy() for k, v in r["val"]["feats"]["clean"].items()}).keys())


def probe_model(job):
    """Fit every probe for one (feature file, feature set) pair."""
    path, only = job
    if path not in _CACHE:
        _CACHE.clear()
        _CACHE[path] = torch.load(path, weights_only=False)
    r = _CACHE[path]
    name = r["model"]
    tr, va = r["train"], r["val"]
    np_ = lambda d: {k: v.float().numpy() for k, v in d.items()}
    Ftr = {c: np_(v) for c, v in tr["feats"].items()}
    Fva = {c: np_(v) for c, v in va["feats"].items()}
    gtr, gva = episodes(tr["labels"]), episodes(va["labels"])
    Ptr, Pva = pose_targets(tr["labels"]["extrinsics"]), pose_targets(va["labels"]["extrinsics"])
    cgtr, cgva = tr["labels"]["camera_group"].astype(int), va["labels"]["camera_group"].astype(int)
    Mtr, Mva = miscal_targets(tr["noise"]["sweep"].numpy()), miscal_targets(va["noise"]["sweep"].numpy())
    rows = []

    def add(**kw):
        rows.append({"model": name, **kw})
        print(json.dumps(rows[-1]), flush=True)

    sets_c_tr, sets_c_va = feature_sets(Ftr["clean"]), feature_sets(Fva["clean"])
    sets_s_tr, sets_s_va = feature_sets(Ftr["sweep"]), feature_sets(Fva["sweep"])
    sets_t_tr, sets_t_va = feature_sets(Ftr["train"]), feature_sets(Fva["train"])
    for fname, (depth, step, Xc_tr) in sets_c_tr.items():
        if fname != only:
            continue
        Xc_va = sets_c_va[fname][2]
        kind = "ctok" if fname[:2] in ("en", "hi", "sa") else ("ext" if fname[0] == "x" and fname != "xdm" else fname[:2])
        glob = Xc_tr.ndim == 2
        cams = ["global"] if glob else (EXT + ["wrist"] if kind in ("ctok", "ext") else EXT)
        if kind in ("ctok", "ext"):
            cams = cams + [2, 3]
        base = dict(feature=fname, kind=kind, depth=depth, step=step)
        for cam in cams:
            xtr, xva = cam_feature(Xc_tr, cam), cam_feature(Xc_va, cam)
            tcams = EXT if cam in ("global",) else ([cam] if cam in (0, 1, 2, 3) else EXT)
            # (A) pose: per target camera; global and wrist-mean features predict the external cameras.
            for tc in tcams:
                if cam == "global" and tc != 0:
                    continue
                tsel = EXT if cam == "global" else [tc]
                cam_label = cam if cam != "wrist" else f"wrist->{tc}"
                for key in ("pos", "fwd", "azel"):
                    if tc in (2, 3) and key == "azel":
                        continue
                    ytr = Ptr[key][:, tsel].reshape(len(xtr), -1)
                    yva = Pva[key][:, tsel].reshape(len(xva), -1)
                    for sh in (False, True):
                        p = ridge(xtr, ytr, xva, gtr, sh)
                        m = {"r2": r2(yva, p)}
                        if key == "pos":
                            m["err_cm"] = float(np.linalg.norm((p - yva).reshape(len(p), -1, 3), axis=-1).mean() * 100)
                        if key == "fwd":
                            m["err_deg"] = float(angle_deg(p.reshape(len(p), -1, 3), yva.reshape(len(p), -1, 3)).mean())
                        if key == "azel":
                            m["err_deg"] = float(np.abs(p - yva).mean())
                        add(**base, cam=cam_label, target=f"A_{key}", split="iid", shuffled=sh, **m)
                if tc in (2, 3):
                    continue
                for sh in (False, True):
                    p = logistic(xtr, cgtr, xva, gtr, sh)
                    add(**base, cam=cam_label, target="A_group", split="iid", shuffled=sh, acc=float((p == cgva).mean()))
                # Leave-one-group-out: train on 5 groups (train split), test on the 6th (val split).
                for key in ("pos", "fwd"):
                    ytr = Ptr[key][:, tsel].reshape(len(xtr), -1)
                    yva = Pva[key][:, tsel].reshape(len(xva), -1)
                    pred, mean_base, near_base = np.zeros_like(yva), np.zeros_like(yva), np.zeros(len(yva))
                    for g in range(1, 7):
                        a, b = cgtr != g, cgva == g
                        pred[b] = ridge(xtr[a], ytr[a], xva[b], gtr[a])
                        mean_base[b] = ytr[a].mean(0)
                        others = np.unique(np.round(ytr[a], 4), axis=0)
                        if key == "pos":
                            d = np.linalg.norm((yva[b][:, None] - others[None]).reshape(b.sum(), len(others), -1, 3), axis=-1).mean(-1)
                        else:
                            d = angle_deg(yva[b][:, None].reshape(b.sum(), 1, -1, 3), others[None].reshape(1, len(others), -1, 3)).mean(-1)
                        near_base[b] = d.min(1)
                    if key == "pos":
                        err = lambda q: float(np.linalg.norm((q - yva).reshape(len(q), -1, 3), axis=-1).mean() * 100)
                        add(**base, cam=cam_label, target="A_pos", split="logo", r2=r2(yva, pred), err_cm=err(pred),
                            mean_baseline_cm=err(mean_base), nearest_other_group_cm=float(near_base.mean() * 100))
                    else:
                        err = lambda q: float(angle_deg(q.reshape(len(q), -1, 3), yva.reshape(len(q), -1, 3)).mean())
                        add(**base, cam=cam_label, target="A_fwd", split="logo", r2=r2(yva, pred), err_deg=err(pred),
                            mean_baseline_deg=err(mean_base), nearest_other_group_deg=float(near_base.mean()))
            # (B) miscalibration: features from the sweep condition, targets of external cameras.
            if cam in (2, 3):
                continue
            xs_tr, xs_va = cam_feature(sets_s_tr[fname][2], cam), cam_feature(sets_s_va[fname][2], cam)
            xt_tr, xt_va = cam_feature(sets_t_tr[fname][2], cam), cam_feature(sets_t_va[fname][2], cam)
            for tc in (EXT if cam in ("global", "wrist") else [cam]):
                if cam == "global" and tc != 0:
                    continue
                tsel = EXT if cam == "global" else [tc]
                cam_label = cam if cam != "wrist" else f"wrist->{tc}"
                for key in ("rotvec", "rot_deg", "trans", "trans_cm"):
                    ytr = Mtr[key][:, tsel].reshape(len(xs_tr), -1)
                    yva = Mva[key][:, tsel].reshape(len(xs_va), -1)
                    for sh in (False, True):
                        p = ridge(xs_tr, ytr, xs_va, gtr, sh)
                        vec = key in ("rotvec", "trans")
                        e = (np.linalg.norm((p - yva).reshape(len(p), -1, 3), axis=-1) if vec else np.abs(p - yva)).mean()
                        add(**base, cam=cam_label, target=f"B_{key}", split="iid", shuffled=sh, r2=r2(yva, p),
                            mae=float(e), mean_baseline_mae=float(((np.linalg.norm((yva - ytr.mean(0)).reshape(len(p), -1, 3), axis=-1)) if vec else np.abs(yva - ytr.mean(0))).mean()))
                for other, (otr, ova) in (("sweep", (xs_tr, xs_va)), ("train", (xt_tr, xt_va))):
                    Xb_tr, Xb_va = np.concatenate([xtr, otr]), np.concatenate([xva, ova])
                    yb_tr = np.r_[np.zeros(len(xtr)), np.ones(len(otr))].astype(int)
                    yb_va = np.r_[np.zeros(len(xva)), np.ones(len(ova))].astype(int)
                    p = logistic(Xb_tr, yb_tr, Xb_va, np.r_[gtr, gtr])
                    add(**base, cam=cam_label, target=f"B_clean_vs_{other}", split="iid", shuffled=False, acc=float((p == yb_va).mean()))
    # Task-only baseline for the group probe: one-hot task id.
    if name.startswith("exp3") and only == "enc":
        oh = lambda t: np.eye(13)[t.astype(int)]
        p = logistic(oh(tr["labels"]["task_id"]), cgtr, oh(va["labels"]["task_id"]), gtr)
        add(feature="task_onehot", kind="baseline", depth=-1, step=None, cam="-", target="A_group", split="iid", shuffled=False, acc=float((p == cgva).mean()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out", default="results/analysis/probes/probe_results.json")
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = [(f, n) for f in a.features for n in feature_names(f)]
        rows = [r for rs in ex.map(probe_model, jobs) for r in rs]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=0))
    print("wrote", a.out, len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
