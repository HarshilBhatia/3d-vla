#!/usr/bin/env python3
"""Collect this week's online results and analysis conclusions into docs/research/dashboard/index.html."""
import glob
import json
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E100 = "/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts"
E20 = "/grogu/datasets/hbhatia/3dfa_online_eval_20rollouts"
OUT = ROOT / "docs/research/dashboard"

LEVELS = ["calibrated", "known", "3/1", "5/5", "10/10", "15/15", "20/20", "30/30", "40/40", "50/50", "60/60"]
G7 = ["clean", "opp 10cm", "opp 20cm", "opp 40cm", "opp 60cm", "opp 100cm", "opp 200cm", "opp 500cm",
      "rot 20°", "rot 90°", "rot 180°", "rot 270°"]

# Campaigns that make up the IDLT task-heldout curve; a method's points are the union over these.
IDLT_CAMPAIGNS = ["idlt_task_heldout_group_fixed_v03_video_k5", "idlt_task_heldout_group_fixed_v02",
                  "idlt_fullpatch_lownoise_task_heldout_group_v01", "idlt_extreme_task_heldout_group_v01",
                  "idlt_scaled_extreme_task_heldout_group_v01", "idlt_scaled_extreme_task_heldout_group_v02_5060",
                  "vdm_rope3d_sweep_task_heldout_group_v01"]
G7_CAMPAIGNS = ["g7_extreme_opposite_v01", "g7_rotation_only_v01", "g7_opposite_camera_sbrs_v01"]

# (series key, label, run id, step label, [method dirs, first match per level wins])
SERIES = [
    ("1a@80k", "exp 1a · per-camera adaLN · 80k", "1a", 80, ["exp1a_adaln_percam_s80k_ema"]),
    ("1a@200k", "exp 1a · per-camera adaLN · 200k", "1a", 200, ["exp1a_adaln_percam_s200k_ema"]),
    ("1b@200k", "exp 1b · pooled adaLN · 200k", "1b", 200, ["exp1b_adaln_pooled_s200k_ema"]),
    ("1c@120k", "exp 1c · static adaLN · 120k", "1c", 120, ["exp1c_adaln_static_s120k_ema"]),
    ("1c@200k", "exp 1c · static adaLN · 200k", "1c", 200, ["exp1c_adaln_static_s200k_ema"]),
    ("2@70k", "exp 2 · Direct δM · 70k", "2", 70, ["exp2_direct_randhist_s70k_ema"]),
    ("2@200k", "exp 2 · Direct δM · 200k", "2", 200, ["exp2_direct_randhist_s200k_ema"]),
    *[(f"3@{s}k", f"exp 3 · layer-wise δM · {s}k", "3", s, [f"exp3_randhist_deltam_s{s}k_ema"]) for s in (50, 80, 110, 140, 170, 200)],
    ("4a@200k", "exp 4a · pre-LN extractor · 200k", "4a", 200, ["exp4a_prenorm_s200k_ema"]),
    ("4b@200k", "exp 4b · input skip · 200k", "4b", 200, ["exp4b_inputskip_s200k_ema"]),
    ("A@160k", "A control · 160k (raw)", "A", 160, ["regtest_a_register_s160k"]),
    ("A@200k", "A control · 200k (raw)", "A", 200, ["regtest_a_register_s200k"]),
    ("A+EMA", "A control + EMA fine-tune", "A", None, ["ema20k_regtest_a_s20k_ema"]),
    ("C@100k", "C no register · 100k (raw)", "C", 100, ["regtest_c_noregister_s100k"]),
    ("C+EMA", "C + EMA fine-tune", "C", None, ["ema20k_regtest_c_s10k_ema"]),
    ("VP@84k", "VideoPooled · 84k (raw)", "VP", 84, ["video_pooled_s084k", "video_deltam_best_s084k"]),
    ("VP+EMA", "VideoPooled + EMA fine-tune", "VP", None, ["ema20k_vidpooled_s15k_ema"]),
    ("RH@50k", "Random history (warm) · 50k", "RH", 50, ["video_pooled_randhist_notime_s050k"]),
    ("RH+EMA", "Random history (warm) + EMA", "RH", None, ["ema20k_randhist_s15k_ema"]),
    ("NVA@140k", "No δM, ordered history (warm) · 140k", "NVA", 140, ["video_pooled_no_view_align_s140k"]),
    ("NVA+EMA", "No δM, ordered history + EMA", "NVA", None, ["ema20k_noviewalign_s15k_ema"]),
    ("FP", "FullPatch (best)", "FP", None, ["video_fullpatch_best"]),
    ("R3D-refine", "FullPatch rope3d Refine (best)", "R3Dr", None, ["vdm-refine-fullpatch-rope3d-best"]),
    ("R3D-direct", "FullPatch rope3d Direct (best)", "R3Dd", None, ["vdm-direct-fullpatch-rope3d-best"]),
    ("Base@160k", "Base · 160k (raw)", "Base", 160, ["base_s160k"]),
    ("Base+EMA", "Base + EMA fine-tune", "Base", None, ["ema20k_base_s20k_ema"]),
    ("DeltaM@140k", "DeltaM (no history) · 140k (raw)", "DeltaM", 140, ["deltam_s140k"]),
]

# run id -> table row; family picks the colour, the rest is the record of what was trained.
RUNS = [
    dict(id="1a", name="exp 1a · per-camera C-token adaLN", family="adaLN", week="Week ending 6 Oct", init="scratch", steps="200k", ema="yes", status="done",
         mech="each camera's C-token modulates its own tokens, every SA layer", config="peract2_orbital_vid_pooled_randhist_adaln_percam_scratch"),
    dict(id="1b", name="exp 1b · pooled C-token adaLN", family="adaLN", week="Week ending 6 Oct", init="scratch", steps="200k", ema="yes", status="done",
         mech="mean C-token modulates every token, every SA layer", config="peract2_orbital_vid_pooled_randhist_adaln_pooled_scratch"),
    dict(id="1c", name="exp 1c · static pooled adaLN", family="adaLN", week="Week ending 6 Oct", init="scratch", steps="200k", ema="yes", status="done",
         mech="mean of head-input C-tokens, never updated", config="peract2_orbital_vid_pooled_randhist_adaln_static_scratch"),
    dict(id="2", name="exp 2 · Direct δM", family="δM scratch", week="Week ending 6 Oct", init="scratch", steps="200k", ema="yes", status="done",
         mech="history extractor predicts one δM per camera", config="peract2_orbital_vid_direct_randhist_scratch"),
    dict(id="3", name="exp 3 · layer-wise δM (refine)", family="δM scratch", week="Week ending 6 Oct", init="scratch", steps="200k", ema="yes", status="done",
         mech="head predicts δM from C-tokens before every SA layer", config="peract2_orbital_vid_pooled_randhist_deltam_scratch"),
    dict(id="4a", name="exp 4a · exp 3 + pre-LN extractor", family="exp 4", week="Week ending 6 Oct", init="scratch", steps="200k", ema="yes", status="training",
         mech="pre-LayerNorm history extractor (stays alive)", config="peract2_orbital_vid_pooled_randhist_deltam_prenorm_scratch"),
    dict(id="4b", name="exp 4b · exp 3 + input skip", family="exp 4", week="Week ending 6 Oct", init="scratch", steps="200k", ema="yes", status="training",
         mech="raw current-frame tokens skipped onto the extractor output", config="peract2_orbital_vid_pooled_randhist_deltam_inputskip_scratch"),
    dict(id="A", name="A · control (no δM, register)", family="control", week="Week ending 29 Sep", init="scratch", steps="200k", ema="no", status="done",
         mech="random history, no δM, history register on", config="peract2_orbital_vid_pooled_noalign_randhist_scratch"),
    dict(id="C", name="C · A without register", family="control", week="Week ending 29 Sep", init="A @ 50k", steps="+150k", ema="no", status="done",
         mech="as A, register off", config="peract2_orbital_vid_pooled_noalign_randhist_noreg_from_a50k"),
    dict(id="VP", name="VideoPooled", family="warm-started", week="Earlier", init="100k ordered pretrain", steps="+84k", ema="no", status="done",
         mech="δM refine, ordered history + time embedding, register", config="peract2_orbital_vid_deltam_pooled"),
    dict(id="RH", name="Random history (warm)", family="warm-started", week="Week ending 29 Sep", init="VideoPooled 84k", steps="+100k", ema="no", status="done",
         mech="δM refine, random unordered history", config="peract2_orbital_vid_deltam_pooled_randhist_notime"),
    dict(id="NVA", name="No δM, ordered history (warm)", family="warm-started", week="Week ending 29 Sep", init="VideoPooled", steps="+300k", ema="no", status="stopped 281k",
         mech="history, no δM", config="peract2_orbital_vid_deltam_pooled_no_view_align"),
    dict(id="FP", name="FullPatch", family="full-patch", week="Earlier", init="warm", steps="best", ema="no", status="done",
         mech="per-patch history extractor, refine", config="peract2_orbital_vid_deltam_fullpatch"),
    dict(id="R3Dr", name="FullPatch rope3d Refine", family="full-patch", week="Earlier", init="warm", steps="best", ema="no", status="done",
         mech="per-patch extractor sees patch xyz; head δM", config="peract2_orbital_vid_deltam_refine_fullpatch_rope3d"),
    dict(id="R3Dd", name="FullPatch rope3d Direct", family="full-patch", week="Earlier", init="warm", steps="best", ema="no", status="done",
         mech="per-patch extractor sees patch xyz; extractor δM", config="peract2_orbital_vid_deltam_direct_fullpatch_rope3d"),
    dict(id="Base", name="Base", family="no history", week="Earlier", init="scratch", steps="200k", ema="no", status="done",
         mech="no history, no δM", config="peract2_orbital_new_external_only_miscal_base"),
    dict(id="DeltaM", name="DeltaM", family="no history", week="Earlier", init="scratch", steps="200k", ema="no", status="done",
         mech="no history, layer-wise δM from encoder C-tokens", config="peract2_orbital_new_external_only_miscal_deltam_external"),
]

# Step ladders: (series key, label, run id, kind, glob pattern with {s} for the step dir suffix)
LADDERS = [
    ("exp 1a SBRS (EMA)", "1a", "sbrs", f"{E20}/video_deltam_checkpoint_ladder_ema_v01_exp1a-adaln-percam-ema_s*"),
    ("exp 1b SBRS (EMA)", "1b", "sbrs", f"{E20}/video_deltam_checkpoint_ladder_ema_v01_exp1b-adaln-pooled-ema_s*"),
    ("exp 1c SBRS (EMA)", "1c", "sbrs", f"{E20}/video_deltam_checkpoint_ladder_ema_v01_exp1c-adaln-static-ema_s*"),
    ("exp 2 SBRS (EMA)", "2", "sbrs", f"{E20}/video_deltam_checkpoint_ladder_ema_v01_exp2-direct-randhist-ema_s*"),
    ("exp 3 SBRS (EMA)", "3", "sbrs", f"{E20}/video_deltam_checkpoint_ladder_ema_v01_exp3-randhist-deltam-ema_s*"),
    ("exp 4a SBRS (EMA)", "4a", "sbrs", f"{E20}/sbrs_sanity_ladder_ema_v01_exp4a-prenorm-ema_s*"),
    ("exp 4b SBRS (EMA)", "4b", "sbrs", f"{E20}/sbrs_sanity_ladder_ema_v01_exp4b-inputskip-ema_s*"),
    ("A SBRS (raw)", "A", "sbrs", f"{E20}/video_deltam_checkpoint_ladder_v01_vid-pooled-noalign-randhist-scratch_s*"),
    ("A robustness ladder (raw)", "A", "rob", f"{E20}/robustness_ladder_v01_regtest_a_raw_s*"),
    ("exp 4a robustness ladder (EMA)", "4a", "rob", f"{E20}/robustness_ladder_v01_exp4a-prenorm-ema_s*"),
    ("exp 4b robustness ladder (EMA)", "4b", "rob", f"{E20}/robustness_ladder_v01_exp4b-inputskip-ema_s*"),
]


def level_of(cal_dir):
    """Map a calibration folder name to a curve level."""
    c = cal_dir
    if c == "calibrated":
        return "calibrated"
    if "known" in c and "unknown" not in c:
        return "known"
    if "supported-e3deg-t1cm" in c:
        return "3/1"
    m = re.search(r"unknown-e(\d+)deg-t(\d+)cm", c)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def g7_of(cal_dir):
    c = cal_dir.replace("g7-", "")
    if c == "clean":
        return "clean"
    m = re.match(r"opposite-(\d+)cm", c)
    if m:
        return f"opp {m.group(1)}cm"
    m = re.match(r"rot-(\d+)deg", c)
    return f"rot {m.group(1)}°" if m else None


def read_mean(f):
    v = list(json.load(open(f)).values())[0]
    return v["mean"] if isinstance(v, dict) else None


def collect(root, campaigns, method, mapper):
    acc = defaultdict(list)
    for camp in campaigns:
        for f in glob.glob(f"{root}/{camp}/{method}/**/results_*.json", recursive=True):
            if f.endswith("manifest.json"):
                continue
            lvl = mapper(f.split("/")[-2])
            m = read_mean(f)
            if lvl and m is not None:
                acc[lvl].append(m)
    return acc


def series_points(methods, campaigns, root, mapper, order):
    # Never average across campaigns: per level, the first campaign (in priority order) holding any alias wins.
    pts = {}
    for camp in campaigns:
        for method in methods:
            for lvl, v in collect(root, [camp], method, mapper).items():
                if lvl not in pts:
                    pts[lvl] = [round(100 * sum(v) / len(v), 1), len(v)]
    return {k: pts[k] for k in order if k in pts}


def ladder_points(pattern, kind):
    out = []
    for d in glob.glob(pattern):
        step = int(re.findall(r"_s(\d+)$", d)[0])
        step = step * 1000 if step < 1000 else step
        if kind == "sbrs":
            v = [m for f in glob.glob(d + "/**/results_*.json", recursive=True) if not f.endswith("manifest.json") for m in [read_mean(f)] if m is not None]
            if len(v) == 6:
                out.append(dict(step=step // 1000, sbrs=round(100 * sum(v) / 6, 1)))
        else:
            row = dict(step=step // 1000)
            for lvl, key in (("calibrated", "cal"), ("e60deg-t60cm", "e60")):
                v = [m for f in glob.glob(d + "/**/results_*.json", recursive=True) if not f.endswith("manifest.json") and lvl in f for m in [read_mean(f)] if m is not None]
                if v:
                    row[key] = round(100 * sum(v) / len(v), 1)
                    row[key + "_n"] = len(v)
            if len(row) > 1:
                out.append(row)
    return sorted(out, key=lambda r: r["step"])


def main():
    series = []
    for key, label, run, step, methods in SERIES:
        idlt = series_points(methods, IDLT_CAMPAIGNS, E100, level_of, LEVELS)
        g7 = series_points(methods, G7_CAMPAIGNS, E20, g7_of, G7)
        series.append(dict(key=key, label=label, run=run, step=step, idlt=idlt, g7=g7))
    ladders = [dict(label=l, run=r, kind=k, points=ladder_points(p, k)) for l, r, k, p in LADDERS]
    todos = json.loads((OUT / "todos.json").read_text())
    data = dict(generated=time.strftime("%Y-%m-%d %H:%M"), levels=LEVELS, g7=G7, runs=RUNS, series=series,
                ladders=ladders, todos=todos)
    (OUT / "data.json").write_text(json.dumps(data, indent=1))
    html = (OUT / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
    (OUT / "index.html").write_text(html)
    filled = sum(1 for s in series if s["idlt"] or s["g7"])
    print(f"wrote {OUT/'index.html'}: {filled}/{len(series)} series with results, {len(ladders)} ladders, {len(todos)} TODOs")


if __name__ == "__main__":
    main()
