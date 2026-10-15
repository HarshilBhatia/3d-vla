#!/usr/bin/env python3
"""Pool sharded and unsharded online-eval results for one method (repeat runs of a demo all count); usage: aggregate_sharded_eval.py <method_id> <version glob>."""
import json, glob, re, ast
root = "/grogu/datasets/hbhatia/3dfa_online_eval_100rollouts/"; logs = "/home/harshilb/3dfa_unified/logs/eval/"
import sys
M = sys.argv[1] if len(sys.argv) > 1 else "exp10f_pooled_refine_s100k_ema"; V = sys.argv[2] if len(sys.argv) > 2 else "0[37]"
def load(f):
    try: return json.loads(open(f, errors="ignore").read().rstrip("\x00 \n\t\r"))
    except (ValueError, OSError): return None
def cellkey(camp, cal):
    return ("G1" if "seen_g1" in camp else "G4", "cal" if cal == "calibrated" else re.sub(r".*-e(\d+)deg-t\d+cm-v01", r"\1/\1", cal))
def per_variation(camp):
    for f in glob.glob(logs + camp.split("_v0")[0] + "_v0*.out"):
        m = re.search(r"num_demos_total=25 over \d+ variations -> (\{[^}]*\})", open(f, errors="ignore").read())
        if m: return ast.literal_eval(m.group(1))
tot = {}
for d in glob.glob(root + f"exp10_*_v{V}*/{M}/*/*/"):
    camp, cal = d.split("/")[-5], d.split("/")[-2]; ck = cellkey(camp, cal); sh = re.search(r"v0[0-9]s(\d)", camp)
    prog = load(d + "results_bimanual_handover_item.progress.json")
    res = load(d + "results_bimanual_handover_item.json")
    t = tot.setdefault(ck, [0, 0])
    if prog:
        t[0] += sum(prog.values()); t[1] += len(prog)
    elif res and sh:
        pv = per_variation(camp) or {}
        for var, n in pv.items():
            t[0] += res["bimanual_handover_item"].get(str(var), 0); t[1] += len([i for i in range(n) if i % 4 == int(sh.group(1))])
    elif res:
        r = res["bimanual_handover_item"]; t[0] += sum(v for k, v in r.items() if k != "mean"); t[1] += 25
order = [("G1", "cal"), ("G4", "cal"), ("G4", "10/10"), ("G4", "15/15"), ("G4", "20/20"), ("G4", "30/30"), ("G4", "40/40"), ("G4", "50/50"), ("G4", "60/60")]
out = []
for ck in order:
    s_, n = tot.get(ck, [0, 0])
    out.append(f"{ck[0]} {ck[1]}={s_}/{n}" + (f" ({100*s_/n:.0f}%)" if n else ""))
print(f"{M}, original + shards pooled (repeat runs of a demo all count):"); print("\n".join(out))
