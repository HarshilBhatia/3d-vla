#!/usr/bin/env python3
"""SBRS ladder recipes every 5k steps, rolling out either raw or EMA weights, for the 20k EMA fine-tunes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
for src in ("video_deltam_checkpoint_ladder_v01", "sbrs_checkpoint_ladder_v01"):
    for weights in ("raw", "ema"):
        plan = json.loads((PLANS / f"{src}.json").read_text())
        plan["protocol_id"] = f"{src.replace('_v01', '')}_ema20k_{weights}_v01"
        plan["description"] = f"20k EMA fine-tune ladder ({weights} weights), every 5k: " + plan["description"]
        plan["min_step"], plan["step_interval"] = 5000, 5000
        plan["runtime"]["overrides"] = {**(plan["runtime"].get("overrides") or {}), "eval_weights": weights}
        out = PLANS / f"{plan['protocol_id']}.json"
        out.write_text(json.dumps(plan, indent=2) + "\n")
        print(out.name, plan["runtime"]["overrides"])
