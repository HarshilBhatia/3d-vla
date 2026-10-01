#!/usr/bin/env python3
"""Current-frame-only history probe on SBRS (trained bias + 3/1, 6 tasks x 20) for the two random-history EMA checkpoints."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "instructions/eval_plans"
TL = ROOT / "train_logs/PerAct2"
# Same condition and runtime as the EMA ladder, so each probe cell pairs with its ladder cell.
SRC = TL / "peract2_orbital_ema20k_randhist/checkpoint_ladder_plans/video_deltam_checkpoint_ladder_ema20k_ema_v01_ema20k-randhist-ema_s015000.json"
METHODS = [{"id": "ema20k_randhist_s15k_ema", "checkpoint": str(TL / "peract2_orbital_ema20k_randhist/interm_step_15000.pth")},
           {"id": "ema20k_regtest_a_s20k_ema", "checkpoint": str(TL / "peract2_orbital_ema20k_regtest_a/interm_step_20000.pth")}]

plan = json.loads(SRC.read_text())
plan["campaign_id"] = "sbrs_hist_current_only_ema_v01"
plan["description"] = "SBRS (trained group bias + 3 deg / 1 cm), 6 tasks x 20 rollouts, EMA weights; eval-only probe: history slots hold the current frame."
plan["methods"] = METHODS
plan["runtime"]["overrides"]["eval_history_probe"] = "current_only"
out = PLANS / f"{plan['campaign_id']}.json"
out.write_text(json.dumps(plan, indent=2) + "\n")
print(out.name, len(METHODS) * len(plan["tasks"]), "cells")
