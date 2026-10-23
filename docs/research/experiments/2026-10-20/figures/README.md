# Figures, week ending 20 Oct

| File | What it shows | Script |
|---|---|---|
| `exp10_success_vs_miscal.png` | Handover success vs camera miscalibration (seen G1 calibrated, held-out G4 calibrated to 60/60): 8d, 10a, 10b at 50k and 10f at 100k. **Interim:** hollow markers with `*` are cells still evaluating (n per cell in the footnote); re-run the script to refresh | `scripts/analysis/plot_exp10_success_vs_miscal.py` (10f cells merged across shards by `scripts/analysis/aggregate_sharded_eval.py`) |
| `exp10_seedmatched_success_vs_miscal.png` | Handover success vs camera miscalibration, 100 seed-matched episodes per point: base (8d), base − proprio (10i), pooled hist + δM (10f), pooled hist + δM − proprio (10h). 10f 50/50 and 60/60 have 90 and 86 episodes (marked *) | `scripts/analysis/plot_exp10_seedmatched.py` (pools shards with `scripts/analysis/aggregate_sharded_eval.py`) |
