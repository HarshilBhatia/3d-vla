# Training entry points

## Current canonical entry points

- `train_peract2_orbital_2d_grogu.slurm` — current 2D SigLIP2 Orbital PerAct2
  training job on Grogu; global batch 256, 100k iterations.
- `train_peract2_standard_2d_grogu.slurm` — 2D SigLIP2 baseline on the
  published standard PerAct2 zarr (front + both wrists), 350k iterations.
- `train_peract2_orbital_2d_grogu_8gpu.slurm` — 8-GPU fallback with the same
  global batch when the 4-GPU layout does not fit.

## Historical / experiment-specific scripts

The remaining files are retained because they correspond to prior 3DFA
experiments or documented checkpoints. They are not the default starting point:

- `train_orbital*.slurm` — earlier Orbital experiments.
- `train_multicam*.slurm` — older multi-camera and miscalibration experiments.
- `train_peract*.slurm` — collected-data and earlier PerAct variants.
- `debug.slurm` — debugging only.

Do not delete historical scripts until the corresponding result provenance in
`docs/REPRODUCE.md` and `docs/status/` has been migrated to an archive index.
