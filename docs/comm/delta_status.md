# delta status

Updated 2026-10-14 13:32 CDT by the Delta Claude session (worktree /work/nvme/bgkz/harshilb/3d-vla-comm, branch 3d-vla-comm).

## Running on Delta
Nothing. Queue is empty.

## NOT running / blocked
- **exp10f pooled_refine**: submitted as job 22900982 with `scripts/comm/delta_exp10.sbatch` (2 x A100, resumes from the 16.7k last.pth in /projects/bgkz/3dfa_train_logs). It crashed at startup: `zarr GroupNotFoundError` on the eval set.
- **Cause**: `/projects/bgkz/3dfa_dense/handover_chunk8/val.zarr` is EMPTY on Delta (dir created 11:44, no files). `train.zarr` is complete (315,785 files). The val push was interrupted.
- **Needed from grogu**: re-run `DELTA_USER=hbhatia1 scripts/data/push_to_delta.sh handover_chunk8` (rsync resumes; it will fill val.zarr and check file counts). Delta cannot reach grogu over ssh (password/Duo), so this has to start from grogu.
- Once val.zarr is there, resubmit: `sbatch scripts/comm/delta_exp10.sbatch pooled_refine peract2_orbital_handover_dense_chunk8_pooled_refine`.
- 10a/10b/10c/10d/10e are not launched on Delta; grogu already runs them. The exp10 evals are held (JobHeldUser) on grogu and need the 50k/100k checkpoints.

## Delta state
- Data: handover_chunk8 train.zarr OK, val.zarr empty.
- Checkpoints in /projects/bgkz/3dfa_train_logs/PerAct2 (snapshot from grogu): deltam interm 30000, rel interm 30000, fullpatch_refine / _x0 / _rel / pooled_refine interm 15000, each with last.pth.
- Env: micromamba env 3dfa works; account bgkz-delta-gpu.
