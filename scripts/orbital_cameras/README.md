# Orbital Camera Data Collection

Scripts for collecting RLBench rollouts with orbital cameras and converting them to zarr.
All commands run from the **repo root**.

---

## Typical workflow

### 1. (One-time) Generate camera group JSON
Produces `instructions/orbital_cameras_grouped.json` — the 6 orbital camera rigs.

```bash
xvfb-run -a bash scripts/orbital_cameras/gen_orbital_cameras.sh
```

### 2. Collect episodes

**Option A — local / interactive (18 sequential CoppeliaSim launches):**
```bash
xvfb-run -a bash scripts/orbital_cameras/run_orbital_datagen.sh
# Override episode count:
N_EPISODES=5 xvfb-run -a bash scripts/orbital_cameras/run_orbital_datagen.sh
```
Output: `data/orbital_rollouts/{task}/{group}/episode_*/`

**Option B — Slurm (54 parallel array jobs, one per task×group):**
```bash
# Smoke test a single index first (index 3 = close_jar / G2):
sbatch --array=3 scripts/orbital_cameras/collect_orbital_slurm.sbatch

# Full run (submits collection + chains zarr merge automatically):
bash scripts/orbital_cameras/submit_orbital_collection.sh

# Dry run (print sbatch commands without submitting):
DRY_RUN=1 bash scripts/orbital_cameras/submit_orbital_collection.sh

# Resubmit failed indices:
sbatch --array=7,23,41 scripts/orbital_cameras/collect_orbital_slurm.sbatch
```

### 3. Merge episodes into zarr
Runs automatically after `submit_orbital_collection.sh`. To run manually:
```bash
sbatch scripts/orbital_cameras/merge_orbital_zarr.sbatch
# Verify output:
python3 -c "import zarr; z=zarr.open('data/orbital_train.zarr'); print({k: z[k].shape for k in z})"
```
Output: `data/orbital_train.zarr`

---

## Debug / validation

Collect 1 episode per group as MP4 + single-episode zarr to validate the pipeline before a full run:
```bash
xvfb-run -a bash scripts/orbital_cameras/run_orbital_debug.sh
# Output: debug_videos/{task}_{group}.mp4  +  debug_videos/{task}_{group}.mp4.zarr/
```

---

## Direct Python CLI (`collect.py`)

The shell scripts call `collect.py` internally. You can also call it directly:
```bash
# Collect episodes for one task + one or more groups:
xvfb-run -a python scripts/orbital_cameras/collect.py \
    --task close_jar --groups G1 G2 --n-episodes 30 \
    --cameras-file instructions/orbital_cameras_grouped.json

# Debug mode (1 episode → MP4 + zarr):
xvfb-run -a python scripts/orbital_cameras/collect.py \
    --task close_jar --groups G1 --video-only --video-dir debug_videos
```

---

## Visualisation tools
See [`vis/README.md`](vis/README.md) for tools to explore and inspect camera positions.
