# Orbital Camera Visualisation Tools

All scripts run from the repo root. Scripts that launch CoppeliaSim need a virtual display (`xvfb-run -a`) and `COPPELIASIM_ROOT` / `LD_LIBRARY_PATH` set.

---

## explore_camera_positions.py
Randomly samples camera positions around the workspace and produces an **interactive HTML gallery** to shortlist viable configs. Works with PerAct and PerAct2 zarrs (no CoppeliaSim needed).

```bash
python scripts/orbital_cameras/vis/explore_camera_positions.py \
    --zarr  Peract_zarr/val.zarr \
    --out   camera_explore/ \
    --n_random 200 --n_frames 15 --n_shortlist 6
# Output: camera_explore/index.html
```

---

## sample_camera_positions.py
Applies a grid of rotations/translations to existing PerAct2 camera rigs and renders each variant as an **HTML summary**. Good for systematic rig sweeps before committing to a group layout.

```bash
python scripts/orbital_cameras/vis/sample_camera_positions.py \
    --zarr Peract2_zarr/bimanual_lift_tray/val.zarr \
    --out camera_samples/ \
    --rotate_z "-30,-15,0,15,30" \
    --translate_x "-0.2,0,0.2" \
    --translate_z "-0.1,0,0.1"
# Output: camera_samples/index.html
```

---

## render_camera_positions.py
Launches CoppeliaSim, resets a task scene, and **renders RGB images** from many random + structured camera positions. Feed the output JSON into the other scripts to inspect shortlisted configs.

```bash
xvfb-run -a python scripts/orbital_cameras/vis/render_camera_positions.py \
    --task close_jar --out camera_render/ --n_random 200 --n_resets 3

# From a shortlist produced by explore_camera_positions.py:
xvfb-run -a python scripts/orbital_cameras/vis/render_camera_positions.py \
    --candidates_json camera_explore/configs.json \
    --task close_jar --out camera_render/
```

---

## visualize_cameras_rerun.py
Launches CoppeliaSim, resets a task, and streams **point clouds + camera frustums** to a [Rerun](https://rerun.io) `.rrd` file for 3D inspection.

```bash
xvfb-run -a python scripts/orbital_cameras/vis/visualize_cameras_rerun.py \
    --task close_jar --out camera_viz.rrd

# Then on your local machine:
rerun camera_viz.rrd
```
