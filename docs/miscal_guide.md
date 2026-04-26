# Miscalibration in Online Eval

## What it does

After capturing real extrinsics (orbital sensors + wrist `obs.misc`), the eval
perturbs them before passing to the model:

```
E_corrupted[:, cam_idx, :3, :3] = R_noise @ E[:, cam_idx, :3, :3]
E_corrupted[:, cam_idx, :3,  3] += t_noise
```

RGB and depth are untouched — only the extrinsics fed to the model are wrong.
This is then used for depth→point cloud, so the model sees a corrupted 3D scene.

## Camera index mapping

The noise is keyed by the names in `instructions/miscalibration_noise.json`:

| cam_idx | orbital name    | noise key in JSON |
|---------|-----------------|-------------------|
| 0       | orbital_left    | `front`           |
| 1       | orbital_right   | `wrist_left`      |
| 2       | wrist           | `wrist_right`     |

## Built-in levels

`instructions/miscalibration_noise.json` has three levels:

| level    | rotation    | translation | point error @ 0.7m |
|----------|-------------|-------------|---------------------|
| `small`  | ~0.5 deg    | ~1 mm       | ~6 mm               |
| `medium` | ~2 deg      | ~5 mm       | ~25 mm              |
| `large`  | (larger)    | (larger)    | (larger)            |

## Enabling it in eval

Pass `miscalibration_noise_level=<level>` to `evaluate_policy.py`. The eval
script already has it wired:

```bash
xvfb-run -a bash scripts/rlbench/eval_orbital_grogu_best.sh \
    miscalibration_noise_level=large
```

Or set it directly in the script:
```bash
miscalibration_noise_level=large   # null to disable
```

## Adding a custom level

Add a new entry to `instructions/miscalibration_noise.json`:

```json
"my_level": {
  "_comment": "description of noise magnitude",
  "front":      { "axis_angle_rad": [rx, ry, rz], "translation_m": [tx, ty, tz] },
  "wrist_left": { "axis_angle_rad": [rx, ry, rz], "translation_m": [tx, ty, tz] },
  "wrist_right":{ "axis_angle_rad": [rx, ry, rz], "translation_m": [tx, ty, tz] }
}
```

`axis_angle_rad` is a 3-vector; its norm is the rotation angle in radians and
its direction is the rotation axis. `translation_m` is additive in world metres.

Then pass `miscalibration_noise_level=my_level` to the eval script.
