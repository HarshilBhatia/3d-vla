# Miscalibration in Online Eval

## What it does

After capturing true camera-to-world extrinsics (orbital sensors + wrist
`obs.misc`), eval can perturb the transforms before passing them to depth
unprojection and the policy:

```
E_applied = E_sampled @ E_group @ E_true
```

`E_group` is an optional fixed per-camera-group miscalibration and `E_sampled`
is an optional stochastic perturbation. RGB and depth are untouched—only the
extrinsics used for depth-to-point-cloud conversion are changed, so the model
receives a geometrically miscalibrated 3D scene.

## Camera index mapping

For the paper-facing bimanual orbital setup, the canonical order is:

| cam_idx | camera | role |
|---------|--------|------|
| 0 | `orbital_left` | external |
| 1 | `orbital_right` | external |
| 2 | `wrist_left` | on-robot |
| 3 | `wrist_right` | on-robot |

Use `miscal_cameras: [0, 1]` (legacy: `miscal_camera_ids`) for the formulation
where on-robot cameras remain calibrated.

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
