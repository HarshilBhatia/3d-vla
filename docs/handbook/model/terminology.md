# Terminology and Paper-Facing Configs

This is the source of truth for new experiments and paper artifacts. Legacy
keys, run names, and checkpoint parameter names remain supported unchanged.

## The experiment

For external camera $e_j$, the implementation and formulation use

\[
\hat T_{e_j}=\varepsilon\,\delta_{i,j}\,T_{e_j}.
\]

`group_miscal` denotes the fixed per-camera-group component $\delta_{i,j}$;
`sampled_miscal` denotes the independently sampled component $\varepsilon$.
The stored extrinsics are camera-to-world. In the bimanual orbital setup,
camera IDs `[0, 1]` are external/orbital and `[2, 3]` are on-robot/wrist.

Use **calibrated** for the identity-miscalibration condition. Do not use plain
“clean,” which is ambiguous between calibrated geometry, no augmentation, and
a model without view alignment.

## The method

**View alignment** is the learned representation-space correction used during
cross-view fusion. For each corrected camera, it predicts an orthogonal RoPE
mixing matrix $\Delta M_j$. It does not estimate a physical extrinsic.

**Layer-wise view alignment** re-predicts $\Delta M_j$ at successive
Transformer depths from evolving per-camera summary features. It is not a
temporal mechanism.

**History feature extraction** is the optional `video_deltam` module
(class `HistoryFeatureExtractor`, historically "Video-DeltaM"). It applies sparse
causal attention over per-camera frame tokens: at each depth, tokens mix across
cameras within a timestep, then causally across timesteps within a camera.

It has two independent axes, and confusing them is the most common error.

`video_deltam_role` -- **what it hands the policy**:

- `refine`: it emits history-refined camera summaries (which replace the
  encoder's, inside `fps_scene_feats`) plus a global history register added to
  the trajectory queries. The policy head still predicts $\Delta M_j$ from those
  summaries, layer-wise.
- `predict_delta_m`: it emits $\Delta M_j$ and nothing else. The policy's scene
  tokens and trajectory queries are untouched -- byte-identical to a model with
  no history stack -- and the policy's own $\Delta M$ head is never constructed.
  One $\Delta M$ is reused for every attention block and every denoising step,
  so `layerwise_view_align` is inert.

`video_deltam_patch_rope3d` -- **whether it sees geometry at all**:

- `false`: positions are two learned lookups, `time_embedding[t]` and
  `camera_embedding[j]`. The visual features are pure appearance (the point
  cloud is never fused into them), so the extractor is blind to the camera
  calibration error it is meant to correct.
- `true`: queries and keys in both stages are rotated by each patch's world xyz
  via `RotaryPositionEncoding3D`. Those xyz come from depth unprojected with the
  *miscalibrated* extrinsics, so cross-camera geometric disagreement becomes
  visible to attention. Requires `video_deltam_full_image=true`.

`video_deltam_full_image` -- **how much detail it sees**:

- `false` (pooled): one pooled token per (timestep, camera).
- `true` (full patch): that image's whole patch grid plus a learned image token.

The two axes are orthogonal, giving four variants:

| `video_deltam_role` | `video_deltam_full_image` | $\Delta M$ predicted by | policy inputs changed? |
|---|---|---|---|
| `refine` | `false` | policy head | yes (refined summaries + register) |
| `refine` | `true` | policy head | yes (refined summaries + register) |
| `predict_delta_m` | `false` | history extractor | no |
| `predict_delta_m` | `true` | history extractor | no |

`video_deltam=true` with `view_align_mode=none` and `role=refine` is also legal:
history refines the scene tokens and no $\Delta M$ is produced anywhere.

| Current internal name | Paper/code concept |
|---|---|
| `camera_token` | global context token; one learned register token per sample |
| `per_img_feats` | camera-summary features; one pooled current-frame feature per camera |
| `current_per_img_feats` | layer-wise camera-summary features |
| `video_frame_feats` | history camera-summary features |
| `history_register` | history context token (was `video_camera`) |
| `history_view_align` | $\Delta M$ produced by the history extractor, not the head |

## Short public config vocabulary

| Public key | Legacy runtime key |
|---|---|
| `view_align_mode` | `predict_extrinsics` + `extrinsics_prediction_mode` |
| `view_align_cameras` | `delta_m_camera_ids` |
| `layerwise_view_align` | `dynamic_rope_from_camtoken` |
| `miscal_cameras` | `miscal_camera_ids` |
| `group_miscal_level` | `orbital_miscal_noise_level` |
| `group_miscal_file` | `orbital_miscal_noise_file` |
| `sampled_miscal_max_rot_deg` | `miscal_max_angle_deg` |
| `sampled_miscal_max_trans_m` | `miscal_max_translation_m` |
| `ee_aux` | `predict_ee_aux` |
| `ee_aux_weight` | `lambda_aux` |
| `ee_aux_cameras` | `ee_aux_cam_ids` |

### Retired keys

`causal_cam_history` and `causal_cam_history_depth` were retired **in favour of**
`video_deltam` and `video_deltam_depth`, not the other way round
(`utils/config_migrations.py`, `_view_align_and_ee_aux`). Old configs still load;
do not write new ones with the retired spelling.

### Class names vs. serialized names

Class names never appear in a `state_dict`, so they are free to be accurate.
Attribute names are `state_dict` paths and are frozen.

| Class (accurate) | Attribute / config key (frozen for checkpoints) |
|---|---|
| `HistoryFeatureExtractor` | `self.video_deltam`, `video_deltam*` config keys |
| `ViewAlignPredictor` | `self.view_align_predictor` (holds no parameters) |
| `RopeViewAlignPredictor` (was `DeltaMExtrinsicsPredictor`) | -- |
| `SE3ViewAlignPredictor` (was `RTExtrinsicsPredictor`) | -- |

`view_align_mode` values:

```text
none          no learned camera correction
rope_6d       6x6 orthogonal RoPE mixing per camera
rope_full     full-D orthogonal RoPE mixing per camera
physical_se3  physical R,T correction baseline
```

The resolver translates public keys into the legacy runtime API before model
construction. Consequently, old CLI overrides, old configs, and old
checkpoints continue to work; do not rename serialized model parameters.

## Preferred paper arms

```text
experiment=paper_external_control
experiment=paper_external_view_align
experiment=paper_external_view_align_eeaux
```

All three explicitly restrict geometric miscalibration to external cameras.
The two method arms also restrict $\Delta M$ to those cameras. They are matched
on the fixed group miscalibration plus sampled perturbation distribution.
