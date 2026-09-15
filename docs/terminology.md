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

**Causal camera history** is the optional `video_deltam` module. It refines
history-by-camera features and produces one global history context token. It
must not be described as directly yielding a per-camera history token for
$\Delta M_j$ until that connection is implemented.

| Current internal name | Paper/code concept |
|---|---|
| `camera_token` | global context token; one learned register token per sample |
| `per_img_feats` | camera-summary features; one pooled current-frame feature per camera |
| `current_per_img_feats` | layer-wise camera-summary features |
| `video_frame_feats` | history camera-summary features |
| `video_camera` | history context token |

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
| `causal_cam_history` | `video_deltam` |
| `causal_cam_history_depth` | `video_deltam_depth` |

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
