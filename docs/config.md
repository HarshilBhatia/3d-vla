# Config System (Hydra)

Base config: `config/config.yaml`. Overrides are `key=value` (no `--`). Three config groups merge at the global level: `data`, `rope_mode`, `experiment`.

```
config/
├── config.yaml                    # all defaults
├── data/{single,two,full,orbital,peract_collected}.yaml
└── experiment/
    ├── default.yaml               # predict_extrinsics=false
    ├── camtoken_deltaM.yaml       # predict_extrinsics=true, delta_m mode
    ├── camtoken_deltaM_full.yaml  # delta_m_full + dynamic_rope + use_front_camera_frame
    └── camtoken_RT.yaml           # rt mode
```

## Important Flags

| Flag | Values / Default | Notes |
|---|---|---|
| `extrinsics_prediction_mode` | `delta_m` \| `delta_m_full` \| `rt` | How camera token predicts RoPE perturbation |
| `dynamic_rope_from_camtoken` | bool, false | Recompute delta_M after every attn block |
| `traj_scene_rope` | bool | 3D RoPE in cross-attn |
| `sa_blocks_use_rope` | bool | 3D RoPE in self-attn |
| `use_front_camera_frame` | bool | Rotate point cloud to front-camera frame |
| `lv2_batch_size` | int, 1 | Re-use same obs encoding with N noise samples |
| `bimanual` | bool | Two-arm tasks; sets `nhand=2` |
| `chunk_size` | int, 1 | Consecutive zarr samples per `__getitem__` |
| `benchmark` / `benchmark_dummy_data` | bool | GPU timing / bypass zarr I/O with random data |
