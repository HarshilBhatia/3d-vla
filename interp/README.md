# Cross-view RoPE diagnostic

`rope_cross_view.py` is an offline-only tensor metric; nothing under `interp/`
is imported by training or online evaluation. For each held-out batch and each
captured dynamic Delta-M layer, call `cross_view_rope_error(clean_xyz,
corrupted_xyz, camera_ids, delta_m, feature_dim)`. It uniformly samples ordinary
external-token / wrist-token pairs and reports clean, uncorrected, and corrected
relative-RoPE Frobenius errors. Average returned values over batches/layers.

It intentionally does **not** search for corresponding points or use content
features, learned Q/K projections, or task success.
