### INIT DEAD CODE
        if self.depth_dir is not None:
            if episode_index_path is None:
                episode_index_path = self.depth_dir / "episode_frame_index.pkl"
            with open(episode_index_path, "rb") as f:
                self._episode_index = pickle.load(f)
            serial_map_path = self.depth_dir / "serial_map.json"
            with open(serial_map_path) as f:
                self._serial_map = json.load(f)
            valid_ids_path = self.depth_dir / "valid_canonical_ids.json"
            if valid_ids_path.exists():
                with open(valid_ids_path) as f:
                    self._valid_canonical_ids = set(json.load(f))
                n_total = len(self._serial_map)
                n_valid = len(self._valid_canonical_ids)
                if n_valid < n_total:
                    print(f"[Gr00tN1d6DataCollator] depth: {n_valid}/{n_total} episodes valid "
                          f"({n_total - n_valid} will have no 3D RoPE conditioning)")

def _load_depth_episode(self, canonical_id: str, serial: str) -> np.ndarray:
        """Load (T, H, W) float32 depth array for an episode/camera, with LRU-1 cache."""
        import blosc
        key = (canonical_id, serial)
        if key not in self._depth_cache:
            if len(self._depth_cache) >= 4:
                self._depth_cache.pop(next(iter(self._depth_cache)))
            ep_dir = self.depth_dir / canonical_id / serial
            shape = np.load(ep_dir / "shape.npy")  # (T, H, W)
            raw = (ep_dir / "depth.blosc").read_bytes()
            arr = np.frombuffer(blosc.decompress(raw), dtype=np.float32).reshape(shape)
            self._depth_cache[key] = arr
        return self._depth_cache[key]

    def _unproject_patches(
        self,
        depth_frame: np.ndarray,       # (180, 320) float32, meters
        intrinsics: np.ndarray,         # [fx, fy, cx, cy] at 180×320
        T_cam2base: np.ndarray,         # (4, 4) float64
        token_seq_positions: np.ndarray,  # indices into the backbone sequence for this camera's tokens
        positions: np.ndarray,          # (seq_len, 3) output array, modified in-place
    ) -> None:
        """Unproject depth patches to 3D world-frame positions and write into positions."""
        import cv2
        rH, rW = self._DROID_RESIZED_H, self._DROID_RESIZED_W
        depth_resized = cv2.resize(depth_frame, (rW, rH), interpolation=cv2.INTER_NEAREST)

        fx180, fy180, cx180, cy180 = intrinsics
        fx = fx180 * rW / 320.0
        fy = fy180 * rH / 180.0
        cx = cx180 * rW / 320.0
        cy = cy180 * rH / 180.0

        stride = self._DROID_TOKEN_STRIDE
        n_cols = self._DROID_GRID_COLS

        for t, seq_pos in enumerate(token_seq_positions):
            row = t // n_cols
            col = t % n_cols
            y0, y1 = row * stride, (row + 1) * stride
            x0, x1 = col * stride, (col + 1) * stride

            patch_depth = depth_resized[y0:y1, x0:x1]
            valid = np.isfinite(patch_depth) & (patch_depth > 0.05) & (patch_depth < 5.0)
            if not valid.any():
                continue

            # Work only on valid pixels to avoid NaN propagation in matmul
            ys_all = np.arange(y0, y1, dtype=np.float32)
            xs_all = np.arange(x0, x1, dtype=np.float32)
            Y_all, X_all = np.meshgrid(ys_all, xs_all, indexing="ij")
            D_v = patch_depth[valid]
            X_cam = (X_all[valid] - cx) * D_v / fx
            Y_cam = (Y_all[valid] - cy) * D_v / fy

            pts = np.stack([X_cam, Y_cam, D_v, np.ones_like(D_v)], axis=-1)  # (N, 4)
            pts_world = pts @ T_cam2base.T  # (N, 4)
            positions[seq_pos] = pts_world[:, :3].mean(axis=0)

    def _compute_token_positions_3d(
        self,
        global_idx: int,
        image_mask: torch.Tensor,  # [seq_len] bool
    ) -> torch.Tensor:
        """Compute [seq_len, 3] world-frame positions for each backbone token.

        Exterior and wrist camera tokens get their patch's average 3D position.
        Text/padding tokens get [0, 0, 0] (identity RoPE = no rotation).
        """
        seq_len = image_mask.shape[0]
        positions = np.zeros((seq_len, 3), dtype=np.float32)

        entry = self._episode_index[global_idx]
        canonical_id = entry["canonical_id"]
        frame_idx = entry["frame_idx"]

        serials = self._serial_map.get(canonical_id)
        if serials is None:
            return torch.from_numpy(positions)

        # Skip depth loading for episodes known to have missing depth/extrinsics
        if self._valid_canonical_ids is not None and canonical_id not in self._valid_canonical_ids:
            return torch.from_numpy(positions)

        # Split image token sequence positions into per-camera blocks
        image_positions = image_mask.nonzero(as_tuple=False).squeeze(-1).numpy()
        n_per_cam = len(image_positions) // 2
        ext1_seq_pos = image_positions[:n_per_cam]
        wrist_seq_pos = image_positions[n_per_cam:]

        # ── Exterior camera (static extrinsics from metadata JSON) ──────────────
        ext1_serial = serials["ext1"]
        try:
            depth_ext1 = self._load_depth_episode(canonical_id, ext1_serial)
            fi_ext1 = min(frame_idx, depth_ext1.shape[0] - 1)
            T_ext1 = self._get_ext1_cam2base(canonical_id)
            if T_ext1 is not None:
                intr_key = (canonical_id, ext1_serial)
                if intr_key not in self._intrinsics_cache:
                    self._intrinsics_cache[intr_key] = np.load(
                        self.depth_dir / canonical_id / ext1_serial / "intrinsics.npy"
                    )
                self._unproject_patches(
                    depth_ext1[fi_ext1],
                    self._intrinsics_cache[intr_key],
                    T_ext1,
                    ext1_seq_pos,
                    positions,
                )
        except Exception as e:
            print("Exterion camera error:", e )
            pass

        # ── Wrist camera (per-timestep extrinsics from trajectory.h5) ───────────
        wrist_serial = serials["wrist"]
        try:
            depth_wrist = self._load_depth_episode(canonical_id, wrist_serial)
            fi_wrist = min(frame_idx, depth_wrist.shape[0] - 1)
            T_wrist = self._get_wrist_cam2base(canonical_id, wrist_serial, frame_idx)
            if T_wrist is not None:
                intr_key = (canonical_id, wrist_serial)
                if intr_key not in self._intrinsics_cache:
                    self._intrinsics_cache[intr_key] = np.load(
                        self.depth_dir / canonical_id / wrist_serial / "intrinsics.npy"
                    )
                self._unproject_patches(
                    depth_wrist[fi_wrist],
                    self._intrinsics_cache[intr_key],
                    T_wrist,
                    wrist_seq_pos,
                    positions,
                )
        except Exception:
            print("Wrist camera error", e )
            pass

        return torch.from_numpy(positions)

    def _get_ext1_cam2base(self, canonical_id: str) -> Optional[np.ndarray]:
        """Return static 4×4 ext1 cam-to-base transform from metadata JSON (cached)."""
        if canonical_id in self._ext_extrinsics_cache:
            return self._ext_extrinsics_cache[canonical_id]

        from scipy.spatial.transform import Rotation
        ep_dir = self.raw_dir / canonical_id
        meta_files = list(ep_dir.glob("metadata_*.json")) if ep_dir.exists() else []
        if not meta_files:
            self._ext_extrinsics_cache[canonical_id] = None
            return None
        meta = json.loads(meta_files[0].read_text())
        dof = meta.get("ext1_cam_extrinsics")
        if dof is None:
            self._ext_extrinsics_cache[canonical_id] = None
            return None
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_rotvec(dof[3:]).as_matrix()
        T[:3, 3] = dof[:3]
        self._ext_extrinsics_cache[canonical_id] = T
        return T

    def _get_wrist_cam2base(
        self, canonical_id: str, wrist_serial: str, frame_idx: int
    ) -> Optional[np.ndarray]:
        """Return per-timestep 4×4 wrist cam-to-base from trajectory.h5 (array cached per episode)."""
        from scipy.spatial.transform import Rotation

        if canonical_id not in self._wrist_extrinsics_cache:
            traj_path = self.raw_dir / canonical_id / "trajectory.h5"
            if not traj_path.exists():
                self._wrist_extrinsics_cache[canonical_id] = None
                return None
            import h5py
            with h5py.File(traj_path, "r") as f:
                key = f"observation/camera_extrinsics/{wrist_serial}_left"
                if key not in f:
                    self._wrist_extrinsics_cache[canonical_id] = None
                    return None
                self._wrist_extrinsics_cache[canonical_id] = f[key][:]  # (T, 6) float64

        dof_array = self._wrist_extrinsics_cache[canonical_id]
        if dof_array is None:
            return None

        fi = min(frame_idx, len(dof_array) - 1)
        dof = dof_array[fi]  # [tx, ty, tz, rx, ry, rz]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_rotvec(dof[3:]).as_matrix()
        T[:3, 3] = dof[:3]
        return T