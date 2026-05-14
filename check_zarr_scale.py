import zarr
import numpy as np

paths = {
    "train (1task_new)": "/grogu/user/harshilb/1task_new.zarr",
    "val (G3_val_set)":  "/grogu/user/harshilb/G3_val_set/val.zarr",
}

np.random.seed(0)


def stats(arr, label, indent=4):
    pad = " " * indent
    print(f"{pad}{label}: min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}")


def backproject_to_world(depth, intrinsics, extrinsics, n_pts=2000):
    """
    depth:      (N, 3, H, W)  float16
    intrinsics: (N, 3, 3, 3)  float16  (camera matrix per cam)
    extrinsics: (N, 3, 4, 4)  float16  (cam-to-world or world-to-cam 4x4)
    Returns: (n_pts, 3) world-space XYZ
    """
    N, C, H, W = depth.shape
    all_pts = []

    for _ in range(n_pts):
        n = np.random.randint(N)
        c = np.random.randint(C)
        u = np.random.randint(W)
        v = np.random.randint(H)

        d = float(depth[n, c, v, u])
        if d <= 0 or np.isnan(d) or np.isinf(d):
            continue

        K = intrinsics[n, c].astype(np.float32)   # 3x3
        E = extrinsics[n, c].astype(np.float32)   # 4x4

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # back-project to camera space
        x_c = (u - cx) * d / fx
        y_c = (v - cy) * d / fy
        z_c = d
        p_cam = np.array([x_c, y_c, z_c, 1.0], dtype=np.float32)

        p_world = E @ p_cam
        all_pts.append(p_world[:3])

    return np.array(all_pts) if all_pts else np.zeros((0, 3))


for name, p in paths.items():
    print(f"\n{'='*60}")
    print(f"{name}")
    z = zarr.open(p, "r")

    N = z["depth"].shape[0]
    print(f"  demos: {N}")

    # --- depth raw values ---
    depth_sample = z["depth"][:].astype(np.float32)  # (N,3,H,W)
    valid = depth_sample[depth_sample > 0]
    print(f"\n  [depth (valid pixels)]")
    stats(valid, "depth (m?)", indent=4)

    # --- back-projected 3D world points ---
    intr = z["intrinsics"][:]   # (N,3,3,3)
    extr = z["extrinsics"][:]   # (N,3,4,4)
    pts = backproject_to_world(depth_sample, intr, extr, n_pts=5000)
    print(f"\n  [3D world points, n={len(pts)}]")
    if len(pts):
        for i, ax in enumerate("xyz"):
            stats(pts[:, i], ax)
        norms = np.linalg.norm(pts, axis=1)
        stats(norms, "|r|")

    # --- camera positions (last col of extrinsics) ---
    cam_pos = extr[:, :, :3, 3]   # (N,3,3) — x,y,z of each camera
    print(f"\n  [camera positions (extrinsics last col)]")
    for i, ax in enumerate("xyz"):
        stats(cam_pos[:, :, i].ravel(), f"cam_{ax}")

    # --- actions (EE poses) ---
    actions = z["action"][:].reshape(-1, 8).astype(np.float32)
    print(f"\n  [action (N={len(actions)}, 8-dim)]")
    labels = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
    for i, lbl in enumerate(labels):
        stats(actions[:, i], lbl)
