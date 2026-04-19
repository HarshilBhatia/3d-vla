"""
Launch RLBench/CoppeliaSim, reset a task scene, move the front camera to
many random (+ structured) positions, and capture proper rendered RGB images.

Requires COPPELIASIM_ROOT, LD_LIBRARY_PATH, QT_QPA_PLATFORM_PLUGIN_PATH
(see scripts/rlbench/README_RLBench.md) and a virtual display (xvfb-run).

Usage (from 3d-vla/):
    xvfb-run -a python scripts/rlbench/render_camera_positions.py \\
        --task close_jar \\
        --out  camera_render/ \\
        --n_random  200 \\
        --n_resets  3

    # Bimanual (PerAct2) scene:
    xvfb-run -a python scripts/rlbench/render_camera_positions.py \\
        --task bimanual_lift_tray --bimanual \\
        --out  camera_render/ --n_random 200

    # Use positions from explore_camera_positions.py:
    xvfb-run -a python scripts/rlbench/render_camera_positions.py \\
        --candidates_json camera_explore/configs.json \\
        --task close_jar --out camera_render/

    # Extract specific indices after viewing the grid pages:
    python scripts/rlbench/render_camera_positions.py \\
        --extract "5,23,41,99,134,177" --configs_json camera_render/configs.json

Outputs:
    <out>/summary_structured.png   -- all 10 structured positions on one page
    <out>/page_001.png ...         -- 20 candidates per page (browse in IDE)
    <out>/configs.json             -- all candidate params indexed by #XXX label
    <out>/candidate_XXXX_*/        -- per-candidate PNGs for all resets (--save_pngs)
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Camera sampling helpers (shared with explore_camera_positions.py) ───────

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def look_at(eye, target, world_up=np.array([0., 0., 1.])):
    """
    Camera-to-world rotation (3x3).
    Camera Z = forward, camera X = right, camera Y = down-in-image.
    """
    z = _normalize(np.asarray(target, float) - np.asarray(eye, float))
    if abs(float(np.dot(z, world_up))) > 0.99:
        world_up = np.array([0., 1., 0.])
    x = _normalize(np.cross(world_up, z))
    y = np.cross(z, x)   # right-handed: z × x = y  (det = +1)
    return np.column_stack([x, y, z])


def apply_roll(R, roll_deg):
    z_world = R[:, 2]
    return ScipyR.from_rotvec(z_world * np.deg2rad(roll_deg)).as_matrix() @ R


def R_to_quat_xyzw(R):
    """Convert 3x3 cam-to-world rotation matrix to [qx,qy,qz,qw]."""
    return ScipyR.from_matrix(R).as_quat()


def structured_cameras(center, radius, z_table):
    cx, cy, cz = center
    r = radius
    configs = [
        ("front",          [cx - r*1.8, cy,          cz + r*0.2]),
        ("front_high",     [cx - r*1.4, cy,          cz + r*0.9]),
        ("left_shoulder",  [cx - r*0.2, cy + r*1.5,  cz + r*1.3]),
        ("right_shoulder", [cx - r*0.2, cy - r*1.5,  cz + r*1.3]),
        ("overhead",       [cx,          cy,          cz + r*2.2]),
        ("left_side",      [cx + r*0.4,  cy + r*1.9,  cz + r*0.4]),
        ("right_side",     [cx + r*0.4,  cy - r*1.9,  cz + r*0.4]),
        ("back_high",      [cx + r*1.6,  cy,          cz + r*1.0]),
        ("front_low",      [cx - r*2.0, cy,          cz - r*0.2]),
        ("wrist_approx",   [cx + r*0.1, cy + r*0.3, cz + r*0.15]),
    ]
    cams = []
    for name, pos_list in configs:
        pos = np.array(pos_list, dtype=float)
        pos[2] = max(pos[2], z_table + 0.05)
        cams.append({"name": name, "pos": pos, "R": look_at(pos, center),
                     "rz": 0.0, "structured": True})
    return cams


def sample_random_cameras(center, radius, z_table, n, seed, jitter_m=0.05, roll_deg=20.0):
    rng   = np.random.default_rng(seed)
    r_min = radius * 0.4
    r_max = radius * 2.8
    z_min = z_table + 0.05
    z_max = center[2] + radius * 3.0

    cameras, attempts = [], 0
    while len(cameras) < n and attempts < n * 30:
        attempts += 1
        while True:
            dp   = rng.uniform(-r_max, r_max, 3)
            dist = np.linalg.norm(dp)
            if r_min <= dist <= r_max:
                break
        pos = center + dp
        if pos[2] < z_min or pos[2] > z_max:
            continue
        target = center + rng.normal(0, jitter_m, 3)
        roll   = float(rng.uniform(-roll_deg, roll_deg))
        R      = apply_roll(look_at(pos, target), roll)
        cameras.append({"name": "rand_{:04d}".format(len(cameras)),
                        "pos": pos, "R": R, "rz": roll, "structured": False})
    return cameras


# ── PNG grid output ─────────────────────────────────────────────────────────

def write_png_grids(candidates, out_dir, title="RLBench Camera Positions", n_cols=5):
    """
    Write page_001.png, page_002.png, ... (20 candidates per page) and
    summary_structured.png (structured candidates only).

    Each cell shows the rendered image with its #IDX label so you can
    cross-reference against configs.json.
    """
    n_per_page = n_cols * 4   # 4 rows per page
    n_pages    = max(1, (len(candidates) + n_per_page - 1) // n_per_page)
    saved      = []

    for p in range(n_pages):
        page = candidates[p * n_per_page : (p + 1) * n_per_page]
        n_rows = max(1, (len(page) + n_cols - 1) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2.6, n_rows * 3.0),
                                 facecolor="#0a0a14",
                                 squeeze=False)
        fig.subplots_adjust(wspace=0.04, hspace=0.45,
                            left=0.01, right=0.99, top=0.93, bottom=0.01)

        flat = axes.flat
        for j, cand in enumerate(page):
            ax   = next(flat)
            gidx = p * n_per_page + j
            img  = cand["render_img"]   # (H, W, 3) uint8

            ax.imshow(img)
            tag   = "[S]" if cand.get("structured") else ""
            label = "#{:03d} {} {}\n[{:.2f},{:.2f},{:.2f}]\nd={:.2f}m".format(
                gidx, tag, cand["name"][:14],
                cand["pos"][0], cand["pos"][1], cand["pos"][2],
                cand.get("dist", 0.0))
            ax.set_title(label, fontsize=5.5, color="#a0c4ff", pad=2)
            ax.axis("off")
            color = "#44aaff" if cand.get("structured") else "#2a2a50"
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

        for ax in flat:   # hide unused cells
            ax.axis("off")
            ax.set_facecolor("#0a0a14")

        fig.suptitle("{} — page {}/{} (#{}-#{})".format(
            title, p + 1, n_pages,
            p * n_per_page,
            min((p + 1) * n_per_page - 1, len(candidates) - 1)),
            color="white", fontsize=8, y=0.97)

        path = os.path.join(out_dir, "page_{:03d}.png".format(p + 1))
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(path)
        print("[PAGE] {}".format(path))

    # Summary: structured candidates only
    struct = [c for c in candidates if c.get("structured")]
    if struct:
        nc    = min(5, len(struct))
        nr    = max(1, (len(struct) + nc - 1) // nc)
        fig2, axes2 = plt.subplots(nr, nc,
                                   figsize=(nc * 3.0, nr * 3.5),
                                   facecolor="#0a0a14",
                                   squeeze=False)
        fig2.subplots_adjust(wspace=0.06, hspace=0.5,
                             left=0.01, right=0.99, top=0.90, bottom=0.01)
        flat2 = axes2.flat
        for j, cand in enumerate(struct):
            ax = next(flat2)
            ax.imshow(cand["render_img"])
            ax.set_title("{}\n[{:.2f},{:.2f},{:.2f}]\nd={:.2f}m".format(
                cand["name"],
                cand["pos"][0], cand["pos"][1], cand["pos"][2],
                cand.get("dist", 0.0)),
                fontsize=7, color="#a0c4ff", pad=3)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#44aaff")
                spine.set_linewidth(2)
        for ax in flat2:
            ax.axis("off"); ax.set_facecolor("#0a0a14")

        fig2.suptitle("Structured camera positions", color="white",
                      fontsize=10, y=0.96)
        summary = os.path.join(out_dir, "summary_structured.png")
        fig2.savefig(summary, dpi=130, bbox_inches="tight",
                     facecolor=fig2.get_facecolor())
        plt.close(fig2)
        saved.insert(0, summary)
        print("[SUMMARY] {}".format(summary))

    return saved


def write_configs_json(candidates, out_dir):
    """Write configs.json: list of all candidates with their #IDX, name, pos, roll."""
    records = []
    for i, c in enumerate(candidates):
        records.append({
            "idx":              i,
            "name":             c["name"],
            "structured":       bool(c.get("structured", False)),
            "pos":              [round(float(v), 4) for v in c["pos"]],
            "roll_deg":         round(float(c.get("rz", 0.0)), 2),
            "dist_from_center": round(float(c.get("dist", 0.0)), 4),
        })
    path = os.path.join(out_dir, "configs.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    return path


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Render random camera positions in RLBench for shortlisting."
    )
    p.add_argument("--task",           default="close_jar",
                   help="Task name (default: close_jar)")
    p.add_argument("--out",            default="camera_render")
    p.add_argument("--n_random",       type=int, default=200)
    p.add_argument("--n_resets",       type=int, default=3,
                   help="Scene resets per candidate (more = more variation coverage)")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--bimanual",       action="store_true",
                   help="Use bimanual (PerAct2) scene and task")
    p.add_argument("--image_size",     type=int, default=256)
    p.add_argument("--variation",      type=int, default=0,
                   help="Task variation index (default: 0)")
    p.add_argument("--save_pngs",      action="store_true",
                   help="Save all reset PNGs to per-candidate subdirectories")
    p.add_argument("--cols",           type=int, default=5,
                   help="Columns per grid page (default: 5)")
    p.add_argument("--candidates_json", default=None,
                   help="JSON file of candidate positions (overrides random sampling)")
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── RLBench imports (require COPPELIASIM_ROOT to be set) ────────────────
    try:
        from pyrep.const import RenderMode
        from rlbench.observation_config import ObservationConfig, CameraConfig
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
    except ImportError as e:
        sys.exit("[ERROR] RLBench/PyRep import failed: {}\n"
                 "Make sure COPPELIASIM_ROOT, LD_LIBRARY_PATH, "
                 "QT_QPA_PLATFORM_PLUGIN_PATH are set.".format(e))

    if args.bimanual:
        from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
        action_mode  = BimanualMoveArmThenGripper(
            BimanualEndEffectorPoseViaPlanning(), BimanualDiscrete())
        robot_setup  = "dual_panda"
        probe_cam    = "front"
        all_cam_names = ["front", "wrist_left", "wrist_right"]
    else:
        action_mode  = MoveArmThenGripper(JointVelocity(), Discrete())
        robot_setup  = "panda"
        probe_cam    = "front"
        all_cam_names = ["over_shoulder_left", "over_shoulder_right", "front"]

    # ── Build obs_config: only probe camera active ──────────────────────────
    off = CameraConfig(); off.set_all(False)
    on  = CameraConfig(rgb=True, depth=False, point_cloud=False, mask=False,
                       image_size=(args.image_size, args.image_size),
                       render_mode=RenderMode.OPENGL3)

    cam_cfgs = {}
    for name in all_cam_names:
        cam_cfgs[name] = on if name == probe_cam else off

    obs_config = ObservationConfig(
        camera_configs=cam_cfgs,
        joint_velocities=False, joint_positions=False, joint_forces=False,
        gripper_open=False, gripper_pose=False, task_low_dim_state=False,
    )

    # ── Launch environment ───────────────────────────────────────────────────
    print("[INFO] Launching RLBench ({})...".format(
          "bimanual" if args.bimanual else "unimanual"))
    env = Environment(action_mode, "", obs_config,
                      headless=True, robot_setup=robot_setup)
    env.launch()

    # ── Load task ────────────────────────────────────────────────────────────
    if args.bimanual:
        import importlib
        task_file = args.task
        class_name = "".join(w[0].upper() + w[1:] for w in task_file.split("_"))
        mod = importlib.import_module("rlbench.bimanual_tasks.{}".format(task_file))
        task_class = getattr(mod, class_name)
    else:
        from rlbench.backend.utils import task_file_to_task_class
        task_class = task_file_to_task_class(args.task)

    task_env = env.get_task(task_class)
    task_env.set_variation(args.variation)

    # ── Get workspace geometry from scene ────────────────────────────────────
    scene      = env._scene
    ws_pos     = np.array(scene._workspace.get_position())   # (3,) world XYZ
    ws_bbox    = scene._workspace.get_bounding_box()         # (6,) min/max
    ws_radius  = max(abs(ws_bbox[1] - ws_bbox[0]),
                     abs(ws_bbox[3] - ws_bbox[2])) / 2.0 + 0.2
    z_table    = ws_pos[2]
    center     = np.array([ws_pos[0], ws_pos[1], ws_pos[2] + 0.2])

    print("[INFO] Workspace: center={} radius={:.3f}m table_z={:.3f}m".format(
          center.round(3), ws_radius, z_table))

    # ── Probe camera sensor ──────────────────────────────────────────────────
    probe = scene.camera_sensors[probe_cam]
    # Enable explicit handling so we can force a re-render after set_pose
    # without stepping the physics simulation (which would move scene objects).
    probe.set_explicit_handling(1)

    # ── Build candidate list ─────────────────────────────────────────────────
    if args.candidates_json:
        with open(args.candidates_json) as f:
            raw = json.load(f)
        cams = []
        for c in raw:
            pos = np.array(c["pos"])
            roll = float(c.get("roll_deg", 0.0))
            R = apply_roll(look_at(pos, center), roll)
            cams.append({"name": c.get("name", "custom"),
                         "pos": pos, "R": R, "rz": roll, "structured": True})
        print("[INFO] Loaded {} candidates from {}".format(
              len(cams), args.candidates_json))
    else:
        cams  = structured_cameras(center, ws_radius, z_table)
        cams += sample_random_cameras(center, ws_radius, z_table,
                                      args.n_random, args.seed)
        print("[INFO] {} candidates ({} structured, {} random)".format(
              len(cams), sum(c["structured"] for c in cams),
              sum(not c["structured"] for c in cams)))

    # ── Render each candidate across n_resets ────────────────────────────────
    candidates = []
    rng = np.random.default_rng(args.seed)

    for i, cam in enumerate(cams):
        pos  = cam["pos"]
        R    = cam["R"]
        quat = R_to_quat_xyzw(R)                 # [qx, qy, qz, qw]
        pose = pos.tolist() + quat.tolist()       # 7-element PyRep pose

        renders = []
        for r_idx in range(args.n_resets):
            try:
                # Use different variations for diversity
                var = int(rng.integers(0, max(1, task_env.variation_count())))
                task_env.set_variation(var)
                _, _ = task_env.reset()
            except Exception:
                try:
                    task_env.set_variation(0)
                    _, _ = task_env.reset()
                except Exception:
                    pass

            # Move probe camera and force a fresh render at the new position.
            # handle_explicitly() re-renders only this sensor without stepping
            # physics (pyrep.step would move scene objects, causing inconsistency).
            probe.set_pose(pose)
            probe.handle_explicitly()
            rgb = probe.capture_rgb()                    # (H, W, 3) float [0,1]
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            renders.append(rgb)

        dist = float(np.linalg.norm(pos - center))
        euler_deg = ScipyR.from_matrix(R).as_euler("xyz", degrees=True)

        print("\n  [{:>4}/{}] {} | pos=[{:.3f},{:.3f},{:.3f}] | "
              "euler_xyz=[{:.1f},{:.1f},{:.1f}]deg | dist={:.2f}m".format(
              i + 1, len(cams), cam["name"],
              pos[0], pos[1], pos[2],
              euler_deg[0], euler_deg[1], euler_deg[2],
              dist), flush=True)

        # First reset image used for the grid thumbnail; save all resets as PNGs
        render_img = renders[0] if renders else np.zeros((256, 256, 3), dtype=np.uint8)

        if args.save_pngs and renders:
            safe = cam["name"].replace("/", "_")
            d = os.path.join(args.out, "candidate_{:04d}_{}".format(i, safe))
            os.makedirs(d, exist_ok=True)
            from PIL import Image as PILImage
            for j, img in enumerate(renders):
                PILImage.fromarray(img).save(
                    os.path.join(d, "reset_{:02d}.png".format(j)))
            # Save camera params alongside the PNGs
            info_lines = [
                "candidate  : {:04d}  {}".format(i, cam["name"]),
                "structured : {}".format(cam.get("structured", False)),
                "pos_xyz    : [{:.6f}, {:.6f}, {:.6f}]".format(
                    pos[0], pos[1], pos[2]),
                "euler_xyz  : [{:.3f}, {:.3f}, {:.3f}] deg".format(
                    euler_deg[0], euler_deg[1], euler_deg[2]),
                "roll_deg   : {:.3f}".format(float(cam.get("rz", 0.0))),
                "dist_m     : {:.4f}".format(dist),
                "quaternion : [{:.6f}, {:.6f}, {:.6f}, {:.6f}]  (xyzw)".format(
                    *quat.tolist()),
                "rot_matrix :",
            ] + ["  [{:.6f}, {:.6f}, {:.6f}]".format(*row) for row in R.tolist()]
            with open(os.path.join(d, "camera_info.txt"), "w") as fh:
                fh.write("\n".join(info_lines) + "\n")

        candidates.append({
            **cam,
            "render_img": render_img,
            "dist":       dist,
        })

        print("  [{:>4}/{}] {}".format(i + 1, len(cams), cam["name"]),
              end="\r", flush=True)

    print()

    env.shutdown()

    # Sort: structured first, then random
    candidates.sort(key=lambda c: (not c["structured"], c["name"]))

    pages  = write_png_grids(candidates, args.out,
                             title="RLBench: {}".format(args.task),
                             n_cols=args.cols)
    cfgp   = write_configs_json(candidates, args.out)

    print("\n[DONE] {} candidates".format(len(candidates)))
    print("  Browse : {}  (page_001.png … page_{:03d}.png)".format(
          args.out, len([p for p in pages if "page_" in p])))
    print("  Summary: {}".format(os.path.join(args.out, "summary_structured.png")))
    print("  Index  : {}".format(cfgp))
    print("\nWorkflow:")
    print("  1. Open summary_structured.png and page_XXX.png in the IDE image viewer")
    print("  2. Note the #IDX numbers of the configs you like")
    print("  3. Look up their params in configs.json (jq '.[5,23,41]' configs.json)")


if __name__ == "__main__":
    main()
