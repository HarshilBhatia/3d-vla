"""
Visualize low_dim_obs.pkl from an orbital episode folder.

Usage:
    python scripts/helpers/vis_low_dim_obs.py <episode_folder>
    python scripts/helpers/vis_low_dim_obs.py <episode_folder> --no-plot
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def load_demo(folder):
    pkl = Path(folder) / "low_dim_obs.pkl"
    if not pkl.exists():
        sys.exit(f"Not found: {pkl}")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def print_summary(demo):
    print(f"\n{'='*60}")
    print(f"Demo: {len(demo)} timesteps  |  variation={demo.variation_number}  |  seed={'set' if demo.random_seed is not None else 'None'}")
    print(f"{'='*60}")

    obs0 = demo[0]

    # Print all non-None fields and their shape/value
    print("\n--- Fields present in each observation (from obs[0]) ---")
    for attr in sorted(vars(obs0).keys()):
        val = getattr(obs0, attr)
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            print(f"  {attr:35s}  shape={val.shape}  dtype={val.dtype}  range=[{val.min():.4f}, {val.max():.4f}]")
        elif isinstance(val, dict):
            print(f"  {attr:35s}  dict keys: {list(val.keys())}")
            for k, v in val.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k:31s}  shape={v.shape}  dtype={v.dtype}")
                else:
                    print(f"    {k:31s}  = {v}")
        else:
            print(f"  {attr:35s}  = {val}")

    # Gripper trajectory summary
    poses = np.array([demo[t].gripper_pose for t in range(len(demo))])  # (T, 7)
    opens = np.array([demo[t].gripper_open for t in range(len(demo))])  # (T,)
    print(f"\n--- Gripper trajectory ({len(demo)} steps) ---")
    print(f"  xyz range:  x=[{poses[:,0].min():.3f}, {poses[:,0].max():.3f}]  "
          f"y=[{poses[:,1].min():.3f}, {poses[:,1].max():.3f}]  "
          f"z=[{poses[:,2].min():.3f}, {poses[:,2].max():.3f}]")
    print(f"  gripper_open: min={opens.min():.2f}  max={opens.max():.2f}  "
          f"mean={opens.mean():.2f}")

    # Per-step table (first 5 + last 5)
    print(f"\n--- Per-step: gripper_pose (xyz) + gripper_open ---")
    indices = list(range(min(5, len(demo)))) + (
        list(range(max(5, len(demo)-5), len(demo))) if len(demo) > 5 else [])
    prev = -1
    for t in sorted(set(indices)):
        if prev != -1 and t != prev + 1:
            print("  ...")
        obs = demo[t]
        p = obs.gripper_pose
        print(f"  t={t:3d}  xyz=({p[0]:6.3f}, {p[1]:6.3f}, {p[2]:6.3f})  "
              f"quat=({p[3]:5.3f},{p[4]:5.3f},{p[5]:5.3f},{p[6]:5.3f})  "
              f"open={obs.gripper_open:.2f}")
        prev = t


def plot_demo(demo, folder):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("\nmatplotlib not available — skipping plots")
        return

    T = len(demo)
    poses = np.array([demo[t].gripper_pose for t in range(T)])
    opens = np.array([demo[t].gripper_open for t in range(T)])
    joints = np.array([demo[t].joint_positions for t in range(T)]) if demo[0].joint_positions is not None else None

    fig = plt.figure(figsize=(14, 5))
    title_base = Path(folder).name

    # 3D trajectory
    ax3d = fig.add_subplot(131, projection='3d')
    sc = ax3d.scatter(poses[:, 0], poses[:, 1], poses[:, 2],
                      c=np.arange(T), cmap='viridis', s=20)
    ax3d.plot(poses[:, 0], poses[:, 1], poses[:, 2], 'k-', alpha=0.3, linewidth=0.8)
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    ax3d.set_title(f'Gripper trajectory\n{title_base}')
    plt.colorbar(sc, ax=ax3d, label='timestep', shrink=0.6)

    # Gripper open/close
    ax2 = fig.add_subplot(132)
    ax2.plot(opens, color='steelblue')
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('timestep'); ax2.set_ylabel('gripper_open')
    ax2.set_title('Gripper open/close')
    ax2.set_ylim(-0.1, 1.1)

    # Joint positions
    ax3 = fig.add_subplot(133)
    if joints is not None:
        for j in range(joints.shape[1]):
            ax3.plot(joints[:, j], label=f'j{j}', alpha=0.8)
        ax3.set_xlabel('timestep'); ax3.set_ylabel('position (rad)')
        ax3.set_title('Joint positions')
        ax3.legend(fontsize=6, ncol=2)
    else:
        ax3.text(0.5, 0.5, 'joint_positions=None', ha='center', va='center')

    plt.tight_layout()
    out = Path(folder) / "vis_low_dim_obs.png"
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="episode folder containing low_dim_obs.pkl")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    demo = load_demo(args.folder)
    print_summary(demo)
    if not args.no_plot:
        plot_demo(demo, args.folder)


if __name__ == "__main__":
    main()
