"""Visualize RGB frames from the zarr dataset.

rgb shape: (N, n_cam, 3, H, W)  uint8 CHW
depth shape: (N, n_cam, H, W)   float16
"""
import argparse
import zarr
import numpy as np
import matplotlib.pyplot as plt


def show_rgb(z, episodes, out):
    rgb = z["rgb"]  # (N, n_cam, 3, H, W)
    N, n_cam = rgb.shape[:2]
    cam_names = ["front", "left_shoulder", "right_shoulder", "wrist"][:n_cam]

    fig, axes = plt.subplots(len(episodes), n_cam, figsize=(n_cam * 4, len(episodes) * 3))
    if len(episodes) == 1:
        axes = axes[np.newaxis, :]
    if n_cam == 1:
        axes = axes[:, np.newaxis]

    for row, ep in enumerate(episodes):
        frames = rgb[ep]  # (n_cam, 3, H, W)
        for col in range(n_cam):
            img = frames[col].transpose(1, 2, 0)  # CHW -> HWC
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(cam_names[col])
        axes[row, 0].set_ylabel(f"ep {ep}", rotation=0, labelpad=40, va="center")

    plt.suptitle("RGB frames", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, out)


def show_depth(z, episodes, out):
    depth = z["depth"]  # (N, n_cam, H, W)
    N, n_cam = depth.shape[:2]
    cam_names = ["front", "left_shoulder", "right_shoulder", "wrist"][:n_cam]

    fig, axes = plt.subplots(len(episodes), n_cam, figsize=(n_cam * 4, len(episodes) * 3))
    if len(episodes) == 1:
        axes = axes[np.newaxis, :]
    if n_cam == 1:
        axes = axes[:, np.newaxis]

    for row, ep in enumerate(episodes):
        frames = depth[ep].astype(np.float32)  # (n_cam, H, W)
        for col in range(n_cam):
            im = axes[row, col].imshow(frames[col], cmap="plasma")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(cam_names[col])
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel(f"ep {ep}", rotation=0, labelpad=40, va="center")

    plt.suptitle("Depth frames", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, out)


def _save_or_show(fig, out):
    if out:
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="/home/harshilb/data/orbital_mini.zarr")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=4)
    parser.add_argument("--mode", choices=["rgb", "depth", "both"], default="rgb")
    parser.add_argument("--out", default=None, help="Save path prefix (e.g. /tmp/out → out_rgb.png)")
    args = parser.parse_args()

    z = zarr.open(args.zarr, "r")
    N = z["rgb"].shape[0]
    episodes = list(range(args.episode, min(args.episode + args.num_episodes, N)))
    print(f"Showing episodes {episodes} from {args.zarr}")

    if args.mode in ("rgb", "both"):
        out = (args.out + "_rgb.png") if args.out else None
        show_rgb(z, episodes, out)

    if args.mode in ("depth", "both"):
        out = (args.out + "_depth.png") if args.out else None
        show_depth(z, episodes, out)


if __name__ == "__main__":
    main()
