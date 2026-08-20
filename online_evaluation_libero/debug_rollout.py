"""Isolate eval failure: (A) servo GT keyposes from a demo — tests the
controller; (B) compare model predictions to GT keyposes on the same frames —
tests the model. Run like evaluate_policy.py (same env vars)."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": k.get("weights_only", False)})

import zarr
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from online_evaluation_rlbench.evaluate_policy import load_models
from online_evaluation_libero.evaluate_policy import (
    Actioner, get_obs_tensors, eef_state, servo_to_keypose, SETTLE_STEPS, CAMERAS, IM_SIZE)
from utils.hydra_utils import get_config, get_config_path


def main():
    args = get_config(overrides=sys.argv[1:], config_name="config",
                      config_path=get_config_path())
    z = zarr.open_group(str(args.eval_data_dir), "r")
    demo_ids = np.array(z["demo_id"])
    task_ids = np.array(z["task_id"])
    # first demo of task 0
    sel = np.where(task_ids == 0)[0]
    d0 = demo_ids[sel[0]]
    rows = np.where(demo_ids == d0)[0]
    print(f"demo {d0}: rows {rows.tolist()}")
    gt_actions = np.array(z["action"][rows[0]:rows[-1] + 1, :, 0])  # (T, A, 8)

    b = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = b.get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_heights=IM_SIZE, camera_widths=IM_SIZE, camera_depths=True)
    env.seed(0)

    # ---- A: GT keypose replay through the servo --------------------------
    # The zarr was built from demo_0 of the HDF5; init from the demo's own
    # first sim state so GT keyposes correspond to the scene.
    import h5py
    h5 = h5py.File(os.path.join(get_libero_path("datasets"),
                                b.get_task_demonstration(0)), "r")
    states0 = h5["data/demo_0/states"][0]
    env.reset()
    obs = env.set_init_state(states0)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(np.zeros(7))

    print("\n=== A: GT keypose replay ===")
    go_cur = True  # episode starts with gripper open
    for t, act in enumerate(gt_actions):
        kp = act[-1] if act.ndim == 2 else act  # last waypoint of segment
        target = kp[:7].astype(np.float64)
        target[3:7] /= np.linalg.norm(target[3:7]) + 1e-8
        go = bool(kp[7] > 0.5)
        obs, done, succ = servo_to_keypose(env, obs, target, go,
                                           gripper_open_cur=go_cur)
        go_cur = go
        reach = np.linalg.norm(obs["robot0_eef_pos"] - target[:3])
        print(f"kp{t}: tgt={target[:3].round(3)} reached_err={reach:.3f} "
              f"grip_open={go} success={succ}")
        if succ:
            print("GT replay SUCCESS — controller OK")
            break
    else:
        print("GT replay FAILED — controller/keypose mismatch is the problem")

    # ---- B: model prediction vs GT on the initial frame ------------------
    print("\n=== B: model prediction on initial frame ===")
    model = load_models(args)
    actioner = Actioner(model, backbone=getattr(args, "text_backbone", None) or args.backbone)
    actioner.load_episode(task.language)
    env.reset()
    obs = env.set_init_state(states0)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(np.zeros(7))
    rgbs, pcds = get_obs_tensors(env, obs)
    print("pcd range:", pcds.min().item(), pcds.max().item())
    nhist = int(getattr(args, "num_history", 1))
    gr = torch.from_numpy(np.stack([eef_state(obs, 1.0)] * nhist)[None]).float().cuda()
    plen = int(getattr(args, "prediction_len", 1) or 1)
    pred = actioner.predict(rgbs, pcds, gr, prediction_len=plen)[0].cpu().numpy()
    gt0 = gt_actions[0][-1] if gt_actions[0].ndim == 2 else gt_actions[0]
    print("pred[last]:", pred[-1].round(3))
    print("gt kp0    :", gt0.round(3))
    print("pos err   :", np.linalg.norm(pred[-1, :3] - gt0[:3]).round(4))
    env.close()


if __name__ == "__main__":
    main()
