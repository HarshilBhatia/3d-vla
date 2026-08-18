"""Asymmetric per-camera miscalibration — offline keypose error + delta_M response.

Tier-3 proposal T3-1 from ``docs/status/deltam_advantage_analysis.md``. The Tier-2
result there is that ``||delta_M - I||_F`` only responds to injected corruption on
cameras 0 and 1, which are exactly ``ee_aux_cam_ids``. That predicts a
**camera x arm interaction**: corrupt one camera at a time and R1c (deltaM +
EE-aux) should hold up better than R1a/R1b when the corrupted camera is one it
supervises, and no better when it is not.

Where :mod:`scripts.eval.offline_deltam_analysis` sweeps a *global* corruption
magnitude over all cameras, this script corrupts exactly **one** camera and leaves
the rest clean. The corruption is a fixed rotation+translation (exact magnitude,
random direction), and each (camera, magnitude) cell is repeated over several
independent directions because single-direction axis sensitivity was a known
artifact of the closed-loop cells.

Conditions per checkpoint::

    clean                                  1 pass
    cam c in {0,1,2,3} x {5deg+5cm, 10deg+10cm} x n_directions

Output is **per-sample**, not aggregated: one row per (pass, val sample) with the
episode id, the task, per-arm position/rotation/gripper error, and the per-camera
``||delta_M - I||_F``. Aggregation, bootstrap CIs over episodes, and the
delta_M response matrix are all built downstream by
:mod:`scripts.eval.analyze_asym_miscal`, so the expensive inference is run once.

Usage (one H200)::

    python scripts/eval/offline_asym_miscal_analysis.py \\
        checkpoints=$CK/base.pth,$CK/deltam.pth,$CK/deltam_eeaux.pth \\
        arm_names=R1a,R1b,R1c \\
        data_path=/k8s-nfs/harsvbha/3dfa/data/orbital_peract2/val.zarr \\
        samples_npz=results/asym_miscal/samples.npz \\
        n_directions=3 num_batches=100 \\
        data=orbital_peract2_nfs bimanual=true dataset=OrbitalPeract2 \\
        num_history=3 batch_size_val=64 num_workers=8

``arm_names`` is optional but recommended: every checkpoint in this experiment is
named ``interm_step_100000.pth``, so the file stem does not identify the arm.
"""
import math
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.eval.eval_utils import (
    BASE_SCRIPT_KEYS,
    extract_script_args,
    load_args,
    load_model,
    make_loader,
    make_preprocessor,
    make_tokenizer,
    parse_csv_list,
    pick_amp_dtype,
)
from scripts.eval.offline_deltam_analysis import (
    _DeltaMRecorder,
    _per_arm_view,
    _quat_geodesic_deg,
)
from utils.trainers.base import relative_to_absolute

# (label, rotation deg, translation m). Both magnitudes are well above the ~3deg
# random top-up the checkpoints saw in training, and 10deg is past the point where
# R1a's closed-loop retention crosses 50% (8.7deg).
MAGNITUDES = [("m5", 5.0, 0.05), ("m10", 10.0, 0.10)]

SCRIPT_KEYS = BASE_SCRIPT_KEYS | {
    "data_path",
    "samples_npz",
    "arm_names",
    "cameras",
    "n_directions",
    "seed",
}


class _DemoIdDataset(Dataset):
    """Inject the zarr's ``demo_id`` so downstream bootstrap can cluster by episode.

    Requires ``chunk_size=1`` (one dataset index per zarr row) and that no row is
    skipped by the base dataset's missing-instruction loop — both hold for the
    orbital PerAct2 val zarr.
    """

    def __init__(self, base_dataset, zarr_root):
        self._base = base_dataset
        z = zarr.open(str(zarr_root), "r")
        self._demo_id = np.array(z["demo_id"][:], dtype=np.int64)
        self._N = len(self._demo_id)
        print(
            f"[demo_id] zarr={Path(zarr_root).name}  N={self._N}  "
            f"n_episodes={len(np.unique(self._demo_id))}",
            flush=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        sample = self._base[idx]
        sample["demo_id"] = torch.tensor(
            [self._demo_id[int(idx) % self._N]], dtype=torch.long
        )
        return sample


# ---------------------------------------------------------------------------
# Single-camera corruption
# ---------------------------------------------------------------------------

def _rodrigues(axis, angle_rad):
    """(3,3) rotation from a unit ``axis`` numpy array and an angle in radians."""
    kx, ky, kz = axis
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def single_camera_noise_T(cam, ncam, angle_deg, translation_m, rng):
    """``(ncam, 4, 4)`` perturbation: identity everywhere except camera ``cam``.

    The corrupted camera gets an *exact* ``angle_deg`` rotation about a uniformly
    random axis and an *exact* ``translation_m`` displacement along an independent
    uniformly random direction. Fixed for the whole pass, so every val sample sees
    the same miscalibration — the corruption is a property of the condition, not
    per-sample noise.
    """
    T = np.tile(np.eye(4), (ncam, 1, 1))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    t_dir = rng.normal(size=3)
    t_dir /= np.linalg.norm(t_dir)
    T[cam, :3, :3] = _rodrigues(axis, math.radians(angle_deg))
    T[cam, :3, 3] = t_dir * translation_m
    return torch.from_numpy(T).float()


class _FixedNoiseInjector:
    """Replace the preprocessor's noise sampler with a fixed ``(ncam,4,4)`` transform.

    Installed *before* :class:`_DeltaMRecorder` wraps ``_get_miscal_noise``, so the
    recorder still sees (and logs) the injected per-camera rotation magnitudes.
    Passing ``None`` leaves the preprocessor clean.
    """

    def __init__(self, preprocessor, noise_T):
        self._preprocessor = preprocessor
        self._noise_T = noise_T
        self._orig = preprocessor._get_miscal_noise

    def __enter__(self):
        noise_T = self._noise_T

        def get_noise(B, ncam, device, dtype, camera_group=None, task=None):
            if noise_T is None:
                return None
            if noise_T.shape[0] != ncam:
                raise ValueError(
                    f"noise_T has {noise_T.shape[0]} cameras but the batch has {ncam}"
                )
            return noise_T.to(device=device, dtype=dtype).unsqueeze(0).expand(
                B, ncam, 4, 4
            )

        self._preprocessor._get_miscal_noise = get_noise
        return self

    def __exit__(self, *exc):
        self._preprocessor._get_miscal_noise = self._orig
        return False


# ---------------------------------------------------------------------------
# One pass over the val set
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_pass(model, tokenizer, preprocessor, loader, num_batches, amp_dtype,
             relative_action, nhand, noise_T):
    """Per-sample keypose errors and per-camera delta_M deviation for one condition.

    Returns a dict of 1-D arrays (plus ``dev`` of shape ``(N, ncam)``), all aligned
    on the same sample order.
    """
    out = {k: [] for k in (
        "demo_id", "task", "pos_arm0", "pos_arm1", "rot_arm0", "rot_arm1",
        "grip_arm0", "grip_arm1", "acc001_arm0", "acc001_arm1",
    )}
    head = model.prediction_head

    with _FixedNoiseInjector(preprocessor, noise_T):
        recorder = _DeltaMRecorder(head, preprocessor)
        with recorder:
            for i, sample in tqdm(enumerate(loader), total=num_batches, desc="  batches"):
                if i >= num_batches:
                    break
                recorder.start_batch()

                action = preprocessor.process_actions(sample["action"])
                proprio = preprocessor.process_proprio(sample["proprioception"])
                rgbs, pcds = preprocessor.process_obs(
                    sample["rgb"], sample.get("rgb2d"), sample["depth"],
                    sample["extrinsics"], sample["intrinsics"],
                    augment=False, task=sample["task"],
                    camera_group=sample.get("camera_group"),
                )
                instr = sample["instr"]
                if tokenizer is not None:
                    instr = tokenizer(instr).cuda(non_blocking=True)
                action_mask = torch.zeros(
                    action.shape[:-1], dtype=torch.bool, device="cuda"
                )

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred_action = model(
                        action, action_mask, rgbs, None, pcds, instr, proprio,
                        run_inference=True,
                    )

                gt_action = action
                if relative_action:
                    prop = proprio[:, :, 0] if proprio.dim() == 4 else proprio
                    pred_action = relative_to_absolute(pred_action[:, :, 0], prop)
                    gt_action = relative_to_absolute(gt_action[:, :, 0], prop)

                pred = _per_arm_view(pred_action.float(), nhand)
                gt = _per_arm_view(gt_action.float(), nhand)

                # (B, T, nhand) -> mean over the trajectory axis, arms kept explicit.
                pos_l2 = (pred[..., :3] - gt[..., :3]).norm(dim=-1)
                rot_deg = _quat_geodesic_deg(pred[..., 3:-1], gt[..., 3:-1])
                grip_ok = ((pred[..., -1] >= 0.5) == (gt[..., -1] >= 0.5)).float()
                acc001 = (pos_l2 < 0.01).float()

                pos_a = pos_l2.mean(1).cpu().numpy()
                rot_a = rot_deg.mean(1).cpu().numpy()
                grip_a = grip_ok.mean(1).cpu().numpy()
                acc_a = acc001.mean(1).cpu().numpy()

                B = pos_a.shape[0]
                out["demo_id"].append(sample["demo_id"].reshape(B).numpy())
                out["task"].extend(str(t) for t in sample["task"])
                for arm in range(2):
                    # nhand==1 would mean a non-bimanual checkpoint; this experiment
                    # is bimanual-only, so index both arms directly.
                    out[f"pos_arm{arm}"].append(pos_a[:, arm])
                    out[f"rot_arm{arm}"].append(rot_a[:, arm])
                    out[f"grip_arm{arm}"].append(grip_a[:, arm])
                    out[f"acc001_arm{arm}"].append(acc_a[:, arm])

                recorder.end_batch(B)

    rec = {"task": np.array(out.pop("task"))}
    rec.update({k: np.concatenate(v) for k, v in out.items()})
    rec["dev"] = (
        np.concatenate(recorder.dev_samples, 0)
        if recorder.dev_samples
        else np.zeros((len(rec["task"]), 0), dtype=np.float32)
    )
    rec["injected_deg"] = (
        np.concatenate(recorder.angle_samples, 0)
        if recorder.angle_samples
        else np.zeros((len(rec["task"]), 0), dtype=np.float32)
    )
    return rec


def main():
    custom, hydra_argv = extract_script_args(sys.argv[1:], SCRIPT_KEYS)

    checkpoints = parse_csv_list(custom.get("checkpoints"))
    arm_names = parse_csv_list(custom.get("arm_names")) or [
        Path(p).stem for p in checkpoints
    ]
    data_path = custom.get("data_path")
    samples_npz = Path(custom.get("samples_npz", "results/asym_miscal/samples.npz"))
    num_batches = int(custom.get("num_batches", 100))
    n_directions = int(custom.get("n_directions", 3))
    seed = int(custom.get("seed", 0))
    cameras = [int(c) for c in (parse_csv_list(custom.get("cameras")) or ["0", "1", "2", "3"])]

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint via checkpoints=path1.pth")
    if not data_path:
        raise ValueError("Pass the val zarr via data_path=/path/val.zarr")
    if len(arm_names) != len(checkpoints):
        raise ValueError(
            f"arm_names has {len(arm_names)} entries for {len(checkpoints)} checkpoints"
        )

    args = load_args(hydra_argv)
    amp_dtype = pick_amp_dtype()

    ncam = 4
    # Corruption directions are drawn from a seed that depends only on
    # (camera, magnitude, direction index), so every arm is evaluated against
    # exactly the same set of miscalibrations.
    conditions = [("clean", -1, "clean", -1, None)]
    for cam in cameras:
        for mag_label, ang, trans in MAGNITUDES:
            for d in range(n_directions):
                rng = np.random.default_rng(
                    (seed * 1000003) ^ (cam * 9176) ^ (int(ang) * 131) ^ d
                )
                conditions.append((
                    f"cam{cam}_{mag_label}_d{d}", cam, mag_label, d,
                    single_camera_noise_T(cam, ncam, ang, trans, rng),
                ))
    print(
        f"AMP dtype: {amp_dtype}  ncam={ncam}  cameras={cameras}  "
        f"n_directions={n_directions}  passes/ckpt={len(conditions)}",
        flush=True,
    )

    records = {}
    for arm, ckpt_path in zip(arm_names, checkpoints):
        args_copy = deepcopy(args)
        model, step = load_model(args_copy, ckpt_path)
        tokenizer = make_tokenizer(args_copy)
        nhand = 2 if getattr(args_copy, "bimanual", False) else 1
        if nhand != 2:
            raise ValueError("This analysis is bimanual-only; pass bimanual=true")
        print(
            f"\n=== {arm} ({Path(ckpt_path).name}, step {step})  predict_extrinsics="
            f"{getattr(args_copy, 'predict_extrinsics', False)}  "
            f"predict_ee_aux={getattr(args_copy, 'predict_ee_aux', False)}  "
            f"ee_aux_cam_ids={getattr(args_copy, 'ee_aux_cam_ids', None)}",
            flush=True,
        )

        # One loader and one preprocessor per checkpoint. The corruption is
        # injected by swapping the preprocessor's noise sampler per pass, so
        # neither object is condition-dependent — and rebuilding a
        # persistent_workers loader ~25x per checkpoint exhausts the file-descriptor
        # limit (Errno 24) well before the sweep finishes.
        preprocessor = make_preprocessor(args_copy)
        loader = make_loader(
            args_copy, data_path, chunk_size=1, dataset_wrapper=_DemoIdDataset
        )

        for cond_name, cam, mag_label, direction, noise_T in conditions:
            # Same sample order and same denoising noise for every (arm, condition),
            # so differences are attributable to the corruption alone.
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(f"\n-- {arm} / {cond_name}", flush=True)
            rec = run_pass(
                model, tokenizer, preprocessor, loader, num_batches, amp_dtype,
                args_copy.relative_action, nhand, noise_T,
            )
            pos = 0.5 * (rec["pos_arm0"] + rec["pos_arm1"])
            print(
                f"   n={len(pos)}  pos_l2={pos.mean():.4f} m  "
                f"arm0={rec['pos_arm0'].mean():.4f}  arm1={rec['pos_arm1'].mean():.4f}  "
                f"dev_per_cam={rec['dev'].mean(0).round(5).tolist() if rec['dev'].size else 'n/a'}",
                flush=True,
            )

            prefix = f"{arm}|{cond_name}|"
            records[prefix + "meta"] = np.array(
                [arm, cond_name, str(cam), mag_label, str(direction), str(step)]
            )
            for k, v in rec.items():
                records[prefix + k] = v
            # Written after every pass so a preempted run is still usable.
            samples_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(samples_npz, **records)

        del loader, preprocessor, model
        torch.cuda.empty_cache()

    print(f"\nPer-sample records -> {samples_npz}")


if __name__ == "__main__":
    main()
