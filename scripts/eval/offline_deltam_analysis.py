"""Offline geometric-error analysis of the deltaM arms under extrinsics corruption.

Open-loop keypose prediction on ground-truth observations (the same
``run_inference=True`` path the trainer's validation uses), swept over a grid of
controlled camera-extrinsics corruptions. Success rate on 10 closed-loop
rollouts/cell resolves ~0.15; the val zarr has ~850 keyposes over 117 episodes,
so continuous position/rotation error on the same conditions is the higher-power
measurement of whether deltaM corrects miscalibration.

Three questions it answers, per checkpoint x condition x task:

1. Does deltaM reduce keypose *position* error under miscalibration, including
   where success rate does not move? (SR is threshold-y; L2 is continuous.)
2. Does that error grow more slowly with corruption magnitude — the correction
   signature, read off a slope rather than a binary outcome?
3. Does the predicted delta_M itself carry calibration information? The head is
   initialized at ~I, so ``||delta_M - I||_F`` is a readout of how much
   correction it thinks it needs. Correlating that against the *injected*
   perturbation magnitude tests whether the token senses miscalibration at all,
   independently of whether the downstream correction succeeds.

Per-arm breakdown is reported throughout: the bimanual EE-aux target is a
midpoint over ``ee_aux_cam_ids``, so R1c's error should be more symmetric across
the two arms than R1b's if the aux head is doing what it is supposed to.

Usage (one H200, ~20 min/checkpoint for the 7-condition grid)::

    python scripts/eval/offline_deltam_analysis.py \
        checkpoints=/path/orbital_miscal_base.pth,/path/orbital_miscal_deltam.pth \
        data_path=/k8s-nfs/harsvbha/3dfa/data/orbital_peract2/val.zarr \
        val_instructions=/k8s-nfs/harsvbha/3dfa/data/orbital_peract2/instructions.json \
        output_csv=results/deltam_offline/errors.csv \
        deltam_csv=results/deltam_offline/deltam_corr.csv \
        num_batches=100

Conditions swept (``conditions=`` to subset), matching the online eval grid:
``clean`` (no corruption), ``base`` (trained fixed per-group base, level 0),
``ood_base`` (held-out fixed base, same magnitude, different directions),
``n2`` / ``n5`` / ``n10`` / ``n15`` (trained base + random top-up of that
magnitude, the composition training used: ``T_rand @ T_base``).

Dataset type and model architecture are auto-detected from each checkpoint's
saved config, exactly as in the sibling offline-MSE scripts.
"""
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
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
    write_csv_rows,
)
from utils.trainers.base import relative_to_absolute

TRAIN_NOISE_FILE = "instructions/orbital_miscalibration_noise.json"
OOD_NOISE_FILE = "instructions/orbital_miscalibration_noise_ood.json"

# condition -> preprocessor miscal kwargs. `base` reproduces the checkpoints'
# training condition at level 0 (fixed per-group base, no random top-up); the
# nXX rows add the random magnitude on top, as `T_rand @ T_base`.
CONDITIONS = {
    "clean": {},
    "base": {"orbital_miscal_noise_level": "medium"},
    "ood_base": {
        "orbital_miscal_noise_level": "medium",
        "orbital_miscal_noise_file": OOD_NOISE_FILE,
    },
    "n2": {
        "orbital_miscal_noise_level": "medium",
        "miscal_max_angle_deg": 2.0,
        "miscal_max_translation_m": 0.02,
    },
    "n5": {
        "orbital_miscal_noise_level": "medium",
        "miscal_max_angle_deg": 5.0,
        "miscal_max_translation_m": 0.05,
    },
    "n10": {
        "orbital_miscal_noise_level": "medium",
        "miscal_max_angle_deg": 10.0,
        "miscal_max_translation_m": 0.10,
    },
    "n15": {
        "orbital_miscal_noise_level": "medium",
        "miscal_max_angle_deg": 15.0,
        "miscal_max_translation_m": 0.15,
    },
}

# Nominal injected rotation magnitude (deg) per condition, for the error-vs-magnitude
# slope. The fixed base contributes ~5.95 deg mean (seed 42, `medium`); the random
# top-up is uniform in [-max, max] per camera, so its expected |angle| is max/2.
BASE_ANGLE_DEG = 5.95
CONDITION_ANGLE_DEG = {
    "clean": 0.0,
    "base": BASE_ANGLE_DEG,
    "ood_base": BASE_ANGLE_DEG,
    "n2": BASE_ANGLE_DEG + 1.0,
    "n5": BASE_ANGLE_DEG + 2.5,
    "n10": BASE_ANGLE_DEG + 5.0,
    "n15": BASE_ANGLE_DEG + 7.5,
}

SCRIPT_KEYS = BASE_SCRIPT_KEYS | {
    "data_path",
    "conditions",
    "deltam_csv",
    "seed",
}

ERROR_HEADER = [
    "ckpt", "step", "condition", "injected_angle_deg", "task", "n_samples",
    "pos_l2_mean", "pos_l2_median", "pos_l2_arm0", "pos_l2_arm1", "pos_l2_arm_asym",
    "rot_deg_mean", "rot_deg_arm0", "rot_deg_arm1",
    "gripper_acc", "pos_acc_001", "pos_acc_005",
]
DELTAM_HEADER = [
    "ckpt", "step", "condition", "n_samples", "ncam",
    "dev_from_identity_mean", "dev_from_identity_std", "dev_per_cam",
    "injected_angle_per_cam", "corr_dev_vs_injected",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat_geodesic_deg(pred_q, gt_q):
    """Geodesic angle (deg) between two xyzw quaternion sets, sign-agnostic.

    Args:
        pred_q, gt_q: (..., 4) xyzw, not assumed normalized.
    """
    pred_q = pred_q / pred_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    gt_q = gt_q / gt_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (pred_q * gt_q).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def _per_arm_view(action, nhand):
    """Reshape a flat action tensor to (B, T, nhand, D) when the arms are packed
    into the last dim, otherwise return it unchanged.

    The bimanual loader yields (T, nhand, 8); the trainer's relative→absolute
    path collapses ``nhand`` for the relative case. Both layouts reach here, so
    normalize to an explicit ``nhand`` axis.
    """
    if action.dim() == 4:
        return action
    B, T, D = action.shape
    if nhand > 1 and D % nhand == 0:
        return action.view(B, T, nhand, D // nhand)
    return action.unsqueeze(2)


def _deltam_deviation(delta_M):
    """Per-camera ``||delta_M - I||_F`` and orthogonality residual ``||MᵀM - I||_F``.

    Args:
        delta_M: (B, ncam, d, d) or (B, d, d).
    Returns:
        (dev, orth) each (B, ncam).
    """
    if delta_M.dim() == 3:
        delta_M = delta_M.unsqueeze(1)
    M = delta_M.float()
    d = M.shape[-1]
    I = torch.eye(d, device=M.device).expand_as(M)
    dev = (M - I).flatten(-2).norm(dim=-1)
    orth = (M.transpose(-1, -2) @ M - I).flatten(-2).norm(dim=-1)
    return dev, orth


def _injected_angle_deg(T_noise):
    """Per-camera rotation magnitude (deg) of an injected ``(B, ncam, 4, 4)`` perturbation."""
    R = T_noise[..., :3, :3].float()
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))


class _DeltaMRecorder:
    """Capture every delta_M the head predicts, and the noise actually injected.

    ``dynamic_rope_from_camtoken`` re-predicts delta_M after every cross-attn and
    self-attn block, and the whole head runs once per denoising step, so
    ``_last_predicted_cam_params`` retains only the final prediction. Wrapping
    ``_predict_from_cam_feat`` — the single site where delta_M is born — records
    all of them, and averaging over predictions is a less arbitrary readout than
    taking the last.

    The injected perturbation is captured from ``_get_miscal_noise`` so the
    correlation is against the magnitude each *sample* actually received, not the
    condition's nominal setting.
    """

    def __init__(self, head, preprocessor):
        self._head = head
        self._preprocessor = preprocessor
        self._orig_predict = head._predict_from_cam_feat
        self._orig_noise = preprocessor._get_miscal_noise
        # Per-batch accumulators, reset by `start_batch`.
        self.batch_dev = []
        self.injected = None
        # Per-sample records, appended by `end_batch`.
        self.dev_samples = []      # (N, ncam) mean deviation over predictions
        self.angle_samples = []    # (N, ncam) injected rotation magnitude

    def __enter__(self):
        def predict(cam_feat):
            out = self._orig_predict(cam_feat)
            delta_M = out[1]
            if delta_M is not None:
                dev, _ = _deltam_deviation(delta_M)
                self.batch_dev.append(dev.detach())
            return out

        def get_noise(*a, **kw):
            T = self._orig_noise(*a, **kw)
            self.injected = None if T is None else _injected_angle_deg(T).detach()
            return T

        self._head._predict_from_cam_feat = predict
        self._preprocessor._get_miscal_noise = get_noise
        return self

    def __exit__(self, *exc):
        self._head._predict_from_cam_feat = self._orig_predict
        self._preprocessor._get_miscal_noise = self._orig_noise
        return False

    def start_batch(self):
        self.batch_dev = []
        self.injected = None

    def end_batch(self, batch_size):
        """Average this batch's predictions and pair them with the injected noise."""
        if not self.batch_dev:
            return
        dev = torch.stack(self.batch_dev, 0).mean(0)  # (B, ncam)
        self.dev_samples.append(dev.cpu().numpy())
        if self.injected is None:
            self.angle_samples.append(np.zeros_like(self.dev_samples[-1]))
        else:
            self.angle_samples.append(self.injected.cpu().numpy())

    def summary(self):
        """Aggregate deviation stats plus the deviation-vs-injected-angle correlation."""
        if not self.dev_samples:
            return {}
        dev = np.concatenate(self.dev_samples, 0)
        ang = np.concatenate(self.angle_samples, 0)
        # Correlate per (sample, camera): does the head deviate further from
        # identity when that camera was perturbed harder? Flat across cameras
        # would mean the token carries no calibration information.
        d, a = dev.reshape(-1), ang.reshape(-1)
        corr = float(np.corrcoef(d, a)[0, 1]) if d.std() > 0 and a.std() > 0 else float("nan")
        return {
            "n_samples": int(dev.shape[0]),
            "ncam": int(dev.shape[1]),
            "dev_mean": float(dev.mean()),
            "dev_std": float(dev.std()),
            "dev_per_cam": dev.mean(0).round(5).tolist(),
            "injected_angle_per_cam": ang.mean(0).round(3).tolist(),
            "corr_dev_vs_injected": corr,
        }


# ---------------------------------------------------------------------------
# One (checkpoint, condition) pass
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_condition(model, tokenizer, preprocessor, loader, num_batches, amp_dtype,
                  relative_action, nhand):
    """Accumulate per-task geometric errors and delta_M statistics.

    Returns ``(per_task, deltam)`` where ``per_task[task]`` holds lists of
    per-sample scalars and ``deltam`` holds the predicted-matrix statistics
    (empty when the checkpoint has no extrinsics predictor).
    """
    per_task = defaultdict(lambda: defaultdict(list))
    head = model.prediction_head
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
            action_mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")

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

            # (B, T, nhand)
            pos_l2 = (pred[..., :3] - gt[..., :3]).norm(dim=-1)
            rot_deg = _quat_geodesic_deg(pred[..., 3:-1], gt[..., 3:-1])
            grip_ok = ((pred[..., -1] >= 0.5) == (gt[..., -1] >= 0.5)).float()

            B = pos_l2.shape[0]
            tasks = sample["task"]
            # Average over the trajectory axis; keep the arm axis explicit.
            pos_a = pos_l2.mean(1)   # (B, nhand)
            rot_a = rot_deg.mean(1)
            grip_a = grip_ok.mean(1)
            for b in range(B):
                t = tasks[b] if isinstance(tasks[b], str) else str(tasks[b])
                d = per_task[t]
                d["pos_l2"].append(float(pos_a[b].mean()))
                d["rot_deg"].append(float(rot_a[b].mean()))
                d["gripper"].append(float(grip_a[b].mean()))
                d["pos_arm0"].append(float(pos_a[b, 0]))
                d["rot_arm0"].append(float(rot_a[b, 0]))
                if pos_a.shape[1] > 1:
                    d["pos_arm1"].append(float(pos_a[b, 1]))
                    d["rot_arm1"].append(float(rot_a[b, 1]))
                # Keypose-level threshold accuracies, over the whole trajectory.
                d["acc_001"].append(float((pos_l2[b] < 0.01).float().mean()))
                d["acc_005"].append(float((pos_l2[b] < 0.05).float().mean()))

            recorder.end_batch(B)

    return per_task, recorder.summary()


def summarize(per_task):
    """Collapse per-sample lists into the CSV row fields, plus an ALL row."""
    rows = {}
    pooled = defaultdict(list)
    for task, d in per_task.items():
        for k, v in d.items():
            pooled[k].extend(v)
        rows[task] = _row_from(d)
    rows["ALL"] = _row_from(pooled)
    return rows


def _row_from(d):
    def m(k):
        return float(np.mean(d[k])) if d.get(k) else float("nan")
    arm0, arm1 = m("pos_arm0"), m("pos_arm1")
    return {
        "n_samples": len(d["pos_l2"]),
        "pos_l2_mean": m("pos_l2"),
        "pos_l2_median": float(np.median(d["pos_l2"])) if d.get("pos_l2") else float("nan"),
        "pos_l2_arm0": arm0,
        "pos_l2_arm1": arm1,
        # Asymmetry between the arms: the EE-aux target is a midpoint, so a
        # smaller value means the head balanced the two arms.
        "pos_l2_arm_asym": abs(arm0 - arm1) if arm1 == arm1 else float("nan"),
        "rot_deg_mean": m("rot_deg"),
        "rot_deg_arm0": m("rot_arm0"),
        "rot_deg_arm1": m("rot_arm1"),
        "gripper_acc": m("gripper"),
        "pos_acc_001": m("acc_001"),
        "pos_acc_005": m("acc_005"),
    }


def main():
    custom, hydra_argv = extract_script_args(sys.argv[1:], SCRIPT_KEYS)

    checkpoints = parse_csv_list(custom.get("checkpoints"))
    data_path = custom.get("data_path")
    output_csv = Path(custom.get("output_csv", "results/deltam_offline/errors.csv"))
    deltam_csv = Path(custom.get("deltam_csv", "results/deltam_offline/deltam.csv"))
    num_batches = int(custom.get("num_batches", 100))
    seed = int(custom.get("seed", 0))
    cond_names = parse_csv_list(custom.get("conditions")) or list(CONDITIONS)

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint via checkpoints=path1.pth")
    if not data_path:
        raise ValueError("Pass the val zarr via data_path=/path/val.zarr")
    unknown = set(cond_names) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions {sorted(unknown)}; available: {list(CONDITIONS)}")

    args = load_args(hydra_argv)
    amp_dtype = pick_amp_dtype()
    print(f"AMP dtype: {amp_dtype}  conditions: {cond_names}  num_batches={num_batches}")

    for ckpt_path in checkpoints:
        args_copy = deepcopy(args)
        model, step = load_model(args_copy, ckpt_path)
        tokenizer = make_tokenizer(args_copy)
        nhand = 2 if getattr(args_copy, "bimanual", False) else 1
        name = Path(ckpt_path).stem
        print(
            f"\n=== {name} (step {step})  predict_extrinsics="
            f"{getattr(args_copy, 'predict_extrinsics', False)}  "
            f"predict_ee_aux={getattr(args_copy, 'predict_ee_aux', False)}  nhand={nhand}"
        )

        for cond in cond_names:
            # Every condition sees the same samples in the same order, and the
            # random top-up is seeded identically per checkpoint, so arm-to-arm
            # differences are not confounded by a different noise draw.
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(f"\n-- condition={cond}  {CONDITIONS[cond]}")
            preprocessor = make_preprocessor(args_copy, **CONDITIONS[cond])
            loader = make_loader(args_copy, data_path)

            per_task, deltam = run_condition(
                model, tokenizer, preprocessor, loader, num_batches,
                amp_dtype, args_copy.relative_action, nhand,
            )
            rows = summarize(per_task)
            all_row = rows["ALL"]
            print(
                f"   n={all_row['n_samples']}  pos_l2={all_row['pos_l2_mean']:.4f} m  "
                f"rot={all_row['rot_deg_mean']:.2f} deg  grip={all_row['gripper_acc']:.3f}  "
                f"arm_asym={all_row['pos_l2_arm_asym']:.4f}"
            )
            write_csv_rows(output_csv, ERROR_HEADER, [
                dict(ckpt=name, step=step, condition=cond,
                     injected_angle_deg=CONDITION_ANGLE_DEG[cond], task=task, **r)
                for task, r in sorted(rows.items())
            ])

            if deltam:
                print(
                    f"   delta_M: ||M-I||_F = {deltam['dev_mean']:.4f} "
                    f"+- {deltam['dev_std']:.4f}  per_cam={deltam['dev_per_cam']}\n"
                    f"            injected_deg_per_cam={deltam['injected_angle_per_cam']}  "
                    f"corr(dev, injected) = {deltam['corr_dev_vs_injected']:+.4f}"
                )
                write_csv_rows(deltam_csv, DELTAM_HEADER, [dict(
                    ckpt=name, step=step, condition=cond,
                    n_samples=deltam["n_samples"], ncam=deltam["ncam"],
                    dev_from_identity_mean=round(deltam["dev_mean"], 6),
                    dev_from_identity_std=round(deltam["dev_std"], 6),
                    dev_per_cam=str(deltam["dev_per_cam"]),
                    injected_angle_per_cam=str(deltam["injected_angle_per_cam"]),
                    corr_dev_vs_injected=round(deltam["corr_dev_vs_injected"], 6),
                )])

            del loader, preprocessor

        del model
        torch.cuda.empty_cache()

    print(f"\nErrors  -> {output_csv}")
    print(f"delta_M -> {deltam_csv}")


if __name__ == "__main__":
    main()
