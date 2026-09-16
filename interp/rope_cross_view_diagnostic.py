"""Offline cross-view RoPE geometry diagnostic for a Delta-M checkpoint.

This is deliberately a diagnostic, not a loss or a policy evaluation.  It draws
ordinary (unmatched) external--wrist point-token pairs from held-out depth maps
and compares the relative rotary operators for clean, corrupted, and
Delta-M-corrected external positions.  There is no cross-view correspondence or
nearest-neighbour matching in this experiment.

The implementation follows the *actual* Delta-M implementation: Delta-M mixes
the six raw 3-D sin/cos channels before a RoPE operator is made.  Thus the
reported corrected term is ``R_delta(hat_p_E)^T R(p_C)``, not an algebraic
post-multiplication of a final RoPE matrix.

Example (one GPU)::

    python interp/rope_cross_view_diagnostic.py \
      checkpoints=/path/to/last.pth \
      data_path=/grogu/user/harshilb/multi_cam/val.zarr \
      output_csv=results/rope_cross_view.csv \
      num_batches=100 pairs_per_camera=256 \
      perturbation_noise_fixed_rot_deg=3 perturbation_noise_fixed_trans_m=0.01 \
      'miscal_cameras=[0,1]'

The perturbation is sampled once per batch item and is applied only to cameras
listed by ``miscal_cameras``.  It is not composed with a file-based fixed
bias: leave ``miscal_mode=none`` for this one-noise protocol.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
)


SCRIPT_KEYS = BASE_SCRIPT_KEYS | {"data_path", "pairs_per_camera", "seed"}
HEADER = [
    "ckpt", "step", "prediction_index", "external_camera", "wrist_camera",
    "n_pairs", "clean_error", "uncorrected_error", "corrected_error",
    "correction_gain",
]


class LayerDeltaMRecorder:
    """Record every Delta-M prediction, preserving its dynamic-layer order.

    This uses the same interception point as the existing offline Delta-M
    analysis, but retains matrices rather than reducing them to ||M-I||.
    """
    def __init__(self, head):
        self.head = head
        self.orig_predict = head._predict_from_cam_feat
        self.calls = []

    def __enter__(self):
        def predict(cam_feat):
            out = self.orig_predict(cam_feat)
            if out[1] is not None:
                self.calls.append(out[1].detach().float())
            return out
        self.head._predict_from_cam_feat = predict
        return self

    def __exit__(self, *exc):
        self.head._predict_from_cam_feat = self.orig_predict
        return False

    def start_batch(self):
        self.calls = []


def _sample_camera_indices(pcd, cam, count, generator):
    """Sample valid pixel indices from one camera's current depth map.

    ``pcd`` is ``(B,ncam,3,H,W)``.  We sample independently for each item and
    discard non-finite / zero-depth pixels; zero is an invalid depth sentinel.
    """
    points = pcd[:, cam].flatten(2).transpose(1, 2)  # (B, HW, 3)
    B, _, _ = points.shape
    chosen = []
    for b in range(B):
        valid = torch.isfinite(points[b]).all(-1) & (points[b].norm(dim=-1) > 1e-6)
        idx = valid.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            # Keep shape stable; these are only used for a pathological row.
            chosen.append(torch.zeros(count, device=pcd.device, dtype=torch.long))
            continue
        draw = idx[torch.randint(idx.numel(), (count,), device=pcd.device, generator=generator)]
        chosen.append(draw)
    return torch.stack(chosen)


def _gather_camera_points(pcd, cam, indices):
    """Gather identical image tokens from clean and corrupted camera clouds."""
    points = pcd[:, cam].flatten(2).transpose(1, 2)
    return torch.gather(points, 1, indices[..., None].expand(-1, -1, 3))


def _relative_frobenius(query_code, key_code, target_code):
    """``||R_q^T R_k - R_target||_F`` without materialising dense RoPE matrices.

    Codes are ``(B,S,C,2)`` from ``RotaryPositionEncoding3D``.  Adjacent C
    entries carry the same phase, so one entry per 2-D rotary block suffices.
    """
    cq, sq = query_code[..., 0][:, :, 0::2], query_code[..., 1][:, :, 0::2]
    ck, sk = key_code[..., 0][:, :, 0::2], key_code[..., 1][:, :, 0::2]
    ct, st = target_code[..., 0][:, :, 0::2], target_code[..., 1][:, :, 0::2]
    # R(q)^T R(k) = R(k-q): (cos, sin) below parameterize that 2x2 block.
    cos_rel = cq * ck + sq * sk
    sin_rel = cq * sk - sq * ck
    # Each 2x2 rotation contributes 2[(dc)^2 + (ds)^2] to Frobenius^2.
    return (2.0 * ((cos_rel - ct).square() + (sin_rel - st).square()).sum(-1)).sqrt()


def _codes(rope, points, delta_m=None):
    """Build implementation-faithful RoPE codes for ``(B,S,3)`` positions."""
    base = rope._compute_sincos_base(points.float())
    return rope._finalize_from_base(base, delta_M=delta_m)


@torch.inference_mode()
def run_checkpoint(model, tokenizer, preprocessor, loader, num_batches, pairs, seed):
    head = model.prediction_head
    rope = head.relative_pe_layer
    generator = torch.Generator(device="cuda").manual_seed(seed)
    stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    amp_dtype = pick_amp_dtype()

    with LayerDeltaMRecorder(head) as recorder:
        for i, sample in tqdm(enumerate(loader), total=num_batches, desc="RoPE pairs"):
            if i >= num_batches:
                break
            recorder.start_batch()

            # Clean cloud is formed before the normal preprocessor call.  The
            # caller config must use only random/fixed perturbation, not a base
            # table, so temporarily disable _get_miscal_noise for this pass.
            original_noise = preprocessor._get_miscal_noise
            preprocessor._get_miscal_noise = lambda *a, **kw: None
            _, clean_pcd = preprocessor.process_obs(
                sample["rgb"], sample.get("rgb2d"), sample["depth"],
                sample["extrinsics"], sample["intrinsics"], augment=False,
                task=sample["task"], camera_group=sample.get("camera_group"),
            )
            preprocessor._get_miscal_noise = original_noise
            rgbs, corrupt_pcd = preprocessor.process_obs(
                sample["rgb"], sample.get("rgb2d"), sample["depth"],
                sample["extrinsics"], sample["intrinsics"], augment=False,
                task=sample["task"], camera_group=sample.get("camera_group"),
            )

            action = preprocessor.process_actions(sample["action"])
            proprio = preprocessor.process_proprio(sample["proprioception"])
            instr = sample["instr"]
            if tokenizer is not None:
                instr = tokenizer(instr).cuda(non_blocking=True)
            mask = torch.zeros(action.shape[:-1], dtype=torch.bool, device="cuda")
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                model(action, mask, rgbs, None, corrupt_pcd, instr, proprio, run_inference=True)

            if not recorder.calls:
                raise RuntimeError("No Delta-M predictions were captured; checkpoint is not a Delta-M model.")
            # History is flattened by process_obs; only latest current-frame cloud
            # is used by the decoder and by this diagnostic.
            if clean_pcd.ndim == 5 and sample["rgb"].ndim == 6:
                B, K = sample["rgb"].shape[:2]
                clean_pcd = clean_pcd.view(B, K, *clean_pcd.shape[1:])[:, -1]
                corrupt_pcd = corrupt_pcd.view(B, K, *corrupt_pcd.shape[1:])[:, -1]

            for ext in (0, 1):
                # The external clean/corrupted positions must describe exactly
                # the same token.  Sampling them independently would turn this
                # into a point-distribution comparison instead of the stated
                # per-token-pair geometry diagnostic.
                ext_indices = _sample_camera_indices(clean_pcd, ext, pairs, generator)
                p_clean = _gather_camera_points(clean_pcd, ext, ext_indices)
                p_bad = _gather_camera_points(corrupt_pcd, ext, ext_indices)
                clean_code = _codes(rope, p_clean)
                bad_code = _codes(rope, p_bad)
                for wrist in (2, 3):
                    wrist_indices = _sample_camera_indices(clean_pcd, wrist, pairs, generator)
                    p_wrist = _gather_camera_points(clean_pcd, wrist, wrist_indices)
                    wrist_code = _codes(rope, p_wrist)
                    target = _relative_frobenius(clean_code, wrist_code, _relative_operator_code(clean_code, wrist_code))
                    # Use the clean relative operator as target represented by its
                    # (cos,sin) code pair; see helper below.
                    uncorrected = _relative_frobenius(bad_code, wrist_code, _relative_operator_code(clean_code, wrist_code))
                    for j, delta in enumerate(recorder.calls):
                        corrected_code = _codes(rope, p_bad, delta[:, ext])
                        corrected = _relative_frobenius(corrected_code, wrist_code, _relative_operator_code(clean_code, wrist_code))
                        key = (j, ext, wrist)
                        s = stats[key]
                        s[0] += float(target.sum())
                        s[1] += float(uncorrected.sum())
                        s[2] += float(corrected.sum())
                        s[3] += int(target.numel())
    return stats


def _relative_operator_code(query_code, key_code):
    """Encode the clean relative operator as cos/sin-shaped code for comparison."""
    cq, sq = query_code[..., 0], query_code[..., 1]
    ck, sk = key_code[..., 0], key_code[..., 1]
    return torch.stack([cq * ck + sq * sk, cq * sk - sq * ck], dim=-1)


def main():
    custom, hydra_argv = extract_script_args(sys.argv[1:], SCRIPT_KEYS)
    required = ("checkpoints", "data_path", "output_csv")
    missing = [k for k in required if k not in custom]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    args = load_args(hydra_argv)
    args.data_path = custom["data_path"]
    # Explicitly enforce the single sampled-noise evaluation protocol.
    args.miscal_group_level = None
    args.miscal_camera_groups = None
    preprocessor = make_preprocessor(args)
    loader = make_loader(args, args.data_path, split="val")
    tokenizer = make_tokenizer(args)
    pairs = int(custom.get("pairs_per_camera", 256))
    batches = int(custom.get("num_batches", 100))
    seed = int(custom.get("seed", 0))
    rows = []
    for checkpoint in parse_csv_list(custom["checkpoints"]):
        model, step = load_model(args, checkpoint)
        stats = run_checkpoint(model, tokenizer, preprocessor, loader, batches, pairs, seed)
        for (index, ext, wrist), (clean, bad, corrected, n) in sorted(stats.items()):
            rows.append({
                "ckpt": Path(checkpoint).name, "step": step,
                "prediction_index": index, "external_camera": ext, "wrist_camera": wrist,
                "n_pairs": n, "clean_error": clean / n, "uncorrected_error": bad / n,
                "corrected_error": corrected / n,
                "correction_gain": (bad - corrected) / n,
            })
    output = Path(custom["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
