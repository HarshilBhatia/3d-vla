"""Sweep miscalibration noise levels and types; measure offline action prediction loss.

Usage (single GPU, no torchrun needed):
    python scripts/eval/sweep_miscal_loss.py \
        checkpoint=path/to/best.pth \
        eval_data_dir=path/to/val.zarr \
        val_instructions=instructions/peract2/instructions.json \
        dataset=Peract2_3dfront_3dwrist \
        bimanual=true \
        [output_csv=results/miscal_sweep.csv] \
        [num_samples=1000] \
        "$@"

Results are written as CSV with columns:
    noise_type, angle_deg, trans_m, n_samples,
    traj_pos_l2, traj_rot_l1, traj_pos_acc_001, traj_rot_acc_0025, traj_gripper

Noise types:
    R_only  — rotation noise only  (angle_deg swept, trans_m=0)
    T_only  — translation noise only (angle_deg=0, trans_m swept)
    RT      — both R and T noise (angle_deg and trans_m swept jointly)

NOTE: Miscalibration noise is only applied when num_history > 1 and the zarr
contains a 'demo_id' field (history-aware path). The default config has
num_history=3, so this works out of the box for standard Peract2 zarrs.
"""
import csv
import re
import sys
from pathlib import Path

# Allow running as `python scripts/eval/sweep_miscal_loss.py` from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import fetch_dataset_class
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.data_preprocessors.rlbench import RLBenchDataPreprocessor
from utils.depth2cloud import fetch_depth2cloud
from utils.hydra_utils import get_config, get_config_path
from utils.trainers.base import base_collate_fn
from utils.trainers.utils import compute_metrics


# ─── Sweep values ─────────────────────────────────────────────────────────────
# R-only sweep: rotate-only noise, 0 → 20 degrees
SWEEP_R_DEG = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]

# T-only sweep: translate-only noise, 0 → 0.20 m
SWEEP_T_M = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

# Build full list of (noise_type, angle_deg, trans_m) configs
NOISE_CONFIGS = (
    [("R_only", r,    0.0) for r in SWEEP_R_DEG]
    + [("T_only", 0.0,  t) for t in SWEEP_T_M]
    + [("RT",     r,    t) for r, t in zip(SWEEP_R_DEG, SWEEP_T_M)]
)
# ──────────────────────────────────────────────────────────────────────────────

METRIC_KEYS = [
    "traj_pos_l2",
    "traj_rot_l1",
    "traj_pos_acc_001",
    "traj_rot_acc_0025",
    "traj_gripper",
]


_SWEEP_RUNTIME_KEYS = frozenset({
    "checkpoint", "eval_data_dir", "data_dir", "output_file",
    "val_instructions", "dataset", "log_dir", "base_log_dir",
})


def _extract_script_args(argv):
    """Split script-specific key=value args from Hydra overrides."""
    custom_keys = {"num_samples", "output_csv"}
    custom, hydra_rest = {}, []
    for arg in argv:
        m = re.match(r"([^=]+)=(.+)", arg)
        if m and m.group(1) in custom_keys:
            custom[m.group(1)] = m.group(2)
        else:
            hydra_rest.append(arg)
    return custom, hydra_rest


def load_model(args):
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Overlay saved training config onto args for all non-eval keys so the
    # caller doesn't need to pass model arch flags on the CLI.
    ckpt_cfg = ckpt.get("config", {})
    if ckpt_cfg:
        for k, v in ckpt_cfg.items():
            if k not in _SWEEP_RUNTIME_KEYS:
                setattr(args, k, v)
    else:
        print("Warning: checkpoint has no saved config — model arch args must be supplied via CLI")

    model_class = fetch_model_class(args.model_type)
    model = model_class(
        backbone=args.backbone,
        text_backbone=getattr(args, "text_backbone", None),
        finetune_backbone=args.finetune_backbone,
        finetune_text_encoder=args.finetune_text_encoder,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        lv2_batch_size=args.lv2_batch_size,
        learn_extrinsics=getattr(args, "learn_extrinsics", False),
        traj_scene_rope=args.traj_scene_rope,
        predict_extrinsics=getattr(args, "predict_extrinsics", False),
        extrinsics_prediction_mode=getattr(args, "extrinsics_prediction_mode", "delta_m"),
        dynamic_rope_from_camtoken=getattr(args, "dynamic_rope_from_camtoken", False),
        rope_type=getattr(args, "rope_type", "normal"),
        use_recursive_set_encoder=getattr(args, "use_recursive_set_encoder", False),
        recursive_set_encoder_num_layers=getattr(args, "recursive_set_encoder_num_layers", 2),
        recursive_set_encoder_ncam=getattr(args, "recursive_set_encoder_ncam", 3),
    )

    # Prefer EMA weights when use_ema=True and ema_weight is present
    use_ema = getattr(args, "use_ema", False)
    weight_key = (
        "ema_weight"
        if use_ema and "ema_weight" in ckpt and ckpt["ema_weight"] is not None
        else "weight"
    )
    print(f"Using checkpoint key: '{weight_key}'")
    # Strip DDP "module." prefix
    state = {k[7:]: v for k, v in ckpt[weight_key].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.cuda()


def make_preprocessor(args, angle_deg, trans_m):
    return RLBenchDataPreprocessor(
        keypose_only=args.keypose_only,
        num_history=args.num_history,
        custom_imsize=getattr(args, "custom_img_size", None),
        depth2cloud=fetch_depth2cloud(args.dataset),
        miscal_max_angle_deg=float(angle_deg),
        miscal_max_translation_m=float(trans_m),
    )


@torch.inference_mode()
def eval_one_config(model, tokenizer, preprocessor, loader, num_samples, amp_dtype):
    """Run inference and accumulate metrics for up to num_samples samples."""
    accum = {}
    seen = 0

    for sample in loader:
        if seen >= num_samples:
            break

        action = preprocessor.process_actions(sample["action"])
        proprio = preprocessor.process_proprio(sample["proprioception"])
        rgbs, pcds = preprocessor.process_obs(
            sample["rgb"],
            sample.get("rgb2d"),
            sample["depth"],
            sample["extrinsics"],
            sample["intrinsics"],
            augment=False,
            task=sample["task"],
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

        losses, _ = compute_metrics(pred_action, action)

        b = action.shape[0]
        for k, v in losses.items():
            accum.setdefault(k, 0.0)
            accum[k] += v.item() * b
        seen += b

    return {k: v / seen for k, v in accum.items()}, seen


def main():
    custom, hydra_argv = _extract_script_args(sys.argv[1:])
    num_samples = int(custom.get("num_samples", 1000))
    output_csv = Path(custom.get("output_csv", "results/miscal_sweep.csv"))

    args = get_config(
        overrides=hydra_argv,
        config_name="config",
        config_path=get_config_path(),
    )

    print("Arguments:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("-" * 80)
    print(f"num_samples={num_samples}  output_csv={output_csv}")
    print(f"Noise configs: {len(NOISE_CONFIGS)} combinations")
    print("-" * 80)

    model = load_model(args)
    print("workspace_normalizer:", model.workspace_normalizer)

    _text_backbone = getattr(args, "text_backbone", None) or args.backbone
    tokenizer = fetch_tokenizers(_text_backbone)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    amp_dtype = torch.float32 if "Quadro RTX 6000" in gpu_name else torch.bfloat16
    print(f"AMP dtype: {amp_dtype}")

    print("Loading val dataset...")
    dataset_class = fetch_dataset_class(args.dataset)
    val_dataset = dataset_class(
        root=args.eval_data_dir,
        instructions=args.val_instructions,
        copies=1,
        relative_action=args.relative_action,
        mem_limit=0.1,
        chunk_size=args.chunk_size,
        num_history=args.num_history,
    )

    nw = args.num_workers
    loader_kwargs = dict(prefetch_factor=2, persistent_workers=True) if nw > 0 else {}
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_val // args.chunk_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=base_collate_fn,
        pin_memory=True,
        drop_last=False,
        **loader_kwargs,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for noise_type, angle_deg, trans_m in tqdm(NOISE_CONFIGS, desc="Noise configs"):
        preprocessor = make_preprocessor(args, angle_deg, trans_m)
        metrics, n = eval_one_config(
            model, tokenizer, preprocessor, val_loader, num_samples, amp_dtype
        )

        row = {"noise_type": noise_type, "angle_deg": angle_deg, "trans_m": trans_m, "n_samples": n}
        for k in METRIC_KEYS:
            row[k] = round(metrics.get(k, float("nan")), 6)
        rows.append(row)

        print(
            f"[{noise_type:6s}] R={angle_deg:5.1f}° T={trans_m:.3f}m | n={n} | "
            + " | ".join(f"{k.split('_', 1)[1]}={row[k]:.4f}" for k in METRIC_KEYS)
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_type", "angle_deg", "trans_m", "n_samples"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    main()
