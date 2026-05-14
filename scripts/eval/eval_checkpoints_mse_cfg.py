"""Evaluate checkpoints on multiple zarr data paths across classifier-free guidance scales.

Usage (single GPU, no torchrun needed):
    python scripts/eval/eval_checkpoints_mse_cfg.py \
        checkpoints=/path/to/ckpt.pth \
        data_paths=/path/to/val1.zarr,/path/to/val2.zarr \
        val_instructions=instructions/peract/instructions.json \
        bimanual=false \
        cfg_scale=4 \
        output_csv=results/checkpoint_mse_cfg/cfg_4.csv \
        num_batches=100

Identical to eval_checkpoints_mse.py but plumbs `cfg_scale` into the model's
inference call so classifier-free guidance is applied. The CSV gains a
`cfg_scale` column. cfg_scale=null/none/<empty> runs vanilla conditional
inference (no uncond pass) for a baseline.

cfg_scale=0 yields pure unconditional generation (the CFG formula collapses to
`out_uncond` when scale=0). It still runs the cond forward pass per timestep
which is wasted, but for offline eval that's fine — no separate script needed.

When running as a SLURM array job (one task per cfg value), pass a per-task
output_csv so parallel tasks never write the same file.

Dataset type, model architecture, and all other config are auto-detected from
each checkpoint's saved config dict — no need to pass arch flags on the CLI.
"""
import csv
import re
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import fetch_dataset_class
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.data_preprocessors import fetch_data_preprocessor
from utils.depth2cloud import fetch_depth2cloud
from utils.hydra_utils import get_config, get_config_path
from utils.trainers.base import base_collate_fn, relative_to_absolute
from utils.trainers.utils import compute_metrics


METRIC_KEYS = [
    "traj_pos_l2",
    "traj_rot_l1",
    "traj_pos_acc_001",
    "traj_rot_acc_0025",
    "traj_gripper",
]

_RUNTIME_KEYS = frozenset({
    "checkpoint", "eval_data_dir", "data_dir", "output_file",
    "val_instructions", "dataset", "log_dir", "base_log_dir",
})

_SCRIPT_KEYS = frozenset({
    "checkpoints", "data_paths", "output_csv", "num_batches", "cfg_scale",
})


def _extract_script_args(argv):
    custom, hydra_rest = {}, []
    for arg in argv:
        m = re.match(r"([^=]+)=(.+)", arg)
        if m and m.group(1) in _SCRIPT_KEYS:
            custom[m.group(1)] = m.group(2)
        else:
            hydra_rest.append(arg)
    return custom, hydra_rest


def _parse_cfg_scale(raw):
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "none", "null"):
        return None
    return float(raw)


def load_model(args, ckpt_path):
    print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_cfg = ckpt.get("config", {})
    if ckpt_cfg:
        for k, v in ckpt_cfg.items():
            if k not in _RUNTIME_KEYS:
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
        lang_dropout_prob=getattr(args, "lang_dropout_prob", 0.0),
    )

    use_ema = getattr(args, "use_ema", False)
    weight_key = (
        "ema_weight"
        if use_ema and "ema_weight" in ckpt and ckpt["ema_weight"] is not None
        else "weight"
    )
    print(f"Using checkpoint key: '{weight_key}'")
    state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt[weight_key].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    ckpt_step = ckpt.get("iter") or 0
    if ckpt_step == 0:
        m = re.search(r"(\d+)(?=\.pth$)", str(ckpt_path))
        if m:
            ckpt_step = int(m.group(1))
    del ckpt
    torch.cuda.empty_cache()
    return model.cuda(), ckpt_step


def make_loader(args, data_path):
    dataset_class = fetch_dataset_class(args.dataset)
    val_dataset = dataset_class(
        root=data_path,
        instructions=args.val_instructions,
        copies=1,
        relative_action=args.relative_action,
        mem_limit=0.1,
        chunk_size=args.chunk_size,
        num_history=args.num_history,
    )
    nw = args.num_workers
    loader_kwargs = dict(prefetch_factor=2, persistent_workers=True) if nw > 0 else {}
    return DataLoader(
        val_dataset,
        batch_size=args.batch_size_val // args.chunk_size,
        shuffle=False,
        num_workers=nw,
        collate_fn=base_collate_fn,
        pin_memory=True,
        drop_last=False,
        **loader_kwargs,
    )


@torch.inference_mode()
def eval_one_datapath(model, tokenizer, preprocessor, loader, num_batches, amp_dtype, relative_action, cfg_scale):
    accum = {}
    seen = 0

    for i, sample in tqdm(enumerate(loader), total=num_batches, desc="  batches"):
        if i >= num_batches:
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
                cfg_scale=cfg_scale,
            )

        gt_action = action
        if relative_action:
            prop = proprio[:, :, 0] if proprio.dim() == 4 else proprio
            pred_action = relative_to_absolute(pred_action[:, :, 0], prop)
            gt_action = relative_to_absolute(gt_action[:, :, 0], prop)

        losses, _ = compute_metrics(pred_action, gt_action)

        b = gt_action.shape[0]
        for k, v in losses.items():
            accum.setdefault(k, 0.0)
            accum[k] += v.item() * b
        seen += b

    return {k: v / seen for k, v in accum.items()}, seen


CSV_HEADER = ["step", "cfg_scale", "dataset", "n_samples"] + METRIC_KEYS


def _cfg_label(cfg_scale):
    if cfg_scale is None:
        return "none"
    return f"{cfg_scale:g}"


def write_csv_rows(output_csv, ckpt_step, cfg_scale, ckpt_results):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_csv.exists()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if is_new:
            writer.writeheader()
        for data_path, (metrics, n) in ckpt_results.items():
            row = {
                "step": ckpt_step,
                "cfg_scale": _cfg_label(cfg_scale),
                "dataset": Path(data_path).stem,
                "n_samples": n,
            }
            for k in METRIC_KEYS:
                row[k] = round(metrics.get(k, float("nan")), 6)
            writer.writerow(row)


def print_results(ckpt_path, ckpt_step, cfg_scale, ckpt_results):
    print(f"\n{'=' * 80}")
    print(f"Checkpoint: {ckpt_path}  (step {ckpt_step})  cfg_scale={_cfg_label(cfg_scale)}")
    print("-" * 80)
    for data_path, (metrics, n) in ckpt_results.items():
        print(f"  Data: {Path(data_path).name}  [n={n} samples]")
        for k in METRIC_KEYS:
            print(f"    {k:<22} {metrics.get(k, float('nan')):.6f}")
    print("=" * 80)


def main():
    custom, hydra_argv = _extract_script_args(sys.argv[1:])

    checkpoints = [p.strip() for p in custom.get("checkpoints", "").split(",") if p.strip()]
    data_paths = [p.strip() for p in custom.get("data_paths", "").split(",") if p.strip()]
    output_csv = Path(custom.get("output_csv", "results/checkpoint_mse_cfg/results.csv"))
    num_batches = int(custom.get("num_batches", 100))
    cfg_scale = _parse_cfg_scale(custom.get("cfg_scale"))

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint path via checkpoints=path.pth")
    if not data_paths:
        raise ValueError("Pass at least one data path via data_paths=path1.zarr,path2.zarr")

    args = get_config(
        overrides=hydra_argv,
        config_name="config",
        config_path=get_config_path(),
    )

    print(f"Checkpoints ({len(checkpoints)}):")
    for p in checkpoints:
        print(f"  {p}")
    print(f"Data paths ({len(data_paths)}):")
    for p in data_paths:
        print(f"  {p}")
    print(f"cfg_scale={_cfg_label(cfg_scale)}  num_batches={num_batches}  output_csv={output_csv}")
    print("-" * 80)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    amp_dtype = torch.float32 if "Quadro RTX 6000" in gpu_name else torch.bfloat16
    print(f"AMP dtype: {amp_dtype}")

    for ckpt_path in checkpoints:
        args_copy = deepcopy(args)
        model, ckpt_step = load_model(args_copy, ckpt_path)

        preprocessor = fetch_data_preprocessor(args_copy.dataset)(
            args_copy.keypose_only,
            args_copy.num_history,
            custom_imsize=getattr(args_copy, "custom_img_size", None),
            depth2cloud=fetch_depth2cloud(args_copy.dataset),
        )

        _text_backbone = getattr(args_copy, "text_backbone", None) or args_copy.backbone
        tokenizer = fetch_tokenizers(_text_backbone)

        ckpt_results = {}
        for data_path in data_paths:
            print(f"\nEvaluating on: {data_path}")
            loader = make_loader(args_copy, data_path)
            metrics, n = eval_one_datapath(
                model, tokenizer, preprocessor, loader,
                num_batches, amp_dtype, args_copy.relative_action, cfg_scale,
            )
            ckpt_results[data_path] = (metrics, n)

        print_results(ckpt_path, ckpt_step, cfg_scale, ckpt_results)
        write_csv_rows(output_csv, ckpt_step, cfg_scale, ckpt_results)
        print(f"Rows written to {output_csv}")

        del model
        torch.cuda.empty_cache()

    print(f"\nAll results written to {output_csv}")


if __name__ == "__main__":
    main()
