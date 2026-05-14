"""Evaluate checkpoints with MSE metrics bucketed by drawer variation (0=bottom, 1=middle, 2=top).

Requires zarrs to have a `variation` array (all open_drawer zarrs do).

Usage:
    python scripts/eval/eval_checkpoints_mse_cfg_variation.py \
        checkpoints=/path/to/ckpt.pth \
        data_paths=/path/to/val.zarr,/path/to/train.zarr \
        val_instructions=instructions/peract/instructions.json \
        bimanual=false \
        cfg_scale=1 \
        output_csv=results/variation/cfg_1/step_70000.csv \
        num_batches=200
"""
import csv
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import zarr
import torch
from torch.utils.data import DataLoader, Dataset
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

VARIATION_LABELS = {0: "bottom", 1: "middle", 2: "top"}

_RUNTIME_KEYS = frozenset({
    "checkpoint", "eval_data_dir", "data_dir", "output_file",
    "val_instructions", "dataset", "log_dir", "base_log_dir",
})

_SCRIPT_KEYS = frozenset({
    "checkpoints", "data_paths", "output_csv", "num_batches", "cfg_scale",
})


# ---------------------------------------------------------------------------
# Dataset wrapper — injects variation id per sample
# ---------------------------------------------------------------------------

class VariationAwareDataset(Dataset):
    def __init__(self, base_dataset, zarr_root):
        self._base = base_dataset
        z = zarr.open(str(zarr_root), "r")
        if "variation" not in z:
            raise ValueError(f"zarr at {zarr_root} has no variation field")
        self._variation = np.array(z["variation"][:], dtype=np.int64)
        self._N = len(self._variation)
        counts = {v: int((self._variation == v).sum()) for v in np.unique(self._variation)}
        print(f"[variation] zarr={Path(zarr_root).name}  N={self._N}  counts={counts}", flush=True)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        sample = self._base[idx]
        zarr_idx = int(idx) % self._N
        sample["variation"] = torch.tensor([self._variation[zarr_idx]], dtype=torch.long)
        return sample


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

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


def _cfg_label(cfg_scale):
    return "none" if cfg_scale is None else f"{cfg_scale:g}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def make_loader(args, data_path):
    dataset_class = fetch_dataset_class(args.dataset)
    base_ds = dataset_class(
        root=data_path,
        instructions=args.val_instructions,
        copies=1,
        relative_action=args.relative_action,
        mem_limit=0.1,
        chunk_size=1,
        num_history=args.num_history,
    )
    dataset = VariationAwareDataset(base_ds, data_path)
    nw = args.num_workers
    loader_kwargs = dict(prefetch_factor=2, persistent_workers=True) if nw > 0 else {}
    return DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=nw,
        collate_fn=base_collate_fn,
        pin_memory=True,
        drop_last=False,
        **loader_kwargs,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_one_datapath(model, tokenizer, preprocessor, loader,
                      num_batches, amp_dtype, relative_action, cfg_scale):
    accum = {v: {} for v in VARIATION_LABELS}
    seen  = {v: 0  for v in VARIATION_LABELS}

    for i, sample in tqdm(enumerate(loader), total=num_batches, desc="  batches"):
        if i >= num_batches:
            break

        # "variation" is in base_collate_fn's list_keys so it arrives as a Python list
        variation_ids = [int(v) for v in sample["variation"]]

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
            gt_action   = relative_to_absolute(gt_action[:, :, 0],  prop)

        for b, var in enumerate(variation_ids):
            if var not in VARIATION_LABELS:
                continue
            losses, _ = compute_metrics(pred_action[b:b+1], gt_action[b:b+1])
            for k, v in losses.items():
                accum[var].setdefault(k, 0.0)
                accum[var][k] += v.item()
            seen[var] += 1

    return {
        var: ({k: v / seen[var] for k, v in accum[var].items()}, seen[var])
        for var in VARIATION_LABELS
        if seen[var] > 0
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_HEADER = ["train_step", "cfg_scale", "dataset", "variation", "variation_name", "n_samples"] + METRIC_KEYS


def write_csv_rows(output_csv, ckpt_step, cfg_scale, data_path, var_results):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_csv.exists()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if is_new:
            writer.writeheader()
        for var, (metrics, n) in var_results.items():
            row = {
                "train_step": ckpt_step,
                "cfg_scale": _cfg_label(cfg_scale),
                "dataset": Path(data_path).stem,
                "variation": var,
                "variation_name": VARIATION_LABELS[var],
                "n_samples": n,
            }
            for k in METRIC_KEYS:
                row[k] = round(metrics.get(k, float("nan")), 6)
            writer.writerow(row)


def print_results(ckpt_path, ckpt_step, cfg_scale, data_path, var_results):
    print(f"\n{'=' * 72}")
    print(f"Checkpoint: {Path(ckpt_path).name}  (step {ckpt_step})  cfg={_cfg_label(cfg_scale)}")
    print(f"Dataset:    {Path(data_path).name}")
    print(f"{'variation':<14} {'pos_l2':>10} {'rot_l1':>10} {'pos_acc%':>10} {'rot_acc%':>10} {'grip%':>8}")
    print("-" * 72)
    for var in sorted(var_results):
        metrics, n = var_results[var]
        label = f"{var} ({VARIATION_LABELS[var]}, n={n})"
        print(
            f"{label:<14} "
            f"{metrics.get('traj_pos_l2', float('nan')):>10.4f} "
            f"{metrics.get('traj_rot_l1', float('nan')):>10.4f} "
            f"{100*metrics.get('traj_pos_acc_001', float('nan')):>10.1f} "
            f"{100*metrics.get('traj_rot_acc_0025', float('nan')):>10.1f} "
            f"{100*metrics.get('traj_gripper', float('nan')):>8.1f}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    custom, hydra_argv = _extract_script_args(sys.argv[1:])

    checkpoints = [p.strip() for p in custom.get("checkpoints", "").split(",") if p.strip()]
    data_paths  = [p.strip() for p in custom.get("data_paths", "").split(",") if p.strip()]
    output_csv  = Path(custom.get("output_csv", "results/variation/results.csv"))
    num_batches = int(custom.get("num_batches", 200))
    cfg_scale   = _parse_cfg_scale(custom.get("cfg_scale"))

    if not checkpoints:
        raise ValueError("Pass at least one checkpoint via checkpoints=path.pth")
    if not data_paths:
        raise ValueError("Pass at least one data path via data_paths=path1.zarr,path2.zarr")

    args = get_config(
        overrides=hydra_argv,
        config_name="config",
        config_path=get_config_path(),
    )

    print(f"cfg_scale={_cfg_label(cfg_scale)}  num_batches={num_batches}  output_csv={output_csv}")
    print("-" * 72)

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

        for data_path in data_paths:
            print(f"\nEvaluating on: {data_path}")
            loader = make_loader(args_copy, data_path)
            var_results = eval_one_datapath(
                model, tokenizer, preprocessor, loader,
                num_batches, amp_dtype, args_copy.relative_action, cfg_scale,
            )
            print_results(ckpt_path, ckpt_step, cfg_scale, data_path, var_results)
            write_csv_rows(output_csv, ckpt_step, cfg_scale, data_path, var_results)

        print(f"\nRows written to {output_csv}")
        del model
        torch.cuda.empty_cache()

    print(f"\nAll results written to {output_csv}")


if __name__ == "__main__":
    main()
