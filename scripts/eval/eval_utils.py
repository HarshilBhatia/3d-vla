"""Shared helpers for the offline-MSE eval scripts in this directory.

The scripts differ along three axes; everything else lives here:
  (a) extra model.__call__ kwargs (e.g. ``cfg_scale``)
  (b) extra preprocessor kwargs (e.g. ``orbital_miscal_noise_level``,
      ``cotrain_miscal_group_ids``)
  (c) bucketing of metrics (none, by ``sample["camera_group"]``, or by a
      zarr-side field injected via a Dataset wrapper)

Typical usage in a script::

    from scripts.eval.eval_utils import (
        BASE_SCRIPT_KEYS, METRIC_KEYS, extract_script_args, load_args,
        load_model, make_loader, make_preprocessor, make_tokenizer,
        pick_amp_dtype, eval_scalar, metric_row, write_csv_rows,
        parse_csv_list,
    )

    script_keys = BASE_SCRIPT_KEYS | {"cfg_scale"}
    custom, hydra_argv = extract_script_args(sys.argv[1:], script_keys)
    ...
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import zarr
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

# Keys that should NOT be overlaid from the checkpoint's saved config — they
# describe the current eval invocation, not the training run.
RUNTIME_KEYS = frozenset({
    "checkpoint", "eval_data_dir", "data_dir", "output_file",
    "val_instructions", "dataset", "log_dir", "base_log_dir",
})

# Args every eval script accepts. Scripts add their own (e.g. ``cfg_scale``)
# by unioning with this set before calling :func:`extract_script_args`.
BASE_SCRIPT_KEYS = frozenset({
    "checkpoints", "data_paths", "output_csv", "num_batches",
})


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def extract_script_args(argv, script_keys):
    """Split ``key=value`` args matching ``script_keys`` from hydra overrides."""
    custom, hydra_rest = {}, []
    for arg in argv:
        m = re.match(r"([^=]+)=(.+)", arg)
        if m and m.group(1) in script_keys:
            custom[m.group(1)] = m.group(2)
        else:
            hydra_rest.append(arg)
    return custom, hydra_rest


def parse_csv_list(raw):
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def parse_cfg_scale(raw):
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "none", "null"):
        return None
    return float(raw)


def cfg_label(cfg_scale):
    return "none" if cfg_scale is None else f"{cfg_scale:g}"


def parse_group_ids(raw):
    if raw is None:
        return None
    s = raw.strip().lstrip("[").rstrip("]")
    return [int(x) for x in s.split(",") if x.strip()]


def parse_miscal_level(raw):
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "none", "null"):
        return None
    return raw


def load_args(hydra_argv):
    return get_config(
        overrides=hydra_argv,
        config_name="config",
        config_path=get_config_path(),
    )


def pick_amp_dtype():
    """fp32 on Quadro RTX 6000 (no bf16 support), bf16 elsewhere."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return torch.float32 if "Quadro RTX 6000" in gpu_name else torch.bfloat16


# ---------------------------------------------------------------------------
# Model + data construction
# ---------------------------------------------------------------------------

def load_model(args, ckpt_path, arch_overrides=None):
    """Load ``ckpt_path``, overlay its saved ``config`` onto ``args`` (skipping
    :data:`RUNTIME_KEYS`), build the model, return ``(model_on_cuda, step)``.

    ``arch_overrides`` is an optional ``{str: value}`` dict applied *after* the
    checkpoint config — use it to disable features at eval time, e.g.
    ``{"predict_extrinsics": False, "dynamic_rope_from_camtoken": False}``.
    """
    print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_cfg = ckpt.get("config", {})
    if ckpt_cfg:
        for k, v in ckpt_cfg.items():
            if k not in RUNTIME_KEYS:
                setattr(args, k, v)
    else:
        print("Warning: checkpoint has no saved config — model arch args must be supplied via CLI")

    if arch_overrides:
        for k, v in arch_overrides.items():
            setattr(args, k, v)
            print(f"[arch_override] {k} = {v}")

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


def make_preprocessor(args, **extra_kwargs):
    """Construct the data preprocessor for ``args.dataset``. ``extra_kwargs``
    are forwarded to the preprocessor constructor (e.g. miscal noise levels).
    """
    return fetch_data_preprocessor(args.dataset)(
        args.keypose_only,
        args.num_history,
        custom_imsize=getattr(args, "custom_img_size", None),
        depth2cloud=fetch_depth2cloud(args.dataset),
        **extra_kwargs,
    )


def make_tokenizer(args):
    text_backbone = getattr(args, "text_backbone", None) or args.backbone
    return fetch_tokenizers(text_backbone)


def make_loader(args, data_path, chunk_size=None, dataset_wrapper=None):
    """Build the eval DataLoader.

    Args:
        chunk_size: override ``args.chunk_size`` (e.g. set to 1 when a dataset
            wrapper injects a per-row field).
        dataset_wrapper: optional ``cls(base_dataset, zarr_root)`` wrapping the
            base dataset to inject extra sample fields.
    """
    cs = args.chunk_size if chunk_size is None else chunk_size
    dataset_class = fetch_dataset_class(args.dataset)
    base_ds = dataset_class(
        root=data_path,
        instructions=args.val_instructions,
        copies=1,
        relative_action=args.relative_action,
        mem_limit=0.1,
        chunk_size=cs,
        num_history=args.num_history,
    )
    ds = dataset_wrapper(base_ds, data_path) if dataset_wrapper is not None else base_ds

    nw = args.num_workers
    loader_kwargs = dict(prefetch_factor=2, persistent_workers=True) if nw > 0 else {}
    return DataLoader(
        ds,
        batch_size=args.batch_size_val // cs,
        shuffle=False,
        num_workers=nw,
        collate_fn=base_collate_fn,
        pin_memory=True,
        drop_last=False,
        **loader_kwargs,
    )


# ---------------------------------------------------------------------------
# Dataset wrappers — inject a zarr-side field as a per-sample tensor.
# Both require ``chunk_size=1`` so each dataset index maps to one zarr row.
# ---------------------------------------------------------------------------

class StepWithinDemoDataset(Dataset):
    """Inject ``step_within_demo`` (0-based index inside each demo).

    Requires the zarr to have a ``demo_id`` array.
    """

    def __init__(self, base_dataset, zarr_root):
        self._base = base_dataset
        z = zarr.open(str(zarr_root), "r")
        if "demo_id" not in z:
            raise ValueError(f"zarr at {zarr_root} has no demo_id — cannot bucket by keypose step")
        demo_ids = np.array(z["demo_id"][:])
        step_within = np.zeros(len(demo_ids), dtype=np.int64)
        prev, count = -1, 0
        for i, did in enumerate(demo_ids):
            if did != prev:
                count = 0
                prev = int(did)
            step_within[i] = count
            count += 1
        self._step_within = step_within
        self._N = len(demo_ids)
        steps_per_demo = np.unique(np.unique(demo_ids, return_counts=True)[1])
        print(
            f"[keystep] zarr={Path(zarr_root).name}  "
            f"N={self._N}  steps_per_demo={steps_per_demo.tolist()}",
            flush=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        sample = self._base[idx]
        zarr_idx = int(idx) % self._N
        sample["step_within_demo"] = torch.tensor(
            [self._step_within[zarr_idx]], dtype=torch.long
        )
        return sample


class CameraGroupFilteredDataset(Dataset):
    """Filter samples to those whose ``camera_group`` is in ``allowed_groups``.

    Requires the zarr to have a ``camera_group`` array and that the base
    dataset was built with ``chunk_size=1`` (so each dataset index maps 1:1
    to a zarr row).
    """

    def __init__(self, base_dataset, zarr_root, allowed_groups):
        self._base = base_dataset
        z = zarr.open(str(zarr_root), "r")
        if "camera_group" not in z:
            raise ValueError(f"zarr at {zarr_root} has no camera_group field")
        groups = np.array(z["camera_group"][:])
        allowed = set(int(g) for g in allowed_groups)
        mask = np.isin(groups, list(allowed))
        self._allowed_indices = np.where(mask)[0].astype(np.int64)
        counts = {int(g): int((groups == g).sum()) for g in sorted(allowed)}
        print(
            f"[cam_group_filter] zarr={Path(zarr_root).name}  "
            f"allowed={sorted(allowed)}  kept={len(self._allowed_indices)}/{len(groups)}  "
            f"per_group={counts}",
            flush=True,
        )
        if len(self._allowed_indices) == 0:
            raise ValueError(
                f"No samples in {zarr_root} match camera_group in {sorted(allowed)}"
            )

    def __len__(self):
        return len(self._allowed_indices)

    def __getitem__(self, idx):
        zarr_idx = int(self._allowed_indices[int(idx) % len(self._allowed_indices)])
        return self._base[zarr_idx]


def make_camera_group_filter(allowed_groups):
    """Build a ``dataset_wrapper`` callable for :func:`make_loader` that
    filters to ``allowed_groups`` (e.g. ``[1, 3]``)."""
    def _wrap(base_dataset, zarr_root):
        return CameraGroupFilteredDataset(base_dataset, zarr_root, allowed_groups)
    return _wrap


class VariationDataset(Dataset):
    """Inject ``variation`` (drawer/etc.) from the zarr's ``variation`` array."""

    def __init__(self, base_dataset, zarr_root):
        self._base = base_dataset
        z = zarr.open(str(zarr_root), "r")
        if "variation" not in z:
            raise ValueError(f"zarr at {zarr_root} has no variation field")
        self._variation = np.array(z["variation"][:], dtype=np.int64)
        self._N = len(self._variation)
        counts = {int(v): int((self._variation == v).sum()) for v in np.unique(self._variation)}
        print(f"[variation] zarr={Path(zarr_root).name}  N={self._N}  counts={counts}", flush=True)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        sample = self._base[idx]
        zarr_idx = int(idx) % self._N
        sample["variation"] = torch.tensor([self._variation[zarr_idx]], dtype=torch.long)
        return sample


# ---------------------------------------------------------------------------
# Inference / metric accumulation
# ---------------------------------------------------------------------------

def run_inference_batch(model, tokenizer, preprocessor, sample, amp_dtype,
                        relative_action, **model_kwargs):
    """Run a single batch through the preprocessor + model. Returns
    ``(pred_action, gt_action)`` in absolute-pose form (i.e. relative→absolute
    conversion is already applied when ``relative_action=True``).

    ``model_kwargs`` are forwarded to ``model.__call__`` (e.g. ``cfg_scale``).
    """
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
            **model_kwargs,
        )

    gt_action = action
    if relative_action:
        prop = proprio[:, :, 0] if proprio.dim() == 4 else proprio
        pred_action = relative_to_absolute(pred_action[:, :, 0], prop)
        gt_action = relative_to_absolute(gt_action[:, :, 0], prop)
    return pred_action, gt_action


@torch.inference_mode()
def eval_scalar(model, tokenizer, preprocessor, loader, num_batches, amp_dtype,
                relative_action, on_sample=None, **model_kwargs):
    """Average metrics over up to ``num_batches`` batches.

    ``on_sample(sample)`` is called (in-place) before preprocessing, e.g. to
    override ``sample["camera_group"]``.

    Returns ``({metric: avg}, n_samples)``.
    """
    accum = {}
    seen = 0
    for i, sample in tqdm(enumerate(loader), total=num_batches, desc="  batches"):
        if i >= num_batches:
            break
        if on_sample is not None:
            on_sample(sample)
        pred_action, gt_action = run_inference_batch(
            model, tokenizer, preprocessor, sample,
            amp_dtype, relative_action, **model_kwargs,
        )
        losses, _ = compute_metrics(pred_action, gt_action)
        b = gt_action.shape[0]
        for k, v in losses.items():
            accum.setdefault(k, 0.0)
            accum[k] += v.item() * b
        seen += b
    return {k: v / seen for k, v in accum.items()}, seen


@torch.inference_mode()
def eval_bucketed(model, tokenizer, preprocessor, loader, num_batches, amp_dtype,
                  relative_action, bucket_fn, on_sample=None, **model_kwargs):
    """Bucket per-sample metrics. ``bucket_fn(sample, B) -> List`` of length ``B``;
    list entries that are ``None`` are dropped, all others are used as dict keys.

    Returns ``{bucket_key: ({metric: avg}, n_samples)}``.
    """
    accum = defaultdict(lambda: defaultdict(float))
    accum_n = defaultdict(int)

    for i, sample in tqdm(enumerate(loader), total=num_batches, desc="  batches"):
        if i >= num_batches:
            break
        if on_sample is not None:
            on_sample(sample)
        pred_action, gt_action = run_inference_batch(
            model, tokenizer, preprocessor, sample,
            amp_dtype, relative_action, **model_kwargs,
        )
        B = gt_action.shape[0]
        buckets = bucket_fn(sample, B)
        _, per_sample = compute_metrics(pred_action, gt_action)
        for k, v in per_sample.items():
            v_flat = v.detach().float().reshape(B, -1).mean(-1).cpu().tolist()
            for g, val in zip(buckets, v_flat):
                if g is None:
                    continue
                accum[g][k] += float(val)
        for g in buckets:
            if g is None:
                continue
            accum_n[g] += 1

    return {
        g: ({k: sums[k] / accum_n[g] for k in sums}, accum_n[g])
        for g, sums in accum.items()
    }


def camera_group_bucket_fn(sample, B):
    """``bucket_fn`` for :func:`eval_bucketed` that groups by ``sample["camera_group"]``
    (or all-zero when the field is absent)."""
    cam_group = sample.get("camera_group")
    if cam_group is None:
        return [0] * B
    return cam_group.long().view(B, -1)[:, 0].tolist()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def metric_row(metrics, **fields):
    """Build a CSV row dict from ``fields`` + rounded :data:`METRIC_KEYS`."""
    row = dict(fields)
    for k in METRIC_KEYS:
        v = metrics.get(k, float("nan"))
        row[k] = round(v, 6) if v == v else float("nan")
    return row


def write_csv_rows(output_csv, header, rows):
    """Append ``rows`` (list of dicts) to ``output_csv``, writing the header if new."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_csv.exists()
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_scalar_results(ckpt_path, ckpt_step, ckpt_results, extra=""):
    """Pretty-print scalar results: one block per data_path with all metrics."""
    print(f"\n{'=' * 80}")
    header = f"Checkpoint: {ckpt_path}  (step {ckpt_step})"
    if extra:
        header += f"  {extra}"
    print(header)
    print("-" * 80)
    for data_path, (metrics, n) in ckpt_results.items():
        print(f"  Data: {Path(data_path).name}  [n={n} samples]")
        for k in METRIC_KEYS:
            print(f"    {k:<22} {metrics.get(k, float('nan')):.6f}")
    print("=" * 80)


def print_bucketed_results(ckpt_path, ckpt_step, data_path, bucket_results,
                           bucket_name="bucket", bucket_labels=None, extra=""):
    """Pretty-print bucketed results: one row per bucket, columns per metric.

    ``bucket_labels`` optionally maps bucket key -> human label.
    """
    print(f"\n{'=' * 80}")
    header = f"Checkpoint: {Path(ckpt_path).name}  (step {ckpt_step})"
    if extra:
        header += f"  {extra}"
    print(header)
    print(f"Dataset: {Path(data_path).name}")
    print("-" * 80)
    for key in sorted(bucket_results, key=lambda x: (isinstance(x, str), x)):
        metrics, n = bucket_results[key]
        label = bucket_labels.get(key, key) if bucket_labels else key
        print(f"  {bucket_name}={label}  [n={n}]")
        for k in METRIC_KEYS:
            print(f"    {k:<22} {metrics.get(k, float('nan')):.6f}")
    print("=" * 80)
