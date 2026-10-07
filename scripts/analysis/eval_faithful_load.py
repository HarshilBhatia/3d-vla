#!/usr/bin/env python3
"""Load a checkpoint the way online eval does: legacy delta_M camera compat, and a chosen visual history length."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from evaluation.checkpoints import overlay_checkpoint_config
from modeling.policy import fetch_model_class
from modeling.policy.construction import build_model_kwargs
from utils.config_migrations import migrate_config
from utils.hydra_utils import get_config, get_config_path, normalize_public_vocabulary_args


def parse_spec(spec):
    """label=checkpoint:raw|ema[:K|ckpt] -> (label, path, weights, K or None)."""
    label, rest = spec.split("=", 1)
    parts = rest.split(":")
    if parts[-1] in ("raw", "ema"):
        return label, ":".join(parts[:-1]), parts[-1], None
    return label, ":".join(parts[:-2]), parts[-2], parts[-1]


def load(path, weights, visual_num_history=None):
    """visual_num_history: None keeps the config default (old offline behaviour), 'ckpt' uses the saved value, or an int."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = ckpt["config"]
    legacy = "dynamic_rope_from_camtoken" in raw and "delta_m_camera_ids" in raw
    cfg = migrate_config(raw)
    args = get_config(overrides=["data=orbital_peract2_nfs"], config_name="config", config_path=get_config_path())
    overlay_checkpoint_config(args, cfg)
    normalize_public_vocabulary_args(args)
    if legacy and getattr(args, "view_align_mode", "none") != "none":
        args.view_align_cameras = None  # same as evaluation.checkpoints.load_model_for_evaluation
    if visual_num_history == "ckpt":
        args.visual_num_history = cfg.get("visual_num_history", args.visual_num_history)
    elif visual_num_history is not None:
        args.visual_num_history = int(visual_num_history)
    cls = fetch_model_class(args.model_type)
    model = cls(**build_model_kwargs(args, cls))
    key = {"raw": "weight", "ema": "ema_weight"}[weights]
    missing, unexpected = model.load_state_dict({k[7:]: v for k, v in ckpt[key].items()}, strict=False)
    bad = [k for k in missing if not k.startswith(("encoder.text_encoder.", "encoder.backbone.", "encoder.normalize."))]
    assert not bad and not unexpected, (bad[:5], unexpected[:5])
    print(f"[load] {path} {key} visual_num_history={args.visual_num_history} legacy_all_cams={legacy} "
          f"view_align_cameras={getattr(args, 'view_align_cameras', None)} scene_sampling={args.scene_sampling}", flush=True)
    return model.cuda().eval(), args
