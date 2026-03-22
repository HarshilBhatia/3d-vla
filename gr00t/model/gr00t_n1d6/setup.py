import json
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.registry import register_model
import numpy as np
from termcolor import colored
import torch
from transformers import AutoModel, AutoProcessor


import pickle
from pathlib import Path as _Path


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d6Pipeline(ModelPipeline):
    model_class = Gr00tN1d6
    processor_class = Gr00tN1d6Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""

        # Build transformers loading kwargs from training config

        use_cached_backbone = getattr(self.config.data, "cached_backbone_dir", None) is not None

        if self.config.training.start_from_checkpoint is not None:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                skip_backbone=use_cached_backbone,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            # Initialize mask_tokens if they are not present in the base checkpoint
            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)

            if mask_token_missing and model.action_head.mask_token is not None:
                # Initialize mask_token
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

        else:
            self.config.model.skip_backbone = use_cached_backbone
            model = self.model_class(
                self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
            )

        # If use_delta_m is requested but the checkpoint config didn't include it,
        # inject DeltaMPredictor post-hoc (new weights, randomly initialized).
        use_delta_m = self.config.model.diffusion_model_cfg.get("use_delta_m", False)
        num_cameras = self.config.model.diffusion_model_cfg.get("num_cameras", 2)
        dit = model.action_head.model
        if use_delta_m and not getattr(dit, "use_delta_m", False):
            from gr00t.model.modules.dit import DeltaMPredictor
            backbone_dim = dit.config.cross_attention_dim or 2048
            dit.use_delta_m = True
            dit.num_cameras = num_cameras
            dit.delta_m_pred = DeltaMPredictor(
                hidden_dim=dit.inner_dim,
                num_cameras=num_cameras,
                backbone_dim=backbone_dim,
            )
            # Move to same device/dtype as the rest of the DiT
            ref_param = next(dit.parameters())
            dit.delta_m_pred = dit.delta_m_pred.to(device=ref_param.device, dtype=ref_param.dtype)
            logging.info(
                f"Injected DeltaMPredictor into DiT post-hoc "
                f"(num_cameras={num_cameras}, backbone_dim={backbone_dim})"
            )

        if self.config.training.reinit_action_head:
            for module in model.action_head.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            logging.info("Action head weights randomly reinitialized (reinit_action_head=True)")

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        print("Model: ", model)

        return model

    def _get_statistics(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""

        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        print(
            colored(
                f"These are all the processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}",
                "yellow",
            )
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        cached_backbone_dir = getattr(self.config.data, "cached_backbone_dir", None)
        if cached_backbone_dir is not None:
            self.processor._collator.cache_dir = _Path(cached_backbone_dir)
            collator = self.processor._collator
            use_3d_rope = getattr(self.config.data, "use_3d_rope", False)
            depth_cache_dir = getattr(self.config.data, "depth_cache_dir", None) or cached_backbone_dir
            depth_shard_files = sorted(_Path(depth_cache_dir).glob("depth_shard_?????.pt")) if use_3d_rope else []
            use_eef_relative_rope = getattr(self.config.data, "use_eef_relative_rope", False)
            use_action_eef_rope = getattr(self.config.data, "use_action_eef_rope", False)
            use_delta_m = getattr(self.config.data, "use_delta_m", False)
            use_camera_positions = getattr(self.config.data, "use_camera_positions", False)
            need_eef = use_eef_relative_rope or use_action_eef_rope
            need_camera_pos = use_delta_m and use_camera_positions
            if depth_shard_files:
                import torch as _torch
                print(f"[setup] Preloading {len(depth_shard_files)} depth shards "
                      f"(use_eef_relative_rope={use_eef_relative_rope}, "
                      f"use_action_eef_rope={use_action_eef_rope}, "
                      f"use_delta_m={use_delta_m}, "
                      f"use_camera_positions={use_camera_positions}) ...")
                collator._depth_shard_cache = {}
                collator._eef_shard_cache = {}
                collator._camera_pos_shard_cache = {}
                for p in depth_shard_files:
                    shard_idx = int(p.stem.split("_")[-1])
                    data = _torch.load(p, weights_only=True, map_location="cpu")
                    # share_memory_() puts the tensor in shared memory so DataLoader
                    # workers get a handle (not a copy) — zero extra RAM per worker.
                    collator._depth_shard_cache[shard_idx] = data["token_positions_3d"].share_memory_()
                    if need_eef and "eef_position_3d" in data:
                        collator._eef_shard_cache[shard_idx] = data["eef_position_3d"].share_memory_()
                    if need_camera_pos and "camera_positions_3d" in data:
                        collator._camera_pos_shard_cache[shard_idx] = data["camera_positions_3d"].share_memory_()
                total_mb = sum(v.nbytes for v in collator._depth_shard_cache.values()) / 1e6
                has_eef = len(collator._eef_shard_cache) > 0
                has_cam_pos = len(collator._camera_pos_shard_cache) > 0
                print(f"[setup] Depth shards loaded ({total_mb:.0f} MB), "
                      f"EEF shards loaded: {has_eef}, "
                      f"camera position shards loaded: {has_cam_pos}")

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d6Config, Gr00tN1d6Pipeline)
