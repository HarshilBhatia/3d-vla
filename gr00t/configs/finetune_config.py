# Finetune config used for single node post-training.
from dataclasses import dataclass, field
from typing import List, Optional

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: EmbodimentTag
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets."""

    base_model_path: str = "nvidia/GR00T-N1.6-3B"
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    cached_backbone_dir: str | None = None
    """
    If set, use precomputed VLM backbone features from this directory (feat_*.pt from
    cache_backbone_features.py). Training will skip the backbone forward and only run the action head.
    Dataset path and shard/seed must match the run that created the cache.
    """

    depth_dir: str | None = None
    """
    If set, load depth frames on-the-fly from this directory to compute 3D token positions
    for 3D RoPE in DiT cross-attention. Directory must contain depth.blosc / intrinsics.npy
    per episode/camera (produced by data_processing/extract_svo_depth.py).
    Also requires episode_frame_index.pkl and serial_map.json (from build_episode_frame_index.py).
    Must be used together with cached_backbone_dir.
    """

    episode_index_path: str | None = None
    """
    Path to episode_frame_index.pkl (maps cache global_idx → canonical_id + frame_idx).
    If None and depth_dir is set, defaults to {depth_dir}/episode_frame_index.pkl.
    """

    labs: Optional[List[str]] = field(default=None)
    """
    If set, only include episodes from these labs (e.g. ["RAIL", "TRI"]).
    Must match the --labs filter used when creating the cache.
    """

    allowed_indices_file: Optional[str] = None
    """
    Path to a selected_episodes.json produced by select_episodes.py.
    When set, uses episode_indices from the file as the allowed set, bypassing --labs.
    Must match the --allowed-indices-file used when creating the cache.
    Cannot be combined with --labs.
    """

    resume_from_checkpoint: Optional[str] = None
    """
    Path to a specific checkpoint to resume from.
    If None, resumes from the latest checkpoint in output_dir (if any).
    """

    eval_only: bool = False
    """
    If True, skip training and run a single forward-pass eval on the train dataset (no gradients).
    Useful for comparing checkpoints. Requires --resume-from-checkpoint to be set.
    """

    rope_position_noise_std: float = 0.0
    """
    Standard deviation of Gaussian noise added to all 3D token positions before computing
    3D RoPE during training. Units are meters (same as the position coordinates).
    0.0 disables noise. Only applies during training, not eval/inference.
    """

    depth_cache_dir: str | None = None
    """
    If set, load depth shards (depth_shard_?????.pt) from this directory instead of
    cached_backbone_dir. Allows using a different extrinsics source for 3D RoPE while
    reusing the same backbone cache. Requires use_3d_rope=True.
    """

    use_3d_rope: bool = False
    """If True, load precomputed depth shard positions and apply 3D RoPE in DiT cross-attention."""

    use_eef_relative_rope: bool = False
    """
    If True, subtract the EEF position from token positions before applying 3D RoPE,
    giving relative positions p_k - p_eef. Requires use_3d_rope=True and depth shards
    regenerated with eef_position_3d (cache_depth_features.py after the EEF caching update).
    """

    use_action_eef_rope: bool = False
    """
    If True, apply the EEF position as the 3D RoPE query position for ALL query tokens
    (state token at index 0 AND all action/trajectory tokens at indices 1-16), not just
    the state token. This gives every query a position-relative view Q·R(p_k - p_eef)·K.
    Requires use_3d_rope=True and use_eef_relative_rope=True.
    """

    rope_base_freq: float = 100.0
    """
    Base frequency for 3D RoPE. Controls the range of spatial frequencies encoded.
    Default 100.0 — tuned for robot workspace positions in meters (±7m range).
    With rope_base_freq=100, all 8 frequency pairs produce meaningful angles over this range.
    (LLaMA uses 10000 for token indices 0..10000, which is wrong for meter-scale positions.)
    """

    use_delta_m: bool = False
    """
    If True, add per-camera register tokens derived from mean-pooled image features.
    Each register token attends through all DiT blocks, and after each image cross-attn block
    its state is used to predict a per-camera orthogonal 6×6 deltaM matrix that rotates the
    3D RoPE sin/cos features before the next image cross-attn block.
    Requires use_3d_rope=True. Register tokens get 3D RoPE positions from camera_positions_3d
    (optical centers stored in depth shards). Must re-run cache_depth_features.py to add them.
    """

    num_cameras: int = 2
    """
    Number of cameras (register tokens) for deltaM. Must match the embodiment.
    OXE_DROID_EXT2: 2 (ext1 + ext2). OXE_DROID: 2 (ext1 + wrist).
    """

    use_camera_positions: bool = False
    """
    If True, load camera_positions_3d from depth shards and use them as 3D RoPE positions
    for deltaM register tokens (optical centers T_cam2base[:3,3]).
    Requires depth shards produced by cache_depth_features.py with --cam2cam-json
    (the campos shards). If False, register tokens get zero 3D RoPE position.
    Only meaningful when use_delta_m=True.
    """

    reinit_action_head: bool = False
    """If True, randomly reinitialize action head weights after loading the pretrained model."""

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6` unless overridden by wandb_project.
    You need to login to wandb to view the logs.
    """

    wandb_project: str | None = None
    """Override wandb project name. If None, uses default `finetune-gr00t-n1d6`."""

    wandb_run_name: str | None = None
    """Optional wandb run name. If None, run name is derived from output_dir."""

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""
