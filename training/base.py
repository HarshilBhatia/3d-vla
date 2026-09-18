from copy import deepcopy
import os
import queue
import threading
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity
from tqdm import trange, tqdm
import wandb
from omegaconf import OmegaConf, DictConfig


# How often to poll for preemption. Each poll is one 4-byte all_reduce, so this
# trades a negligible amount of hot-loop time for reaction latency that is
# still two orders of magnitude inside the 120 s grace window.
_PREEMPT_CHECK_EVERY = 10


def _atomic_save(obj, path):
    """Write to a temp file then rename so a killed job never leaves a partial checkpoint."""
    path = str(path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        os.close(tmp_fd)
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _cpu_snapshot(obj):
    """Deep-copy tensors to CPU so a background writer never races the training loop."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_snapshot(v) for v in obj)
    return obj


class _AsyncSaver:
    """Serialise checkpoints on a worker thread so the GPU never waits on NFS."""

    def __init__(self, maxsize=4):
        self._q = queue.Queue(maxsize=maxsize)
        self._error = None
        self._thread = threading.Thread(target=self._run, name="ckpt-writer", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                obj, path = item
                _atomic_save(obj, path)
            except BaseException as exc:
                self._error = exc
            finally:
                self._q.task_done()

    def save(self, obj, path):
        """Queue a write. The caller must already have moved tensors to CPU."""
        self._raise_pending()
        self._q.put((obj, path))

    def flush(self):
        """Block until every queued write has landed. Use before exiting."""
        self._q.join()
        self._raise_pending()

    def close(self):
        self.flush()
        self._q.put(None)
        self._thread.join(timeout=60)

    def _raise_pending(self):
        if self._error is not None:
            err, self._error = self._error, None
            raise err


def _args_to_dict(args):
    if isinstance(args, DictConfig):
        return OmegaConf.to_container(args, resolve=True)
    return vars(args)

from modeling.encoder.text import fetch_tokenizers
from modeling.policy.construction import (
    assert_model_kwargs_complete,
    build_model_kwargs,
)
from utils.common_utils import count_parameters
from data.geometry import fetch_depth2cloud
from data.preprocessing import fetch_data_preprocessor
from utils.ema import EMA
from utils.schedulers import fetch_scheduler
# The resilience layer is deliberately model-agnostic and imports nothing from
# this repo, so the dependency only ever points this way.
from scripts.multinode.resilience import (
    PreemptionGuard,
    SkipAheadSampler,
    capture_rng,
    gather_rng,
    restore_rng,
    seed_everything,
)
from data.batch import actions_collate_fn, base_collate_fn, relative_to_absolute
from common.metrics import compute_metrics
from .utils import BenchmarkLogger
from datasets.samplers import DiverseChunkBatchSampler


class BaseTrainTester:
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args, dataset_cls, model_cls):
        """Initialize."""
        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls
        # Single semantic for train vs offline eval (Option B: derived from eval_only; can migrate to --mode later)
        self.run_mode = "eval_offline" if getattr(args, "eval_only", False) else "train"
        self._saver = _AsyncSaver()

        self.benchmark_logger = None

        self.preprocessor = fetch_data_preprocessor(self.args.dataset)(
            self.args.keypose_only,
            self.args.visual_num_history,
            proprio_num_history=getattr(self.args, 'proprio_num_history', self.args.visual_num_history),
            custom_imsize=self.args.custom_img_size,
            depth2cloud=fetch_depth2cloud(self.args.dataset),
            miscal_mode=self.args.miscal_mode,
            perturbation_noise_rot_deg=self.args.perturbation_noise_rot_deg,
            perturbation_noise_trans_m=self.args.perturbation_noise_trans_m,
            perturbation_noise_fixed_rot_deg=self.args.perturbation_noise_fixed_rot_deg,
            perturbation_noise_fixed_trans_m=self.args.perturbation_noise_fixed_trans_m,
            miscal_group_level=self.args.miscal_group_level,
            miscal_group_file=self.args.miscal_group_file,
            miscal_camera_groups=self.args.miscal_camera_groups,
            miscal_cameras=self.args.miscal_cameras,
        )

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        # fp32 keeps SDPA on the mem-efficient kernel; bf16 needs head_dim % 8 == 0.
        amp = str(getattr(self.args, "amp_dtype", "auto") or "auto").lower()
        if amp == "float32":
            self.amp_dtype = torch.float32
        elif amp == "bfloat16":
            self.amp_dtype = torch.bfloat16
        elif amp == "auto":
            self.amp_dtype = torch.float32 if "Quadro RTX 6000" in gpu_name else torch.bfloat16
        else:
            raise ValueError(f"amp_dtype must be auto, float32 or bfloat16, got {amp!r}")
        if dist.get_rank() == 0:
            print(f"[amp] dtype={self.amp_dtype} (amp_dtype={amp}, gpu={gpu_name})", flush=True)

        if dist.get_rank() == 0 and self.run_mode == "train":
            self.writer = SummaryWriter(log_dir=args.log_dir)

            # Initialize wandb if enabled
            if getattr(args, 'use_wandb', True):
                wandb.init(
                    entity=getattr(args, 'wandb_entity', None),
                    project=getattr(args, 'wandb_project', '3d_flowmatch_actor'),
                    name=getattr(args, 'wandb_run_name', None) or args.log_dir.name,
                    config=vars(args),
                    dir=args.log_dir,
                    resume='allow',
                    id=getattr(args, 'wandb_run_id', None)
                )
                print("Wandb logging enabled")
            else:
                print("Wandb logging disabled (using TensorBoard only)")

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        visual_num_history = getattr(self.args, 'visual_num_history', 1)
        preload = getattr(self.args, 'preload', False)
        print(self.args.train_data_dir)
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit,
            chunk_size=self.args.chunk_size,
            visual_num_history=visual_num_history,
            proprio_num_history=getattr(self.args, 'proprio_num_history', visual_num_history),
            preload=preload,
        )
        val_dataset = self.dataset_cls(
            root=self.args.eval_data_dir,
            instructions=self.args.val_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            chunk_size=self.args.chunk_size,
            visual_num_history=visual_num_history,
            proprio_num_history=getattr(self.args, 'proprio_num_history', visual_num_history),
            preload=preload,
        )
        return train_dataset, val_dataset

    def get_loaders(self):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Datasets
        train_dataset, val_dataset = self.get_datasets()
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        # Video-DeltaM can preserve per-update diversity while retaining a
        # small episode/time working set in each loader worker.  Disabled by
        # default to keep all existing experiments bit-for-bit unchanged.
        cache_batches = getattr(self.args, 'video_deltam_cache_batches', 0)
        use_diverse_chunk_sampler = cache_batches > 0
        if use_diverse_chunk_sampler and 'demo_id' not in train_dataset.annos:
            raise ValueError(
                'video_deltam_cache_batches requires zarr annotations with demo_id'
            )

        train_sampler = None

        # Divide batch size by world size to keep effective batch size constant
        world_size = dist.get_world_size()
        per_gpu_batch_size = self.args.batch_size // (self.args.chunk_size * world_size)

        if dist.get_rank() == 0:
            print(f"World size: {world_size}")
            print(f"Global batch size: {self.args.batch_size}")
            print(f"Per-GPU batch size: {per_gpu_batch_size}")

        prefetch = getattr(self.args, 'prefetch_factor', 4)
        loader_process_opts = {}
        if self.args.num_workers > 0:
            loader_process_opts = {
                'prefetch_factor': prefetch,
                'persistent_workers': True,
            }
        if use_diverse_chunk_sampler:
            train_sampler = DiverseChunkBatchSampler(
                train_dataset,
                batch_size=per_gpu_batch_size,
                num_replicas=world_size,
                rank=dist.get_rank(),
                drop_last=True,
                seed=getattr(self.args, 'video_deltam_sampler_seed', 0),
                cache_batches=cache_batches,
                cache_span=getattr(self.args, 'video_deltam_cache_span', 8),
                # A DataLoader with no workers does not dispatch lanes.
                num_workers=max(1, self.args.num_workers),
            )
            if dist.get_rank() == 0:
                print(
                    'Using diverse chunk sampler: one sample per demo in each '
                    f'batch; cache working set lasts {cache_batches} batches.'
                )
            # Wrapping preserves the inner order exactly: a fresh run is
            # bit-for-bit unchanged (skip=0), while a resume discards only the
            # prefix it already saw. Discarding happens at the sampler, so the
            # skipped batches cost index arithmetic, not data loading.
            train_sampler = SkipAheadSampler(train_sampler, mode="batch")
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=self.args.num_workers,
                worker_init_fn=seed_worker,
                collate_fn=base_collate_fn,
                pin_memory=True,
                generator=g,
                **loader_process_opts,
            )
        else:
            train_sampler = SkipAheadSampler(
                DistributedSampler(train_dataset, drop_last=True, shuffle=True),
                mode="index", batch_size=per_gpu_batch_size,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=per_gpu_batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                worker_init_fn=seed_worker,
                collate_fn=base_collate_fn,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g,
                **loader_process_opts,
            )
        # Val loader on all ranks so every rank participates in eval (avoids NCCL timeout).
        # Each rank independently iterates the full val set; only rank 0 logs metrics.
        g_val = torch.Generator()
        g_val.manual_seed(0)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size_val // self.args.chunk_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=base_collate_fn,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            **loader_process_opts,
            generator=g_val,
        )
        return train_loader, val_loader, train_sampler

    def get_model(self):
        """Initialize the model."""
        # Model kwargs come from the shared helper that the eval loader also uses,
        # so a flag can never be honoured at eval but dropped in training.
        model_kwargs = build_model_kwargs(self.args, self.model_cls)
        # Crash at step 0 rather than train with a silently-defaulted flag.
        assert_model_kwargs_complete(self.args, self.model_cls, model_kwargs)

        if dist.get_rank() == 0:
            print(f'model_kwargs: {model_kwargs}')

        print(f"[Rank {dist.get_rank()}] Instantiating model (loads backbone weights)...", flush=True)
        _model = self.model_cls(**model_kwargs)
        print(f"[Rank {dist.get_rank()}] Model instantiated.", flush=True)

        # Frozen visual backbones can be stored in bf16 when AMP compute is bf16.
        if not self.args.finetune_backbone and hasattr(_model, "encoder") and hasattr(_model.encoder, "backbone"):
            _model.encoder.backbone.eval()
            if self.amp_dtype == torch.bfloat16:
                _model.encoder.backbone.to(dtype=torch.bfloat16)
                if dist.get_rank() == 0:
                    print("Frozen vision backbone cast to bfloat16 (AMP bf16 mode).")

        # Print basic modules' parameters
        if dist.get_rank() == 0:
            count_parameters(_model)
            # delta_M is a RoPE-space correction, whereas the legacy RT mode is
            # a physical transform. Keep the underlying flag for compatibility.
            mode = getattr(getattr(_model, 'prediction_head', None), 'view_align_mode', 'none')
            if mode != 'none':
                label = 'physical SE(3) correction' if mode == 'physical_se3' else 'per-camera RoPE correction'
                print(f"\n{label} enabled (view_align_mode={mode})")

        # Useful for some models to ensure parameters are contiguous
        for name, param in _model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        return _model

    @torch.no_grad()
    def get_workspace_normalizer(self, ndims=3):
        print("Computing workspace normalizer...")

        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            actions_only=True,
            chunk_size=self.args.chunk_size
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=max(self.args.batch_size, 64) // self.args.chunk_size,
            collate_fn=actions_collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        # Loop and compute action min-max
        min_, max_ = torch.ones(ndims) * 10000, -torch.ones(ndims) * 10000
        for sample in tqdm(data_loader):
            action = sample["action"][..., :ndims].reshape([-1, ndims])
            min_ = torch.min(min_, action.min(0).values)
            max_ = torch.max(max_, action.max(0).values)

        min_ = min_ - self.args.workspace_normalizer_buffer
        max_ = max_ + self.args.workspace_normalizer_buffer

        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        if self.args.finetune_backbone:
            optimizer_grouped_parameters.append({
                "params": [], "weight_decay": self.args.wd,
                "lr": self.args.backbone_lr
            })

        # Collect names of all norm parameters
        norm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LocalResponseNorm,
            torch.nn.RMSNorm
        )
        norm_param_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, norm_types):
                for param_name, _ in module.named_parameters(recurse=False):
                    norm_param_names.add(f"{module_name}.{param_name}")

        # Now split parameters based on name
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in norm_param_names or name.endswith(".bias"):
                optimizer_grouped_parameters[0]["params"].append(param)
            elif self.args.finetune_backbone and 'backbone' in name:
                optimizer_grouped_parameters[2]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)

        for i, g in enumerate(optimizer_grouped_parameters):
            print(i, len(g['params']))


        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95),
            # foreach=True
            fused=True,
        )
        return optimizer

    def _seed(self):
        """Seed every RNG stream from (args.seed, rank), if a seed was given.

        Off by default: no run to date has been seeded -- torch.manual_seed is
        never called, so two identical submissions already produce different
        weights. Making it unconditional would change experiments in flight, so
        it is opt-in. With a seed set, a run becomes reproducible and a resume
        becomes verifiable by comparing weights against an uninterrupted run.
        """
        seed = getattr(self.args, "seed", None)
        if seed is None:
            return
        seed_everything(int(seed), dist.get_rank())
        if dist.get_rank() == 0:
            print(f"Seeded all RNG streams from seed={seed} "
                  f"(per-rank offset applied)", flush=True)

    def main(self):
        """Run main training/testing pipeline."""
        rank = dist.get_rank()

        # Before the loaders and the model, so dataset shuffling and weight
        # init are both covered.
        self._seed()

        print(f"[Rank {rank}] Building data loaders...", flush=True)
        train_loader, val_loader, train_sampler = self.get_loaders()
        print(f"[Rank {rank}] Data loaders ready.", flush=True)

        # If train_epochs is set, derive train_iters from dataset size so runs are
        # batch-size invariant in samples-seen. len(train_loader) already accounts
        # for batch_size and world_size (DistributedSampler), so doubling batch_size
        # halves train_iters and samples_seen = train_epochs * len(dataset) is fixed.
        train_epochs = getattr(self.args, 'train_epochs', None)
        if train_epochs is not None:
            steps_per_epoch = len(train_loader)
            derived_iters = int(train_epochs) * steps_per_epoch
            if rank == 0:
                print(
                    f"train_epochs={train_epochs} → train_iters={derived_iters} "
                    f"(steps_per_epoch={steps_per_epoch})"
                )
                if self.args.train_iters is not None:
                    print(f"  (overriding configured train_iters={self.args.train_iters})")
            self.args.train_iters = derived_iters

        print(f"[Rank {rank}] Building model...", flush=True)
        model = self.get_model()
        print(f"[Rank {rank}] Loading tokenizer...", flush=True)
        _text_backbone = getattr(self.args, 'text_backbone', None) or self.args.backbone
        self.tokenizer = fetch_tokenizers(_text_backbone)
        print(f"[Rank {rank}] Model + tokenizer ready.", flush=True)

        dummy = getattr(self.args, 'benchmark_dummy_data', False)
        if not dummy and not (self.args.checkpoint and os.path.exists(self.args.checkpoint)):
            normalizer = self.get_workspace_normalizer()
            model.workspace_normalizer.copy_(normalizer)
            dist.barrier(device_ids=[torch.cuda.current_device()])

        print(f"[Rank {rank}] Moving model to GPU and wrapping DDP...", flush=True)
        # Move model to devices FIRST before creating optimizer
        if torch.cuda.is_available():
            model = model.cuda()
            # Enable TF32 for faster matmuls on Ampere+ GPUs
            torch.set_float32_matmul_precision('high')

        # Compile before DDP so torch.compile sees the raw module graph
        if self.args.use_compile:
            model.compute_loss = torch.compile(model.compute_loss, fullgraph=True)


        # Wrap in DDP
        # Note: find_unused_parameters=False for better performance
        # If you get unused parameter warnings, it means some model parameters
        # don't receive gradients, which should be investigated and fixed
        model = DistributedDataParallel(
            model, device_ids=[self.args.local_rank],
            static_graph=True,
            find_unused_parameters=False,
            bucket_cap_mb=10,
            gradient_as_bucket_view=True,
        )

        print(f"[Rank {rank}] DDP ready.", flush=True)

        # NOW create optimizer with CUDA/DDP parameters
        optimizer = self.get_optimizer(model)
        if self.run_mode == "train" and self.args.train_iters is None:
            raise ValueError(
                "train_iters must be set for training. Set in config (e.g. experiment yaml) or CLI: train_iters=100000"
            )
        lr_scheduler = fetch_scheduler(
            self.args.lr_scheduler, optimizer, self.args.train_iters
        )

        # Watch model with wandb
        if dist.get_rank() == 0 and self.run_mode == "train":
            if getattr(self.args, 'wandb_watch_model', False):
                # Reuses the validation cadence deliberately: gradient histograms are
                # expensive and only interesting at the same points we validate.
                wandb.watch(model, log='all', log_freq=self.args.val_interval_steps)

        # Initialize EMA copy
        ema_model = deepcopy(model)
        self.ema = EMA()

        # Check for a checkpoint (skipped when benchmark_dummy_data=true so start_iter stays 0)
        # Priority: resume checkpoint (run's own last.pth) > pretrained (fine-tuning seed) > scratch.
        # This means re-submitting the same slurm script safely resumes without editing it.
        start_iter, best_loss = 0, None
        pretrained_ckpt = getattr(self.args, 'pretrained_checkpoint', None)
        resume_ckpt = self.args.checkpoint
        if not dummy and resume_ckpt and os.path.exists(resume_ckpt):
            start_iter, best_loss = self.load_checkpoint(model, ema_model, optimizer)
            print(f"[Rank {dist.get_rank()}] Loaded checkpoint: {resume_ckpt} (resuming from step {start_iter})")
        elif not dummy and pretrained_ckpt and os.path.exists(pretrained_ckpt):
            self.load_pretrained_checkpoint(model, ema_model, pretrained_ckpt)
            print(f"[Rank {dist.get_rank()}] Loaded pretrained weights: {pretrained_ckpt} (training from step 0)")
        elif not dummy and pretrained_ckpt:
            print(f"[Rank {dist.get_rank()}] Pretrained checkpoint not found: {pretrained_ckpt} — starting from scratch")
        elif not dummy and resume_ckpt:
            print(f"[Rank {dist.get_rank()}] Checkpoint not found: {resume_ckpt} — starting from scratch")
        else:
            print(f"[Rank {dist.get_rank()}] No checkpoint specified — starting from scratch")
        print(model.module.workspace_normalizer)

        # Eval only (offline validation, no training)
        if self.run_mode == "eval_offline":
            if dist.get_rank() == 0:
                print("Test evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id=-1,
                    val_iters=-1
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return ema_model if self.args.use_ema else model

        # Step the lr scheduler to the current step
        for _ in range(start_iter):
            lr_scheduler.step()

        # Step the sampler to the currect "epoch"
        samples_per_epoch = len(train_loader)
        epoch = start_iter // samples_per_epoch + 1
        train_sampler.set_epoch(epoch)  # ensures new batches are sampled
        # ...and to the right point *inside* that epoch. Setting only the epoch
        # restarts at batch 0, so a resumed run re-consumed the first
        # start_iter % samples_per_epoch batches and diverged from the data
        # order it would have seen uninterrupted.
        batch_in_epoch = start_iter % samples_per_epoch
        if batch_in_epoch:
            train_sampler.set_skip(batch_in_epoch)
            if dist.get_rank() == 0:
                print(f"Resuming mid-epoch: skipping {batch_in_epoch} batch(es) "
                      f"of epoch {epoch} ({samples_per_epoch} batches/epoch)")

        # Initialize per-rank benchmark logger (enabled via benchmark=true in config/CLI)
        bench_warmup = getattr(self.args, 'benchmark_warmup_steps', 0)
        self._profiler = None
        self._profile_start_step = None
        self._profile_n_steps = None
        if getattr(self.args, 'benchmark', False):
            bench_freq = getattr(self.args, 'benchmark_log_freq', 50)
            rank = dist.get_rank()
            bench_path = self.args.log_dir / f"benchmark_rank{rank}.txt"
            self.benchmark_logger = BenchmarkLogger(bench_path, rank, dist.get_world_size(), bench_freq)
            if rank == 0:
                print(f"Benchmark logging enabled → {bench_path} "
                      f"(warmup={bench_warmup} steps, log every {bench_freq} steps)")
            # torch.profiler runs on rank 0 only, starting after warmup.  A
            # zero value is useful for lightweight loader benchmarks.
            n = getattr(self.args, 'benchmark_profile_steps', 10)
            if dist.get_rank() == 0 and n > 0:
                self._profile_start_step = start_iter + bench_warmup
                self._profile_n_steps = n
                profiler_dir = self.args.log_dir / "profiler"
                profiler_dir.mkdir(parents=True, exist_ok=True)
                self._profiler = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir)),
                )
                print(f"Torch profiler will capture {n} steps starting at step {self._profile_start_step}")
                print(f"Trace will be written to {profiler_dir} (view via `tensorboard --logdir {self.args.log_dir}` with the torch-tb-profiler plugin)")

        # Preemption: grogu signals a preempted job and kills it GraceTime
        # (120 s) later. The checkpoint here is trainable-params only (~17 MB)
        # and writes in well under a second, so that window is ample -- but it
        # is only usable if we actually listen for the signal.
        preempt = PreemptionGuard()

        # Training loop
        model.train()
        iter_loader = iter(train_loader)
        for step_id in trange(start_iter, self.args.train_iters):
            t_step_start = time.perf_counter()
            try:
                sample = next(iter_loader)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                iter_loader = iter(train_loader)
                sample = next(iter_loader)
            data_ms = (time.perf_counter() - t_step_start) * 1000

            # Enter profiler before the step so all n steps are fully captured
            if self._profiler is not None and step_id == self._profile_start_step:
                self._profiler.__enter__()

            try:
                timing = self.train_one_step(model, optimizer, lr_scheduler, sample, step_id)
            except Exception as e:
                # Save an emergency checkpoint before dying so the next torchrun restart
                # (--max-restarts) picks up from the current step instead of the last
                # periodic save.  Only rank 0 writes to avoid races.
                print(f"[Rank {dist.get_rank()}] Step {step_id} failed: {e}", flush=True)
                if dist.get_rank() == 0:
                    emergency_path = self.args.log_dir / "last.pth"
                    self._saver.flush()
                    _atomic_save({
                        "weight": self._trainable_state_dict(model),
                        "ema_weight": self._trainable_state_dict(ema_model) if self.args.use_ema else None,
                        "optimizer": _cpu_snapshot(optimizer.state_dict()),
                        "iter": step_id,
                        "best_loss": best_loss,
                        "config": _args_to_dict(self.args),
                    }, emergency_path)
                    print(f"Emergency checkpoint saved to {emergency_path}", flush=True)
                raise  # re-raise so the process exits and torchrun can restart

            self.ema.step(model, ema_model, self.args.use_ema, step_id)

            if self.benchmark_logger is not None and timing is not None:
                if step_id == bench_warmup - 1:
                    # warmup just finished — reset peak so measured window is clean
                    torch.cuda.reset_peak_memory_stats()
                if step_id >= bench_warmup:
                    total_ms = (time.perf_counter() - t_step_start) * 1000
                    batch_size = sample['action'].shape[0]
                    self.benchmark_logger.record(
                        data_ms, timing['fwd_ms'], timing['bwd_ms'], timing['opt_ms'], total_ms, batch_size
                    )
                    if (step_id + 1) % self.benchmark_logger.log_freq == 0:
                        self.benchmark_logger.flush(step_id + 1)

            if self._profiler is not None and step_id >= self._profile_start_step:
                self._profiler.step()

                steps_profiled = step_id - self._profile_start_step + 1
                if steps_profiled >= self._profile_n_steps:
                    self._profiler.__exit__(None, None, None)
                    stats = self._profiler.key_averages()
                    print("\n" + "─" * 80)
                    print(f"Torch Profiler — top ops by CUDA time (steps {self._profile_start_step}–{step_id})")
                    print("─" * 80)
                    print(stats.table(sort_by="cuda_time_total", row_limit=20))
                    print("─" * 80)
                    def _cuda_us(s):
                        for attr in ('cuda_time_total', 'self_cuda_time_total',
                                     'device_time_total', 'self_device_time_total'):
                            if hasattr(s, attr):
                                return getattr(s, attr)
                        return s.cpu_time_total

                    nccl_stats = sorted(
                        [s for s in stats if "nccl" in s.key.lower()],
                        key=_cuda_us, reverse=True,
                    )
                    if nccl_stats:
                        print(f"\nNCCL communication ops:")
                        print(f"  {'Name':<40} | {'CUDA Total (us)':<16} | Calls")
                        print(f"  {'-'*40}-+-{'-'*16}-+------")
                        for s in nccl_stats:
                            print(f"  {s.key:<40} | {_cuda_us(s):<16.2f} | {s.count}")
                    else:
                        print("\nNo NCCL ops captured (all-reduce may be async/overlapped).")

                    self._profiler = None  # don't profile again

            # A preempted job should spend its grace window saving, not dying.
            # The flag is agreed across ranks first: signal delivery is not
            # simultaneous, and a one-sided save would deadlock on the gather.
            # Checked on a fixed cadence rather than every step so the extra
            # collective stays off the hot path; 10 steps is a few seconds of
            # latency against a 120 s grace window.
            stopping = False
            if (step_id + 1) % _PREEMPT_CHECK_EVERY == 0:
                stopping = self._agree(preempt.check())

            if stopping or (step_id + 1) % self.args.ckpt_interval_steps == 0:
                # gather_rng is collective, so every rank must reach it.
                rng_all = gather_rng(capture_rng(), dist.get_world_size())
                if dist.get_rank() == 0:
                    self._save_rolling_checkpoint(
                        model, ema_model, optimizer, step_id, best_loss,
                        rng_all=rng_all, epoch=epoch,
                        batch_in_epoch=(step_id + 1) % samples_per_epoch,
                        samples_per_epoch=samples_per_epoch,
                    )
            if stopping:
                if dist.get_rank() == 0:
                    self._saver.flush()
                    preempt.confirm()
                    print(f"Preempted at step {step_id + 1}; checkpoint written, "
                          f"exiting for requeue", flush=True)
                dist.barrier(device_ids=[torch.cuda.current_device()])
                break

            interm_freq = getattr(self.args, "interm_ckpt_interval_steps", None)
            if interm_freq and (step_id + 1) % interm_freq == 0 and dist.get_rank() == 0:
                self._save_interm_checkpoint(model, ema_model, step_id, best_loss)

            if (step_id + 1) % self.args.val_interval_steps == 0:
                model.eval()
                if dist.get_rank() == 0:
                    print("Train evaluation.......")
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    train_loader, step_id,
                    val_iters=10,
                    split='train'
                )
                if dist.get_rank() == 0:
                    print("Test evaluation.......")
                new_loss = self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id,
                    val_iters=1250
                )
                if dist.get_rank() == 0:
                    best_loss = self.save_checkpoint(
                        model, ema_model, optimizer, step_id,
                        new_loss, best_loss
                    )
                model.train()
            dist.barrier(device_ids=[torch.cuda.current_device()])

        self._saver.close()
        return ema_model if self.args.use_ema else model

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        pass  # implement in children

    def _model_forward(self, model, sample, training=True, augment=None):
        # `augment` defaults to `training` but can be forced off so the validation
        # pass can score the training loss on un-augmented observations.
        if augment is None:
            augment = training
        with torch.profiler.record_function("step/prepare_batch"):
            action, action_mask, rgbs, rgb2d, pcds, instr, prop = self.prepare_batch(
                sample, augment=augment
            )
        if self.args.pre_tokenize:
            with torch.profiler.record_function("step/tokenize"):
                instr = self.tokenizer(instr).cuda(non_blocking=True)
        with torch.profiler.record_function("step/model_forward"):
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                out = model(
                    action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                    run_inference=not training,
                )
        return out  # loss if training, else action

    def _forward_loss(self, model, sample):
        """Training-objective loss on a batch, no grad and no augmentation.

        `run_inference=True` (the validation path) returns a predicted action, not
        a loss, so the flow-matching objective has to be re-run with
        `run_inference=False` to get a comparable scalar.
        """
        return self._model_forward(model, sample, training=True, augment=False)

    def train_one_step(self, model, optimizer, lr_scheduler, sample, step_id=None):
        """Run a single training step. Returns GPU timing dict when benchmark_logger is set."""
        benchmark = getattr(self, 'benchmark_logger', None) is not None
        if benchmark:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            opt_start = torch.cuda.Event(enable_timing=True)
            opt_end = torch.cuda.Event(enable_timing=True)

        optimizer.zero_grad()


        if benchmark:
            fwd_start.record()
        loss = self._model_forward(model, sample, training=True)
        if benchmark:
            fwd_end.record()

        if benchmark:
            bwd_start.record()
        loss.backward()

        # Clip gradients
        for p in model.parameters():
            if p.grad is not None and not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        # Update
        if benchmark:
            bwd_end.record()
            opt_start.record()
        optimizer.step()
        if benchmark:
            opt_end.record()

        # Step the lr scheduler
        lr_scheduler.step()

        # Buffered logging: accumulate detached tensors on-device, sync once per
        # `train_log_freq` steps so the wandb path costs ~1 .item() window instead
        # of ~10+ per step. Loss/grad_norm/extrinsics are averaged over the window.
        if dist.get_rank() == 0 and step_id is not None:
            if not hasattr(self, '_log_buf'):
                self._log_buf = {
                    'loss': [], 'grad_norm': [], 'extrinsics_learn': [],
                    'extrinsics_pred': [], 'view_align': [], 'ee_aux_loss': [],
                }
                self._log_freq = getattr(self.args, 'train_log_freq', 50)

            self._log_buf['loss'].append(loss.detach())
            self._log_buf['grad_norm'].append(grad_norm.detach())

            base_model = model.module if hasattr(model, 'module') else model

            base_ee_aux = getattr(base_model, '_last_ee_aux_loss', None)
            if base_ee_aux is not None:
                self._log_buf['ee_aux_loss'].append(base_ee_aux)
            prediction_head = base_model.prediction_head

            # NOTE: BATCH STATISTIC FOR SINGLE SCENE -- doesn't work for multi-scene.
            if hasattr(prediction_head, 'predict_extrinsics') and prediction_head.predict_extrinsics:
                if getattr(prediction_head, '_last_predicted_cam_params', None) is not None:
                    cam_params = prediction_head._last_predicted_cam_params
                    if cam_params.dim() == 2 and cam_params.shape[-1] == 6:
                        # Keep full (B, 6) so flush can compute window-wide mean+std in one shot
                        self._log_buf['extrinsics_pred'].append(cam_params.detach())
                    elif cam_params.dim() >= 3 and cam_params.shape[-1] == cam_params.shape[-2]:
                        # delta_M is a representation-space view-alignment
                        # matrix. Log it separately from the legacy physical-RT
                        # diagnostics so dashboards do not imply calibration
                        # recovery.
                        self._log_buf['view_align'].append(cam_params.detach())

            if (step_id + 1) % self._log_freq == 0:
                # Stack everything first (no sync), then a single .tolist() drains the window.
                loss_mean      = torch.stack(self._log_buf['loss']).mean()
                grad_norm_mean = torch.stack(self._log_buf['grad_norm']).mean()

                metrics = {
                    'train/loss':          loss_mean.item(),
                    'train/grad_norm':     grad_norm_mean.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                }

                if self._log_buf['ee_aux_loss']:
                    metrics['train/ee_aux_loss'] = torch.stack(self._log_buf['ee_aux_loss']).mean().item()

                if self._log_buf['extrinsics_learn']:
                    mean = torch.stack(self._log_buf['extrinsics_learn']).mean(dim=0)  # (6,)
                    rot_mag   = torch.norm(mean[:3])
                    trans_mag = torch.norm(mean[3:])
                    vals = mean.tolist()
                    for i, axis in enumerate(('x', 'y', 'z')):
                        metrics[f'extrinsics/cam_axis_angle_{axis}']  = vals[i]
                        metrics[f'extrinsics/cam_translation_{axis}'] = vals[3 + i]
                    metrics['extrinsics/rotation_angle_rad']    = rot_mag.item()
                    metrics['extrinsics/translation_magnitude'] = trans_mag.item()

                if self._log_buf['extrinsics_pred']:
                    all_params = torch.cat(self._log_buf['extrinsics_pred'], dim=0)  # (N_window*B, 6)
                    mean = all_params.mean(dim=0)
                    std  = all_params.std(dim=0)
                    rot_mag   = torch.norm(mean[:3])
                    trans_mag = torch.norm(mean[3:])
                    rot_std   = std[:3].mean()
                    trans_std = std[3:].mean()
                    vals = mean.tolist()
                    for i, axis in enumerate(('x', 'y', 'z')):
                        metrics[f'extrinsics/cam_axis_angle_{axis}']  = vals[i]
                        metrics[f'extrinsics/cam_translation_{axis}'] = vals[3 + i]
                    metrics['extrinsics/rotation_angle_rad']    = rot_mag.item()
                    metrics['extrinsics/translation_magnitude'] = trans_mag.item()
                    metrics['extrinsics/rotation_std']          = rot_std.item()
                    metrics['extrinsics/translation_std']       = trans_std.item()

                if self._log_buf['view_align']:
                    matrices = torch.cat(self._log_buf['view_align'], dim=0).float()
                    identity = torch.eye(
                        matrices.shape[-1], device=matrices.device, dtype=matrices.dtype
                    )
                    deviation = torch.linalg.matrix_norm(matrices - identity, ord='fro', dim=(-2, -1))
                    metrics['view_align/frob_from_identity_mean'] = deviation.mean().item()
                    metrics['view_align/frob_from_identity_std'] = deviation.std().item()
                    if deviation.dim() == 2:
                        for cam_id, value in enumerate(deviation.mean(dim=0).tolist()):
                            metrics[f'view_align/frob_from_identity_camera_{cam_id}'] = value

                if getattr(self.args, 'use_wandb', True):
                    wandb.log(metrics, step=step_id)

                for k in self._log_buf:
                    self._log_buf[k].clear()

        if benchmark:
            # Synchronize once after all events are queued; event elapsed-time
            # measurements otherwise stay asynchronous and under-report work.
            torch.cuda.synchronize()
            return {
                'fwd_ms': fwd_start.elapsed_time(fwd_end),
                'bwd_ms': bwd_start.elapsed_time(bwd_end),
                'opt_ms': opt_start.elapsed_time(opt_end),
            }

        return None

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        values = {}
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break

            pred_action = self._model_forward(model, sample, training=False)
            gt_action = sample["action"].to(device='cuda', non_blocking=True)
            if self.args.relative_action:
                prop = sample["proprioception"].to(device='cuda', non_blocking=True)[:, :, 0]
                pred_action = relative_to_absolute(pred_action[:, :, 0], prop)
                gt_action = relative_to_absolute(gt_action[:, :, 0], prop)

            losses, losses_B = compute_metrics(pred_action, gt_action)

            # The training objective itself, so train/val loss are directly comparable.
            values.setdefault(f"{split}-losses/mean/loss", []).append(
                self._forward_loss(model, sample).detach().float()
            )

            # Gather global statistics — collect into lists, stack once at the end
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                values.setdefault(key, []).append(l)

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    values.setdefault(key, []).append(l[tasks == task].mean())

        # Log all statistics
        values = {k: torch.stack(v).mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                # Log to TensorBoard
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

                # Log to wandb
                if getattr(self.args, 'use_wandb', True):
                    wandb_metrics = {key.replace('-', '/'): val for key, val in values.items()}
                    wandb.log(wandb_metrics, step=step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return -values[f'{split}-losses/mean/traj_pos_acc_001']

    def load_pretrained_checkpoint(self, model, ema_model, ckpt_path):
        """Load weights from a pretrained checkpoint — no optimizer restore, iter resets to 0.

        Useful for fine-tuning: new heads (e.g. ee_predictor, camera_trunk) start randomly
        initialised when missing from the checkpoint; unexpected keys are ignored.
        """
        print(f"=> loading pretrained weights from '{ckpt_path}'")
        model_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "config" in model_dict:
            print(f"Pretrained checkpoint config: {model_dict['config']}")

        # ``strict=False`` still raises on same-name tensors whose shapes differ.
        # A warm start across a changed proprio-history length should retain every
        # compatible visual/Video-DeltaM/decoder weight and explicitly leave only
        # length-dependent proprio modules at their fresh initialization.
        current = model.state_dict()
        source = model_dict["weight"]
        compatible = {
            key: value for key, value in source.items()
            if key in current and current[key].shape == value.shape
        }
        skipped_shapes = {
            key: (tuple(value.shape), tuple(current[key].shape))
            for key, value in source.items()
            if key in current and current[key].shape != value.shape
        }
        msn, unxpct = model.load_state_dict(compatible, strict=False)
        if skipped_shapes:
            print("[pretrained] Shape-mismatched keys kept freshly initialized:")
            for key, (old_shape, new_shape) in skipped_shapes.items():
                print(f"  {key}: checkpoint={old_shape}, model={new_shape}")
        if msn:
            print(f"[pretrained] Missing keys (will be randomly initialised): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"[pretrained] Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("[pretrained] All keys matched.")

        if model_dict.get("ema_weight") is not None and ema_model is not None:
            ema_current = ema_model.state_dict()
            ema_compatible = {
                key: value for key, value in model_dict["ema_weight"].items()
                if key in ema_current and ema_current[key].shape == value.shape
            }
            msn_e, unxpct_e = ema_model.load_state_dict(ema_compatible, strict=False)
            if msn_e:
                print(f"[pretrained EMA] Missing keys: {len(msn_e)}")
            if unxpct_e:
                print(f"[pretrained EMA] Unexpected keys (ignored): {len(unxpct_e)}")

        del model_dict
        torch.cuda.empty_cache()
        print(f"=> pretrained weights loaded from '{ckpt_path}'")

    def load_checkpoint(self, model, ema_model, optimizer):
        """Load from checkpoint."""
        print("=> trying checkpoint '{}'".format(self.args.checkpoint))
        if not os.path.exists(self.args.checkpoint):
            print('Warning: checkpoint was not found, starting from scratch')
            print('The main process will compute workspace bounds')
            return 0, None

        model_dict = torch.load(
            self.args.checkpoint,
            map_location="cpu",
            weights_only=False
        )
        if "config" in model_dict:
            print(f"Checkpoint config: {model_dict['config']}")
        # Load weights flexibly
        msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")
        # EMA weights
        if model_dict.get("ema_weight") is not None:
            msn_ema, unxpct_ema = ema_model.load_state_dict(model_dict["ema_weight"], strict=False)
            if msn_ema:
                print(f"EMA missing keys (not found in checkpoint): {len(msn_ema)}")
                print(msn_ema)
            if unxpct_ema:
                print(f"EMA unexpected keys (ignored): {len(unxpct_ema)}")
                print(unxpct_ema)
            if not msn_ema and not unxpct_ema:
                print("EMA: all keys matched successfully!")
        if self.run_mode == "train":
            if 'optimizer' in model_dict:
                optimizer.load_state_dict(model_dict["optimizer"])
        start_iter = model_dict.get("iter", 0)
        best_loss = model_dict.get("best_loss", None)

        # Restore the RNG streams, so flow-matching noise and dropout continue
        # the original sequence. Checkpoints written before this existed simply
        # have no "rng" key and fall back to the old behaviour.
        rng_all = model_dict.get("rng")
        if rng_all is None:
            if dist.get_rank() == 0:
                print("=> checkpoint carries no RNG state (written by an older "
                      "revision); resuming with fresh RNG — the loss curve will "
                      "not match the original run exactly")
        elif model_dict.get("world_size") != dist.get_world_size():
            if dist.get_rank() == 0:
                print(f"=> checkpoint was written at world_size="
                      f"{model_dict.get('world_size')} but this run is "
                      f"{dist.get_world_size()}; per-rank RNG cannot be restored, "
                      f"resuming with fresh RNG")
        else:
            restore_rng(rng_all[dist.get_rank()])
            if dist.get_rank() == 0:
                print(f"=> restored RNG state for {len(rng_all)} rank(s)")

        print("=> loaded successfully '{}' (step {})".format(
            self.args.checkpoint, model_dict.get("iter", 0)
        ))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss

    @staticmethod
    def _agree(local: bool) -> bool:
        """True if *any* rank says True.

        Used for the preemption flag. Slurm does not deliver the signal to every
        rank at the same instant, so without agreeing first some ranks would
        enter the RNG gather and others would not, and the job would hang
        instead of checkpointing.
        """
        if dist.get_world_size() == 1:
            return local
        flag = torch.tensor([1 if local else 0], dtype=torch.int32,
                            device=torch.cuda.current_device())
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _save_rolling_checkpoint(self, model, ema_model, optimizer, step_id, best_loss,
                                 rng_all=None, epoch=None, batch_in_epoch=None,
                                 samples_per_epoch=None):
        """Save rolling checkpoint to last.pth only (no per-step files, no frozen backbone)."""
        state = {
            "weight": self._trainable_state_dict(model),
            "ema_weight": self._trainable_state_dict(ema_model) if ema_model is not None else None,
            "optimizer": _cpu_snapshot(optimizer.state_dict()),
            "iter": step_id + 1,
            "best_loss": best_loss,
            "config": _args_to_dict(self.args),
        }
        # Resume state. Without the RNG block a resumed run restores weights and
        # data order correctly and still draws different flow-matching noise and
        # dropout masks, so its loss curve silently departs from the
        # uninterrupted one. Recorded per rank because RNG state is per rank.
        if rng_all is not None:
            state["rng"] = rng_all
            state["world_size"] = dist.get_world_size()
        if epoch is not None:
            state["epoch"] = epoch
            state["batch_in_epoch"] = batch_in_epoch
            state["samples_per_epoch"] = samples_per_epoch
        self._saver.save(state, self.args.log_dir / "last.pth")

    def _trainable_state_dict(self, model):
        """State dict with only trainable params + workspace_normalizer (skips frozen backbone)."""
        base = model.module if hasattr(model, "module") else model
        trainable = {n for n, p in base.named_parameters() if p.requires_grad}
        full = model.state_dict()
        # DDP prefixes keys with "module." — normalise to match base.named_parameters()
        prefix = "module." if next(iter(full)).startswith("module.") else ""
        keep = {}
        for k, v in full.items():
            bare = k[len(prefix):]
            if bare in trainable or bare == "workspace_normalizer":
                keep[k] = v.cpu()
        return keep

    def _save_interm_checkpoint(self, model, ema_model, step_id, best_loss):
        """Save periodic checkpoint (trainable weights only for eval)."""
        ckpt_path = self.args.log_dir / f"interm_step_{step_id + 1}.pth"
        state = {
            "weight": self._trainable_state_dict(model),
            "ema_weight": self._trainable_state_dict(ema_model) if ema_model is not None else None,
            "iter": step_id + 1,
            "best_loss": best_loss,
            "config": _args_to_dict(self.args),
        }
        self._saver.save(state, ckpt_path)
        print(f"Saved periodic checkpoint: {ckpt_path}", flush=True)
        self._submit_checkpoint_evals(step_id + 1)

    def _submit_checkpoint_evals(self, step: int) -> None:
        """Submit every configured eval recipe for one newly saved checkpoint.

        Each recipe delegates to the ledger-backed checkpoint_ladder launcher,
        constrained to this exact step.  This is intentionally asynchronous:
        training only invokes ``sbatch`` and never waits for evaluation work.
        """
        entries = getattr(self.args, "checkpoint_evals", None) or []
        if isinstance(entries, DictConfig):
            entries = OmegaConf.to_container(entries, resolve=True)
        if not entries:
            return
        repo_root = Path(__file__).resolve().parents[2]
        launcher = repo_root / "scripts/eval/checkpoint_ladder.py"
        for raw_entry in entries:
            entry = OmegaConf.to_container(raw_entry, resolve=True) if isinstance(raw_entry, DictConfig) else dict(raw_entry)
            recipe = entry.get("recipe")
            method_id = entry.get("method_id")
            if not recipe or not method_id:
                raise ValueError("Each checkpoint_evals entry requires recipe and method_id")
            min_step = entry.get("min_step")
            max_step = entry.get("max_step")
            if (min_step is not None and step < int(min_step)) or (max_step is not None and step > int(max_step)):
                continue
            command = [
                sys.executable, str(launcher),
                "--checkpoint-dir", str(self.args.log_dir),
                "--method-id", str(method_id),
                "--config", str(recipe),
                "--min-step", str(step), "--max-step", str(step), "--submit",
            ]
            try:
                result = subprocess.run(command, cwd=repo_root, check=True, text=True, capture_output=True)
                print(f"[checkpoint eval] step={step}, recipe={recipe}, method={method_id}: {result.stdout.strip()}", flush=True)
            except subprocess.CalledProcessError as error:
                # A failed eval submission must never kill or stall training;
                # the saved checkpoint remains available for manual submission.
                print(
                    f"[checkpoint eval] submission failed at step={step}, recipe={recipe}: "
                    f"{error.stderr.strip() or error.stdout.strip()}",
                    flush=True,
                )

    def save_checkpoint(self, model, ema_model, optimizer,
                        step_id, new_loss, best_loss):
        """Save best and intermediate checkpoints; rolling last-K is handled separately."""
        model_state = self._trainable_state_dict(model)
        ema_state = self._trainable_state_dict(ema_model) if self.args.use_ema else None
        config_container = _args_to_dict(self.args)

        optimizer_state = _cpu_snapshot(optimizer.state_dict())

        def _save(path):
            self._saver.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "optimizer": optimizer_state,
                "iter": step_id + 1,
                "best_loss": best_loss,
                "config": config_container,
            }, path)
            if getattr(self.args, 'wandb_save_checkpoints', True):
                wandb.save(str(path), base_path=str(self.args.log_dir))

        # Best checkpoint
        if best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            _save(self.args.log_dir / "best.pth")

        return best_loss


# Compatibility imports for old checkpoints, scripts, and external launchers.
# New code must import these helpers from data.batch.
