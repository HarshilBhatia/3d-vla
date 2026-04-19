import torch


def compute_metrics(pred, gt):
    # pred/gt are (B, L, 3+rot+1)
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    # symmetric quaternion eval
    quat_l1 = (pred[..., 3:-1] - gt[..., 3:-1]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:-1] + gt[..., 3:-1]).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
    # gripper openess
    openess = ((pred[..., -1:] >= 0.5) == (gt[..., -1:] >= 0.5)).bool()
    tr = 'traj_'

    # Trajectory metrics
    ret_1, ret_2 = {
        tr + 'pos_l2': pos_l2.mean(),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
        tr + 'rot_l1': quat_l1.mean(),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
        tr + 'gripper': openess.flatten().float().mean()
    }, {
        tr + 'pos_l2': pos_l2.mean(-1),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
        tr + 'rot_l1': quat_l1.mean(-1),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
    }

    return ret_1, ret_2


class BenchmarkLogger:
    """Per-rank benchmark logger for DDP training profiling.

    Measures per-step: data loading, forward pass, backward pass (includes DDP
    NCCL allreduce), optimizer step, GPU memory, and throughput. Each DDP rank
    writes its own file so load imbalance is visible.
    """

    def __init__(self, log_path, rank, world_size, log_freq=50):
        self.log_path = log_path
        self.rank = rank
        self.world_size = world_size
        self.log_freq = log_freq
        self._reset_accumulators()
        torch.cuda.reset_peak_memory_stats()

        with open(log_path, 'a') as f:
            f.write(f"\n# Benchmark log  rank={rank}  world_size={world_size}\n")
            f.write(
                f"# step | data_ms | fwd_ms | bwd_ms (incl. DDP allreduce) | "
                f"opt_ms | total_ms | samp/s | mem_alloc_gb | mem_peak_gb\n"
            )
            f.write("#" + "-" * 110 + "\n")

    def _reset_accumulators(self):
        self._data_ms = []
        self._fwd_ms = []
        self._bwd_ms = []
        self._opt_ms = []
        self._total_ms = []
        self._n_samples = []

    def record(self, data_ms, fwd_ms, bwd_ms, opt_ms, total_ms, batch_size):
        self._data_ms.append(data_ms)
        self._fwd_ms.append(fwd_ms)
        self._bwd_ms.append(bwd_ms)
        self._opt_ms.append(opt_ms)
        self._total_ms.append(total_ms)
        self._n_samples.append(batch_size)

    def flush(self, step):
        if not self._data_ms:
            return

        n = len(self._data_ms)
        avg = lambda lst: sum(lst) / n  # noqa: E731

        avg_data = avg(self._data_ms)
        avg_fwd = avg(self._fwd_ms)
        avg_bwd = avg(self._bwd_ms)
        avg_opt = avg(self._opt_ms)
        avg_total = avg(self._total_ms)
        avg_bs = avg(self._n_samples)

        # Throughput over the window
        samp_per_s = avg_bs * 1000.0 / avg_total if avg_total > 0 else 0.0

        # Memory snapshot (current allocated + peak since last reset)
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        line = (
            f"step={step:7d} | "
            f"data={avg_data:7.1f}ms | "
            f"fwd={avg_fwd:7.1f}ms | "
            f"bwd={avg_bwd:7.1f}ms | "
            f"opt={avg_opt:7.1f}ms | "
            f"total={avg_total:7.1f}ms | "
            f"tput={samp_per_s:7.1f}samp/s | "
            f"mem={mem_alloc:.3f}GB | "
            f"peak={mem_peak:.3f}GB\n"
        )

        with open(self.log_path, 'a') as f:
            f.write(line)

        self._reset_accumulators()
