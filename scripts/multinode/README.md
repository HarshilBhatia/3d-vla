# Multi-node training resilience layer

A launcher plus resilience wrapper around an arbitrary `train.py`. Model-agnostic
and hardware-agnostic: nothing here imports anything model-specific.

```
submit.py            submit loop, constraint fallback, fail-fast validation
job.sbatch           generic job body — holds no settings, all of them arrive as env
preflight.py         in-allocation checks, fails closed
runlog.py            live pool census + run-log aggregation
resilience/          importable package: checkpointer, RNG/data state, signals
tests/               the kill/resume correctness test (build this first)
bench/scaling.py     N=1,2,3,4 efficiency harness
config/example.yaml  every knob, annotated
```

## The contract with your trainer

Two touchpoints, both via environment variables so `train_cmd` can be any shell
string (a conda wrapper, `torchrun ...`) without the launcher appending argv to
it — appending breaks the moment the command is a wrapper.

| variable | meaning |
|---|---|
| `RESILIENCE_CKPT_DIR` | where to save and resume from |
| `RESILIENCE_CKPT_INTERVAL` | seconds between periodic checkpoints |
| `RESILIENCE_SENTINEL` | file that appears when preemption is imminent |
| `RESILIENCE_DONE_MARKER` | touch this once the checkpoint is on disk |

Minimal wiring:

```python
from resilience import Checkpointer, PreemptionGuard, ResumableDistributedSampler, StatefulLoader

ckpt  = Checkpointer(os.environ["RESILIENCE_CKPT_DIR"],
                     interval_seconds=float(os.environ["RESILIENCE_CKPT_INTERVAL"]),
                     rank=rank, world_size=world)
guard = PreemptionGuard()
step  = ckpt.resume(apply_state)          # 0 if nothing to resume

while step < total:
    train_one_step(...)
    step += 1
    if guard.check():                      # preemption signalled
        ckpt.maybe_save(step, build_state, force=True)
        guard.confirm()                    # lets the batch script requeue at once
        break
    ckpt.maybe_save(step, build_state)     # cheap no-op when not due
```

`build_state()` returns the model-specific half (weights, optimizer, ...).
`Checkpointer` owns the model-agnostic half: step, per-rank RNG, data position,
world size, provenance.

## Quick start

```bash
python scripts/multinode/runlog.py pools -p all --nodes 3   # what can satisfy N=3 right now
cp scripts/multinode/config/example.yaml myrun.yaml         # edit partition/N/ckpt_dir/train_cmd
python scripts/multinode/submit.py myrun.yaml --dry-run     # validate, print the sbatch lines
python scripts/multinode/submit.py myrun.yaml               # submit and supervise
python scripts/multinode/runlog.py summary                  # what actually happened
```

## Planning a placement

`plan.py` answers "what should I request right now, and what will I get?" from
live cluster state, then writes a config `submit.py` can consume.

```
python scripts/multinode/plan.py --gpus 4 --per-gpu-batch 8
python scripts/multinode/plan.py --gpus 4 --emit config/my_run.yaml \
    --train-cmd "bash -lc '...checkpoint=<log_dir>/last.pth'"
```

It exists because counting free GPUs is not enough. A request sized sensibly
by GPU count sat with a **two-day** start estimate: the node had 6 free GPUs
and 53 GB of 503 GB free host memory. Meanwhile every node with plenty of free
RAM and >=4 GPUs was a 2080Ti, which this model cannot run on at all. So the
planner considers GPUs, CPUs, memory, GPU model and recorded-bad nodes
together, and prints the specific blocker per node when nothing fits.

Placement is ranked by **throughput per wall-clock**: a configuration that can
start now beats a faster one that cannot, because queue time is dead time. It
picks the GPU model and how to spread the ranks, but takes `--per-gpu-batch`
from you — batch sets the global batch, which changes the experiment, so it is
not the scheduler's to choose.

Predictions come from the measured tables at the top of the file
(`RANK_THROUGHPUT_B8`, `INTRA_NODE_EFF`, `INTER_NODE_EFF`, `BATCH_SCALE`), not
from a model of the hardware. Two deliberate refusals: `rtx2080ti` is reported
as unusable rather than slow, and `H200` / `6000Blackwell` / `A6000Ada` are
skipped as never-benchmarked rather than guessed at. Re-measure with
`bench/scaling.py` if the model or data pipeline changes.

## Tests

```bash
python -m pytest scripts/multinode/tests/test_kill_resume.py -v   # CPU, ~20s
TOY_DEVICE=cuda python -m pytest scripts/multinode/tests/ -v      # on a GPU node
```

`test_kill_resume_matches_uninterrupted` runs 100 steps, kills at 50, resumes,
and asserts both the sample order and the per-step losses match the
uninterrupted run. It is the only component with a real correctness criterion;
everything else is plumbing around it.

It earned its keep immediately. The first run failed with matching data order
but losses diverging at step 51, because **every call to `iter(DataLoader)`
draws one int64 from the global torch RNG** to derive `_base_seed`. A resumed
run creates an iterator the uninterrupted run never created, so the global
stream ends up one draw out of step and every later dropout mask differs —
weights and data order both restored correctly, losses still silently wrong.
`StatefulLoader` now owns a dedicated generator and checkpoints it.

### Testing the real trainer: why not compare weights

`tests/run_resume_matrix.sh` runs four paired jobs against `main.py` in
parallel and `--compare` checks them. The obvious assertion — weights after N
steps should match an uninterrupted run — **does not work on this stack**, and
the control proves it: two runs with the same seed, no interruption, differ by
`6.3e-03`. `torch.compile`, cudnn autotune and scatter atomics make the
forward nondeterministic, so the noise floor is larger than a broken resume's
signature. Any weight comparison here passes or fails for the wrong reason.

`compare_indices.py` asserts two *discrete* quantities instead, both immune to
float nondeterminism:

* **RNG state bytes** at the same step, per rank. RNG advances by draw *count*,
  not draw *values*, so a missing or extra random draw shows up exactly.
* **The index sequence** — which samples each batch contained, from an
  audit trail the sampler writes when `RESILIENCE_INDEX_LOG` is set (no
  overhead otherwise).

Comparing index sequences on their *overlap* is not sufficient, and that
mistake hid a real bug for one round: a resume that restarts at batch 0
re-yields the same head indices, so the overlap matches while the run has
trained the first half twice and never reached the second. The assertion that
discriminates is **coverage**: a 40-step run must cover batches 0..39 with no
gaps, and the resumed half must report a non-zero `skip=`. Under the bug it
covered only 0..21.

### The bug that assertion found

`SkipAheadSampler`'s skip was one-shot — consumed by the first `__iter__`.
`_MultiProcessingDataLoaderIter` constructs the sampler iterator **twice**,
once in `_BaseDataLoaderIter.__init__` and again in `_reset`, so the throwaway
iterator ate the skip and the real one restarted at batch 0. It reproduces in
four lines and is invisible without workers:

```
num_workers=0, batch_sampler: first batch [80,81,82,83]   correct
num_workers=2, batch_sampler: first batch [0,1,2,3]       WRONG
```

Index mode only *accidentally* worked: PyTorch wraps a plain sampler in a
`BatchSampler` whose `__iter__` is a generator, so the discarded iterator never
touched the wrapped sampler. So the bug hit exactly one of the two branches —
the `DiverseChunkBatchSampler` path — which is why a single end-to-end check
would have missed it.

The skip is now sticky for the duration of an epoch and cleared by
`set_epoch`, which is robust to the iterator being built any number of times.
`test_batch_sampler_skip_with_dataloader_workers` covers both branches.

## Cluster facts this is built around

Measured on grogu, 2026-09-16. Re-derive rather than trust: pool numbers move
within hours.

**memlock.** RDMA registers pinned memory. The login node's memlock hard limit
is 64 KB and `PropagateResourceLimits=ALL` copies it into every job, while the
compute nodes' own hard limit is `unlimited`. Without `ulimit -l unlimited`
NCCL dies in `ibv_create_cq` with *Cannot allocate memory* — it does **not**
fall back to sockets, it crashes. `--propagate=NONE` alone is not sufficient,
so the batch script raises the limit explicitly as well.

**Two NCCL paths, chosen separately.** `NCCL_SOCKET_IFNAME` scopes only the
bootstrap/out-of-band socket; `NCCL_IB_HCA` selects the verbs data path. A job
can report `ib0` everywhere and still move gradients over TCP. Preflight parses
NCCL's own debug log and asserts `NET/IB`. Pinning `mlx5_0` also matters because
row-2 nodes expose Intel `irdma0/irdma1` (RoCE on the ethernet NIC) alongside
it — unpinned, measured bandwidth depends on which nodes you land on.

**Bandwidth is not a fixed property.** The same two A6000 nodes measured
7.31 GB/s busbw when idle and 4.12 GB/s once neighbours filled them to 6/8 and
7/8 GPUs. Socket fallback is ~1.29 GB/s. So the floor has to sit below
contended-but-healthy: `min_busbw: 4.0` discriminates, `6.0` would reject a
perfectly good allocation.

**`GraceTime` is 120 s and is not extendable.** It is set per-partition by the
admins. `--signal=B:USR1@N` only moves the *time-limit* warning earlier; it does
nothing for preemption. So a checkpoint that takes longer than ~120 s cannot be
written on the way out, and the **periodic timer is the real safety net** — the
signal handler is best-effort. `Checkpointer` warns above 90 s and tells you
that raising the signal lead will not help.

**`EnforcePartLimits=NO`.** A job whose walltime exceeds the partition's MaxTime
is *accepted* and pends forever with `reason=PartitionTimeLimit`. `submit.py`
validates walltime against MaxTime before submitting, and classifies pend
reasons into fatal vs transient — a plain "cancel after 15 min and try the next
constraint" loop would cycle through every pool, blame capacity, and never
report the real cause.

**`scontrol requeue` keeps the original `--constraint`.** A preempted job
returns to the same pool that just evicted it. `submit.py` counts requeues and
rotates to the next constraint after `max_requeues_per_constraint`.

**`/tmp` is node-local** (63 GB local disk, not the NAS). A checkpoint written
there is gone on requeue. `Checkpointer` refuses node-local paths outright, and
preflight write-tests the directory from every node.

**Pool sizes constrain N.** `--constraint` narrows to a homogeneous allocation
but the pools are small, and three of them cannot serve N>1 at all:

| constraint | nodes in `all` | step time | note |
|---|---|---|---|
| `A6000` | 6–7 | **111 ms** (1.00×) | fastest; often only 2 nodes with a free GPU |
| `A5000` | 6–7 | **139 ms** (1.25×) | largest usable pool — the default second choice |
| `rtx3090` | 2 | 151 ms (1.36×) | too small for N≥3 |
| `rtx6000` | 3 | **473 ms (4.25×)** | Turing: no tf32/bf16, 91% of step in forward |
| `rtx2080ti` | 4 | **does not run** | Turing sm_75: ptxas fails under torch.compile + bf16 |
| `H200`, `6000Blackwell`, `A6000Ada` | 1 each | — | cannot do multi-node |

Both Turing pools are excluded. That is worth stating plainly because
`grogu-3-25` and `grogu-3-30` are the only nodes in `all` that belong to **no**
lab partition, so nothing on the cluster can preempt you off them — but they
are 2080Ti, and a pool you cannot run on is not a pool. `rtx6000` runs and is
preemption-quiet but costs 4.25× per step, which no amount of scheduling luck
repays.

Step times: 1 GPU, batch 8, identical config, 40-step warmup discarded,
2026-09-16. `data_ms` was 0.4–0.6 ms on every card, so the loader is not the
bottleneck — forward is 68–91% of the step.

`submit.py` drops pools smaller than N with a message rather than letting you
discover it by timeout. Unpinned gets you up to 26 nodes but a mixed
allocation runs at the speed of its slowest GPU, so preflight aborts on
heterogeneity unless `allow_heterogeneous: true`.

**Preemption risk compounds with N.** Preemption kills the *whole* allocation —
`PreemptMode=REQUEUE`, no partial survival. Measured over 45 days of `all`, by
how long the job ran: <10 min 0.1%, 10–60 min 0.2%, 1–4 h 0.8%, 4–12 h 1.1%,
12–24 h 1.8% (essentially all 1-node jobs). Compounding as `1-(1-p)^N`, a
12–24 h run is ~5% at N=3 and ~14% at N=8. Low enough that
requeue-from-checkpoint covers it; high enough that the checkpoint must be
correct.

## Verified on real multi-node training

2 nodes x 1 GPU, A5000, PerAct2 orbital 2d on the real zarr, via
`submit.py` -> `job.sbatch` -> preflight -> `main.py`:

* **600-step run COMPLETED** (3m41s), checkpoint carrying `world_size=2`,
  `epoch/batch`, and RNG state for **both ranks**.
* **Resume**: `compare_indices.py` PASS at step 200 — RNG byte-identical across
  both ranks, 416 batches index-identical, `skips=[0, 100]` on each rank.
* **Preemption**: USR1 -> checkpoint on disk in **16.0 s**, trainer confirmed,
  `scontrol requeue` issued. With `ckpt_interval_steps` set equal to
  `train_iters`, `last.pth` could only have come from the signal path.

### Measured weak scaling

Per-GPU batch fixed at 8, A5000, 200 steps with 40 discarded as warmup. The
same three nodes for every leg, each at 1/8 load (only this job), and the legs
run **sequentially** via `--dependency`:

| N | nodes | step ms | fwd | bwd | Δbwd | samp/s/rank | aggregate | speedup | eff |
|---|---|---|---|---|---|---|---|---|---|
| 1 | grogu-2-20 | 125.7 | 91.1 | 13.0 | — | 63.6 | 63.6 | 1.00× | 100% |
| 2 | grogu-2-[20,25] | 131.7 | 93.1 | 18.0 | +5.0 | 60.8 | **121.5** | **1.91×** | 95.5% |
| 3 | grogu-2-[10,20,25] | 144.1 | 95.3 | 23.7 | +10.7 | 55.5 | **166.6** | **2.62×** | 87.3% |

**The cost is per-node synchronisation, not bandwidth.** Δbwd roughly doubles
from N=2 to N=3 (+5.0 → +10.7 ms) while ring-allreduce volume rises only 33%
(2(N-1)/N: 1.00 → 1.33). A 10.2 MB gradient at 9 GB/s is ~1 ms of wire time;
the other ~10 ms is kernel launch and bucket sync. So a bandwidth calculation
is a lower bound on the real cost, not a prediction of it — the first estimate
made here was 3-10x optimistic for exactly this reason.

Practical read: 2 nodes is clearly worth it. 3 still gains but costs 13%
overhead. Past that, for a 2.55 M-param model the per-step sync grows faster
than the work being distributed.

**Do not measure this by running the legs in parallel.** The first attempt here
launched N=1..4 concurrently on overlapping nodes (N=2 on 2-[20,25], N=3 on
2-[20,25,30], N=4 on 2-[10,20,25,30]) so the legs contended with each other,
and the N=1 baseline happened to land on a node already 5/8 busy. That produced
apparent *superlinear* scaling at N=2 and N=3, which should have been the
tell. Pin the nodes, chain the jobs, and record each node's load at launch.

### Is it the fabric, or the collective? (it is the collective)

Same GPU count, intra-node vs inter-node, so the interconnect is the only
variable. Per-GPU batch 8, A5000, sequential on pinned idle nodes:

| case | ranks | step ms | bwd | Δbwd vs 1 GPU | samp/s/rank | aggregate |
|---|---|---|---|---|---|---|
| 1 GPU | 1 | 125.7 | 13.0 | — | 63.6 | 63.6 |
| 2 GPU, 1 node | 2 | 131.0 | 15.6 | +2.6 | 61.1 | 122.2 |
| 2 GPU, 2 nodes | 2 | 134.9 | 18.5 | +5.5 | 59.3 | 118.6 |
| 3 GPU, 1 node | 3 | 140.4 | 23.6 | +10.6 | 57.0 | 171.1 |
| 3 GPU, 3 nodes | 3 | 147.8 | **23.6** | +10.6 | 54.2 | 162.5 |
| 8 GPU, 1 node | 8 | 147.7 | 31.4 | +18.4 | 54.2 | 433.5 |

**Going multi-node costs only 3-5%** at a given GPU count, and at 3 ranks the
backward time — which contains the allreduce — is *identical* whether
gradients cross PCIe inside one box or 100 Gb InfiniBand between three.

The cost is DDP's per-rank collective overhead, and the proof is that it
appears with **no network at all**: bwd goes 13.0 → 15.6 → 23.6 → 31.4 ms for
1 → 2 → 3 → 8 ranks entirely within one node. A bandwidth argument cannot
explain a 4x jump in allreduce cost from 2 to 3 ranks on the same machine.

Two consequences:

* **Prefer more GPUs on one node over more nodes** for throughput: 8 GPUs in
  one box gives 433 samp/s at the same 85% efficiency that 3 nodes gives at
  163 samp/s.
* **Multi-node's value is availability, not speed.** It lets a run use
  scattered idle GPUs when no single node has enough free, for a 3-5%
  premium. That is a scheduling win, and the resilience layer is what makes
  it safe.
* **Raise per-GPU batch before adding ranks.** Batch 16 gave +18% single-GPU
  throughput (63.6 → 75.1 samp/s) and moved 3-rank efficiency from 87.3% to
  92.2%.

### How to measure this without fooling yourself

Three mistakes made here, all of which produced confident wrong numbers:

1. **Running the legs concurrently.** The first N=1..4 sweep launched all four
   at once on overlapping nodes, so they contended with each other and
   produced apparent *superlinear* scaling. Chain them with `--dependency`.
2. **An uncontrolled baseline.** That sweep's N=1 leg landed on a node already
   5/8 busy, inflating every efficiency figure derived from it. Pin the nodes
   and record each one's load at launch.
3. **Too few samples.** 3-GPU and 4-GPU came out at *exactly* 140.4 ms /
   57.02 samp/s on 8 benchmark rows each, which read as "the 4th GPU is
   free". An interleaved repeat (3,4,3,4) at 400 steps / 18 rows showed
   run-to-run spread of 0.06-0.5% and a real **2.3%** per-rank cost from 3 to
   4 ranks — about 5x the noise. Interleave, repeat, and check `data_ms` is
   still negligible so the run is not secretly data-bound.

### Contention costs more than communication

The single most useful result here. Same config, 1 GPU on an A5000, varying
only how much of the node other jobs held:

| other jobs' GPUs | ms/step | samp/s | vs quiet |
|---|---|---|---|
| 1 of 8 | 125.7 | 63.6 | — |
| 2 of 8 | 129.5 | 61.8 | −2.8% |
| 5 of 8 | 144.4 | 55.4 | **−12.9%** |

Against that, multi-node communication costs 4.5% at N=2 and 12.7% at N=3. So
a busy neighbour can cost as much as adding two more nodes — and it compounds,
because a DDP step runs at the speed of its slowest rank. A contaminated N=4
run that included the 5/8-busy node fell to 53.6 samp/s/rank, *below* the
single-GPU run on that same node. One noisy neighbour taxes every rank.

It hits bandwidth too, not just compute. The pair grogu-1-30 + grogu-2-5
measured 1.64 and later 1.16 GB/s busbw — socket-tier numbers on a working IB
fabric — while grogu-1-30 sat at 88% GPU load. Other jobs' traffic saturates
the HCA as well as the SMs, so one threshold catches both failure modes.

Hence `max_node_load` (default 1.0 = warn only; the example config sets 0.75).
Load is transient, so a rejection is not recorded against the node — preflight
fails, and the supervisor simply retries for a different allocation. Verified:
it rejected grogu-1-30 at 88%, the launcher rotated to A5000, and the next
allocation trained cleanly.

### The fabric is not uniform either

`audit_nodes.sh --report` now also lists each node's HCA, because
`NCCL_IB_HCA` is a single job-wide value:

* **`grogu-2-35` is the only node with `mlx4_0` and no `mlx5_0`.** A job pinned
  to `mlx5_0` cannot include it: NCCL fails to find the device and dies with
  `ncclInternalError ... socketFinalizeAccept: wrong type 3 != 4`. It broke two
  runs here before being excluded. There is no setting that works for a mixed
  mlx4/mlx5 allocation, so the node has to be left out of multi-node jobs.
* 8 nodes expose `irdma0`/`irdma1` (RoCE on the ethernet NIC) next to
  `mlx5_0` — the reason the HCA must be pinned rather than auto-selected.
* 4 nodes also have `mlx5_1`.

### Node pairing matters, and the scheduler does not know it

A 2-rank allreduce measured **9.70, 9.22, 4.36 and 1.64 GB/s** on different
node pairs. The 1.64 GB/s pair was `grogu-1-30` + `grogu-2-5` — different rows.
With `TopologyPlugin=none` and `SwitchType=none` the scheduler has no idea
which nodes share a switch, so it will happily hand out a cross-row pair that
runs at a fifth of the bandwidth of an in-row one. Preflight's `min_busbw`
floor is what turns that into a fast, clear rejection instead of a slow run:
it rejected the 1.64 GB/s pair, the supervisor rotated pools, and the next
allocation was fine.

## Bugs this found in the pipeline itself

Listed because each was silent, and each would have cost real training time.

1. **Requeue without resume.** `checkpoint: null` is 3DFA's default, so a
   `train_cmd` that omits `checkpoint=` saves a checkpoint on preemption, comes
   back, and **restarts from step 0** — forever, while looking like it
   recovered. `submit.py` now refuses such a config outright.
2. **Mislabelled node.** `grogu-4-13` advertises `ActiveFeatures=A6000` while
   physically holding an **RTX 3080 Ti**, so `--constraint=A6000` produced a
   heterogeneous allocation. Preflight caught it, recorded the node, and
   `submit.py` now excludes it on subsequent submits.
3. **Partially faulty node.** On `grogu-3-20` one card of eight (PCI
   `0000:1B:00.0`) is dead: `nvidia-smi` errors on it and silently omits it
   while `/proc/driver/nvidia` lists all eight and Slurm keeps `Gres=gpu:8`.
   Whether a job gets the dead GPU is luck. Checking that *a* GPU name came
   back is not enough, so preflight compares the device count against the
   allocation and probes every device.
4. **`sacct` field that does not exist.** Asking for `Restarts` (absent in
   Slurm 20.11) made `sacct` fail, which turned every finished job into
   `UNKNOWN_GONE` and made the supervisor exit the instant a job ended.
5. **Preflight rejection treated as fatal.** An unusable allocation is a
   property of the nodes, not the run, so it now rotates pools and retries
   rather than giving up.
6. **Waiting out a timeout on a full pool.** The launcher now leads with a
   constraint that has capacity *now*, instead of submitting to the preferred
   pool and burning the whole pend timeout before rotating.

`audit_nodes.sh` cross-checks every node's real GPU against its label. It reads
`/proc/driver/nvidia` rather than running `nvidia-smi`, so it needs
`--gres=gpu:0` and can therefore inspect saturated nodes — which are exactly
the ones worth checking. It does not detect *health*, only the model; that is
preflight's job.

## Out of scope, deliberately

No elastic torchrun, no DiLoCo, no async or dynamic shard queue. Preemption is
all-or-nothing and infrequent; requeue-from-checkpoint covers it completely,
and every one of those alternatives trades a correctness risk for a problem
this cluster does not have.

## Known gap in the existing trainer

`training/base.py` already does atomic saves and resumes from `last.pth`, but
its checkpoint holds only `weight / ema_weight / optimizer / iter / best_loss`
— **no RNG state and no dataloader position**. So a requeued 3DFA run today
resumes to a different data order and different dropout draws. (The LR
scheduler is fine: `base.py:492` replays `scheduler.step()` `start_iter` times.)
Porting `ResumableDistributedSampler` + `StatefulLoader` into
`training/base.py` is the follow-up that makes that trainer actually resumable;
the test here is what will prove it.
