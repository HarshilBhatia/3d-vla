# System optimisation notes

## Resilient Slurm online-evaluation tasks

### Problem

An online-evaluation array task can fail before inference because the allocated
node exposes no usable CUDA device to the Apptainer process.  The observed
signature is a failure during `torch.cuda._lazy_init()` such as:

```text
RuntimeError: CUDA unknown error ... CUDA_VISIBLE_DEVICES ... available devices to be zero
```

This is infrastructure failure, not a bad checkpoint or a policy result.  A
recent checkpoint-ladder evaluation had five such failures while a sibling task
using the same checkpoint/configuration completed; resubmitting those five
tasks on A5000 nodes completed normally.

### Proposed implementation

Add a small bounded retry wrapper to
`scripts/eval/online_eval_plan.slurm` (roughly 40--70 lines).

1. Run the evaluator while capturing an attempt log and its exit status.
2. Exit immediately when the task's result JSON already exists.  Evaluation is
   therefore idempotent across retries and requeues.
3. On failure, classify only an allowlist of known transient infrastructure
   signatures, initially:
   - CUDA initialization / zero visible device errors;
   - selected Apptainer GPU-device setup failures;
   - optionally, clearly identified transient CoppeliaSim/X startup failures.
4. If the failure is allowlisted and a retry budget has not been exhausted,
   requeue the *same Slurm array element* with `scontrol requeue
   "$SLURM_JOB_ID"`.  Use `SLURM_RESTART_COUNT` (or a durable sidecar counter)
   and cap retries at two.
5. For ordinary Python exceptions, checkpoint/config errors, missing data, or
   policy failures, preserve the non-zero exit immediately.  Never classify
   them as retryable.
6. Emit a compact retry record: job/array ID, allocated node, restart count,
   checkpoint, classifier match, and path to the attempt log.

### Why requeue instead of retrying in place?

Retrying within the same allocation cannot repair a node/container allocation
that has no visible GPU.  Requeuing returns the array task to Slurm for a fresh
placement.  The retry cap prevents a pathological node or a misclassified code
error from looping indefinitely.

### Interaction with checkpoint-ladder watching

The existing checkpoint-ladder ledger already prevents duplicate *submission*
for a checkpoint.  The task-level wrapper complements it: it guarantees that a
submitted array element either writes its result JSON, exhausts a small
infrastructure retry budget with useful logs, or fails visibly for a genuine
application problem.  No watcher redesign is required.

### Validation plan

- Unit-test the shell classifier with representative retryable and non-retryable
  log snippets.
- Dry-run one array element with a forced retryable exit and confirm it is
  requeued exactly twice at most.
- Confirm a valid existing result JSON skips evaluation.
- Confirm a deliberate configuration error fails once without requeueing.
