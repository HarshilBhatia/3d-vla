# In-job patches for upstream 3DFA evaluation

## `upstream_3dfa_eval.patch`

Applies to upstream `nickgkan/3d_flowmatch_actor` at commit `ab70932`. Two
independent fixes, both in the bimanual online-eval path.

### 1. `num_demos` support (episode matching)

Upstream's bimanual eval always calls `get_stored_demos(amount=-1)` and has no
way to cap the episode count, so it would run all 100 staged test seeds per
variation. Our campaign runs 25 (the benchmark convention, and what our own
checkpoint was evaluated with), so the comparison has to be episode-matched.

The patch threads `num_demos` from the `evaluate_policy.py` CLI through
`evaluate_task_on_multiple_variations` into `_evaluate_task_on_one_variation`,
where it truncates `var_demos`. This mirrors the fix already present in our fork.

Note this is a *different* failure mode than the fork needed: upstream does not
raise a `TypeError`, because upstream's `evaluate_policy.py` never passes
`num_demos` in the first place. Without the patch the job would silently
evaluate 100 episodes instead of 25 — a wrong-but-not-crashing result, which is
why it is patched rather than ignored.

### 2. `Mover` drops the `ret_obs=True` kwarg

Upstream's `Mover.__call__` calls `self._task.step(action_collision, ret_obs=True)`,
but the PerAct2 RLBench fork baked into the eval image (markusgrotz, at
`/opt/src/RLBench2`) has `TaskEnvironment.step(self, action)` with no `ret_obs`
parameter, so the first executed action dies with:

    TypeError: TaskEnvironment.step() got an unexpected keyword argument 'ret_obs'

`ret_obs` is redundant against this RLBench: `step()` already returns
`(Observation, reward, terminate)`, which is exactly what the call site unpacks,
so dropping the kwarg is behavior-preserving rather than a semantic change. Our
fork carries the identical one-line difference. Upstream presumably developed
against a slightly different RLBench revision that accepted the flag.

Applied in-job by `scripts/sky/peract2_upstream_eval.yaml` with
`git apply --3way`.
