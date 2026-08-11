# In-job patches for upstream 3DFA evaluation

## `upstream_3dfa_num_demos.patch`

Applies to upstream `nickgkan/3d_flowmatch_actor` at commit `ab70932`.

Upstream's bimanual online-eval path always calls `get_stored_demos(amount=-1)`
and has no way to cap the episode count, so it would run all 100 staged test
seeds per variation. Our campaign runs 25 (the benchmark convention, and what
our own checkpoint was evaluated with), so the comparison has to be
episode-matched.

The patch threads a `num_demos` argument from the `evaluate_policy.py` CLI
through `evaluate_task_on_multiple_variations` into
`_evaluate_task_on_one_variation`, where it truncates `var_demos`. This mirrors
the same fix already present in our fork
(`online_evaluation_rlbench/utils_with_bimanual_rlbench.py`).

Note this is a *different* failure mode than the fork needed: upstream does not
raise a `TypeError`, because upstream's `evaluate_policy.py` never passes
`num_demos` in the first place. Without the patch the job would silently
evaluate 100 episodes instead of 25 — a wrong-but-not-crashing result, which is
why it is patched rather than ignored.

Applied in-job by `scripts/sky/peract2_upstream_eval.yaml` with
`git apply --3way`.
