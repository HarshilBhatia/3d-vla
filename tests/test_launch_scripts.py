"""Launch scripts must only pass config keys that still exist.

A config rename that misses a .slurm/.sh/.yaml launcher does not fail until
someone submits the job, which can be days later and on a cluster. This caught
eight scripts still passing retired miscal keys after the miscal collapse.
"""

import re
from pathlib import Path

import pytest

from utils.config_migrations import RETIRED_KEYS
from utils.hydra_utils import get_config, get_config_path

REPO = Path(__file__).resolve().parent.parent
# Hydra config groups are selected as `group=name`, not overridden as keys.
CONFIG_GROUPS = {"data", "eval", "logging", "experiment", "miscal", "dataset"}
# Runtime-injected, never present in config/config.yaml.
RUNTIME_KEYS = {"log_dir", "local_rank", "run_mode"}
# Keys reaching a component other than the Hydra config (eval backends take
# these directly, hence the `++` form in the scripts that pass them).
NON_CONFIG_KEYS = {"miscal_camera_indices"}

SCRIPTS = sorted(
    p for p in (REPO / "scripts").rglob("*")
    if p.is_file() and p.suffix in (".sh", ".slurm", ".yaml", ".py")
)


@pytest.fixture(scope="module")
def valid_keys():
    args = get_config(overrides=[], config_name="config", config_path=get_config_path())
    return set(vars(args)) | CONFIG_GROUPS | RUNTIME_KEYS | NON_CONFIG_KEYS


# Exact names only: a substring match flags ordinary Python kwargs such as
# `extrinsics=` in plotting code.


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(REPO)))
def test_script_passes_only_live_config_keys(script, valid_keys):
    offenders = {
        m.group(1)
        for m in re.finditer(r"(?<![\w./-])([a-z_][a-z0-9_]*)=(?![=])", script.read_text())
        if m.group(1) in RETIRED_KEYS and m.group(1) not in valid_keys
    }
    assert not offenders, (
        f"{script.relative_to(REPO)} passes config keys that no longer exist: "
        f"{sorted(offenders)}. Rename them, and add a @migration in "
        f"utils/config_migrations.py if checkpoints carry the old spelling."
    )


EVAL_PLANS = sorted((REPO / "instructions" / "eval_plans").glob("*.json"))


@pytest.mark.parametrize("plan", EVAL_PLANS, ids=lambda p: p.name)
def test_eval_plan_uses_only_live_config_keys(plan, valid_keys):
    """Eval plans carry config overrides as JSON, so the script scan above misses
    them. A stale key here fails the rollout at Hydra startup, hours later."""
    import json

    offenders = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in RETIRED_KEYS and key not in valid_keys:
                    offenders.add(key)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(plan.read_text()))
    assert not offenders, (
        f"{plan.name} passes retired config keys {sorted(offenders)}; "
        f"migrate them and add a @migration if checkpoints carry the old spelling."
    )
