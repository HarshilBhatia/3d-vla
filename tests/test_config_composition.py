"""Config-group composition: later groups must not silently clobber earlier ones."""

import pytest

from utils.hydra_utils import get_config, get_config_path


def cfg(*overrides):
    return get_config(overrides=list(overrides), config_name="config",
                      config_path=get_config_path())


@pytest.fixture(scope="module")
def probe_experiment(tmp_path_factory):
    """An experiment config that sets miscal under its legacy key names."""
    path = get_config_path() / "experiment" / "_probe_legacy_miscal.yaml"
    path.write_text("miscal_mode: group\nmiscal_group_level: medium\n")
    yield "_probe_legacy_miscal"
    path.unlink()


def test_experiment_miscal_survives_the_miscal_group(probe_experiment):
    """The miscal group loads after experiment. If it re-nulls keys an experiment
    set, those values vanish silently -- the class of bug that cost ~30 points."""
    args = cfg(f"experiment={probe_experiment}")
    assert args.miscal_mode == "group"
    assert args.miscal_group_level == "medium"


def test_no_miscal_by_default():
    args = cfg()
    assert args.miscal_mode == "none"
    for key in ("miscal_group_level", "miscal_camera_groups",
                "perturbation_noise_rot_deg", "perturbation_noise_trans_m"):
        assert getattr(args, key) is None


@pytest.mark.parametrize("group, key, expected", [
    ("orbital_medium", "miscal_group_level", ["medium"]),
    ("orbital_fixed_medium_randnoise", "perturbation_noise_rot_deg", 3.0),
    ("orbital_fixed_medium_randnoise", "miscal_mode", "group"),
    ("orbital_fixed_medium", "perturbation_noise_rot_deg", None),
    ("cotrain_g1g2_medium", "miscal_camera_groups", [1, 2]),
    ("cotrain_g1g2_medium", "miscal_group_level", "medium"),
    ("cotrain_g1g2_random_large", "miscal_mode", "none"),
    ("cotrain_g1g2_random_large", "perturbation_noise_rot_deg", 5.0),
])
def test_miscal_groups_still_apply(group, key, expected):
    assert getattr(cfg(f"miscal={group}"), key) == expected


def test_eval_and_logging_groups_merge_at_root():
    """Keys moved into the eval/ and logging/ groups must stay flat at the root."""
    args = cfg()
    for key in ("task", "fov_deg", "max_tries", "eval_use_depth2cloud",
                "wandb_project", "benchmark_log_freq"):
        assert hasattr(args, key), key


def test_experiment_overrides_eval_and_logging_groups():
    """eval/logging load before experiment, so an experiment can still override them."""
    args = cfg("experiment=peract2_orbital_vid_deltam_fullpatch")
    assert args.interm_ckpt_interval_steps == 10000
