"""Old checkpoints must rebuild the architecture they were trained with."""

import pytest

from utils.config_migrations import MIGRATIONS, migrate_config


@pytest.mark.parametrize("old, expected", [
    (dict(traj_scene_rope=True,  use_learned_abs_pe=False, use_proprio_rope=False), "rope3d"),
    (dict(traj_scene_rope=True,  use_learned_abs_pe=False, use_proprio_rope=True),  "rope3d_proprio"),
    (dict(traj_scene_rope=True,  use_learned_abs_pe=True,  use_proprio_rope=False), "learned_abs"),
    (dict(traj_scene_rope=False, use_learned_abs_pe=False, use_proprio_rope=False), "none"),
])
def test_head_positional_encoding_recovered(old, expected):
    assert migrate_config(old)["head_positional_encoding"] == expected


def test_retired_keys_are_dropped():
    out = migrate_config(dict(
        traj_scene_rope=True, use_learned_abs_pe=False, use_proprio_rope=False,
        rope_type="stopgrad", rope_schedule_type="linear", rope_schedule_start_k=3,
        rope_schedule_end_k=0, rope_schedule_steps=100, keep_last_k=3,
        embedding_dim=120,
    ))
    assert out["embedding_dim"] == 120
    for key in ("traj_scene_rope", "use_learned_abs_pe", "use_proprio_rope", "rope_type",
                "rope_schedule_type", "rope_schedule_start_k", "rope_schedule_end_k",
                "rope_schedule_steps", "keep_last_k"):
        assert key not in out


def test_current_config_is_unchanged():
    """A checkpoint already in the current vocabulary must pass through untouched."""
    cur = dict(head_positional_encoding="learned_abs", embedding_dim=120, bimanual=True)
    assert migrate_config(cur) == cur


def test_migrations_are_idempotent():
    once = migrate_config(dict(traj_scene_rope=False, rope_type="normal"))
    assert migrate_config(once) == once


def test_input_is_not_mutated():
    old = dict(traj_scene_rope=False)
    migrate_config(old)
    assert old == dict(traj_scene_rope=False)


def test_every_migration_is_registered():
    assert len(MIGRATIONS) >= 3


# --- miscalibration vs perturbation noise ------------------------------------
@pytest.mark.parametrize("old, mode, level, rot", [
    # a fixed per-group base alone is miscalibration, no perturbation
    (dict(orbital_miscal_noise_level="medium"), "group", "medium", None),
    # the old magnitude pair was always a per-sample draw: perturbation noise
    (dict(miscal_max_angle_deg=3.0, miscal_max_translation_m=0.01), "none", None, 3.0),
    # both: persistent error plus jitter on top
    (dict(orbital_miscal_noise_level="medium", miscal_max_angle_deg=3.0), "group", "medium", 3.0),
    # cotrain-with-a-level: miscalibration restricted to some groups
    (dict(cotrain_miscal_group_ids=[1, 2], cotrain_miscal_level="medium"), "group", "medium", None),
])
def test_miscal_and_perturbation_are_separated(old, mode, level, rot):
    out = migrate_config(old)
    assert out["miscal_mode"] == mode
    assert out["miscal_group_level"] == level
    assert out["perturbation_noise_rot_deg"] == rot


def test_cotrain_group_ids_become_camera_groups():
    out = migrate_config(dict(cotrain_miscal_group_ids=[1, 2], miscal_max_angle_deg=5.0))
    assert out["miscal_camera_groups"] == [1, 2]
    assert out["miscal_mode"] == "none"          # no persistent error, only jitter
    assert out["perturbation_noise_rot_deg"] == 5.0


def test_every_retired_key_is_migrated():
    """RETIRED_KEYS is the registry the launch-script test checks against, so a
    key listed there must actually be translated and dropped by a migration."""
    from utils.config_migrations import RETIRED_KEYS
    cfg = {k: None for k in RETIRED_KEYS}
    # give the ones whose migration branches on a value something to branch on
    cfg.update(traj_scene_rope=True, predict_extrinsics=False,
               image_space_sampling=False, rope_type="normal")
    out = migrate_config(cfg)
    assert not (RETIRED_KEYS & set(out)), sorted(RETIRED_KEYS & set(out))
