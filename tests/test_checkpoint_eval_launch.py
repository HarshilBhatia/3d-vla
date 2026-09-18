"""The checkpoint-eval launcher must be reachable from the trainer.

A package move once shifted training/base.py one directory shallower without
updating its parents[N] repo-root index. Submission then pointed one level above
the repo, and fifteen checkpoint evals failed silently -- the failure is caught
and logged so training is never killed, which is why it went unnoticed.
"""

from pathlib import Path

from training.base import REPO_ROOT


def test_repo_root_is_the_repo():
    assert (REPO_ROOT / "main.py").is_file()
    assert (REPO_ROOT / "config" / "config.yaml").is_file()


def test_checkpoint_ladder_launcher_exists():
    assert (REPO_ROOT / "scripts" / "eval" / "checkpoint_ladder.py").is_file()


def test_repo_root_matches_this_test_file():
    assert REPO_ROOT == Path(__file__).resolve().parents[1]
