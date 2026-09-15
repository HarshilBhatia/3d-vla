"""Compatibility imports for migrated checkpoint-ladder planning."""
from evaluation.planning.checkpoint_ladder import *  # noqa: F403
if __name__ == "__main__":
    from evaluation.planning.checkpoint_ladder import main
    main()
