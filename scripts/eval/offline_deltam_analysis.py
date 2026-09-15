"""Compatibility executable for ``evaluation.analysis.offline_deltam``."""
from evaluation.analysis.offline_deltam import *  # noqa: F403
if __name__ == "__main__":
    from evaluation.analysis.offline_deltam import main
    main()
