"""Compatibility executable for the migrated result collector."""

import runpy


if __name__ == "__main__":
    runpy.run_module("evaluation.analysis.collect_results", run_name="__main__")
