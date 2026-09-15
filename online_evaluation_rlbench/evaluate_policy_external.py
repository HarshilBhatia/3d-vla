"""Compatibility executable for the migrated external-policy evaluator."""

import runpy


if __name__ == "__main__":
    runpy.run_module("evaluation.online.external_policy", run_name="__main__")
