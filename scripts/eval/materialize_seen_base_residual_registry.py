"""Compatibility executable for migrated calibration materialization."""
import runpy
if __name__ == "__main__":
    runpy.run_module("evaluation.planning.materialize_seen_base_residual_registry", run_name="__main__")
