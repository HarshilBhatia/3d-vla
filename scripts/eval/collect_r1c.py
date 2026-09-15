"""Compatibility executable for migrated R1C collection."""
import runpy
if __name__ == "__main__":
    runpy.run_module("evaluation.planning.collect_r1c", run_name="__main__")
