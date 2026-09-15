"""Compatibility executable for migrated evaluation S3 transfer."""
import runpy
if __name__ == "__main__":
    runpy.run_module("evaluation.integrations.s3_transfer", run_name="__main__")
