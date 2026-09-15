"""Compatibility executable for migrated SBRS plotting."""
import runpy
if __name__ == "__main__":
    runpy.run_module("evaluation.analysis.plot_sbrs_deltam_loss_vs_performance", run_name="__main__")
