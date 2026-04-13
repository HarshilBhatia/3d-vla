"""
Create task-to-camera-group mapping for orbital data generation.
Delegates to orbital.task_mapping; run from the repo root.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data_generation.orbital.task_mapping import main

if __name__ == "__main__":
    main()
