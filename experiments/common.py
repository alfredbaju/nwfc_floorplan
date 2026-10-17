"""Shared utilities for the E1-E5 experiment scripts."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tileset_core.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
os.makedirs(RESULTS_DIR, exist_ok=True)


def result_path(filename):
    return os.path.join(RESULTS_DIR, filename)