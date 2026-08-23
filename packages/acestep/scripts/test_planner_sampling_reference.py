from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPOSITORY / "scripts/planner_sampling_reference.py"
SPEC = importlib.util.spec_from_file_location("planner_sampling_reference", REFERENCE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {REFERENCE_PATH}")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class PlannerSamplingReferenceTests(unittest.TestCase):
    def test_committed_vectors_and_realistic_sparse_vocabulary(self) -> None:
        REFERENCE.check_vectors()


if __name__ == "__main__":
    unittest.main()
