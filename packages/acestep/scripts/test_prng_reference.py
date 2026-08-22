#!/usr/bin/env python3
"""Known-answer and invariant tests for the independent PRNG oracle."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPOSITORY / "scripts/prng_reference.py"
SPEC = importlib.util.spec_from_file_location("prng_reference", REFERENCE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {REFERENCE_PATH}")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class PrngReferenceTests(unittest.TestCase):
    def test_committed_cross_language_vectors(self) -> None:
        REFERENCE.verify_vectors()

    def test_random123_primitive_known_answer(self) -> None:
        self.assertEqual(
            REFERENCE.philox4x32_10((0, 0, 0, 0), (0, 0)),
            (0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8),
        )

    def test_gaussian_mapping_has_exact_odd_symmetry(self) -> None:
        for word in (0, 1, 17, 0x0635_3F7C, 0x1234_5678, 0x7FFF_FFFF):
            negative = REFERENCE.float32_bits(
                REFERENCE.gaussian_from_word(word)
            )
            positive = REFERENCE.float32_bits(
                REFERENCE.gaussian_from_word(0xFFFF_FFFF - word)
            )
            self.assertEqual(negative ^ 0x8000_0000, positive)

    def test_categorical_validation_is_fail_closed(self) -> None:
        invalid_weights = ((), (0.0, 0.0), (1.0, -1.0), (1.0, float("nan")))
        for weights in invalid_weights:
            with self.subTest(weights=weights), self.assertRaises(
                (TypeError, ValueError)
            ):
                REFERENCE.categorical_token_from_word(weights, 0)

    def test_vector_tampering_is_rejected(self) -> None:
        vectors = json.loads(REFERENCE.VECTORS.read_text(encoding="utf-8"))
        vectors["gaussian"][0]["outputF32Bits"] ^= 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors.json"
            path.write_text(json.dumps(vectors), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "minimum-word"):
                REFERENCE.verify_vectors(path)


if __name__ == "__main__":
    unittest.main()
