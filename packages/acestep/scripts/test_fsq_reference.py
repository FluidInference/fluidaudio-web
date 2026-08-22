#!/usr/bin/env python3
"""Known-answer tests for the independent ResidualFSQ oracle."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPOSITORY / "scripts/fsq_reference.py"
SPEC = importlib.util.spec_from_file_location("fsq_reference", REFERENCE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {REFERENCE_PATH}")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class FsqReferenceTests(unittest.TestCase):
    def test_committed_cross_language_vectors(self) -> None:
        REFERENCE.verify_vectors()

    def test_level_eight_known_answers_pin_bfloat16_rounding(self) -> None:
        records = [REFERENCE.scalar_record(8, digit) for digit in range(8)]
        self.assertEqual(
            [record["bfloat16Bits"] for record in records],
            ["bf80", "bf37", "bedb", "be12", "3e12", "3edb", "3f37", "3f80"],
        )
        self.assertEqual(
            [record["fp16BitsFromBfloat16"] for record in records],
            ["bc00", "b9b8", "b6d8", "b090", "3090", "36d8", "39b8", "3c00"],
        )

    def test_full_codebook_known_answer_hashes(self) -> None:
        self.assertEqual(
            REFERENCE.full_codebook_hashes(),
            (
                "368cfc217ff17d9c3a87f66488f67b13d5a3ff16fc11ab2a43bd082dfa57ce1c",
                "d1ba8fed5f32390647416340be61ec96755065ae1d7c142a98eeef691820e083",
            ),
        )

    def test_raw_fp16_is_derived_from_bfloat16(self) -> None:
        ideal_direct_fp16 = int.from_bytes(
            __import__("struct").pack("<e", REFERENCE.ideal_scalar(8, 1)),
            "little",
        )
        rounded_bfloat16 = REFERENCE.bfloat16_bits_from_float32(
            REFERENCE.ideal_scalar(8, 1)
        )
        from_bfloat16 = REFERENCE.float16_bits_from_bfloat16_bits(
            rounded_bfloat16
        )
        self.assertEqual(ideal_direct_fp16, 0xB9B7)
        self.assertEqual(from_bfloat16, 0xB9B8)
        self.assertNotEqual(ideal_direct_fp16, from_bfloat16)

    def test_vector_tampering_is_rejected(self) -> None:
        vectors = json.loads(REFERENCE.VECTORS.read_text(encoding="utf-8"))
        vectors["vectors"][0]["bfloat16Bits"][0] = "0000"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vectors.json"
            path.write_text(json.dumps(vectors, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ"):
                REFERENCE.verify_vectors(path)


if __name__ == "__main__":
    unittest.main()
