from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import make_fixture_manifest as fixture_manifest


class FixtureEvidenceTests(unittest.TestCase):
    def write_values(self, values: list[float]) -> Path:
        temporary = tempfile.NamedTemporaryFile(suffix=".f32le", delete=False)
        self.addCleanup(lambda: Path(temporary.name).unlink(missing_ok=True))
        with temporary:
            temporary.write(struct.pack(f"<{len(values)}f", *values))
        return Path(temporary.name)

    def test_exact_evidence_preserves_signed_zero_bits(self) -> None:
        path = self.write_values([-0.0, 0.0, -2.5, 1.25, 3.0, -4.0, 0.5, 8.0])
        evidence = fixture_manifest.inspect_f32le(path, 8)
        self.assertEqual(evidence["finiteCount"], 8)
        self.assertEqual(evidence["nonzeroCount"], 6)
        self.assertEqual(evidence["minimum"], -4.0)
        self.assertEqual(evidence["maximum"], 8.0)
        self.assertEqual(evidence["headF32Bits"][:2], ["80000000", "00000000"])
        self.assertEqual(evidence["headF32Bits"], evidence["tailF32Bits"])

    def test_all_zero_and_nonfinite_inputs_fail_closed(self) -> None:
        for values in ([0.0] * 8, [1.0] * 7 + [float("inf")]):
            with self.subTest(values=values):
                with self.assertRaises(SystemExit):
                    fixture_manifest.inspect_f32le(self.write_values(values), 8)

    def test_element_count_mismatch_fails_closed(self) -> None:
        with self.assertRaises(SystemExit):
            fixture_manifest.inspect_f32le(self.write_values([1.0] * 8), 9)


if __name__ == "__main__":
    unittest.main()
