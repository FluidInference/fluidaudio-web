#!/usr/bin/env python3
"""Adversarial tests for the dependency-free golden contract validator."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPOSITORY / "scripts/validate-golden.py"
SPEC = importlib.util.spec_from_file_location("validate_golden", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {VALIDATOR_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class ValidatorHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture_path = (
            self.root / "golden/fixtures/direct-lyrics-short.json"
        )
        self.fixture_path.parent.mkdir(parents=True)
        shutil.copy2(
            REPOSITORY / "golden/fixtures/direct-lyrics-short.json",
            self.fixture_path,
        )
        self.original_root = VALIDATOR.ROOT
        VALIDATOR.ROOT = self.root

    def tearDown(self) -> None:
        VALIDATOR.ROOT = self.original_root
        self.temporary.cleanup()

    def fixture(self) -> dict[str, object]:
        return json.loads(self.fixture_path.read_text(encoding="utf-8"))

    def write_fixture(self, fixture: dict[str, object]) -> None:
        self.fixture_path.write_text(
            json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def test_duplicate_nested_key_is_rejected_before_update(self) -> None:
        original = self.fixture_path.read_text(encoding="utf-8")
        duplicate = original.replace(
            '"samplerMode": "euler",',
            '"samplerMode": "euler",\n      "samplerMode": "heun",',
            1,
        )
        self.fixture_path.write_text(duplicate, encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "duplicate object key 'samplerMode'"
        ):
            VALIDATOR.validate_fixture(self.fixture_path, update=True)

        self.assertEqual(self.fixture_path.read_text(encoding="utf-8"), duplicate)

    def test_update_rejects_unknown_nested_control_without_rehashing(self) -> None:
        fixture = self.fixture()
        original_hash = fixture["contractSha256"]
        fixture["contract"]["diffusion"]["heunCorrection"] = True
        self.write_fixture(fixture)

        with self.assertRaisesRegex(ValueError, "unknown=\\['heunCorrection'\\]"):
            VALIDATOR.validate_fixture(self.fixture_path, update=True)

        self.assertEqual(self.fixture()["contractSha256"], original_hash)

    def test_update_rejects_postprocess_change_without_rehashing(self) -> None:
        fixture = self.fixture()
        original_hash = fixture["contractSha256"]
        fixture["contract"]["postprocess"]["normalize"] = False
        self.write_fixture(fixture)

        with self.assertRaisesRegex(ValueError, "postprocess profile differs"):
            VALIDATOR.validate_fixture(self.fixture_path, update=True)

        self.assertEqual(self.fixture()["contractSha256"], original_hash)

    def test_bool_integer_masquerade_is_rejected(self) -> None:
        fixture = self.fixture()
        original_hash = fixture["contractSha256"]
        fixture["contract"]["postprocess"]["normalize"] = 1
        self.write_fixture(fixture)

        with self.assertRaisesRegex(ValueError, "postprocess profile differs"):
            VALIDATOR.validate_fixture(self.fixture_path, update=True)

        self.assertEqual(self.fixture()["contractSha256"], original_hash)


if __name__ == "__main__":
    unittest.main()
