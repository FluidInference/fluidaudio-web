from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silence_latent import (  # noqa: E402
    MemberContract,
    extract_validated_zip_member,
)


class SilenceLatentTests(unittest.TestCase):
    def test_extracts_only_validated_storage_without_unpickling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.pt"
            values = {"x/data.pkl": b"never execute me", "x/data/0": b"raw-f32"}
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
                for name, payload in values.items():
                    output.writestr(name, payload)
            contracts = {
                name: MemberContract(len(payload), hashlib.sha256(payload).hexdigest())
                for name, payload in values.items()
            }
            destination = root / "tensor.bin"
            extract_validated_zip_member(
                archive,
                destination,
                contracts,
                "x/data/0",
            )
            self.assertEqual(destination.read_bytes(), b"raw-f32")

            bad_contracts = dict(contracts)
            bad_contracts["x/data.pkl"] = MemberContract(16, "0" * 64)
            with self.assertRaises(ValueError):
                extract_validated_zip_member(
                    archive,
                    root / "bad.bin",
                    bad_contracts,
                    "x/data/0",
                )


if __name__ == "__main__":
    unittest.main()
