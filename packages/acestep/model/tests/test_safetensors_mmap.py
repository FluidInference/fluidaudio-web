from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors_mmap import (  # noqa: E402
    SafetensorsFile,
    SafetensorsFormatError,
)
from source_contract import SafetensorsContract  # noqa: E402


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    header: dict[str, object] = {"__metadata__": {"fixture": "yes"}}
    cursor = 0
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(item[3] for item in tensors))


class SafetensorsTests(unittest.TestCase):
    def test_reads_inventory_and_bounded_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.safetensors"
            write_safetensors(
                path,
                [
                    ("a", "BF16", [2], b"\x00\x00\x80\x3f"),
                    ("b", "F32", [1], b"\x00\x00\x00\x40"),
                ],
            )
            with SafetensorsFile(path) as checkpoint:
                self.assertEqual(checkpoint.parameter_count, 3)
                self.assertEqual(checkpoint.read_tensor_region("a", 2, 2), b"\x80\x3f")
                self.assertEqual(list(checkpoint.iter_tensor_chunks("b", chunk_bytes=2)), [b"\x00\x00", b"\x00\x40"])
                contract = SafetensorsContract(
                    tensor_count=2,
                    parameter_count=3,
                    header_length=checkpoint.header_length,
                    header_sha256=checkpoint.header_sha256,
                    inventory_sha256=checkpoint.inventory_sha256,
                )
                checkpoint.assert_contract(contract)

    def test_rejects_gap_overlap_and_shape_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, offsets, shape in (
                ("gap", [1, 3], [1]),
                ("shape", [0, 2], [2]),
            ):
                header = {"a": {"dtype": "BF16", "shape": shape, "data_offsets": offsets}}
                encoded = json.dumps(header).encode()
                path = root / f"{label}.safetensors"
                path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * max(offsets[1], 2))
                with self.assertRaises(SafetensorsFormatError):
                    SafetensorsFile(path)

    def test_rejects_duplicate_json_tensor_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.safetensors"
            record = '{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}'
            encoded = f'{{"a":{record},"a":{record}}}'.encode()
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0\0")
            with self.assertRaisesRegex(SafetensorsFormatError, "Duplicate JSON key"):
                SafetensorsFile(path)


if __name__ == "__main__":
    unittest.main()
