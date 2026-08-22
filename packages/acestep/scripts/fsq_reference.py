#!/usr/bin/env python3
"""Independent Python 3.13 oracle for the pinned ACE ResidualFSQ codebook.

The upstream ``FSQ`` layer builds its implicit codebook in binary32. Moving the
ACE model to bfloat16 then rounds that non-persistent buffer before
``ResidualFSQ.get_output_from_indices`` reads it. The browser must therefore
decode the rounded bfloat16 values, not recompute the ideal fractions.

The raw-fp16 package profile is also sourced from the same bfloat16 model. Its
expected half values below are conversions of those rounded bfloat16 scalars,
not direct conversions of the ideal binary32 codebook.

This module uses only Python's binary packing primitives. It is deliberately
independent of both PyTorch and the TypeScript implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "test/fsq-codebook-vectors.json"

SOURCE_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
VECTOR_QUANTIZE_PYTORCH_VERSION = "1.27.20"
LEVELS = (8, 8, 8, 5, 5, 5)
BASES = (1, 8, 64, 512, 2_560, 12_800)
CODEBOOK_SIZE = 64_000
VECTOR_CODES = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    63,
    64,
    511,
    512,
    1_536,
    2_559,
    2_560,
    12_799,
    12_800,
    27_763,
    63_999,
)


def float32(value: float) -> float:
    """Round a Python binary64 value to IEEE-754 binary32."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def float32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def ideal_scalar(level: int, digit: int) -> float:
    """Mirror the binary32 operation order in FSQ._scale_and_shift_inverse."""

    scale = float32(2.0 / (level - 1))
    product = float32(float32(float(digit)) * scale)
    return float32(product - float32(1.0))


def bfloat16_bits_from_float32(value: float) -> int:
    """Convert finite binary32 to bfloat16 with round-to-nearest-even."""

    bits = float32_bits(value)
    upper = bits >> 16
    lower = bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1) != 0):
        upper += 1
    return upper & 0xFFFF


def float32_from_bfloat16_bits(bits: int) -> float:
    if bits < 0 or bits > 0xFFFF:
        raise ValueError("bfloat16 bits must fit in 16 bits")
    return float32_from_bits(bits << 16)


def float16_bits_from_bfloat16_bits(bits: int) -> int:
    """Convert through the bfloat16 value, matching package preparation."""

    value = float32_from_bfloat16_bits(bits)
    return struct.unpack("<H", struct.pack("<e", value))[0]


def scalar_record(level: int, digit: int) -> dict[str, Any]:
    ideal = ideal_scalar(level, digit)
    bfloat16_bits = bfloat16_bits_from_float32(ideal)
    return {
        "digit": digit,
        "idealF32Bits": f"{float32_bits(ideal):08x}",
        "bfloat16Bits": f"{bfloat16_bits:04x}",
        "bfloat16Value": float32_from_bfloat16_bits(bfloat16_bits),
        "fp16BitsFromBfloat16": (
            f"{float16_bits_from_bfloat16_bits(bfloat16_bits):04x}"
        ),
    }


def decode_record(code: int) -> dict[str, Any]:
    if code < 0 or code >= CODEBOOK_SIZE:
        raise ValueError(f"code must be in [0, {CODEBOOK_SIZE})")
    digits = [(code // base) % level for base, level in zip(BASES, LEVELS)]
    scalars = [scalar_record(level, digit) for level, digit in zip(LEVELS, digits)]
    return {
        "code": code,
        "digits": digits,
        "bfloat16Bits": [record["bfloat16Bits"] for record in scalars],
        "bfloat16Values": [record["bfloat16Value"] for record in scalars],
        "fp16BitsFromBfloat16": [
            record["fp16BitsFromBfloat16"] for record in scalars
        ],
    }


def full_codebook_hashes() -> tuple[str, str]:
    bfloat16_hash = hashlib.sha256()
    fp16_hash = hashlib.sha256()
    for code in range(CODEBOOK_SIZE):
        for base, level in zip(BASES, LEVELS):
            digit = (code // base) % level
            bits = bfloat16_bits_from_float32(ideal_scalar(level, digit))
            bfloat16_hash.update(struct.pack("<H", bits))
            fp16_hash.update(
                struct.pack("<H", float16_bits_from_bfloat16_bits(bits))
            )
    return bfloat16_hash.hexdigest(), fp16_hash.hexdigest()


def build_vectors() -> dict[str, Any]:
    bfloat16_hash, fp16_hash = full_codebook_hashes()
    return {
        "schema": "ace-fsq-bfloat16-codebook-v1",
        "sourceRevision": SOURCE_REVISION,
        "vectorQuantizePytorchVersion": VECTOR_QUANTIZE_PYTORCH_VERSION,
        "construction": (
            "ideal-f32-implicit-codebook then model.to(bfloat16); "
            "raw-fp16 converts those bfloat16 values"
        ),
        "levels": list(LEVELS),
        "bases": list(BASES),
        "codebookSize": CODEBOOK_SIZE,
        "scalarValues": {
            "level8": [scalar_record(8, digit) for digit in range(8)],
            "level5": [scalar_record(5, digit) for digit in range(5)],
        },
        "vectors": [decode_record(code) for code in VECTOR_CODES],
        "fullCodebook": {
            "shape": [CODEBOOK_SIZE, len(LEVELS)],
            "elementOrder": "row-major-code-then-dimension",
            "bfloat16LittleEndianSha256": bfloat16_hash,
            "fp16LittleEndianFromBfloat16Sha256": fp16_hash,
        },
    }


def encoded_vectors() -> str:
    return json.dumps(build_vectors(), indent=2) + "\n"


def verify_vectors(path: Path = VECTORS) -> None:
    expected = encoded_vectors()
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError(f"FSQ vectors differ from the independent oracle: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("vectors", nargs="?", type=Path, default=VECTORS)
    args = parser.parse_args()
    if args.write:
        args.vectors.write_text(encoded_vectors(), encoding="utf-8")
        print(f"wrote ACE FSQ vectors: {args.vectors}")
        return 0
    try:
        verify_vectors(args.vectors)
    except (OSError, ValueError) as error:
        print(f"FSQ vector validation failed: {error}")
        return 1
    print(f"validated ACE FSQ vectors: {args.vectors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
