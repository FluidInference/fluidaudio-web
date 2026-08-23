#!/usr/bin/env python3
"""Independent Python 3.13 oracle for the browser-defined ACE PRNG contract.

This module deliberately avoids ``random``, NumPy, and host transcendental
functions. It mirrors the pinned binary64 operation order, then rounds Gaussian
outputs to binary32. The committed JSON vectors are the cross-language boundary;
this file is a readable reference and validator, not a runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "golden/prng/ace-seed-v1-vectors.json"

MAX_UINT32 = 0xFFFF_FFFF
MAX_UINT64 = 0xFFFF_FFFF_FFFF_FFFF
MAX_WORD_INDEX = (MAX_UINT64 << 2) | 3
UINT32_MIDPOINT_SCALE = 1.0 / 0x1_0000_0000
ACKLAM_LOW_TAIL = 0.02425
ACKLAM_HIGH_TAIL = 1.0 - ACKLAM_LOW_TAIL
NATURAL_LOG_OF_TWO = 0.6931471805599453

PHILOX_MULTIPLIER_0 = 0xD251_1F53
PHILOX_MULTIPLIER_1 = 0xCD9E_8D57
PHILOX_WEYL_0 = 0x9E37_79B9
PHILOX_WEYL_1 = 0xBB67_AE85
PHILOX_ROUNDS = 10

STREAM_DOMAINS = {
    "diffusion-noise": (0x4449_4646, 0x4E4F_4953),
    "planner-sampling": (0x504C_414E, 0x5341_4D50),
}

ACKLAM_A = (
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
)
ACKLAM_B = (
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
)
ACKLAM_C = (
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
)
ACKLAM_D = (
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
)


def _uint32(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an unsigned 32-bit integer")
    if value < 0 or value > MAX_UINT32:
        raise ValueError(f"{label} must be an unsigned 32-bit integer")
    return value


def _multiply_high_low(left: int, right: int) -> tuple[int, int]:
    product = _uint32(left, "left") * _uint32(right, "right")
    return (product >> 32) & MAX_UINT32, product & MAX_UINT32


def philox4x32_10(
    counter: Sequence[int], key: Sequence[int]
) -> tuple[int, int, int, int]:
    if len(counter) != 4 or len(key) != 2:
        raise ValueError("Philox requires four counter words and two key words")
    word0, word1, word2, word3 = (
        _uint32(value, f"counter[{index}]")
        for index, value in enumerate(counter)
    )
    key0, key1 = (
        _uint32(value, f"key[{index}]") for index, value in enumerate(key)
    )

    for _ in range(PHILOX_ROUNDS):
        high0, low0 = _multiply_high_low(PHILOX_MULTIPLIER_0, word0)
        high1, low1 = _multiply_high_low(PHILOX_MULTIPLIER_1, word2)
        word0, word1, word2, word3 = (
            (high1 ^ word1 ^ key0) & MAX_UINT32,
            low1,
            (high0 ^ word3 ^ key1) & MAX_UINT32,
            low0,
        )
        key0 = (key0 + PHILOX_WEYL_0) & MAX_UINT32
        key1 = (key1 + PHILOX_WEYL_1) & MAX_UINT32
    return word0, word1, word2, word3


def random_words(seed: str, stream: str, block_index: int) -> tuple[int, ...]:
    if (
        not isinstance(seed, str)
        or len(seed) != 16
        or any(character not in "0123456789abcdef" for character in seed)
    ):
        raise ValueError("seed must be 16 lowercase hexadecimal digits")
    if stream not in STREAM_DOMAINS:
        raise ValueError(f"unknown ACE random stream: {stream!r}")
    if (
        isinstance(block_index, bool)
        or not isinstance(block_index, int)
        or block_index < 0
        or block_index > MAX_UINT64
    ):
        raise ValueError("block index must fit in an unsigned 64-bit integer")

    seed_value = int(seed, 16)
    seed_low = seed_value & MAX_UINT32
    seed_high = seed_value >> 32
    domain0, domain1 = STREAM_DOMAINS[stream]
    return philox4x32_10(
        (block_index & MAX_UINT32, block_index >> 32, 0, 0),
        (seed_low ^ domain0, seed_high ^ domain1),
    )


def random_word(seed: str, stream: str, word_index: int) -> int:
    if (
        isinstance(word_index, bool)
        or not isinstance(word_index, int)
        or word_index < 0
        or word_index > MAX_WORD_INDEX
    ):
        raise ValueError("word index must fit in 66 unsigned bits")
    return random_words(seed, stream, word_index >> 2)[word_index & 3]


def open_unit_float64_from_word(word: int) -> float:
    return (_uint32(word, "word") + 0.5) * UINT32_MIDPOINT_SCALE


def _deterministic_natural_log(value: float) -> float:
    if not (value > 0.0) or value == float("inf"):
        raise ValueError("deterministic log requires a positive finite value")
    mantissa = value
    exponent = 0
    while mantissa < 1.0:
        mantissa *= 2.0
        exponent -= 1
    while mantissa >= 2.0:
        mantissa *= 0.5
        exponent += 1
    y = (mantissa - 1.0) / (mantissa + 1.0)
    y_squared = y * y
    term = y
    total = y
    for denominator in range(3, 50, 2):
        term *= y_squared
        total += term / denominator
    return 2.0 * total + exponent * NATURAL_LOG_OF_TWO


def _deterministic_square_root(value: float) -> float:
    if value < 0.0 or value == float("inf") or value != value:
        raise ValueError("deterministic square root requires a finite value")
    if value == 0.0:
        return 0.0
    estimate = value if value >= 1.0 else 1.0
    for _ in range(10):
        estimate = 0.5 * (estimate + value / estimate)
    return estimate


def _acklam_tail(q: float) -> float:
    numerator = (
        ((((ACKLAM_C[0] * q + ACKLAM_C[1]) * q + ACKLAM_C[2]) * q + ACKLAM_C[3])
         * q + ACKLAM_C[4])
        * q
        + ACKLAM_C[5]
    )
    denominator = (
        (((ACKLAM_D[0] * q + ACKLAM_D[1]) * q + ACKLAM_D[2]) * q + ACKLAM_D[3])
        * q
        + 1.0
    )
    return numerator / denominator


def gaussian_from_word(word: int) -> float:
    probability = open_unit_float64_from_word(word)
    if probability < ACKLAM_LOW_TAIL:
        q = _deterministic_square_root(
            -2.0 * _deterministic_natural_log(probability)
        )
        normal = _acklam_tail(q)
    elif probability > ACKLAM_HIGH_TAIL:
        q = _deterministic_square_root(
            -2.0 * _deterministic_natural_log(1.0 - probability)
        )
        normal = -_acklam_tail(q)
    else:
        q = probability - 0.5
        r = q * q
        numerator = (
            (((((ACKLAM_A[0] * r + ACKLAM_A[1]) * r + ACKLAM_A[2]) * r
               + ACKLAM_A[3]) * r + ACKLAM_A[4]) * r + ACKLAM_A[5])
            * q
        )
        denominator = (
            ((((ACKLAM_B[0] * r + ACKLAM_B[1]) * r + ACKLAM_B[2]) * r
              + ACKLAM_B[3]) * r + ACKLAM_B[4])
            * r
            + 1.0
        )
        normal = numerator / denominator
    return struct.unpack("<f", struct.pack("<f", normal))[0]


def float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise ValueError("categorical weights must fit binary32") from error


def categorical_token_from_word(weights: Sequence[float], word: int) -> int:
    if not weights:
        raise ValueError("categorical weights must not be empty")
    total = 0.0
    final_positive_token = -1
    for token, weight in enumerate(weights):
        if not isinstance(weight, (float, int)) or isinstance(weight, bool):
            raise TypeError("categorical weights must be numeric")
        value = _float32(float(weight))
        if value < 0.0 or value == float("inf") or value != value:
            raise ValueError("categorical weights must be finite and nonnegative")
        if value > 0.0:
            final_positive_token = token
        total += value
    if not (total > 0.0) or total == float("inf"):
        raise ValueError("categorical weights must have a finite positive sum")

    threshold = open_unit_float64_from_word(word) * total
    cumulative = 0.0
    for token, weight in enumerate(weights):
        value = _float32(float(weight))
        cumulative += value
        if value > 0.0 and cumulative > threshold:
            return token
    return final_positive_token


def verify_vectors(path: Path = VECTORS) -> None:
    vectors = json.loads(path.read_text(encoding="utf-8"))
    for vector in vectors["philox"]:
        actual = random_words(
            vector["seed"], vector["stream"], int(vector["blockIndexHex"], 16)
        )
        if list(actual) != vector["outputWords"]:
            raise ValueError(f"Philox vector {vector['id']} does not match")
    for vector in vectors["gaussian"]:
        actual = float32_bits(gaussian_from_word(vector["word"]))
        if actual != vector["outputF32Bits"]:
            raise ValueError(f"Gaussian vector {vector['id']} does not match")
    for vector in vectors["streamGaussian"]:
        word = random_word(vector["seed"], "diffusion-noise", vector["wordIndex"])
        actual = float32_bits(gaussian_from_word(word))
        if word != vector["word"] or actual != vector["outputF32Bits"]:
            raise ValueError(f"stream Gaussian vector {vector['id']} does not match")
    for vector in vectors["categorical"]:
        actual = categorical_token_from_word(vector["weights"], vector["word"])
        if actual != vector["token"]:
            raise ValueError(f"categorical vector {vector['id']} does not match")
    for vector in vectors["plannerStream"]:
        word = random_word(vector["seed"], "planner-sampling", vector["drawIndex"])
        actual = categorical_token_from_word(vector["weights"], word)
        if word != vector["word"] or actual != vector["token"]:
            raise ValueError(f"planner stream vector {vector['id']} does not match")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vectors", nargs="?", type=Path, default=VECTORS)
    args = parser.parse_args()
    try:
        verify_vectors(args.vectors)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"PRNG vector validation failed: {error}")
        return 1
    print(f"validated ACE PRNG vectors: {args.vectors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
