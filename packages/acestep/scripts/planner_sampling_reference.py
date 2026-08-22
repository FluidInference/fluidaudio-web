"""Independent ACE browser-v1 planner-sampling reference.

The TypeScript production sampler owns its categorical distribution and does
not claim bit identity with a Torch device softmax. This Python 3.13 module
independently mirrors its fixed arithmetic, checks committed word/hash vectors,
and bounds the result against 80-digit ``Decimal.exp``. Coverage includes
adversarial exponent ranges, top-p boundaries, and a 217,204-row vocabulary
with exactly the 64,000 semantic-code rows populated.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = ROOT / "test" / "planner-sampling-vectors.json"

BROWSER_SOFTMAX_LN2 = 0.6931471805599453
BROWSER_EXP_TAYLOR = (
    1.0,
    1.0,
    0.5,
    0.16666666666666666,
    0.041666666666666664,
    0.008333333333333333,
    0.001388888888888889,
    0.0001984126984126984,
    0.0000248015873015873,
    0.0000027557319223985893,
    2.755731922398589e-7,
    2.505210838544172e-8,
    2.08767569878681e-9,
)

getcontext().prec = 80


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


NEGATIVE_POWERS_OF_TWO = [1.0]
for _index in range(151):
    NEGATIVE_POWERS_OF_TWO.append(NEGATIVE_POWERS_OF_TWO[-1] * 0.5)


def browser_exp_f32(value: float) -> float:
    """Mirror ``acePlannerBrowserExpF32`` without using ``math.exp``."""

    value = f32(value)
    if not math.isfinite(value) or value > 0:
        raise ValueError("browser exp input must be finite and non-positive")
    if value == 0:
        return f32(1.0)
    exponent = math.floor(value / BROWSER_SOFTMAX_LN2)
    if exponent < -151:
        return f32(0.0)
    remainder = value - exponent * BROWSER_SOFTMAX_LN2
    polynomial = BROWSER_EXP_TAYLOR[-1]
    for coefficient in reversed(BROWSER_EXP_TAYLOR[:-1]):
        polynomial = polynomial * remainder + coefficient
    return f32(polynomial * NEGATIVE_POWERS_OF_TWO[-exponent])


def stable_order(values: list[float]) -> list[int]:
    return sorted(
        (index for index, value in enumerate(values) if math.isfinite(value)),
        key=lambda index: (-values[index], index),
    )


def mask(values: Iterable[float], allowed: set[int]) -> list[float]:
    source = [f32(value) for value in values]
    return [value if index in allowed else -math.inf for index, value in enumerate(source)]


def cfg_on_allowed(cond: list[float], uncond: list[float], scale: float) -> list[float]:
    scale_f32 = f32(scale)
    output: list[float] = []
    for conditional, unconditional in zip(cond, uncond, strict=True):
        if conditional == -math.inf and unconditional == -math.inf:
            output.append(-math.inf)
            continue
        output.append(
            f32(unconditional + f32(scale_f32 * f32(conditional - unconditional)))
        )
    return output


def repetition(values: list[float], seen: Iterable[int], penalty: float) -> list[float]:
    output = list(values)
    penalty_f32 = f32(penalty)
    for token in dict.fromkeys(seen):
        value = output[token]
        if value == -math.inf:
            continue
        output[token] = f32(value * penalty_f32 if value < 0 else value / penalty_f32)
    return output


def top_k(values: list[float], count: int) -> list[float]:
    output = list(values)
    order = stable_order(output)
    if count == 0 or count >= len(order):
        return output
    threshold = output[order[count - 1]]
    return [value if value >= threshold else -math.inf for value in output]


def browser_softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    temperature_f32 = f32(temperature)
    scaled = [
        -math.inf if value == -math.inf else f32(f32(value) / temperature_f32)
        for value in values
    ]
    maximum = max(value for value in scaled if math.isfinite(value))
    weights: list[float] = []
    total = f32(0.0)
    for value in scaled:
        weight = (
            0.0
            if value == -math.inf
            else browser_exp_f32(f32(value - maximum))
        )
        weights.append(weight)
        total = f32(total + weight)
    return [0.0 if weight == 0 else f32(weight / total) for weight in weights]


def browser_top_p_keep(values: list[float], probability: float) -> bytearray:
    order = stable_order(values)
    keep = bytearray(len(values))
    if probability == 1:
        for token in order:
            keep[token] = 1
        return keep
    weights = browser_softmax(values)
    threshold = f32(probability)
    cumulative = f32(0.0)
    previous_crossed = False
    for token in order:
        if not previous_crossed:
            keep[token] = 1
        cumulative = f32(cumulative + weights[token])
        previous_crossed = cumulative > threshold
    return keep


def top_p(values: list[float], probability: float) -> list[float]:
    keep = browser_top_p_keep(values, probability)
    return [value if keep[index] else -math.inf for index, value in enumerate(values)]


def categorical(weights: list[float], word: int) -> int:
    rounded = [f32(weight) for weight in weights]
    total = sum(rounded)
    threshold = ((word + 0.5) / 2**32) * total
    cumulative = 0.0
    final_positive = -1
    for token, weight in enumerate(rounded):
        cumulative += weight
        if weight > 0:
            final_positive = token
            if cumulative > threshold:
                return token
    if final_positive < 0:
        raise ValueError("no categorical mass")
    return final_positive


def evaluate(vector: dict[str, object]) -> dict[str, object]:
    conditional = [float(value) for value in vector["conditional"]]  # type: ignore[index]
    unconditional_raw = vector.get("unconditional")
    allowed = {int(value) for value in vector["allowedTokenIds"]}  # type: ignore[index]
    cond_allowed = mask(conditional, allowed)
    if unconditional_raw is None:
        combined = cond_allowed
    else:
        uncond_allowed = mask(
            [float(value) for value in unconditional_raw],  # type: ignore[arg-type]
            allowed,
        )
        combined = cfg_on_allowed(
            cond_allowed,
            uncond_allowed,
            float(vector["guidanceScale"]),
        )
    penalized = repetition(
        combined,
        [int(value) for value in vector["seenTokenIds"]],  # type: ignore[index]
        float(vector["repetitionPenalty"]),
    )
    kept_k = top_k(penalized, int(vector["topK"]))
    kept_p = top_p(kept_k, float(vector["topP"]))
    weights = browser_softmax(kept_p, float(vector["temperature"]))
    return {
        "combinedWords": [f32_word(value) for value in combined],
        "penalizedWords": [f32_word(value) for value in penalized],
        "postTopKFinite": [index for index, value in enumerate(kept_k) if math.isfinite(value)],
        "postTopPFinite": [index for index, value in enumerate(kept_p) if math.isfinite(value)],
        "weightWords": [f32_word(value) for value in weights],
        "sampledTokenId": categorical(weights, int(vector["word"])),
    }


def decimal_softmax(values: list[float], temperature: float = 1.0) -> list[Decimal]:
    temperature_f32 = f32(temperature)
    scaled = [
        None if value == -math.inf else f32(f32(value) / temperature_f32)
        for value in values
    ]
    maximum = max(value for value in scaled if value is not None)
    cache: dict[Decimal, Decimal] = {}
    exponents: list[Decimal] = []
    for value in scaled:
        if value is None:
            exponents.append(Decimal(0))
            continue
        reduced = Decimal.from_float(float(f32(value - maximum)))
        exponent = cache.get(reduced)
        if exponent is None:
            exponent = reduced.exp()
            cache[reduced] = exponent
        exponents.append(exponent)
    total = sum(exponents, Decimal(0))
    return [value / total for value in exponents]


def decimal_top_p_keep(values: list[float], probability: float) -> bytearray:
    probabilities = decimal_softmax(values)
    threshold = Decimal.from_float(float(f32(probability)))
    keep = bytearray(len(values))
    cumulative = Decimal(0)
    previous_crossed = False
    for token in stable_order(values):
        if not previous_crossed:
            keep[token] = 1
        cumulative += probabilities[token]
        previous_crossed = cumulative > threshold
    return keep


def realistic_sparse_logits() -> list[float]:
    vocabulary = 217_204
    first_code = 151_669
    code_count = 64_000
    logits = [-math.inf] * vocabulary
    for offset in range(code_count):
        phase = (offset * 73 + 19) % 257
        logits[first_code + offset] = f32(-12.0 + phase / 16.0)
    for offset, value in ((0, 8.0), (1, 7.5), (31_999, 7.0), (63_999, 6.5)):
        logits[first_code + offset] = f32(value)
    return logits


def words_sha256(values: list[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<I", f32_word(value)))
    return digest.hexdigest()


def scalar_exp_errors() -> dict[str, float]:
    maximum_normal_relative = Decimal(0)
    maximum_subnormal_absolute = Decimal(0)
    minimum_normal = Decimal(2) ** Decimal(-126)
    # Include range-reduction edges plus a dense, deterministic full range.
    points = [f32(-104.0 * index / 8_192) for index in range(8_193)]
    points.extend(
        f32(-BROWSER_SOFTMAX_LN2 * exponent + delta)
        for exponent in range(0, 151)
        for delta in (-1e-7, 0.0, 1e-7)
        if -BROWSER_SOFTMAX_LN2 * exponent + delta <= 0
    )
    for point in points:
        exact = Decimal.from_float(float(point)).exp()
        actual = Decimal.from_float(float(browser_exp_f32(point)))
        error = abs(actual - exact)
        if exact >= minimum_normal:
            maximum_normal_relative = max(maximum_normal_relative, error / exact)
        else:
            maximum_subnormal_absolute = max(maximum_subnormal_absolute, error)
    return {
        "maximumNormalRelative": float(maximum_normal_relative),
        "maximumSubnormalAbsolute": float(maximum_subnormal_absolute),
    }


def realistic_validation() -> dict[str, object]:
    logits = realistic_sparse_logits()
    browser_keep = browser_top_p_keep(logits, 0.9)
    decimal_keep = decimal_top_p_keep(logits, 0.9)
    filtered = [
        value if browser_keep[index] else -math.inf
        for index, value in enumerate(logits)
    ]
    weights = browser_softmax(filtered, 0.85)
    decimal_weights = decimal_softmax(filtered, 0.85)
    maximum_absolute = Decimal(0)
    distribution_l1 = Decimal(0)
    for actual, expected in zip(weights, decimal_weights, strict=True):
        error = abs(Decimal.from_float(float(actual)) - expected)
        maximum_absolute = max(maximum_absolute, error)
        distribution_l1 += error

    browser_keep_count = sum(browser_keep)
    decimal_keep_count = sum(decimal_keep)
    decimal_untempered = decimal_softmax(logits)
    browser_retained_mass = sum(
        (decimal_untempered[index] for index, keep in enumerate(browser_keep) if keep),
        Decimal(0),
    )
    threshold = Decimal.from_float(float(f32(0.9)))
    return {
        "vocabularySize": len(logits),
        "firstSemanticTokenId": 151_669,
        "semanticTokenCount": 64_000,
        "topP": 0.9,
        "temperature": 0.85,
        "topPKeepSha256": hashlib.sha256(browser_keep).hexdigest(),
        "weightWordsSha256": words_sha256(weights),
        "positiveWeightCount": sum(value > 0 for value in weights),
        "browserKeepCount": browser_keep_count,
        "decimalKeepCount": decimal_keep_count,
        "keepCountDelta": abs(browser_keep_count - decimal_keep_count),
        "decimalRetainedMassError": float(abs(browser_retained_mass - threshold)),
        "maximumWeightAbsoluteError": float(maximum_absolute),
        "distributionL1Error": float(distribution_l1),
    }


def validation_report() -> dict[str, object]:
    return {
        "scalarExp": scalar_exp_errors(),
        "realisticSparse": realistic_validation(),
    }


def check_vectors() -> None:
    payload = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    if payload["status"] != "accepted-browser-softmax-v1":
        raise AssertionError("planner sampling fixture is not production-accepted")
    for vector in payload["vectors"]:
        actual = evaluate(vector)
        if actual != vector["expected"]:
            raise AssertionError(f"{vector['id']}: {actual!r} != {vector['expected']!r}")

    report = validation_report()
    fixture = payload["browserSoftmaxValidation"]
    realistic = report["realisticSparse"]
    expected_realistic = fixture["realisticSparse"]
    for key in (
        "vocabularySize",
        "firstSemanticTokenId",
        "semanticTokenCount",
        "topP",
        "temperature",
        "topPKeepSha256",
        "weightWordsSha256",
        "positiveWeightCount",
        "browserKeepCount",
        "decimalKeepCount",
    ):
        if realistic[key] != expected_realistic[key]:  # type: ignore[index]
            raise AssertionError(
                f"realisticSparse.{key}: {realistic[key]!r} != "  # type: ignore[index]
                f"{expected_realistic[key]!r}"  # type: ignore[index]
            )

    bounds = fixture["errorBounds"]
    scalar = report["scalarExp"]
    checks = (
        (scalar["maximumNormalRelative"], bounds["normalExpRelative"]),  # type: ignore[index]
        (scalar["maximumSubnormalAbsolute"], bounds["subnormalExpAbsolute"]),  # type: ignore[index]
        (realistic["maximumWeightAbsoluteError"], bounds["realisticWeightAbsolute"]),  # type: ignore[index]
        (realistic["distributionL1Error"], bounds["realisticDistributionL1"]),  # type: ignore[index]
        (realistic["keepCountDelta"], bounds["realisticTopPKeepCountDelta"]),  # type: ignore[index]
        (realistic["decimalRetainedMassError"], bounds["realisticTopPRetainedMass"]),  # type: ignore[index]
    )
    for actual, bound in checks:
        if actual > bound:
            raise AssertionError(f"browser softmax error {actual!r} exceeds {bound!r}")


def main() -> None:
    if sys.argv[1:] == ["--check"]:
        check_vectors()
        return
    if sys.argv[1:] == ["--print-validation"]:
        print(json.dumps(validation_report(), indent=2, sort_keys=True))
        return
    raise SystemExit(
        "usage: planner_sampling_reference.py (--check | --print-validation)"
    )


if __name__ == "__main__":
    main()
