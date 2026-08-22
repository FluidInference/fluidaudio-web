#!/usr/bin/env python3
"""Validate committed golden contracts without downloading model weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "golden"
FIXTURES = GOLDEN / "fixtures"
MANIFEST = GOLDEN / "MANIFEST.json"
HASHED_FILES = (
    Path("README.md"),
    Path("fixture.schema.json"),
    Path("taps.json"),
    Path("prng/ace-seed-v1.json"),
    Path("prng/ace-seed-v1-vectors.json"),
    Path("expected/README.md"),
    Path("expected/status.json"),
)

FIXTURE_PROFILES: dict[str, dict[str, Any]] = {
    "direct-instrumental-short": {
        "planner": False,
        "instrumental": True,
        "duration": 12,
        "seed": "0000000000c0ffee",
        "initialNoiseSha256": "0334c9c4176e8945f33aa2c9538d0fa74fe46855141dcfac1f822455310d5e29",
        "coverage": [
            "direct",
            "instrumental",
            "short-conditioning",
            "short-duration",
            "dcw-default",
        ],
    },
    "direct-lyrics-long-condition": {
        "planner": False,
        "instrumental": False,
        "duration": 20,
        "seed": "00000000deadbeef",
        "initialNoiseSha256": "3e6e13598e8849f66702a80e8d89e0710497f74850d18371a6b0e5bb448af532",
        "coverage": [
            "direct",
            "supplied-lyrics",
            "long-conditioning",
            "short-duration",
            "dcw-default",
        ],
    },
    "direct-lyrics-short": {
        "planner": False,
        "instrumental": False,
        "duration": 16,
        "seed": "00000000a205b064",
        "initialNoiseSha256": "2a4acd57ee71340608b8cf5943db60ffa8bfe138c5b62f597a8cf5af0d6767be",
        "coverage": [
            "direct",
            "supplied-lyrics",
            "short-conditioning",
            "short-duration",
            "dcw-default",
        ],
    },
    "planner-lyrics-short": {
        "planner": True,
        "instrumental": False,
        "duration": 12,
        "seed": "000000000badc0de",
        "initialNoiseSha256": "c60e0c8886e32109a6d1317280f95d2406a6b9aa0e440f76fddac36766af3edd",
        "coverage": [
            "planner",
            "supplied-lyrics",
            "short-conditioning",
            "short-duration",
            "two-row-cfg",
            "dcw-think-default",
        ],
    },
}

TOP_LEVEL_KEYS = {"schemaVersion", "fixtureId", "contractSha256", "contract"}
CONTRACT_KEYS = {
    "coverage",
    "source",
    "request",
    "planner",
    "diffusion",
    "postprocess",
    "random",
    "expected",
}
REQUEST_KEYS = {
    "batchSize",
    "taskType",
    "resolvedTaskType",
    "instruction",
    "caption",
    "lyrics",
    "instrumental",
    "vocalLanguage",
    "bpm",
    "keyScale",
    "timeSignature",
    "durationSeconds",
    "referenceAudio",
    "sourceAudio",
    "audioCodes",
    "audioCoverStrength",
    "coverNoiseStrength",
    "chunkMaskMode",
    "repaintingStart",
    "repaintingEnd",
    "retakeVariance",
    "retakeSeed",
}
PLANNER_BASE_KEYS = {
    "enabled",
    "thinking",
    "useCotMetas",
    "useCotCaption",
    "useCotLyrics",
    "useCotLanguage",
    "useConstrainedDecoding",
    "temperature",
    "cfgScale",
    "topK",
    "topP",
    "repetitionPenalty",
    "negativePrompt",
}
PLANNER_ENABLED_KEYS = {
    "semanticRateHz",
    "codeVocabularySize",
    "samplingBatchRows",
    "userMetadataWins",
}
DIFFUSION_KEYS = {
    "inferenceSteps",
    "shift",
    "customTimesteps",
    "resolvedTimesteps",
    "effectiveTimestepsBfloat16",
    "inferMethod",
    "samplerMode",
    "guidanceScale",
    "useAdg",
    "cfgIntervalStart",
    "cfgIntervalEnd",
    "velocityNormThreshold",
    "velocityEmaFactor",
    "dcw",
    "latentShift",
    "latentRescale",
}
DCW_KEYS = {
    "enabled",
    "mode",
    "lowScaler",
    "highScaler",
    "wavelet",
    "levels",
    "boundaryMode",
    "computeDtype",
}
POSTPROCESS_KEYS = {
    "vaeDtype",
    "sampleRate",
    "channels",
    "normalize",
    "normalizationDb",
    "fadeInSeconds",
    "fadeOutSeconds",
    "captureRawBeforeNormalization",
}
RANDOM_BASE_KEYS = {
    "userSeed",
    "useRandomSeed",
    "algorithmId",
    "generator",
    "diffusionStream",
    "plannerStream",
    "gaussianMapping",
    "categoricalMapping",
    "initialNoise",
    "referenceInitialNoiseInjectionRequired",
    "randomTransformCaptureStatus",
}
INITIAL_NOISE_KEYS = {"tapId", "dtype", "shape", "layout", "sha256"}

SOURCE_PROFILE = {
    "aceRepositoryCommit": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
    "mainModelId": "ACE-Step/Ace-Step1.5",
    "mainModelRevision": "19671f406d603126926c1b7e2adc169acbcade22",
    "mainModelVariant": "acestep-v15-turbo",
    "plannerModelId": "ACE-Step/acestep-5Hz-lm-0.6B",
    "plannerModelRevision": "148d8ea0225bdab342ee1ae3a354275ccd60ca80",
}
POSTPROCESS_PROFILE = {
    "vaeDtype": "float32",
    "sampleRate": 48_000,
    "channels": 2,
    "normalize": True,
    "normalizationDb": -1.0,
    "fadeInSeconds": 0.0,
    "fadeOutSeconds": 0.0,
    "captureRawBeforeNormalization": True,
}
RESOLVED_TIMESTEPS = [
    1.0,
    0.9545454545454546,
    0.9,
    0.8333333333333334,
    0.75,
    0.6428571428571429,
    0.5,
    0.3,
]
EFFECTIVE_TIMESTEPS_BFLOAT16 = [
    1.0,
    0.953125,
    0.8984375,
    0.83203125,
    0.75,
    0.64453125,
    0.5,
    0.30078125,
]


class DuplicateJsonKeyError(ValueError):
    """Raised when an object contains a key more than once."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateJsonKeyError,
        ValueError,
    ) as exc:
        raise ValueError(f"{display_path(path)}: invalid JSON: {exc}") from exc


def require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise ValueError(f"{display_path(path)}: {message}")


def same_json_value(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""
    return canonical_json(actual) == canonical_json(expected)


def require_exact_keys(
    value: Any, expected: Iterable[str], path: Path, label: str
) -> dict[str, Any]:
    require(isinstance(value, dict), path, f"{label} must be an object")
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    unknown = sorted(actual_set - expected_set)
    details = []
    if missing:
        details.append(f"missing={missing}")
    if unknown:
        details.append(f"unknown={unknown}")
    require(not details, path, f"{label} keys mismatch ({'; '.join(details)})")
    return value


def expected_geometry(duration: int) -> dict[str, int]:
    latent = max(128, duration * 25)
    return {
        "latentFrames": latent,
        "patchedTokens": math.ceil(latent / 2),
        "audioSamplesPerChannel": duration * 48_000,
    }


def validate_request(
    request_value: Any, profile: dict[str, Any], path: Path
) -> dict[str, Any]:
    request = require_exact_keys(request_value, REQUEST_KEYS, path, "contract.request")
    planner_enabled = profile["planner"]
    fixed = {
        "batchSize": 1,
        "taskType": "text2music",
        "resolvedTaskType": "cover" if planner_enabled else "text2music",
        "instruction": (
            "Generate audio semantic tokens based on the given conditions:"
            if planner_enabled
            else "Fill the audio semantic mask based on the given conditions:"
        ),
        "instrumental": profile["instrumental"],
        "durationSeconds": profile["duration"],
        "referenceAudio": None,
        "sourceAudio": None,
        "audioCodes": None if planner_enabled else "",
        "audioCoverStrength": 1.0,
        "coverNoiseStrength": 0.0,
        "chunkMaskMode": "auto",
        "repaintingStart": 0.0,
        "repaintingEnd": None,
        "retakeVariance": 0.0,
        "retakeSeed": None,
    }
    for key, expected in fixed.items():
        require(
            same_json_value(request[key], expected),
            path,
            f"request.{key} must be {expected!r}",
        )

    require(
        isinstance(request["caption"], str) and 0 < len(request["caption"]) <= 512,
        path,
        "request.caption length invalid",
    )
    require(
        isinstance(request["lyrics"], str) and len(request["lyrics"]) <= 4096,
        path,
        "request.lyrics length invalid",
    )
    if request["instrumental"]:
        require(
            request["lyrics"] == "[Instrumental]",
            path,
            "instrumental fixture must use the canonical lyrics marker",
        )
    else:
        require(bool(request["lyrics"]), path, "vocal fixture requires supplied lyrics")

    bpm = request["bpm"]
    require(
        bpm is None or (type(bpm) is int and 30 <= bpm <= 300),
        path,
        "request.bpm must be null or an integer in 30..300",
    )
    for key in ("vocalLanguage", "keyScale", "timeSignature"):
        require(isinstance(request[key], str), path, f"request.{key} must be a string")
    if planner_enabled:
        require(
            bpm is None
            and request["keyScale"] == ""
            and request["timeSignature"] == ""
            and request["vocalLanguage"] == "unknown",
            path,
            "planner fixture must leave metadata and language unresolved for CoT",
        )
    else:
        require(type(bpm) is int, path, "direct fixture requires a fixed integer bpm")
        require(bool(request["keyScale"]), path, "direct fixture requires keyScale")
        require(
            bool(request["timeSignature"]),
            path,
            "direct fixture requires timeSignature",
        )
        expected_language = "unknown" if request["instrumental"] else "en"
        require(
            request["vocalLanguage"] == expected_language,
            path,
            f"direct fixture vocalLanguage must be {expected_language!r}",
        )
    return request


def validate_planner(
    planner_value: Any, planner_enabled: bool, path: Path
) -> dict[str, Any]:
    expected = {
        "enabled": planner_enabled,
        "thinking": planner_enabled,
        "useCotMetas": planner_enabled,
        "useCotCaption": planner_enabled,
        "useCotLyrics": False,
        "useCotLanguage": planner_enabled,
        "useConstrainedDecoding": True,
        "temperature": 0.85,
        "cfgScale": 2.0,
        "topK": 0,
        "topP": 0.9,
        "repetitionPenalty": 1.0,
        "negativePrompt": "NO USER INPUT",
    }
    if planner_enabled:
        expected.update(
            {
                "semanticRateHz": 5,
                "codeVocabularySize": 64_000,
                "samplingBatchRows": 2,
                "userMetadataWins": True,
            }
        )
    keys = PLANNER_BASE_KEYS | (PLANNER_ENABLED_KEYS if planner_enabled else set())
    planner = require_exact_keys(planner_value, keys, path, "contract.planner")
    require(
        same_json_value(planner, expected),
        path,
        "planner profile differs from pinned defaults",
    )
    return planner


def validate_diffusion(
    diffusion_value: Any, planner_enabled: bool, path: Path
) -> dict[str, Any]:
    diffusion = require_exact_keys(
        diffusion_value, DIFFUSION_KEYS, path, "contract.diffusion"
    )
    dcw = require_exact_keys(diffusion["dcw"], DCW_KEYS, path, "contract.diffusion.dcw")
    expected_dcw = {
        "enabled": True,
        "mode": "double",
        "lowScaler": 0.02 if planner_enabled else 0.05,
        "highScaler": 0.06 if planner_enabled else 0.02,
        "wavelet": "haar",
        "levels": 1,
        "boundaryMode": "zero",
        "computeDtype": "float32",
    }
    expected = {
        "inferenceSteps": 8,
        "shift": 3.0,
        "customTimesteps": None,
        "resolvedTimesteps": RESOLVED_TIMESTEPS,
        "effectiveTimestepsBfloat16": EFFECTIVE_TIMESTEPS_BFLOAT16,
        "inferMethod": "ode",
        "samplerMode": "euler",
        "guidanceScale": 1.0,
        "useAdg": False,
        "cfgIntervalStart": 0.0,
        "cfgIntervalEnd": 1.0,
        "velocityNormThreshold": 0.0,
        "velocityEmaFactor": 0.0,
        "dcw": expected_dcw,
        "latentShift": 0.0,
        "latentRescale": 1.0,
    }
    require(
        same_json_value(dcw, expected_dcw),
        path,
        "DCW profile differs from pinned defaults",
    )
    require(
        same_json_value(diffusion, expected),
        path,
        "diffusion profile differs from pinned defaults",
    )
    return diffusion


def validate_random(
    random_value: Any, profile: dict[str, Any], path: Path
) -> dict[str, Any]:
    planner_enabled = profile["planner"]
    keys = RANDOM_BASE_KEYS | (
        {"referencePlannerWordsInjectionRequired"} if planner_enabled else set()
    )
    random = require_exact_keys(random_value, keys, path, "contract.random")
    initial_noise = require_exact_keys(
        random["initialNoise"], INITIAL_NOISE_KEYS, path, "contract.random.initialNoise"
    )
    duration = profile["duration"]
    expected_noise = {
        "tapId": "diffusion.initial-noise",
        "dtype": "float32",
        "shape": [1, expected_geometry(duration)["latentFrames"], 64],
        "layout": "row-major-little-endian",
        "sha256": profile["initialNoiseSha256"],
    }
    expected = {
        "userSeed": profile["seed"],
        "useRandomSeed": False,
        "algorithmId": "ace-seed-v1",
        "generator": "philox4x32-10",
        "diffusionStream": "diffusion-noise",
        "plannerStream": "planner-sampling",
        "gaussianMapping": "probit-acklam-binary64-f32-v1",
        "categoricalMapping": "u32-midpoint-binary64-cdf-v1",
        "initialNoise": expected_noise,
        "referenceInitialNoiseInjectionRequired": True,
        "randomTransformCaptureStatus": "ready-for-native-capture",
    }
    if planner_enabled:
        expected["referencePlannerWordsInjectionRequired"] = True
    require(
        same_json_value(initial_noise, expected_noise),
        path,
        "initial-noise contract mismatch",
    )
    require(
        same_json_value(random, expected),
        path,
        "random profile differs from pinned defaults",
    )
    return random


def validate_fixture(path: Path, update: bool = False) -> tuple[str, str]:
    fixture = require_exact_keys(load_json(path), TOP_LEVEL_KEYS, path, "fixture")
    require(fixture["schemaVersion"] == 1, path, "schemaVersion must be 1")
    fixture_id = fixture["fixtureId"]
    require(
        isinstance(fixture_id, str) and path.stem == fixture_id,
        path,
        "fixtureId must equal filename",
    )
    require(fixture_id in FIXTURE_PROFILES, path, f"unknown v1 fixture ID {fixture_id!r}")
    profile = FIXTURE_PROFILES[fixture_id]
    require(
        isinstance(fixture["contractSha256"], str)
        and len(fixture["contractSha256"]) == 64
        and all(c in "0123456789abcdef" for c in fixture["contractSha256"]),
        path,
        "contractSha256 must be canonical lowercase SHA-256",
    )

    contract = require_exact_keys(fixture["contract"], CONTRACT_KEYS, path, "contract")
    require(
        contract["coverage"] == profile["coverage"],
        path,
        "coverage does not match the fixture matrix",
    )
    source = require_exact_keys(
        contract["source"], SOURCE_PROFILE, path, "contract.source"
    )
    require(
        same_json_value(source, SOURCE_PROFILE),
        path,
        "source revisions differ from pinned profile",
    )
    validate_request(contract["request"], profile, path)
    validate_planner(contract["planner"], profile["planner"], path)
    validate_diffusion(contract["diffusion"], profile["planner"], path)

    postprocess = require_exact_keys(
        contract["postprocess"], POSTPROCESS_KEYS, path, "contract.postprocess"
    )
    require(
        same_json_value(postprocess, POSTPROCESS_PROFILE),
        path,
        "postprocess profile differs from pinned defaults",
    )
    validate_random(contract["random"], profile, path)

    geometry = expected_geometry(profile["duration"])
    expected = dict(geometry)
    expected["semanticCodes"] = profile["duration"] * 5 if profile["planner"] else 0
    if profile["planner"]:
        expected["semanticFrames"] = geometry["latentFrames"]
    actual_expected = require_exact_keys(
        contract["expected"], expected, path, "contract.expected"
    )
    require(
        same_json_value(actual_expected, expected),
        path,
        "expected geometry does not match request",
    )

    contract_hash = sha256(canonical_json(contract))
    if not update:
        require(
            fixture["contractSha256"] == contract_hash,
            path,
            "contractSha256 does not match canonical contract",
        )
    return fixture_id, contract_hash


def validate_prng_contract() -> None:
    path = GOLDEN / "prng/ace-seed-v1.json"
    expected = {
        "schemaVersion": 1,
        "contractId": "ace-seed-v1",
        "authority": "browser-defined",
        "seedSyntax": "uint64-lowercase-hex-16",
        "generator": "philox4x32-10",
        "counterOrder": "little-endian-word-index",
        "streams": {
            "diffusion-noise": {"domainWords": [1145652806, 1313818963]},
            "planner-sampling": {"domainWords": [1347174734, 1396788560]},
        },
        "philox": {
            "rounds": 10,
            "multipliers": [3528531795, 3449720151],
            "weyl": [2654435769, 3144134277],
            "arithmetic": "all operations modulo 2^32",
        },
        "knownAnswer": {
            "counterWords": [0, 0, 0, 0],
            "keyWords": [0, 0],
            "outputWords": [1713891541, 3781805453, 3159862348, 2600524760],
        },
        "wordAddressing": {
            "blockIndex": "floor(globalWordIndex / 4)",
            "laneOrder": ["word0", "word1", "word2", "word3"],
            "maximumGlobalWordIndexHex": "3ffffffffffffffff",
        },
        "gaussian": {
            "mappingId": "probit-acklam-binary64-f32-v1",
            "uniform": "u=(uint32(word)+0.5)*2^-32, evaluated exactly in binary64",
            "algorithm": (
                "Acklam inverse-standard-normal rational approximation with fixed "
                "coefficient and operation order"
            ),
            "deterministicLog": (
                "binary64 power-of-two range reduction plus 25 atanh terms through "
                "denominator 49"
            ),
            "deterministicSqrt": (
                "10 fixed binary64 Newton iterations, initial estimate max(x,1)"
            ),
            "hostTranscendentals": False,
            "output": (
                "one IEEE-754 binary32 value, round-to-nearest-ties-to-even"
            ),
            "consumption": (
                "one diffusion-noise word per row-major "
                "[batch,latent-frame,channel] element"
            ),
            "executionLocation": "dedicated-worker-cpu",
        },
        "categorical": {
            "mappingId": "u32-midpoint-binary64-cdf-v1",
            "uniform": "u=(uint32(word)+0.5)*2^-32, evaluated exactly in binary64",
            "weights": (
                "original-token-ID array; each finite nonnegative value rounds once "
                "to binary32"
            ),
            "plannerPipeline": (
                "CFG combine for semantic codes, constrained FSM mask, repetition "
                "penalty, top-k, top-p on untempered FP32 logits, temperature "
                "division, FP32 softmax, then this mapping; descending-logit ties "
                "use ascending token ID"
            ),
            "accumulation": (
                "binary64, ascending token ID, first total pass then second "
                "cumulative pass"
            ),
            "selection": (
                "first positive-weight token with cumulative>u*total; final positive "
                "token is the upper-rounding guard"
            ),
            "consumption": (
                "one planner-sampling word per emitted token, including forced "
                "single-candidate and stop tokens"
            ),
            "plannerOrder": (
                "continuous draw ordinal across CoT then semantic-code phases; "
                "autoregressive step then item; CFG model rows share one sampled "
                "token and consume no extra word"
            ),
            "executionLocation": "dedicated-worker-cpu",
        },
        "vectors": {
            "path": "prng/ace-seed-v1-vectors.json",
            "pythonOracle": "scripts/prng_reference.py",
            "typescriptTest": "test/seed.test.ts",
        },
        "transformStatus": {
            "uniformWords": "pinned",
            "diffusionGaussianMapping": "pinned",
            "plannerCategoricalMapping": "pinned",
            "nativeCapture": "pending",
        },
        "referenceInterop": (
            "Inject the exact browser-produced initial-noise tensor and raw planner "
            "words into the pinned upstream capture. A numeric seed is never "
            "presumed to match PyTorch or MLX RNG."
        ),
    }
    require(
        same_json_value(load_json(path), expected),
        path,
        "PRNG contract differs from ace-seed-v1",
    )


def validate_prng_vectors() -> None:
    path = GOLDEN / "prng/ace-seed-v1-vectors.json"
    vectors = require_exact_keys(
        load_json(path),
        {
            "schemaVersion",
            "contractId",
            "gaussianMapping",
            "categoricalMapping",
            "philox",
            "gaussian",
            "streamGaussian",
            "categorical",
            "plannerStream",
        },
        path,
        "PRNG vectors",
    )
    require(vectors["schemaVersion"] == 1, path, "schemaVersion must be 1")
    require(vectors["contractId"] == "ace-seed-v1", path, "wrong PRNG contract ID")
    require(
        vectors["gaussianMapping"] == "probit-acklam-binary64-f32-v1",
        path,
        "wrong Gaussian mapping ID",
    )
    require(
        vectors["categoricalMapping"] == "u32-midpoint-binary64-cdf-v1",
        path,
        "wrong categorical mapping ID",
    )

    collection_contracts = {
        "philox": (
            {"id", "seed", "stream", "blockIndexHex", "outputWords"},
            {
                "zero-seed-diffusion-block-zero",
                "zero-seed-planner-block-zero",
                "mixed-seed-diffusion-block-one",
                "max-seed-planner-max-block",
                "fixture-seed-diffusion-high-counter",
            },
        ),
        "gaussian": (
            {"id", "word", "outputF32Bits"},
            {
                "minimum-word",
                "next-minimum-word",
                "lower-tail-last-word",
                "central-first-word",
                "center-negative",
                "center-positive",
                "central-last-word",
                "upper-tail-first-word",
                "next-maximum-word",
                "maximum-word",
            },
        ),
        "streamGaussian": (
            {"id", "seed", "wordIndex", "word", "outputF32Bits"},
            {
                "zero-seed-first-lane",
                "zero-seed-fourth-lane",
                "zero-seed-next-block-second-lane",
                "zero-seed-word-index-32-bit-boundary-minus-one",
                "zero-seed-word-index-32-bit-boundary",
                "fixture-seed-first-lane",
                "fixture-seed-third-lane",
                "fixture-seed-high-word-index",
            },
        ),
        "categorical": (
            {"id", "weights", "word", "token"},
            {
                "two-equal-low",
                "two-equal-lower-boundary",
                "two-equal-upper-boundary",
                "two-equal-high",
                "zero-weights-low",
                "zero-weights-high",
                "skew-low",
                "skew-first-boundary-below",
                "skew-first-boundary-above",
                "skew-second-boundary-below",
                "skew-high",
                "tiny-edge-mass",
            },
        ),
        "plannerStream": (
            {"id", "seed", "drawIndex", "word", "weights", "token"},
            {
                "planner-fixture-draw-zero",
                "planner-fixture-draw-three",
                "planner-fixture-next-block",
                "planner-fixture-draw-seven",
                "planner-fixture-draw-nine",
            },
        ),
    }
    for collection_name, (entry_keys, expected_ids) in collection_contracts.items():
        collection = vectors[collection_name]
        require(
            isinstance(collection, list),
            path,
            f"{collection_name} vectors must be an array",
        )
        actual_ids = set()
        for index, value in enumerate(collection):
            entry = require_exact_keys(
                value, entry_keys, path, f"{collection_name}[{index}]"
            )
            vector_id = entry["id"]
            require(
                isinstance(vector_id, str) and bool(vector_id),
                path,
                f"{collection_name}[{index}].id must be non-empty",
            )
            require(
                vector_id not in actual_ids,
                path,
                f"duplicate {collection_name} vector ID {vector_id!r}",
            )
            actual_ids.add(vector_id)
        require(
            actual_ids == expected_ids,
            path,
            f"{collection_name} vector IDs differ from the pinned set",
        )

    reference_path = ROOT / "scripts/prng_reference.py"
    spec = importlib.util.spec_from_file_location("ace_prng_reference", reference_path)
    require(
        spec is not None and spec.loader is not None,
        reference_path,
        "could not load the independent PRNG oracle",
    )
    if spec is None or spec.loader is None:
        raise ValueError("could not load the independent PRNG oracle")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.verify_vectors(path)


def validate_fixture_noise_hashes() -> None:
    """Recompute every fixture's full realized f32 noise identity."""

    reference_path = ROOT / "scripts/prng_reference.py"
    spec = importlib.util.spec_from_file_location("ace_prng_noise_reference", reference_path)
    require(
        spec is not None and spec.loader is not None,
        reference_path,
        "could not load the independent PRNG oracle for fixture noise",
    )
    if spec is None or spec.loader is None:
        raise ValueError("could not load the independent PRNG oracle")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.verify_vectors(GOLDEN / "prng/ace-seed-v1-vectors.json")

    for fixture_id, profile in FIXTURE_PROFILES.items():
        count = expected_geometry(profile["duration"])["latentFrames"] * 64
        digest = hashlib.sha256()
        for word_index in range(count):
            word = reference.random_word(
                profile["seed"], "diffusion-noise", word_index
            )
            digest.update(struct.pack("<f", reference.gaussian_from_word(word)))
        require(
            digest.hexdigest() == profile["initialNoiseSha256"],
            FIXTURES / f"{fixture_id}.json",
            "pinned initial-noise SHA-256 differs from ace-seed-v1 recomputation",
        )


def validate_tap_contract() -> None:
    path = GOLDEN / "taps.json"
    contract = require_exact_keys(
        load_json(path),
        {"schemaVersion", "layout", "symbols", "captureRules", "taps"},
        path,
        "tap contract",
    )
    require(contract["schemaVersion"] == 1, path, "schemaVersion must be 1")
    require(
        contract["layout"] == "contiguous-little-endian",
        path,
        "tap layout must be contiguous little-endian",
    )
    taps = contract["taps"]
    require(isinstance(taps, list), path, "taps must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for index, tap_value in enumerate(taps):
        tap = require_exact_keys(
            tap_value,
            {"id", "stage", "required", "dtype", "shape", "capture"},
            path,
            f"taps[{index}]",
        )
        tap_id = tap["id"]
        require(
            isinstance(tap_id, str) and bool(tap_id),
            path,
            f"taps[{index}].id must be non-empty",
        )
        require(tap_id not in by_id, path, f"duplicate tap ID {tap_id!r}")
        by_id[tap_id] = tap

    required_taps = {
        "planner.sample.words": {
            "id": "planner.sample.words",
            "stage": "planner",
            "required": "planner",
            "dtype": "uint32",
            "shape": ["R"],
            "capture": "full",
        },
        "semantic.fsq.output": {
            "id": "semantic.fsq.output",
            "stage": "semantic",
            "required": "planner",
            "dtype": "bfloat16",
            "shape": ["B", "C5", 2048],
            "capture": "slice+stats",
        },
        "semantic.hints.25hz": {
            "id": "semantic.hints.25hz",
            "stage": "semantic",
            "required": "planner",
            "dtype": "bfloat16",
            "shape": ["B", "T", 64],
            "capture": "slice+stats",
        },
        "diffusion.initial-noise": {
            "id": "diffusion.initial-noise",
            "stage": "dit",
            "required": "all",
            "dtype": "float32",
            "shape": ["B", "T", 64],
            "capture": "full",
        },
    }
    for tap_id, expected in required_taps.items():
        require(tap_id in by_id, path, f"missing required tap {tap_id!r}")
        require(
            same_json_value(by_id[tap_id], expected),
            path,
            f"tap {tap_id!r} differs from its pinned shape/type",
        )


def validate_capture_status(fixtures: dict[str, str]) -> None:
    path = GOLDEN / "expected/status.json"
    status = require_exact_keys(
        load_json(path), {"schemaVersion", "captures"}, path, "capture status"
    )
    require(status["schemaVersion"] == 1, path, "schemaVersion must be 1")
    captures = require_exact_keys(
        status["captures"], fixtures, path, "capture status.captures"
    )
    for fixture_id, capture in captures.items():
        capture = require_exact_keys(
            capture, {"status", "reason"}, path, f"captures.{fixture_id}"
        )
        require(capture["status"] == "pending", path, "v1 captures must remain pending")
        require(
            isinstance(capture["reason"], str) and bool(capture["reason"]),
            path,
            f"captures.{fixture_id}.reason must be non-empty",
        )


def all_manifest_paths() -> list[Path]:
    return [
        *HASHED_FILES,
        *(Path("fixtures") / path.name for path in sorted(FIXTURES.glob("*.json"))),
    ]


def build_manifest(fixtures: dict[str, str]) -> dict[str, Any]:
    entries = []
    for relative in all_manifest_paths():
        payload = (GOLDEN / relative).read_bytes()
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": sha256(payload),
            }
        )
    identity = sha256(canonical_json(entries))
    return {
        "schemaVersion": 1,
        "manifestId": f"ace-golden-contract-{identity}",
        "fixtures": fixtures,
        "files": entries,
    }


def write_fixture_hash(path: Path, contract_hash: str) -> None:
    fixture = load_json(path)
    fixture["contractSha256"] = contract_hash
    path.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def preflight_json() -> None:
    """Reject duplicate keys/non-finite values before --update writes anything."""
    paths = [
        GOLDEN / "fixture.schema.json",
        GOLDEN / "taps.json",
        GOLDEN / "prng/ace-seed-v1.json",
        GOLDEN / "prng/ace-seed-v1-vectors.json",
        GOLDEN / "expected/status.json",
        *sorted(FIXTURES.glob("*.json")),
    ]
    if MANIFEST.exists():
        paths.append(MANIFEST)
    for path in paths:
        load_json(path)


def validate_repository(update: bool = False) -> dict[str, Any]:
    preflight_json()
    fixture_paths = sorted(FIXTURES.glob("*.json"))
    validated = [(path, *validate_fixture(path, update)) for path in fixture_paths]
    fixtures = {fixture_id: digest for _, fixture_id, digest in validated}
    require(
        set(fixtures) == set(FIXTURE_PROFILES),
        FIXTURES,
        "fixture IDs must equal the four-fixture v1 matrix",
    )
    validate_prng_contract()
    validate_prng_vectors()
    validate_fixture_noise_hashes()
    validate_tap_contract()
    validate_capture_status(fixtures)
    if update:
        for path, _, digest in validated:
            write_fixture_hash(path, digest)
    manifest = build_manifest(fixtures)
    if update:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        require(
            same_json_value(load_json(MANIFEST), manifest),
            MANIFEST,
            "manifest is stale; run with --update",
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update", action="store_true", help="refresh contract hashes and MANIFEST.json"
    )
    args = parser.parse_args()
    try:
        manifest = validate_repository(args.update)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"golden validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"validated {len(FIXTURE_PROFILES)} golden fixture contracts "
        f"({manifest['manifestId']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
