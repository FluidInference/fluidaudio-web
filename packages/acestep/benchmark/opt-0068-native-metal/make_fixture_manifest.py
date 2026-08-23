#!/usr/bin/env python3
"""Deterministically assemble the OPT-0068 fixture manifest from captured files."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


ACTIVATIONS = {
    "self-modulated": 2_048,
    "self-merged-attention": 2_048,
    "cross-normalized": 2_048,
    "cross-merged-attention": 2_048,
    "mlp-modulated": 2_048,
    "mlp-gated-activation": 6_144,
}
CASES = (
    ("self-query", "self-modulated", "ace.decoder.layers.0.self_attn.q_proj.weight", 2_048, 2_048),
    ("self-key", "self-modulated", "ace.decoder.layers.0.self_attn.k_proj.weight", 2_048, 1_024),
    ("self-value", "self-modulated", "ace.decoder.layers.0.self_attn.v_proj.weight", 2_048, 1_024),
    ("self-output", "self-merged-attention", "ace.decoder.layers.0.self_attn.o_proj.weight", 2_048, 2_048),
    ("cross-query", "cross-normalized", "ace.decoder.layers.0.cross_attn.q_proj.weight", 2_048, 2_048),
    ("cross-output", "cross-merged-attention", "ace.decoder.layers.0.cross_attn.o_proj.weight", 2_048, 2_048),
    ("mlp-gate", "mlp-modulated", "ace.decoder.layers.0.mlp.gate_proj.weight", 2_048, 6_144),
    ("mlp-up", "mlp-modulated", "ace.decoder.layers.0.mlp.up_proj.weight", 2_048, 6_144),
    ("mlp-down", "mlp-gated-activation", "ace.decoder.layers.0.mlp.down_proj.weight", 6_144, 2_048),
)
MAIN_MANIFEST_SHA256 = "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6"
REQUEST_SHA256 = "031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f"
EVALUATION_0_SHA256 = "d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286"


def inspect_f32le(path: Path, expected_elements: int) -> dict[str, object]:
    digest = hashlib.sha256()
    finite_count = 0
    nonzero_count = 0
    minimum = math.inf
    maximum = -math.inf
    observed_elements = 0
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1_024 * 1_024):
            if len(chunk) % 4:
                raise SystemExit(f"{path} is not a complete float32-le stream")
            digest.update(chunk)
            values = array("f")
            values.frombytes(chunk)
            if sys.byteorder != "little":
                values.byteswap()
            observed_elements += len(values)
            finite_count += sum(math.isfinite(value) for value in values)
            nonzero_count += sum(value != 0.0 for value in values)
            if values:
                minimum = min(minimum, min(values))
                maximum = max(maximum, max(values))
    if observed_elements != expected_elements:
        raise SystemExit(
            f"{path} contains {observed_elements} elements; expected {expected_elements}"
        )
    if finite_count != expected_elements or nonzero_count == 0:
        raise SystemExit(
            f"{path} must contain {expected_elements} finite values and at least one nonzero"
        )
    with path.open("rb") as handle:
        head = handle.read(32)
        handle.seek(-32, 2)
        tail = handle.read(32)
    return {
        "sha256": digest.hexdigest(),
        "finiteCount": finite_count,
        "nonzeroCount": nonzero_count,
        "minimum": minimum,
        "maximum": maximum,
        "headF32Bits": [f"{bits:08x}" for bits in struct.unpack("<8I", head)],
        "tailF32Bits": [f"{bits:08x}" for bits in struct.unpack("<8I", tail)],
    }


def hashed_file(
    capture: Path,
    filename: str,
    identity: str,
    shape: list[int],
    expected_sha256: str | None = None,
) -> dict[str, object]:
    path = capture / filename
    elements = math.prod(shape)
    expected = elements * 4
    if not path.is_file() or path.stat().st_size != expected:
        raise SystemExit(f"missing or wrong-sized {path}; expected {expected} bytes")
    evidence = inspect_f32le(path, elements)
    if expected_sha256 is not None and evidence["sha256"] != expected_sha256:
        raise SystemExit(
            f"{path} SHA-256 {evidence['sha256']} does not match {expected_sha256}"
        )
    return {
        "id": identity,
        "path": filename,
        "dtype": "float32-le",
        "shape": shape,
        "elementCount": elements,
        "byteLength": expected,
        **evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--capture-commit", required=True)
    parser.add_argument("--capture-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.capture_commit) != 40 or any(c not in "0123456789abcdef" for c in args.capture_commit):
        raise SystemExit("capture commit must be 40 lowercase hex characters")
    if len(args.capture_source_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.capture_source_sha256
    ):
        raise SystemExit("capture source SHA-256 must be 64 lowercase hex characters")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    activations = [
        hashed_file(
            args.capture_directory,
            f"activation-{identity}.f32le",
            identity,
            [2_250, width],
        )
        for identity, width in ACTIVATIONS.items()
    ]
    outputs = [
        hashed_file(
            args.capture_directory,
            f"output-{identity}.f32le",
            identity,
            [2_250, columns],
        )
        for identity, _, _, _, columns in CASES
    ]
    evaluation_output = hashed_file(
        args.capture_directory,
        "evaluation-0-result.f32le",
        "evaluation-0-result",
        [288_000],
        EVALUATION_0_SHA256,
    )
    cases = [
        {
            "id": identity,
            "activation": activation,
            "acceptedWebGPUOutput": identity,
            "weightTensor": tensor,
            "rows": 2_250,
            "inner": inner,
            "columns": columns,
        }
        for identity, activation, tensor, inner, columns in CASES
    ]
    manifest = {
        "schema": "ace-opt-0068-m2250-native-fixture-v1",
        "authority": {
            "experimentId": "OPT-0068",
            "aceSourceCommit": "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
            "aceSnapshot": "19671f406d603126926c1b7e2adc169acbcade22",
            "mainManifestSha256": MAIN_MANIFEST_SHA256,
            "goldenFixtureManifestSha256": "cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb",
            "packageManifestSha256": "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f",
            "packageConverterRevision": 7,
            "requestId": "ace-turbo-v1-correctness",
            "requestSha256": REQUEST_SHA256,
            "requestByteLength": 366,
            "plannerEnabled": False,
            "durationSeconds": 180,
            "sampler": "shift-3-euler-8-evaluations",
            "dcwMode": "double-haar",
            "lowFrequencyStrength": 0.05,
            "highFrequencyStrength": 0.02,
            "evaluation": 0,
            "layer": 0,
            "conditionTokens": 98,
            "expectedEvaluation0Sha256": EVALUATION_0_SHA256,
            "captureCommit": args.capture_commit,
            "captureSourceSha256": args.capture_source_sha256,
        },
        "evaluationOutput": evaluation_output,
        "activations": activations,
        "acceptedWebGPUOutputs": outputs,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
