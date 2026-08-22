#!/usr/bin/env python3
"""Optional MLX corroboration for OPT-0068.

This runner is deliberately diagnostic-only: MLX 0.32 materializes an FP16
matmul result before the explicit FP32 cast, and its Python API exposes fenced
wall time but not the raw GPU timestamps required by the primary gate.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_SHA256 = "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f"
ACE_COMMIT = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
ACE_SNAPSHOT = "19671f406d603126926c1b7e2adc169acbcade22"
GOLDEN_SHA256 = "cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb"
MAIN_MANIFEST_SHA256 = "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6"
REQUEST_SHA256 = "031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f"
EVALUATION_0_SHA256 = "d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286"
CONSENT = "I_UNDERSTAND_OPT_0068_BENCHMARK_ONLY"
ROWS = 2_250

CASES = (
    ("self-query", "self-modulated", "self-query", "ace.decoder.layers.0.self_attn.q_proj.weight", 2_048, 2_048),
    ("self-key", "self-modulated", "self-key", "ace.decoder.layers.0.self_attn.k_proj.weight", 2_048, 1_024),
    ("self-value", "self-modulated", "self-value", "ace.decoder.layers.0.self_attn.v_proj.weight", 2_048, 1_024),
    ("self-output", "self-merged-attention", "self-output", "ace.decoder.layers.0.self_attn.o_proj.weight", 2_048, 2_048),
    ("cross-query", "cross-normalized", "cross-query", "ace.decoder.layers.0.cross_attn.q_proj.weight", 2_048, 2_048),
    ("cross-output", "cross-merged-attention", "cross-output", "ace.decoder.layers.0.cross_attn.o_proj.weight", 2_048, 2_048),
    ("mlp-gate", "mlp-modulated", "mlp-gate", "ace.decoder.layers.0.mlp.gate_proj.weight", 2_048, 6_144),
    ("mlp-up", "mlp-modulated", "mlp-up", "ace.decoder.layers.0.mlp.up_proj.weight", 2_048, 6_144),
    ("mlp-down", "mlp-gated-activation", "mlp-down", "ace.decoder.layers.0.mlp.down_proj.weight", 6_144, 2_048),
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1_024 * 1_024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_f32le(path: Path) -> dict[str, Any]:
    finite_count = 0
    nonzero_count = 0
    minimum = math.inf
    maximum = -math.inf
    element_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1_024 * 1_024):
            require(len(chunk) % 4 == 0, f"incomplete float32-le stream: {path}")
            values = array("f")
            values.frombytes(chunk)
            if sys.byteorder != "little":
                values.byteswap()
            element_count += len(values)
            finite_count += sum(math.isfinite(value) for value in values)
            nonzero_count += sum(value != 0.0 for value in values)
            if values:
                minimum = min(minimum, min(values))
                maximum = max(maximum, max(values))
    with path.open("rb") as handle:
        head = handle.read(32)
        handle.seek(-32, 2)
        tail = handle.read(32)
    return {
        "elementCount": element_count,
        "finiteCount": finite_count,
        "nonzeroCount": nonzero_count,
        "minimum": minimum,
        "maximum": maximum,
        "headF32Bits": [f"{bits:08x}" for bits in struct.unpack("<8I", head)],
        "tailF32Bits": [f"{bits:08x}" for bits in struct.unpack("<8I", tail)],
    }


def validate_captured_file(
    entry: dict[str, Any],
    root: Path,
    identity: str,
    filename: str,
    shape: list[int],
    expected_sha256: str | None = None,
) -> Path:
    elements = math.prod(shape)
    require(entry["id"] == identity and entry["path"] == filename, f"{identity} ID/path changed")
    require(entry["dtype"] == "float32-le" and entry["shape"] == shape, f"{identity} metadata changed")
    require(entry["elementCount"] == elements, f"{identity} element count changed")
    require(entry["byteLength"] == 4 * elements, f"{identity} byte length changed")
    path = root / filename
    require(path.is_file() and path.stat().st_size == entry["byteLength"], f"missing/short {path}")
    observed_sha = sha256_file(path)
    require(observed_sha == entry["sha256"], f"SHA changed for {path}")
    if expected_sha256 is not None:
        require(observed_sha == expected_sha256, f"frozen SHA changed for {path}")
    evidence = inspect_f32le(path)
    require(evidence["finiteCount"] == elements and evidence["nonzeroCount"] > 0, f"{path} is non-finite/all-zero")
    require(
        all(entry.get(key) == value for key, value in evidence.items()),
        f"float evidence changed for {path}",
    )
    return path


def authenticate(package_dir: Path, fixture_path: Path) -> list[dict[str, Any]]:
    manifest_path = package_dir / "manifest.json"
    require(manifest_path.is_file(), f"missing revision-7 manifest: {manifest_path}")
    require(sha256_file(manifest_path) == MANIFEST_SHA256, "revision-7 manifest SHA changed")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["format"] == "ace-step-webgpu-v1", "package format changed")
    require(manifest["profile"] == "fp16-dit-dense-experimental", "package profile changed")
    provenance = manifest["provenance"]
    require(provenance["converterRevision"] == 7, "converter revision changed")
    require(provenance["referenceCommit"] == ACE_COMMIT, "ACE source changed")
    require(provenance["aceSnapshot"] == ACE_SNAPSHOT, "ACE snapshot changed")
    require(fixture_path.is_file(), f"missing actual activation fixture: {fixture_path}")
    fixture = json.loads(fixture_path.read_text())
    require(fixture["schema"] == "ace-opt-0068-m2250-native-fixture-v1", "fixture schema changed")
    authority = fixture["authority"]
    expected_authority = {
        "experimentId": "OPT-0068",
        "aceSourceCommit": ACE_COMMIT,
        "aceSnapshot": ACE_SNAPSHOT,
        "mainManifestSha256": MAIN_MANIFEST_SHA256,
        "goldenFixtureManifestSha256": GOLDEN_SHA256,
        "packageManifestSha256": MANIFEST_SHA256,
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
    }
    require(all(authority.get(k) == v for k, v in expected_authority.items()), "fixture authority changed")
    capture_commit = authority.get("captureCommit", "")
    capture_source_sha256 = authority.get("captureSourceSha256", "")
    require(
        len(capture_commit) == 40 and all(c in "0123456789abcdef" for c in capture_commit),
        "capture commit must be 40 lowercase hex characters",
    )
    require(
        len(capture_source_sha256) == 64
        and all(c in "0123456789abcdef" for c in capture_source_sha256),
        "capture source hash must be 64 lowercase hex characters",
    )
    activations = {entry["id"]: entry for entry in fixture["activations"]}
    outputs = {entry["id"]: entry for entry in fixture["acceptedWebGPUOutputs"]}
    require(len(activations) == 6 and len(outputs) == 9, "fixture IDs are incomplete/duplicated")
    declared = {entry["id"]: entry for entry in fixture["cases"]}
    require(len(declared) == 9, "case IDs are incomplete/duplicated")
    root = fixture_path.parent
    verified_files: set[Path] = set()
    verified_shards: set[Path] = set()
    file_entries = {entry["name"]: entry for entry in manifest["files"]}
    resolved: list[dict[str, Any]] = []
    evaluation_output = validate_captured_file(
        fixture["evaluationOutput"],
        root,
        "evaluation-0-result",
        "evaluation-0-result.f32le",
        [288_000],
        EVALUATION_0_SHA256,
    )
    verified_files.add(evaluation_output)

    def validate_once(
        entry: dict[str, Any], identity: str, filename: str, shape: list[int]
    ) -> Path:
        path = root / filename
        if path not in verified_files:
            validate_captured_file(entry, root, identity, filename, shape)
            verified_files.add(path)
        return path

    for case_id, activation_id, output_id, tensor_name, inner, columns in CASES:
        case = declared.get(case_id)
        require(case is not None, f"fixture omits {case_id}")
        require(
            case == {
                "id": case_id,
                "activation": activation_id,
                "acceptedWebGPUOutput": output_id,
                "weightTensor": tensor_name,
                "rows": ROWS,
                "inner": inner,
                "columns": columns,
            },
            f"case {case_id} changed",
        )
        paths = [
            validate_once(
                activations[activation_id], activation_id,
                f"activation-{activation_id}.f32le", [ROWS, inner]
            ),
            validate_once(
                outputs[output_id], output_id,
                f"output-{output_id}.f32le", [ROWS, columns]
            ),
        ]
        tensor = manifest["tensors"].get(tensor_name)
        require(tensor is not None, f"package omits {tensor_name}")
        require(tensor["logicalShape"] == [columns, inner], f"{tensor_name} shape changed")
        require(tensor["dtype"] == "float16", f"{tensor_name} is not FP16")
        require(tensor["layout"] == "dit-gemm-n256-k32-tile-major-v1", f"{tensor_name} layout changed")
        require(tensor["byteLength"] == 2 * inner * columns, f"{tensor_name} length changed")
        shard = package_dir / tensor["shard"]
        if shard not in verified_shards:
            file_entry = file_entries[tensor["shard"]]
            require(shard.is_file() and shard.stat().st_size == file_entry["byteLength"], f"missing/short {shard}")
            require(sha256_file(shard) == file_entry["sha256"], f"SHA changed for {shard}")
            verified_shards.add(shard)
        resolved.append(
            dict(
                id=case_id, activation=paths[0], expected=paths[1],
                tensor=tensor, shard=shard, rows=ROWS, inner=inner, columns=columns,
            )
        )
    require(len(verified_files) == 16, "captured activation/output paths must be distinct")
    return resolved


@dataclass
class ThermalMonitor:
    path: Path

    def __post_init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        require(not self.path.exists(), f"thermal trace already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, name="opt0068-mlx-thermal", daemon=True)
        self.thread.start()

    def gate(self) -> list[dict[str, Any]]:
        deadline = time.monotonic() + 900
        nominal_start: int | None = None
        seen = 0
        while time.monotonic() < deadline:
            if self.error:
                raise self.error
            snapshot = list(self.observations)
            if len(snapshot) > seen:
                latest = snapshot[-1]
                if len(snapshot) > 1 and latest["atEpochMilliseconds"] - snapshot[-2]["atEpochMilliseconds"] > 1_800:
                    nominal_start = None
                if latest["level"] == 0:
                    nominal_start = nominal_start or latest["atEpochMilliseconds"]
                    if latest["atEpochMilliseconds"] - nominal_start >= 30_000:
                        return [item for item in snapshot if item["atEpochMilliseconds"] >= nominal_start]
                else:
                    nominal_start = None
                seen = len(snapshot)
            time.sleep(0.05)
        raise ContractError("30-second continuous thermal level-0 gate timed out")

    def await_after(self, epoch_ms: int) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.error:
                raise self.error
            if self.observations and self.observations[-1]["atEpochMilliseconds"] > epoch_ms:
                return
            time.sleep(0.05)
        raise ContractError("thermal trace did not observe MLX cleanup")

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.error:
            raise self.error
        require(bool(self.observations), "thermal trace is empty")
        gaps = [b["atEpochMilliseconds"] - a["atEpochMilliseconds"] for a, b in zip(self.observations, self.observations[1:])]
        return {
            "source": "macOS notifyutil thermal pressure",
            "command": "/usr/bin/notifyutil -g com.apple.system.thermalpressurelevel",
            "rawTraceSha256": sha256_file(self.path),
            "observationCount": len(self.observations),
            "maximumGapMilliseconds": max(gaps, default=0),
            "nonNominalCount": sum(item["level"] != 0 for item in self.observations),
            "observations": self.observations,
        }

    def _loop(self) -> None:
        try:
            with self.path.open("x", buffering=1) as handle:
                while not self.stop_event.is_set():
                    result = subprocess.run(
                        ["/usr/bin/notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
                        check=True,
                        text=True,
                        capture_output=True,
                    )
                    raw = result.stdout.strip()
                    observation = {
                        "atEpochMilliseconds": round(time.time() * 1_000),
                        "level": int(raw.split()[-1]),
                        "rawValue": raw,
                    }
                    handle.write(json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    self.observations.append(observation)
                    self.stop_event.wait(0.9)
        except BaseException as error:  # preserve exact monitor failure
            self.error = error


def metrics(np: Any, candidate: Any, reference: Any) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    require(candidate.size == reference.size and candidate.size > 0, "comparison size changed")
    candidate_finite = np.isfinite(candidate)
    reference_finite = np.isfinite(reference)
    both = candidate_finite & reference_finite
    error = candidate[both].astype(np.float64) - reference[both].astype(np.float64)
    reference64 = reference[both].astype(np.float64)
    rms_error = float(np.sqrt(np.mean(error * error))) if error.size else float("inf")
    rms_reference = float(np.sqrt(np.mean(reference64 * reference64))) if reference64.size else 0.0
    signed_zero = int(np.count_nonzero((candidate == 0) & (reference == 0) & (np.signbit(candidate) != np.signbit(reference))))
    return {
        "compared": int(candidate.size),
        "finiteClassMismatchCount": int(np.count_nonzero(candidate_finite != reference_finite)),
        "nonFiniteClassMismatchCount": int(np.count_nonzero((~both) & ~(np.isnan(candidate) & np.isnan(reference)) & (candidate != reference))),
        "signedZeroMismatchCount": signed_zero,
        "zeroClassMismatchCount": int(np.count_nonzero((candidate == 0) != (reference == 0))),
        "maximumAbsoluteError": float(np.max(np.abs(error), initial=0)),
        "nrmse": rms_error / max(rms_reference, 1e-30),
        "snrDB": 20 * __import__("math").log10(max(rms_reference, 1e-30) / max(rms_error, 1e-30)),
        "pearson": float(np.corrcoef(candidate[both], reference[both])[0, 1]) if np.count_nonzero(both) > 1 else 1.0,
    }


def unpack_weight(np: Any, case: dict[str, Any]) -> Any:
    inner, columns = case["inner"], case["columns"]
    with case["shard"].open("rb") as handle:
        handle.seek(case["tensor"]["byteOffset"])
        physical = np.fromfile(handle, dtype="<f2", count=inner * columns)
    return physical.reshape(columns // 256, inner // 32, 32, 256).transpose(0, 3, 1, 2).reshape(columns, inner).T.copy()


def adversarial_fixtures(np: Any) -> list[dict[str, Any]]:
    specs = (
        ("signed-zero", 33, 32, 256),
        ("cancellation", 33, 32, 256),
        ("finite-range", 33, 32, 256),
        ("long-k-cancellation", 3, 6_144, 256),
        ("benign-tail", 33, 2_048, 256),
    )
    result = []
    for kind, rows, inner, columns in specs:
        activation = np.empty((rows, inner), dtype=np.float16)
        weight = np.empty((inner, columns), dtype=np.float16)
        for row in range(rows):
            for k in range(inner):
                if kind == "signed-zero":
                    values = (-0.0, +0.0, -1, 1, -(2**-24), 2**-24)
                    value = values[(k + row) % len(values)]
                elif kind == "cancellation":
                    values = (2_048, 1, -2_048, 0.5, 1_024, -0.5, -1_024, 2**-10)
                    value = values[(k + row * 3) % len(values)]
                elif kind == "finite-range":
                    value = (65_504 if row % 2 == 0 else -65_504) if k == 0 else 2**-14 if k == 1 else 2**-24 if k == 2 else +0.0 if k % 2 == 0 else -0.0
                elif kind == "long-k-cancellation":
                    values = (1, 2**-10, -1, 2**-11, 0.5, -(2**-12), -0.5, 2**-13)
                    value = values[(k + row) % len(values)]
                else:
                    value = (((row * 17 + k * 13 + 3) % 31) - 15) / 64
                activation[row, k] = value
        for k in range(inner):
            for column in range(columns):
                if kind == "signed-zero":
                    values = (1, -1, -0.0, +0.0, 0.5, -0.5)
                    value = values[(k + column) % len(values)]
                elif kind == "cancellation":
                    value = 1 if column % 4 < 2 else -1
                elif kind == "finite-range":
                    value = (2, -2, 1, 0.5)[column % 4] if k == 0 else (1 if column % 2 == 0 else -1) if k in (1, 2) else 0
                elif kind == "long-k-cancellation":
                    value = 1 if column % 4 == 0 else -1 if column % 4 == 1 else 0.5
                else:
                    value = (((column * 13 + k * 7 + 7) % 29) - 14) / 64
                weight[k, column] = value
        cpu = np.zeros((rows, columns), dtype=np.float32)
        for k in range(inner):
            product = np.multiply(
                activation[:, k, None].astype(np.float32),
                weight[k, :][None, :].astype(np.float32),
                dtype=np.float32,
            )
            cpu = np.add(cpu, product, dtype=np.float32)
        result.append(
            dict(
                id=f"adversarial-{kind}",
                kind=kind,
                rows=rows,
                inner=inner,
                columns=columns,
                activation=activation,
                weight=weight,
                expected=cpu,
            )
        )
    return result


def execute(args: argparse.Namespace, resolved: list[dict[str, Any]]) -> None:
    require(args.consent == CONSENT, f"MLX execution requires --execute-native-gpu {CONSENT}")
    require(args.output and args.thermal_trace and args.harness_commit, "MLX execution requires output/trace/commit")
    require(len(args.harness_commit) == 40 and all(c in "0123456789abcdefABCDEF" for c in args.harness_commit), "invalid commit")
    require(platform.machine() == "arm64", "MLX gate requires Apple arm64")
    require(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.model"], text=True).strip() == "Mac15,12", "wrong Mac")
    require(subprocess.check_output(["/usr/bin/sw_vers", "-productVersion"], text=True).strip() == "26.5.2", "wrong macOS")
    import numpy as np
    import mlx.core as mx

    prepared: list[dict[str, Any]] = []
    for case in resolved:
        activation = np.fromfile(case["activation"], dtype="<f4").reshape(ROWS, case["inner"]).astype(np.float16)
        weight = unpack_weight(np, case)
        expected = np.fromfile(case["expected"], dtype="<f4").reshape(ROWS, case["columns"])
        a, b = mx.array(activation), mx.array(weight)
        mx.eval(a, b)
        mx.synchronize()
        prepared.append({**case, "a": a, "b": b, "expectedArray": expected})

    def run(case: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        rows = case["rows"]
        started = time.perf_counter_ns()
        output16 = mx.matmul(case["a"], case["b"])
        output32 = output16.astype(mx.float32)
        mx.eval(output32)
        mx.synchronize()
        materialized = np.array(output32, copy=True)
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        digest = hashlib.sha256(materialized.astype("<f4", copy=False).tobytes()).hexdigest()
        return {
            "caseId": case["id"],
            "wallMilliseconds": wall_ms,
            "flops": 2 * rows * case["inner"] * case["columns"],
            "tflops": 2 * rows * case["inner"] * case["columns"] / max(wall_ms, 1e-9) / 1e9,
            "matmulResultDtype": str(output16.dtype),
            "materializedDtype": str(output32.dtype),
            "outputSha256": digest,
        }, materialized

    def profile(case: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter_ns()
        output16 = mx.matmul(case["a"], case["b"])
        mx.eval(output16)
        mx.synchronize()
        contraction_ms = (time.perf_counter_ns() - started) / 1e6
        started = time.perf_counter_ns()
        output32 = output16.astype(mx.float32)
        mx.eval(output32)
        mx.synchronize()
        epilogue_ms = (time.perf_counter_ns() - started) / 1e6
        started = time.perf_counter_ns()
        _ = np.array(output32, copy=True)
        materialization_ms = (time.perf_counter_ns() - started) / 1e6
        return {
            "caseId": case["id"],
            "contractionFencedWallMilliseconds": contraction_ms,
            "biasMode": "absent-by-ACE-repeated-layer-contract",
            "fp16ToFp32EpilogueFencedWallMilliseconds": epilogue_ms,
            "materializationWallMilliseconds": materialization_ms,
            "rawGpuTimestampsAvailable": False,
        }

    correctness = []
    mandatory = True
    for case in prepared:
        first, first_array = run(case)
        second, second_array = run(case)
        comparison = metrics(np, second_array, case["expectedArray"])
        deterministic = bool(np.array_equal(first_array.view(np.uint32), second_array.view(np.uint32)))
        mandatory &= (
            deterministic
            and comparison["finiteClassMismatchCount"] == 0
            and comparison["nonFiniteClassMismatchCount"] == 0
            and comparison["signedZeroMismatchCount"] == 0
            and comparison["zeroClassMismatchCount"] == 0
        )
        correctness.append({"id": case["id"], "first": first, "second": second, "metrics": comparison, "deterministic": deterministic})
    adversarial = []
    for fixture in adversarial_fixtures(np):
        case = {
            **fixture,
            "a": mx.array(fixture["activation"]),
            "b": mx.array(fixture["weight"]),
        }
        mx.eval(case["a"], case["b"])
        first, first_array = run(case)
        second, second_array = run(case)
        comparison = metrics(np, second_array, fixture["expected"])
        deterministic = bool(np.array_equal(first_array.view(np.uint32), second_array.view(np.uint32)))
        passed = (
            deterministic
            and comparison["finiteClassMismatchCount"] == 0
            and comparison["nonFiniteClassMismatchCount"] == 0
            and comparison["signedZeroMismatchCount"] == 0
            and comparison["zeroClassMismatchCount"] == 0
            and (fixture["kind"] != "long-k-cancellation" or comparison["maximumAbsoluteError"] < 0.01)
        )
        mandatory &= passed
        adversarial.append(
            {"id": fixture["kind"], "first": first, "second": second, "metrics": comparison, "deterministic": deterministic}
        )
    require(mandatory, "MLX numerical/class/determinism screen failed; timing refused")
    for _ in range(args.warmups):
        for case in prepared:
            run(case)

    monitor = ThermalMonitor(Path(args.thermal_trace))
    monitor.start()
    monitor.gate()
    samples: list[dict[str, Any]] = []
    stage_profiles: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] | None = None
    if args.mode == "measure":
        for repetition in range(args.samples):
            order = prepared if repetition % 2 == 0 else list(reversed(prepared))
            for case in order:
                sample, _ = run(case)
                sample["repetition"] = repetition
                samples.append(sample)
        stage_profiles = [profile(case) for case in prepared]
    else:
        intervals = []
        overall = time.perf_counter()
        interval_start = overall
        interval_flops = 0
        repetition = 0
        while time.perf_counter() - overall < 60:
            for case in prepared:
                sample, _ = run(case)
                sample["repetition"] = repetition
                samples.append(sample)
                interval_flops += sample["flops"]
            repetition += 1
            now = time.perf_counter()
            if now - interval_start >= 5:
                intervals.append({
                    "index": len(intervals),
                    "seconds": now - interval_start,
                    "flops": interval_flops,
                    "tflops": interval_flops / (now - interval_start) / 1e12,
                })
                interval_start, interval_flops = now, 0
    del case, first_array, second_array, prepared
    mx.clear_cache()
    cleanup_ms = round(time.time() * 1_000)
    monitor.await_after(cleanup_ms)
    thermal = monitor.stop()
    repetitions = sorted({item["repetition"] for item in samples})
    walls = [sum(item["wallMilliseconds"] for item in samples if item["repetition"] == rep) for rep in repetitions]
    total_flops = sum(2 * ROWS * inner * columns for _, _, _, _, inner, columns in CASES)
    rates = [total_flops / wall / 1e9 for wall in walls]
    retention = None
    if intervals and len(intervals) >= 3:
        third = max(1, len(intervals) // 3)
        retention = statistics.median(item["tflops"] for item in intervals[-third:]) / max(
            statistics.median(item["tflops"] for item in intervals[:third]), 1e-30
        )
    receipt = {
        "schema": "ace-opt-0068-mlx-diagnostic-v1",
        "experimentId": "OPT-0068",
        "status": "diagnostic-only",
        "gateAuthorizing": False,
        "nonAuthorizingReasons": [
            "MLX f16 matmul materializes f16 before explicit f32 cast",
            "MLX Python exposes fenced wall but no raw GPU timestamps",
            "full actual-output independent CPU contract is implemented by the Swift gate, not duplicated here",
        ],
        "identity": {
            "model": "Mac15,12",
            "osVersion": "26.5.2",
            "python": sys.version,
            "mlx": mx.__version__ if hasattr(mx, "__version__") else "0.32.0",
            "numpy": np.__version__,
            "harnessCommit": args.harness_commit,
            "sourceSha256": sha256_file(Path(__file__)),
        },
        "correctness": correctness,
        "adversarial": adversarial,
        "performance": {
            "samples": samples,
            "stageProfiles": stage_profiles,
            "weightedWallMilliseconds": walls,
            "weightedTFLOPS": rates,
            "medianWeightedWallMilliseconds": statistics.median(walls),
            "medianWeightedTFLOPS": statistics.median(rates),
            "intervals": intervals,
            "sustainedRetention": retention,
        },
        "thermal": thermal,
    }
    output = Path(args.output)
    require(not output.exists(), f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("describe", "inspect", "measure", "sustained"), default="describe")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--output")
    parser.add_argument("--thermal-trace")
    parser.add_argument("--harness-commit")
    parser.add_argument("--execute-native-gpu", dest="consent")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "describe":
        print(__doc__)
        return
    require(args.package_dir is not None and args.fixture_manifest is not None, "non-describe modes require package/fixture")
    resolved = authenticate(args.package_dir, args.fixture_manifest)
    if args.mode == "inspect":
        print(f"authenticated {len(resolved)} actual M2250 cases; MLX was not imported")
        return
    execute(args, resolved)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"OPT-0068 MLX FAIL CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
