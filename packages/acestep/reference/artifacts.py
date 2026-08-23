"""Transactional native-capture artifact writer and hash replay verifier."""

from __future__ import annotations

import os
import re
import shutil
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .constants import CAPTURE_SCHEMA_VERSION
from .contracts import FixtureContract, TapContract
from .jsonio import (
    canonical_json_bytes,
    ensure_finite_numbers,
    load_json,
    require_exact_keys,
    require_object,
    require_sha256,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)


_CAPTURE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_TAP_ID = re.compile(r"^[a-z0-9][a-z0-9._-]+$")
_DTYPE_BYTES = {
    "bool": 1,
    "uint32": 4,
    "int64": 8,
    "bfloat16": 2,
    "float32": 4,
}


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    tapId: str
    path: str
    dtype: str
    logicalShape: list[int]
    storedShape: list[int]
    capture: str
    byteLength: int
    sha256: str
    logicalTensorSha256: str | None
    selection: dict[str, object] | None
    stats: dict[str, int | float] | None


@dataclass(frozen=True, slots=True)
class CaptureReplayContract:
    """Trusted repository state against which a capture is replayed."""

    fixture: FixtureContract
    taps: TapContract
    golden_manifest_id: str
    reference_tool: dict[str, object]
    source: dict[str, object]
    browser_package: dict[str, object]
    input_bundle: dict[str, object]
    environment: dict[str, object]


def _safe_relative(value: str, *, name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {name} path {value!r}")
    return path


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class CaptureWriter:
    """Write one complete capture under an atomic ``golden-local`` directory."""

    def __init__(
        self,
        *,
        output_root: Path,
        capture_id: str,
        fixture: FixtureContract,
        taps: TapContract,
    ) -> None:
        if _CAPTURE_ID.fullmatch(capture_id) is None:
            raise ValueError("capture-id must be a lowercase filesystem-safe identifier")
        self.fixture = fixture
        self.taps = taps
        self.capture_id = capture_id
        fixture_root = output_root / fixture.fixture_id
        self.final_root = fixture_root / capture_id
        self.partial_root = fixture_root / f".{capture_id}.{uuid.uuid4().hex}.partial"
        if self.final_root.exists():
            raise ValueError(f"capture already exists: {self.final_root}")
        self.partial_root.mkdir(parents=True, exist_ok=False)
        self._records: dict[str, ArtifactRecord] = {}
        self._auxiliary: list[dict[str, object]] = []
        expected = fixture.expected
        self._shape_symbols: dict[str, int] = {
            "B": 1,
            "T": int(expected["latentFrames"]),
            "P": int(expected["patchedTokens"]),
            "C5": int(expected["semanticCodes"]),
            "S": int(expected["audioSamplesPerChannel"]),
            "probe-steps": 3,
        }
        self._closed = False

    def abort(self) -> None:
        if not self._closed:
            shutil.rmtree(self.partial_root, ignore_errors=True)
            self._closed = True

    def has_tensor(self, tap_id: str) -> bool:
        return tap_id in self._records

    def add_tensor(
        self,
        tap_id: str,
        payload: bytes,
        *,
        dtype: str,
        logical_shape: Iterable[int],
        stored_shape: Iterable[int] | None = None,
        capture: str = "full",
        logical_tensor_sha256: str | None = None,
        selection: dict[str, object] | None = None,
        stats: dict[str, int | float] | None = None,
    ) -> ArtifactRecord:
        if self._closed:
            raise RuntimeError("capture writer is closed")
        if tap_id not in self.taps.specs:
            raise ValueError(f"unknown tap {tap_id!r}")
        if tap_id in self._records:
            raise ValueError(f"tap {tap_id!r} was captured more than once")
        if _TAP_ID.fullmatch(tap_id) is None:
            raise ValueError(f"tap id cannot be represented safely: {tap_id!r}")
        spec = self.taps.specs[tap_id]
        if dtype != spec.dtype or dtype not in _DTYPE_BYTES:
            raise ValueError(
                f"{tap_id}: dtype {dtype!r} differs from contract {spec.dtype!r}"
            )
        logical = list(logical_shape)
        stored = logical if stored_shape is None else list(stored_shape)
        if not logical or not stored or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            for dimension in logical + stored
        ):
            raise ValueError(f"{tap_id}: invalid shape")
        self._validate_logical_shape(tap_id, logical)
        expected_bytes = _DTYPE_BYTES[dtype]
        for dimension in stored:
            expected_bytes *= dimension
        if len(payload) != expected_bytes:
            raise ValueError(
                f"{tap_id}: got {len(payload)} payload bytes, expected {expected_bytes}"
            )
        if capture not in {"full", "slice+stats"} or capture != spec.capture:
            raise ValueError(f"{tap_id}: capture mode differs from taps.json")
        if capture == "full":
            actual_full_sha = sha256_bytes(payload)
            if logical_tensor_sha256 is not None and logical_tensor_sha256 != actual_full_sha:
                raise ValueError(f"{tap_id}: supplied full-tensor SHA-256 is wrong")
            logical_tensor_sha256 = actual_full_sha
            if selection is not None:
                raise ValueError(f"{tap_id}: full capture cannot have a slice selection")
        else:
            if selection is None or stats is None:
                raise ValueError(f"{tap_id}: slice+stats requires selection and stats")
            if logical_tensor_sha256 is not None:
                require_sha256(logical_tensor_sha256, name="logicalTensorSha256")
        if stats is not None:
            _validate_stats(stats, tap_id=tap_id)
        ensure_finite_numbers(selection)
        ensure_finite_numbers(stats)
        relative = f"tensors/{tap_id}.bin"
        destination = self.partial_root / relative
        _write_exclusive(destination, payload)
        record = ArtifactRecord(
            tapId=tap_id,
            path=relative,
            dtype=dtype,
            logicalShape=logical,
            storedShape=stored,
            capture=capture,
            byteLength=len(payload),
            sha256=sha256_bytes(payload),
            logicalTensorSha256=logical_tensor_sha256,
            selection=selection,
            stats=stats,
        )
        self._records[tap_id] = record
        return record

    def _validate_logical_shape(self, tap_id: str, actual: list[int]) -> None:
        expected = self.taps.specs[tap_id].shape
        if len(actual) != len(expected):
            raise ValueError(
                f"{tap_id}: rank {len(actual)} differs from contract rank {len(expected)}"
            )
        for axis, (actual_dimension, expected_dimension) in enumerate(
            zip(actual, expected, strict=True)
        ):
            if isinstance(expected_dimension, int):
                resolved = expected_dimension
            elif expected_dimension == "variable":
                if actual_dimension <= 0:
                    raise ValueError(f"{tap_id}: variable axis {axis} is empty")
                continue
            elif isinstance(expected_dimension, str):
                resolved = self._shape_symbols.get(expected_dimension)
                if resolved is None:
                    if actual_dimension <= 0:
                        raise ValueError(
                            f"{tap_id}: symbolic axis {expected_dimension} is empty"
                        )
                    self._shape_symbols[expected_dimension] = actual_dimension
                    continue
            else:
                raise ValueError(f"{tap_id}: malformed shape contract")
            if actual_dimension != resolved:
                raise ValueError(
                    f"{tap_id}: axis {axis} is {actual_dimension}, expected "
                    f"{expected_dimension}={resolved}"
                )

    def add_auxiliary(self, relative: str, payload: bytes, *, kind: str) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("capture writer is closed")
        pure = _safe_relative(relative, name="auxiliary")
        if pure.parts[0] == "tensors" or relative == "capture.json":
            raise ValueError("auxiliary path overlaps reserved capture paths")
        if any(item["path"] == pure.as_posix() for item in self._auxiliary):
            raise ValueError(f"duplicate auxiliary artifact {relative!r}")
        _write_exclusive(self.partial_root / pure, payload)
        record = {
            "kind": kind,
            "path": pure.as_posix(),
            "byteLength": len(payload),
            "sha256": sha256_bytes(payload),
        }
        self._auxiliary.append(record)
        return record

    def add_float32_stereo_output(
        self,
        planar_f32le: bytes,
        *,
        samples_per_channel: int,
        sample_rate: int = 48_000,
    ) -> None:
        """Write final normalized planar raw audio and an interleaved float WAV."""

        if len(planar_f32le) != samples_per_channel * 2 * 4:
            raise ValueError("stereo float32 audio byte length mismatch")
        values = struct.unpack(f"<{samples_per_channel * 2}f", planar_f32le)
        if any(not (-float("inf") < value < float("inf")) for value in values):
            raise ValueError("final audio contains a non-finite sample")
        interleaved = bytearray(len(planar_f32le))
        left = values[:samples_per_channel]
        right = values[samples_per_channel:]
        for index, (left_sample, right_sample) in enumerate(zip(left, right, strict=True)):
            struct.pack_into("<ff", interleaved, index * 8, left_sample, right_sample)
        wav = _float_wav(bytes(interleaved), channels=2, sample_rate=sample_rate)
        self.add_auxiliary("output.raw-f32le", planar_f32le, kind="final-audio-planar-f32le")
        self.add_auxiliary("output.wav", wav, kind="final-audio-wav-f32le")

    def finalize(
        self,
        *,
        replay: CaptureReplayContract,
        random_injection: dict[str, object],
        environment: dict[str, object],
    ) -> Path:
        if self._closed:
            raise RuntimeError("capture writer is closed")
        if replay.fixture != self.fixture or replay.taps is not self.taps:
            raise ValueError("capture writer/replay contracts differ")
        required = set(self.taps.required_ids(self.fixture))
        missing = sorted(required - set(self._records))
        unknown = sorted(set(self._records) - set(self.taps.specs))
        if missing or unknown:
            raise ValueError(f"capture tap inventory mismatch; missing={missing}, unknown={unknown}")
        auxiliary_paths = {item["path"] for item in self._auxiliary}
        if auxiliary_paths != {"output.raw-f32le", "output.wav"}:
            raise ValueError("complete capture requires exactly output.raw-f32le and output.wav")
        if random_injection.get("initialNoiseInjected") is not True:
            raise ValueError("capture cannot finish without injected initial noise")
        if (
            self.fixture.planner_enabled
            and random_injection.get("plannerWordsInjected") is not True
        ):
            raise ValueError("planner capture cannot finish without injected planner words")
        if environment.get("backend") != "pytorch-eager":
            raise ValueError("capture backend must be pytorch-eager")
        if environment.get("attentionImplementation") != "eager":
            raise ValueError("capture attention implementation must be eager")
        if environment.get("dcwBackend") != "pytorch-wavelets-haar":
            raise ValueError("capture DCW backend must be active PyTorch Haar")
        if environment != replay.environment:
            raise ValueError("live environment differs from replay contract")

        artifact_values = [
            {
                "tapId": record.tapId,
                "path": record.path,
                "dtype": record.dtype,
                "logicalShape": record.logicalShape,
                "storedShape": record.storedShape,
                "capture": record.capture,
                "byteLength": record.byteLength,
                "sha256": record.sha256,
                "logicalTensorSha256": record.logicalTensorSha256,
                "selection": record.selection,
                "stats": record.stats,
            }
            for _, record in sorted(self._records.items())
        ]
        auxiliary = sorted(self._auxiliary, key=lambda item: str(item["path"]))
        artifact_material = {"artifacts": artifact_values, "auxiliary": auxiliary}
        artifact_set_sha = sha256_bytes(canonical_json_bytes(artifact_material))
        capture_without_identity = {
            "schemaVersion": CAPTURE_SCHEMA_VERSION,
            "captureId": self.capture_id,
            "fixtureId": self.fixture.fixture_id,
            "fixtureContractSha256": self.fixture.contract_sha256,
            "tapContractSha256": self.taps.contract_sha256,
            "goldenManifestId": replay.golden_manifest_id,
            "referenceTool": replay.reference_tool,
            "source": replay.source,
            "browserPackage": replay.browser_package,
            "inputBundle": replay.input_bundle,
            "randomInjection": random_injection,
            "environment": environment,
            "requiredTapIds": sorted(required),
            "artifacts": artifact_values,
            "auxiliary": auxiliary,
            "artifactSetSha256": artifact_set_sha,
        }
        identity_material = {
            key: value
            for key, value in capture_without_identity.items()
            if key != "captureId"
        }
        capture = {
            **capture_without_identity,
            "captureIdentitySha256": sha256_bytes(
                canonical_json_bytes(identity_material)
            ),
        }
        ensure_finite_numbers(capture)
        write_json_atomic(self.partial_root / "capture.json", capture)
        verify_capture(self.partial_root, replay=replay)
        self.final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.partial_root, self.final_root)
        self._closed = True
        return self.final_root

    def __del__(self) -> None:
        # Best-effort cleanup only; callers should use abort after an exception.
        if not getattr(self, "_closed", True):
            shutil.rmtree(getattr(self, "partial_root", Path("/nonexistent")), ignore_errors=True)


def _validate_stats(stats: dict[str, int | float], *, tap_id: str) -> None:
    keys = {
        "count",
        "finiteCount",
        "nanCount",
        "infCount",
        "min",
        "max",
        "meanF64",
        "rmsF64",
        "l1F64",
    }
    require_exact_keys(stats, keys, name=f"{tap_id}.stats")
    integer_keys = {"count", "finiteCount", "nanCount", "infCount"}
    if any(
        not isinstance(stats[key], int) or isinstance(stats[key], bool) or stats[key] < 0
        for key in integer_keys
    ):
        raise ValueError(f"{tap_id}: invalid statistics counts")
    if stats["finiteCount"] + stats["nanCount"] + stats["infCount"] != stats["count"]:
        raise ValueError(f"{tap_id}: inconsistent statistics counts")
    numeric_keys = {"min", "max", "meanF64", "rmsF64", "l1F64"}
    if any(
        not isinstance(stats[key], (int, float)) or isinstance(stats[key], bool)
        for key in numeric_keys
    ):
        raise ValueError(f"{tap_id}: statistics values must be finite numbers")
    if stats["min"] > stats["max"] or stats["rmsF64"] < 0 or stats["l1F64"] < 0:
        raise ValueError(f"{tap_id}: statistics values are inconsistent")
    ensure_finite_numbers(stats)


def _float_wav(interleaved_f32le: bytes, *, channels: int, sample_rate: int) -> bytes:
    """Build a standards-compliant WAVE_FORMAT_IEEE_FLOAT container."""

    block_align = channels * 4
    byte_rate = sample_rate * block_align
    fmt = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, 32)
    riff_size = 4 + 8 + len(fmt) + 8 + len(interleaved_f32le)
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(interleaved_f32le))
        + interleaved_f32le
    )


_CAPTURE_KEYS = {
    "schemaVersion",
    "captureId",
    "fixtureId",
    "fixtureContractSha256",
    "tapContractSha256",
    "goldenManifestId",
    "referenceTool",
    "source",
    "browserPackage",
    "inputBundle",
    "randomInjection",
    "environment",
    "requiredTapIds",
    "artifacts",
    "auxiliary",
    "artifactSetSha256",
    "captureIdentitySha256",
}


def verify_capture(
    capture_root: Path,
    *,
    replay: CaptureReplayContract,
) -> dict[str, Any]:
    """Replay all byte/identity checks without loading an upstream model."""

    capture = require_object(load_json(capture_root / "capture.json"), name="capture.json")
    require_exact_keys(capture, _CAPTURE_KEYS, name="capture.json")
    if capture["schemaVersion"] != CAPTURE_SCHEMA_VERSION:
        raise ValueError("unsupported capture schema")
    require_sha256(capture["fixtureContractSha256"], name="fixtureContractSha256")
    require_sha256(capture["tapContractSha256"], name="tapContractSha256")
    require_sha256(capture["artifactSetSha256"], name="artifactSetSha256")
    require_sha256(capture["captureIdentitySha256"], name="captureIdentitySha256")
    if (
        capture["fixtureId"] != replay.fixture.fixture_id
        or capture["fixtureContractSha256"] != replay.fixture.contract_sha256
        or capture["tapContractSha256"] != replay.taps.contract_sha256
        or capture["goldenManifestId"] != replay.golden_manifest_id
    ):
        raise ValueError("capture belongs to another committed golden contract")
    _validate_reference_tool(capture["referenceTool"])
    _validate_source(capture["source"])
    _validate_browser_package(capture["browserPackage"])
    for field, expected in (
        ("referenceTool", replay.reference_tool),
        ("source", replay.source),
        ("browserPackage", replay.browser_package),
        ("inputBundle", replay.input_bundle),
        ("environment", replay.environment),
    ):
        if capture[field] != expected:
            raise ValueError(f"capture {field} differs from authenticated provenance")
    artifacts = capture["artifacts"]
    auxiliary = capture["auxiliary"]
    if not isinstance(artifacts, list) or not isinstance(auxiliary, list):
        raise ValueError("capture artifact inventories must be arrays")
    paths: set[str] = set()
    tap_ids: set[str] = set()
    shape_symbols: dict[str, int] = {
        "B": 1,
        "T": int(replay.fixture.expected["latentFrames"]),
        "P": int(replay.fixture.expected["patchedTokens"]),
        "C5": int(replay.fixture.expected["semanticCodes"]),
        "S": int(replay.fixture.expected["audioSamplesPerChannel"]),
        "probe-steps": 3,
    }
    for index, raw in enumerate(artifacts):
        record = require_object(raw, name=f"artifacts[{index}]")
        require_exact_keys(
            record,
            {
                "tapId",
                "path",
                "dtype",
                "logicalShape",
                "storedShape",
                "capture",
                "byteLength",
                "sha256",
                "logicalTensorSha256",
                "selection",
                "stats",
            },
            name=f"artifacts[{index}]",
        )
        tap_id = record["tapId"]
        if not isinstance(tap_id, str) or tap_id in tap_ids:
            raise ValueError("invalid or duplicate captured tap")
        tap_ids.add(tap_id)
        if (
            tap_id not in replay.taps.specs
            or record["dtype"] != replay.taps.specs[tap_id].dtype
            or record["capture"] != replay.taps.specs[tap_id].capture
        ):
            raise ValueError(f"capture tap contract mismatch for {tap_id}")
        _validate_tensor_record(
            record,
            tap_id=tap_id,
            spec=replay.taps.specs[tap_id],
            symbols=shape_symbols,
        )
        _verify_record_file(capture_root, record, paths=paths)
    for index, raw in enumerate(auxiliary):
        record = require_object(raw, name=f"auxiliary[{index}]")
        require_exact_keys(record, {"kind", "path", "byteLength", "sha256"}, name="aux")
        _validate_auxiliary_record(record, replay.fixture)
        _verify_record_file(capture_root, record, paths=paths)
    expected_required = set(capture["requiredTapIds"])
    if tap_ids != expected_required:
        raise ValueError("captured and required tap inventories differ")
    if expected_required != set(replay.taps.required_ids(replay.fixture)):
        raise ValueError("capture requiredTapIds differs from current taps.json")
    identity_material = {"artifacts": artifacts, "auxiliary": auxiliary}
    if sha256_bytes(canonical_json_bytes(identity_material)) != capture["artifactSetSha256"]:
        raise ValueError("capture artifact-set identity mismatch")
    capture_identity_material = {
        key: value
        for key, value in capture.items()
        if key not in {"captureId", "captureIdentitySha256"}
    }
    if (
        sha256_bytes(canonical_json_bytes(capture_identity_material))
        != capture["captureIdentitySha256"]
    ):
        raise ValueError("capture provenance identity mismatch")
    actual_paths = {
        path.relative_to(capture_root).as_posix()
        for path in capture_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_paths = paths | {"capture.json"}
    if actual_paths != expected_paths:
        raise ValueError(
            "capture file inventory mismatch; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unknown={sorted(actual_paths - expected_paths)}"
        )
    _validate_audio_payloads(capture_root, capture, replay.fixture)
    _validate_environment(capture["environment"])
    _validate_random_injection(capture, replay)
    return capture


def _verify_record_file(
    capture_root: Path,
    record: dict[str, Any],
    *,
    paths: set[str],
) -> None:
    relative = record["path"]
    if not isinstance(relative, str):
        raise ValueError("artifact path must be a string")
    pure = _safe_relative(relative, name="artifact")
    normalized = pure.as_posix()
    if normalized in paths:
        raise ValueError(f"duplicate artifact path {normalized}")
    paths.add(normalized)
    require_sha256(record["sha256"], name=f"{normalized}.sha256")
    path = capture_root / pure
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe artifact {normalized}")
    if path.stat().st_size != record["byteLength"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"artifact identity mismatch for {normalized}")


def _validate_shape(
    raw: object,
    *,
    name: str,
    allow_zero: bool = True,
) -> list[int]:
    if (
        not isinstance(raw, list)
        or not raw
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < (0 if allow_zero else 1)
            for value in raw
        )
    ):
        raise ValueError(f"{name} must contain valid integer dimensions")
    return raw


def _elements(shape: list[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _validate_logical_shape_replay(
    tap_id: str,
    actual: list[int],
    expected: tuple[object, ...],
    symbols: dict[str, int],
) -> None:
    if len(actual) != len(expected):
        raise ValueError(f"{tap_id}: logical rank differs from tap contract")
    for axis, (dimension, contract) in enumerate(zip(actual, expected, strict=True)):
        if isinstance(contract, int):
            resolved = contract
        elif contract == "variable":
            if dimension <= 0:
                raise ValueError(f"{tap_id}: variable logical axis {axis} is empty")
            continue
        elif isinstance(contract, str):
            resolved = symbols.get(contract)
            if resolved is None:
                if dimension <= 0:
                    raise ValueError(f"{tap_id}: symbolic logical axis {contract} is empty")
                symbols[contract] = dimension
                continue
        else:
            raise ValueError(f"{tap_id}: malformed tap shape contract")
        if dimension != resolved:
            raise ValueError(f"{tap_id}: logical axis {axis} differs from {contract}={resolved}")


def _validate_tensor_record(
    record: dict[str, Any],
    *,
    tap_id: str,
    spec: object,
    symbols: dict[str, int],
) -> None:
    if record["path"] != f"tensors/{tap_id}.bin":
        raise ValueError(f"{tap_id}: tensor path does not follow the canonical convention")
    dtype = record["dtype"]
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"{tap_id}: unsupported tensor dtype")
    logical = _validate_shape(record["logicalShape"], name=f"{tap_id}.logicalShape")
    stored = _validate_shape(record["storedShape"], name=f"{tap_id}.storedShape")
    _validate_logical_shape_replay(tap_id, logical, spec.shape, symbols)
    byte_length = record["byteLength"]
    if (
        not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or byte_length != _elements(stored) * _DTYPE_BYTES[dtype]
    ):
        raise ValueError(f"{tap_id}: byteLength differs from dtype/storedShape")
    logical_sha = record["logicalTensorSha256"]
    if logical_sha is not None:
        require_sha256(logical_sha, name=f"{tap_id}.logicalTensorSha256")
    if record["capture"] == "full":
        if (
            stored != logical
            or logical_sha != record["sha256"]
            or record["selection"] is not None
            or record["stats"] is not None
        ):
            raise ValueError(f"{tap_id}: malformed full-tensor metadata")
        return
    if record["capture"] != "slice+stats":
        raise ValueError(f"{tap_id}: unknown capture mode")
    if logical_sha is not None:
        raise ValueError(f"{tap_id}: sliced capture cannot claim a full-tensor SHA")
    _validate_selection(
        record["selection"],
        tap_id=tap_id,
        logical=logical,
        stored=stored,
    )
    stats = require_object(record["stats"], name=f"{tap_id}.stats")
    _validate_stats(stats, tap_id=tap_id)
    if stats["count"] != _elements(logical):
        raise ValueError(f"{tap_id}: statistics count differs from logical tensor size")


def _validate_selection(
    raw: object,
    *,
    tap_id: str,
    logical: list[int],
    stored: list[int],
) -> None:
    selection = require_object(raw, name=f"{tap_id}.selection")
    require_exact_keys(
        selection,
        {
            "sequenceAxis",
            "sequenceIndices",
            "channelAxis",
            "channelStart",
            "channelEnd",
        },
        name=f"{tap_id}.selection",
    )
    sequence_axis = selection["sequenceAxis"]
    indices = selection["sequenceIndices"]
    if (
        not isinstance(sequence_axis, int)
        or isinstance(sequence_axis, bool)
        or not 0 <= sequence_axis < len(logical)
        or not isinstance(indices, list)
        or not indices
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < logical[sequence_axis]
            for index in indices
        )
        or indices != sorted(set(indices))
        or stored[sequence_axis] != len(indices)
    ):
        raise ValueError(f"{tap_id}: invalid sliced sequence selection")
    channel_axis = selection["channelAxis"]
    expected_stored = list(logical)
    expected_stored[sequence_axis] = len(indices)
    if channel_axis is None:
        if selection["channelStart"] is not None or selection["channelEnd"] is not None:
            raise ValueError(f"{tap_id}: inconsistent absent channel selection")
    else:
        start = selection["channelStart"]
        end = selection["channelEnd"]
        if (
            not isinstance(channel_axis, int)
            or isinstance(channel_axis, bool)
            or not 0 <= channel_axis < len(logical)
            or channel_axis == sequence_axis
            or start != 0
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 < end <= logical[channel_axis]
        ):
            raise ValueError(f"{tap_id}: invalid channel selection")
        expected_stored[channel_axis] = end
    if stored != expected_stored:
        raise ValueError(f"{tap_id}: storedShape differs from slice selection")


def _validate_auxiliary_record(record: dict[str, Any], fixture: FixtureContract) -> None:
    expected = {
        "output.raw-f32le": (
            "final-audio-planar-f32le",
            int(fixture.expected["audioSamplesPerChannel"]) * 2 * 4,
        ),
        "output.wav": (
            "final-audio-wav-f32le",
            44 + int(fixture.expected["audioSamplesPerChannel"]) * 2 * 4,
        ),
    }
    path = record["path"]
    if path not in expected:
        raise ValueError("capture contains an unknown auxiliary artifact")
    kind, byte_length = expected[path]
    if record["kind"] != kind or record["byteLength"] != byte_length:
        raise ValueError(f"{path}: auxiliary kind/length differs from fixture")


def _validate_audio_payloads(
    capture_root: Path,
    capture: dict[str, Any],
    fixture: FixtureContract,
) -> None:
    samples = int(fixture.expected["audioSamplesPerChannel"])
    raw_path = capture_root / "output.raw-f32le"
    wav_path = capture_root / "output.wav"
    raw = raw_path.read_bytes()
    values = struct.unpack(f"<{samples * 2}f", raw)
    if any(not (-float("inf") < value < float("inf")) for value in values):
        raise ValueError("captured raw audio contains a non-finite sample")
    artifacts = {
        record["tapId"]: record
        for record in capture["artifacts"]
        if isinstance(record, dict)
    }
    final_audio = artifacts.get("audio.final-output")
    if final_audio is None or final_audio.get("sha256") != sha256_bytes(raw):
        raise ValueError("raw audio differs from audio.final-output tensor")
    interleaved = bytearray(len(raw))
    left = values[:samples]
    right = values[samples:]
    for index, (left_sample, right_sample) in enumerate(zip(left, right, strict=True)):
        struct.pack_into("<ff", interleaved, index * 8, left_sample, right_sample)
    expected_wav = _float_wav(bytes(interleaved), channels=2, sample_rate=48_000)
    if wav_path.read_bytes() != expected_wav:
        raise ValueError("WAV artifact is not the canonical 48 kHz stereo float container")


def _validate_environment(raw: object) -> None:
    environment = require_object(raw, name="environment")
    require_exact_keys(
        environment,
        {
            "backend",
            "attentionImplementation",
            "dcwBackend",
            "vaeComputeDtype",
            "vaeChunkSize",
            "vaeOverlap",
            "vaeOffloadWavToCpu",
            "device",
            "python",
            "platform",
            "packages",
            "deterministicAlgorithms",
            "tf32",
            "accelerator",
        },
        name="environment",
    )
    if (
        environment["backend"] != "pytorch-eager"
        or environment["attentionImplementation"] != "eager"
        or environment["dcwBackend"] != "pytorch-wavelets-haar"
        or environment["vaeComputeDtype"] != "float32"
        or environment["vaeChunkSize"] != 256
        or environment["vaeOverlap"] != 64
        or environment["vaeOffloadWavToCpu"] is not True
    ):
        raise ValueError("capture was not produced by the eager PyTorch/Haar reference path")
    if environment["device"] not in {"cuda", "xpu"}:
        raise ValueError("capture environment device is not an accepted BF16 backend")
    if (
        not isinstance(environment["python"], str)
        or not environment["python"].startswith("3.12.")
        or not isinstance(environment["platform"], str)
        or not environment["platform"]
    ):
        raise ValueError("capture environment has an invalid Python/platform identity")
    if environment["deterministicAlgorithms"] is not True or environment["tf32"] is not False:
        raise ValueError("capture environment did not preserve deterministic FP32 controls")
    packages = require_object(environment["packages"], name="environment.packages")
    require_exact_keys(
        packages,
        {
            "torch",
            "pytorchWavelets",
            "pyWavelets",
            "transformers",
            "diffusers",
            "numpy",
            "einops",
        },
        name="environment.packages",
    )
    if any(not isinstance(version, str) or not version for version in packages.values()):
        raise ValueError("capture environment contains an empty package version")
    expected_common = {
        "pytorchWavelets": "1.3.0",
        "pyWavelets": "1.9.0",
        "transformers": "4.57.6",
        "diffusers": "0.37.1",
        "numpy": "2.3.5",
        "einops": "0.8.2",
    }
    if any(packages[key] != version for key, version in expected_common.items()):
        raise ValueError("capture package versions differ from the pinned source lock")
    accelerator = require_object(environment["accelerator"], name="environment.accelerator")
    require_exact_keys(
        accelerator,
        {
            "backend",
            "deviceIndex",
            "name",
            "capability",
            "runtimeVersion",
            "driverVersion",
            "totalMemoryBytes",
        },
        name="environment.accelerator",
    )
    if accelerator["backend"] != environment["device"]:
        raise ValueError("capture accelerator backend differs from environment device")
    if (
        not isinstance(accelerator["deviceIndex"], int)
        or isinstance(accelerator["deviceIndex"], bool)
        or accelerator["deviceIndex"] < 0
        or not isinstance(accelerator["totalMemoryBytes"], int)
        or isinstance(accelerator["totalMemoryBytes"], bool)
        or accelerator["totalMemoryBytes"] <= 0
        or any(
            not isinstance(accelerator[key], str) or not accelerator[key]
            for key in ("name", "capability", "runtimeVersion", "driverVersion")
        )
    ):
        raise ValueError("capture accelerator receipt is incomplete")
    allowed_torch = {
        "cuda": {"2.10.0+cu128", "2.10.0+cu130"},
        "xpu": {"2.10.0"},
    }
    if packages["torch"] not in allowed_torch[environment["device"]]:
        raise ValueError("capture Torch version differs from the pinned accelerator lock")


def _validate_reference_tool(raw: object) -> None:
    tool = require_object(raw, name="referenceTool")
    require_exact_keys(tool, {"version", "fileCount", "fileSetSha256"}, name="referenceTool")
    require_sha256(tool["fileSetSha256"], name="referenceTool.fileSetSha256")
    if (
        not isinstance(tool["version"], int)
        or isinstance(tool["version"], bool)
        or tool["version"] <= 0
        or not isinstance(tool["fileCount"], int)
        or isinstance(tool["fileCount"], bool)
        or tool["fileCount"] <= 0
    ):
        raise ValueError("referenceTool contains invalid numeric identity fields")


def _validate_source(raw: object) -> None:
    source = require_object(raw, name="source")
    require_exact_keys(
        source,
        {
            "repository",
            "commit",
            "checkoutRemote",
            "aceModelRevision",
            "plannerModelRevision",
            "rawModelArtifactSetSha256",
        },
        name="source",
    )
    if any(not isinstance(value, str) or not value for value in source.values()):
        raise ValueError("source provenance contains an empty identity")
    require_sha256(
        source["rawModelArtifactSetSha256"],
        name="source.rawModelArtifactSetSha256",
    )


def _validate_browser_package(raw: object) -> None:
    package = require_object(raw, name="browserPackage")
    require_exact_keys(
        package,
        {
            "format",
            "profile",
            "manifestSha256",
            "manifestBytes",
            "packageBytesIncludingManifest",
            "payloadHashesVerified",
        },
        name="browserPackage",
    )
    require_sha256(package["manifestSha256"], name="browserPackage.manifestSha256")
    if (
        package["profile"] != "reference"
        or package["payloadHashesVerified"] is not True
        or any(
            not isinstance(package[key], int)
            or isinstance(package[key], bool)
            or package[key] <= 0
            for key in ("manifestBytes", "packageBytesIncludingManifest")
        )
    ):
        raise ValueError("browser package provenance is not a deep-authenticated reference package")


def _validate_input_bundle_identity(raw: object, fixture: FixtureContract) -> dict[str, Any]:
    identity = require_object(raw, name="inputBundle")
    require_exact_keys(
        identity,
        {
            "manifestSha256",
            "algorithmId",
            "gaussianMapping",
            "initialNoise",
            "plannerSampling",
        },
        name="inputBundle",
    )
    require_sha256(identity["manifestSha256"], name="inputBundle.manifestSha256")
    random_contract = fixture.contract["random"]
    if (
        identity["algorithmId"] != random_contract["algorithmId"]
        or identity["gaussianMapping"] != random_contract["gaussianMapping"]
    ):
        raise ValueError("capture input bundle uses different random mappings")
    initial = require_object(identity["initialNoise"], name="inputBundle.initialNoise")
    require_exact_keys(
        initial,
        {"shape", "byteLength", "sha256"},
        name="inputBundle.initialNoise",
    )
    require_sha256(initial["sha256"], name="inputBundle.initialNoise.sha256")
    expected_initial = random_contract["initialNoise"]
    if (
        initial["shape"] != expected_initial["shape"]
        or initial["byteLength"] != 4 * _shape_elements(expected_initial["shape"])
        or initial["sha256"] != expected_initial["sha256"]
    ):
        raise ValueError("capture input noise differs from the committed fixture")
    planner = identity["plannerSampling"]
    if not fixture.planner_enabled:
        if planner is not None:
            raise ValueError("direct capture unexpectedly declares planner input")
        return identity
    planner_value = require_object(planner, name="inputBundle.plannerSampling")
    require_exact_keys(
        planner_value,
        {"mapping", "drawStart", "wordCapacity", "byteLength", "sha256"},
        name="inputBundle.plannerSampling",
    )
    require_sha256(planner_value["sha256"], name="inputBundle.plannerSampling.sha256")
    capacity = planner_value["wordCapacity"]
    if (
        planner_value["mapping"] != random_contract["categoricalMapping"]
        or planner_value["drawStart"] != 0
        or not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or capacity < fixture.expected["semanticCodes"]
        or capacity > 1_000_000
        or planner_value["byteLength"] != capacity * 4
    ):
        raise ValueError("capture planner input identity is invalid")
    return identity


def _shape_elements(shape: object) -> int:
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in shape
        )
    ):
        raise ValueError("capture input shape is invalid")
    count = 1
    for value in shape:
        count *= value
    return count


def _validate_random_injection(
    capture: dict[str, Any], replay: CaptureReplayContract
) -> None:
    input_bundle = _validate_input_bundle_identity(capture["inputBundle"], replay.fixture)
    if input_bundle != replay.input_bundle:
        raise ValueError("capture input bundle differs from authenticated replay input")
    injection = require_object(capture["randomInjection"], name="randomInjection")
    base_keys = {
        "algorithmId",
        "gaussianMapping",
        "initialNoiseInjected",
        "initialNoiseSha256",
        "plannerWordsInjected",
    }
    planner_keys = {
        "plannerMapping",
        "plannerWordsInputSha256",
        "plannerWordsConsumed",
        "plannerWordCapacity",
        "phaseDraws",
    }
    require_exact_keys(
        injection,
        base_keys | (planner_keys if replay.fixture.planner_enabled else set()),
        name="randomInjection",
    )
    initial = require_object(input_bundle["initialNoise"], name="inputBundle.initialNoise")
    if (
        injection["algorithmId"] != input_bundle["algorithmId"]
        or injection["gaussianMapping"] != input_bundle["gaussianMapping"]
        or injection["initialNoiseInjected"] is not True
        or injection["initialNoiseSha256"] != initial["sha256"]
    ):
        raise ValueError("capture did not inject the authenticated initial noise")
    records = {
        record["tapId"]: record
        for record in capture["artifacts"]
        if isinstance(record, dict) and isinstance(record.get("tapId"), str)
    }
    noise_record = records.get("diffusion.initial-noise")
    if noise_record is None or noise_record.get("logicalTensorSha256") != initial["sha256"]:
        raise ValueError("captured initial-noise tap differs from injected input")
    if not replay.fixture.planner_enabled:
        if injection["plannerWordsInjected"] is not False:
            raise ValueError("direct capture claims planner-word injection")
        return

    planner = require_object(input_bundle["plannerSampling"], name="inputBundle.plannerSampling")
    phase_draws = require_object(injection["phaseDraws"], name="randomInjection.phaseDraws")
    require_exact_keys(phase_draws, {"cot", "codes"}, name="randomInjection.phaseDraws")
    consumed = injection["plannerWordsConsumed"]
    capacity = planner["wordCapacity"]
    semantic_codes = replay.fixture.expected["semanticCodes"]
    if (
        injection["plannerWordsInjected"] is not True
        or injection["plannerMapping"] != planner["mapping"]
        or injection["plannerWordsInputSha256"] != planner["sha256"]
        or injection["plannerWordCapacity"] != capacity
        or not isinstance(consumed, int)
        or isinstance(consumed, bool)
        or consumed <= 0
        or consumed > capacity
        or not isinstance(phase_draws["cot"], int)
        or isinstance(phase_draws["cot"], bool)
        or phase_draws["cot"] < 0
        or phase_draws["codes"] != semantic_codes + 1
        or phase_draws["cot"] + phase_draws["codes"] != consumed
    ):
        raise ValueError("capture planner injection receipt is inconsistent")
    words_record = records.get("planner.sample.words")
    if (
        words_record is None
        or words_record.get("logicalShape") != [consumed]
        or words_record.get("byteLength") != consumed * 4
    ):
        raise ValueError("captured planner-word tap has the wrong consumed prefix")
    from .inputs import load_prng_oracle

    oracle = load_prng_oracle()
    seed = replay.fixture.contract["random"]["userSeed"]
    expected_payload = b"".join(
        struct.pack("<I", oracle.random_word(seed, "planner-sampling", index))
        for index in range(consumed)
    )
    if words_record.get("logicalTensorSha256") != sha256_bytes(expected_payload):
        raise ValueError("captured planner words differ from the deterministic input prefix")


_ENVIRONMENT_RECEIPT_KEYS = {
    "schemaVersion",
    "captureIdentitySha256",
    "environmentSha256",
    "environment",
}


def environment_receipt_path(capture_root: Path) -> Path:
    return capture_root.with_name(f"{capture_root.name}.environment.json")


def write_environment_receipt(capture_root: Path) -> Path:
    """Write a review candidate outside the immutable capture directory."""

    capture = require_object(load_json(capture_root / "capture.json"), name="capture.json")
    environment = require_object(capture.get("environment"), name="environment")
    _validate_environment(environment)
    capture_identity = require_sha256(
        capture.get("captureIdentitySha256"),
        name="captureIdentitySha256",
    )
    receipt = {
        "schemaVersion": 1,
        "captureIdentitySha256": capture_identity,
        "environmentSha256": sha256_bytes(canonical_json_bytes(environment)),
        "environment": environment,
    }
    path = environment_receipt_path(capture_root)
    if path.exists():
        raise ValueError(f"environment receipt already exists: {path}")
    write_json_atomic(path, receipt)
    return path


def load_environment_receipt(
    path: Path,
    *,
    capture_identity_sha256: str,
) -> dict[str, object]:
    """Load externally reviewed expected environment bytes for replay."""

    raw_bytes = path.read_bytes()
    receipt = require_object(load_json(path), name="environment contract")
    require_exact_keys(receipt, _ENVIRONMENT_RECEIPT_KEYS, name="environment contract")
    if raw_bytes != canonical_json_bytes(receipt, newline=True):
        raise ValueError("environment contract must use canonical JSON bytes")
    if receipt["schemaVersion"] != 1:
        raise ValueError("unsupported environment-contract schema")
    identity = require_sha256(
        receipt["captureIdentitySha256"],
        name="environment contract captureIdentitySha256",
    )
    if identity != capture_identity_sha256:
        raise ValueError("environment contract belongs to another capture identity")
    expected_sha = require_sha256(
        receipt["environmentSha256"],
        name="environment contract environmentSha256",
    )
    environment = require_object(receipt["environment"], name="environment contract environment")
    if sha256_bytes(canonical_json_bytes(environment)) != expected_sha:
        raise ValueError("environment contract SHA-256 mismatch")
    _validate_environment(environment)
    return environment


def compare_captures(
    first_root: Path,
    second_root: Path,
    *,
    replay: CaptureReplayContract,
) -> None:
    """Require two fresh-process captures to have identical artifact identities."""

    first = verify_capture(first_root, replay=replay)
    second = verify_capture(second_root, replay=replay)
    if first["captureIdentitySha256"] != second["captureIdentitySha256"]:
        raise ValueError("fresh-process capture identities differ")
