"""Golden fixture, tap, and binding contracts for native capture."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    ACE_MODEL_ID,
    ACE_MODEL_REVISION,
    ACE_SOURCE_REVISION,
    PLANNER_MODEL_ID,
    PLANNER_MODEL_REVISION,
    REFERENCE_TOOL_VERSION,
)
from .jsonio import (
    canonical_json_bytes,
    load_json,
    require_exact_keys,
    require_object,
    require_sha256,
    sha256_bytes,
    sha256_file,
)


_BRACE = re.compile(r"\{([^{}]+)\}")


def _expand_braces(identifier: str) -> tuple[str, ...]:
    match = _BRACE.search(identifier)
    if match is None:
        return (identifier,)
    choices = match.group(1).split(",")
    if not choices or any(not choice for choice in choices):
        raise ValueError(f"malformed tap expansion {identifier!r}")
    expanded: list[str] = []
    for choice in choices:
        replacement = identifier[: match.start()] + choice + identifier[match.end() :]
        expanded.extend(_expand_braces(replacement))
    return tuple(expanded)


@dataclass(frozen=True, slots=True)
class FixtureContract:
    path: Path
    fixture_id: str
    contract_sha256: str
    contract: dict[str, Any]

    @property
    def planner_enabled(self) -> bool:
        return self.contract["planner"]["enabled"] is True

    @property
    def duration_seconds(self) -> float:
        return float(self.contract["request"]["durationSeconds"])

    @property
    def expected(self) -> dict[str, Any]:
        return self.contract["expected"]


@dataclass(frozen=True, slots=True)
class TapSpec:
    tap_id: str
    stage: str
    required: str
    dtype: str
    shape: tuple[object, ...]
    capture: str


class TapContract:
    """Expanded tap inventory with fixture-specific required-set resolution."""

    def __init__(self, path: Path) -> None:
        raw = require_object(load_json(path), name="taps.json")
        require_exact_keys(
            raw,
            {"schemaVersion", "layout", "symbols", "captureRules", "taps"},
            name="taps.json",
        )
        if raw["schemaVersion"] != 1 or raw["layout"] != "contiguous-little-endian":
            raise ValueError("unsupported tap contract")
        self.path = path
        self.contract_sha256 = sha256_file(path)
        if not isinstance(raw["taps"], list):
            raise ValueError("taps.json.taps must be an array")
        self.capture_rules = require_object(raw["captureRules"], name="captureRules")
        self.pattern_ids: list[str] = []
        self.specs: dict[str, TapSpec] = {}
        for index, item in enumerate(raw["taps"]):
            record = require_object(item, name=f"taps[{index}]")
            require_exact_keys(
                record,
                {"id", "stage", "required", "dtype", "shape", "capture"},
                name=f"taps[{index}]",
            )
            if not isinstance(record["id"], str) or not record["id"]:
                raise ValueError(f"taps[{index}].id must be non-empty")
            if not isinstance(record["shape"], list):
                raise ValueError(f"taps[{index}].shape must be an array")
            self.pattern_ids.append(record["id"])
            for tap_id in _expand_braces(record["id"]):
                if tap_id in self.specs:
                    raise ValueError(f"duplicate expanded tap {tap_id!r}")
                self.specs[tap_id] = TapSpec(
                    tap_id=tap_id,
                    stage=str(record["stage"]),
                    required=str(record["required"]),
                    dtype=str(record["dtype"]),
                    shape=tuple(record["shape"]),
                    capture=str(record["capture"]),
                )

    def required_ids(self, fixture: FixtureContract) -> tuple[str, ...]:
        required: list[str] = []
        for tap_id, spec in self.specs.items():
            include = spec.required == "all"
            include = include or (spec.required == "planner" and fixture.planner_enabled)
            include = include or (
                spec.required == "short-fixtures" and fixture.duration_seconds <= 30.0
            )
            if include:
                required.append(tap_id)
        return tuple(sorted(required))


def load_fixture(path: Path) -> FixtureContract:
    raw = require_object(load_json(path), name=str(path))
    require_exact_keys(
        raw,
        {"schemaVersion", "fixtureId", "contractSha256", "contract"},
        name=str(path),
    )
    if raw["schemaVersion"] != 1:
        raise ValueError(f"{path}: unsupported fixture schema")
    fixture_id = raw["fixtureId"]
    if not isinstance(fixture_id, str) or path.stem != fixture_id:
        raise ValueError(f"{path}: fixtureId/path mismatch")
    contract = require_object(raw["contract"], name=f"{fixture_id}.contract")
    digest = require_sha256(raw["contractSha256"], name="contractSha256")
    actual = sha256_bytes(canonical_json_bytes(contract))
    if digest != actual:
        raise ValueError(f"{fixture_id}: contract SHA-256 mismatch")
    source = require_object(contract.get("source"), name=f"{fixture_id}.source")
    expected_source = {
        "aceRepositoryCommit": ACE_SOURCE_REVISION,
        "mainModelId": ACE_MODEL_ID,
        "mainModelRevision": ACE_MODEL_REVISION,
        "plannerModelId": PLANNER_MODEL_ID,
        "plannerModelRevision": PLANNER_MODEL_REVISION,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(
                f"{fixture_id}: {key} is {source.get(key)!r}, expected {expected!r}"
            )
    return FixtureContract(path, fixture_id, digest, contract)


def verify_golden_manifest(golden_root: Path) -> str:
    """Verify all committed golden contract bytes and fixture identities."""

    manifest = require_object(load_json(golden_root / "MANIFEST.json"), name="MANIFEST")
    require_exact_keys(
        manifest,
        {"schemaVersion", "manifestId", "fixtures", "files"},
        name="MANIFEST",
    )
    if manifest["schemaVersion"] != 1:
        raise ValueError("unsupported golden manifest")
    fixtures = require_object(manifest["fixtures"], name="MANIFEST.fixtures")
    files = manifest["files"]
    if not isinstance(files, list):
        raise ValueError("MANIFEST.files must be an array")
    seen: set[str] = set()
    for index, item in enumerate(files):
        record = require_object(item, name=f"MANIFEST.files[{index}]")
        require_exact_keys(record, {"path", "bytes", "sha256"}, name="manifest file")
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise ValueError(f"unsafe or duplicate golden path {relative!r}")
        seen.add(relative)
        path = golden_root / relative
        expected_sha = require_sha256(record["sha256"], name=f"{relative}.sha256")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != expected_sha:
            raise ValueError(f"golden manifest mismatch for {relative}")
    fixture_paths = sorted((golden_root / "fixtures").glob("*.json"))
    loaded = {fixture.fixture_id: fixture for fixture in map(load_fixture, fixture_paths)}
    if set(loaded) != set(fixtures):
        raise ValueError("golden fixture inventory mismatch")
    for fixture_id, digest in fixtures.items():
        if loaded[fixture_id].contract_sha256 != digest:
            raise ValueError(f"golden fixture digest mismatch for {fixture_id}")
    manifest_id = "ace-golden-contract-" + sha256_bytes(canonical_json_bytes(files))
    if manifest["manifestId"] != manifest_id:
        raise ValueError("golden manifestId mismatch")
    return manifest_id


def validate_bindings(taps: TapContract, bindings_path: Path) -> None:
    raw = require_object(load_json(bindings_path), name="upstream bindings")
    require_exact_keys(raw, {"schemaVersion", "sourceCommit", "bindings"}, name="bindings")
    if raw["schemaVersion"] != 1 or raw["sourceCommit"] != ACE_SOURCE_REVISION:
        raise ValueError("binding source identity mismatch")
    bindings = raw["bindings"]
    if not isinstance(bindings, list):
        raise ValueError("bindings must be an array")
    identifiers: list[str] = []
    for index, item in enumerate(bindings):
        binding = require_object(item, name=f"bindings[{index}]")
        require_exact_keys(
            binding,
            {"tapId", "strategy", "source", "value"},
            name=f"bindings[{index}]",
        )
        if not all(isinstance(binding[key], str) and binding[key] for key in binding):
            raise ValueError(f"bindings[{index}] contains an empty field")
        identifiers.append(binding["tapId"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate upstream tap binding")
    if set(identifiers) != set(taps.pattern_ids):
        raise ValueError(
            "upstream binding/tap mismatch; "
            f"missing={sorted(set(taps.pattern_ids) - set(identifiers))}, "
            f"unknown={sorted(set(identifiers) - set(taps.pattern_ids))}"
        )


_REFERENCE_TOOL_FILES = (
    "reference/__init__.py",
    "reference/artifacts.py",
    "reference/capture.py",
    "reference/constants.py",
    "reference/contracts.py",
    "reference/input.schema.json",
    "reference/inputs.py",
    "reference/instrumentation.py",
    "reference/jsonio.py",
    "reference/native.py",
    "reference/preflight.py",
    "reference/pyproject.toml",
    "reference/runner.py",
    "reference/upstream-bindings.json",
    "reference/uv.lock",
    "model/__init__.py",
    "model/canonical-packages.json",
    "model/package_format.py",
    "model/source_contract.py",
    "scripts/prng_reference.py",
)


def reference_tool_identity(repository_root: Path) -> dict[str, object]:
    """Hash the exact repository-owned code and contracts used for capture.

    The compact aggregate is embedded in every capture and recomputed during
    replay. A reference implementation edit therefore invalidates old capture
    provenance even if its integer tool version was not advanced accidentally.
    """

    records: list[dict[str, object]] = []
    for relative in _REFERENCE_TOOL_FILES:
        path = repository_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe reference-tool file {relative}")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "version": REFERENCE_TOOL_VERSION,
        "fileCount": len(records),
        "fileSetSha256": sha256_bytes(canonical_json_bytes(records)),
    }
