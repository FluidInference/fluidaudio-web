"""Fail-closed source, model, package, and random-input preflight."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    ACE_MODEL_REVISION,
    ACE_SOURCE_REPOSITORY,
    ACE_SOURCE_REVISION,
    PLANNER_MODEL_REVISION,
)
from .contracts import (
    FixtureContract,
    TapContract,
    load_fixture,
    reference_tool_identity,
    validate_bindings,
    verify_golden_manifest,
)
from .inputs import InputBundle, load_input_bundle
from .jsonio import canonical_json_bytes, load_json, require_object, sha256_bytes, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class PreflightResult:
    fixture: FixtureContract
    taps: TapContract
    golden_manifest_id: str
    reference_tool: dict[str, object]
    source: dict[str, str]
    browser_package: dict[str, object]
    inputs: InputBundle | None
    deep_payload_hashes: bool
    blockers: tuple[str, ...]

    @property
    def capture_authorized(self) -> bool:
        return self.deep_payload_hashes and self.inputs is not None and not self.blockers

    def report(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "captureAuthorized": self.capture_authorized,
            "deepPayloadHashes": self.deep_payload_hashes,
            "fixtureId": self.fixture.fixture_id,
            "fixtureContractSha256": self.fixture.contract_sha256,
            "goldenManifestId": self.golden_manifest_id,
            "referenceTool": self.reference_tool,
            "source": self.source,
            "browserPackage": self.browser_package,
            "inputBundle": (
                None
                if self.inputs is None
                else {
                    "path": str(self.inputs.manifest_path),
                    "algorithmId": self.inputs.algorithm_id,
                    "gaussianMapping": self.inputs.gaussian_mapping,
                    "initialNoiseSha256": self.inputs.initial_noise.sha256,
                    "plannerWordsAvailable": self.inputs.planner is not None,
                    "plannerWordsSha256": (
                        self.inputs.planner.words.sha256
                        if self.inputs.planner is not None
                        else None
                    ),
                }
            ),
            "blockers": list(self.blockers),
        }


def _git(source_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "")
        raise ValueError(f"cannot inspect ACE source Git identity: {detail or error}") from error
    return completed.stdout.strip()


def verify_source_checkout(source_root: Path) -> dict[str, str]:
    if not (source_root / "acestep").is_dir():
        raise ValueError(f"not an ACE-Step source checkout: {source_root}")
    commit = _git(source_root, "rev-parse", "HEAD")
    if commit != ACE_SOURCE_REVISION:
        raise ValueError(f"ACE source is {commit}, expected {ACE_SOURCE_REVISION}")
    status = _git(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(
            "ACE source checkout is dirty; reference capture refuses modified or "
            f"untracked source files:\n{status}"
        )
    remote = _git(source_root, "remote", "get-url", "origin")
    normalized = remote.removesuffix(".git").lower()
    if not (
        normalized.endswith("github.com/ace-step/ace-step-1.5")
        or normalized == ACE_SOURCE_REPOSITORY.removesuffix(".git").lower()
    ):
        raise ValueError(f"unexpected ACE source remote {remote!r}")
    return {
        "repository": ACE_SOURCE_REPOSITORY,
        "commit": commit,
        "checkoutRemote": remote,
    }


def _load_model_contract_modules() -> tuple[Any, Any]:
    root_text = str(REPOSITORY_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from model import package_format, source_contract

    return source_contract, package_format


def verify_raw_model_cache(cache_root: Path, *, verify_hashes: bool) -> dict[str, object]:
    source_contract, _ = _load_model_contract_modules()
    verified_bytes = 0
    resolved = dict(authenticated_source_artifacts(cache_root))
    for artifact in source_contract.SOURCE_ARTIFACTS:
        path = resolved[artifact.key]
        actual_bytes = path.stat().st_size
        if actual_bytes != artifact.byte_length:
            raise ValueError(
                f"raw model artifact length mismatch for {artifact.key}: "
                f"{actual_bytes} != {artifact.byte_length}"
            )
        if verify_hashes and sha256_file(path) != artifact.sha256:
            raise ValueError(f"raw model artifact SHA-256 mismatch for {artifact.key}")
        verified_bytes += actual_bytes
    return {
        "aceSnapshot": ACE_MODEL_REVISION,
        "plannerSnapshot": PLANNER_MODEL_REVISION,
        "artifactCount": len(source_contract.SOURCE_ARTIFACTS),
        "artifactBytes": verified_bytes,
        "payloadHashesVerified": verify_hashes,
        "artifactSetSha256": raw_model_contract_identity(),
    }


def raw_model_contract_identity() -> str:
    source_contract, _ = _load_model_contract_modules()
    records = [
        {
            "key": artifact.key,
            "repository": artifact.repository,
            "revision": artifact.revision,
            "path": artifact.path,
            "byteLength": artifact.byte_length,
            "sha256": artifact.sha256,
        }
        for artifact in source_contract.SOURCE_ARTIFACTS
    ]
    return sha256_bytes(canonical_json_bytes(records))


def authenticated_source_artifacts(cache_root: Path) -> tuple[tuple[str, Path], ...]:
    """Resolve only the individual immutable files declared by source contract."""

    source_contract, _ = _load_model_contract_modules()
    resolved: list[tuple[str, Path]] = []
    if cache_root.is_symlink():
        raise ValueError("raw model cache root must not be a symlink")
    root = cache_root.resolve()
    for artifact in source_contract.SOURCE_ARTIFACTS:
        path = artifact.cache_path(cache_root)
        try:
            relative = path.relative_to(cache_root)
        except ValueError as error:
            raise ValueError(f"raw model artifact escapes cache root: {artifact.key}") from error
        cursor = cache_root
        if any((cursor := cursor / part).is_symlink() for part in relative.parts):
            raise ValueError(f"raw model artifact path contains a symlink: {artifact.key}")
        if not path.is_file():
            raise ValueError(f"missing or unsafe raw model artifact {artifact.key}: {path}")
        actual = path.resolve(strict=True)
        if root not in actual.parents:
            raise ValueError(f"raw model artifact escapes cache root: {artifact.key}")
        resolved.append((artifact.key, actual))
    return tuple(resolved)


def verify_browser_package(
    package_root: Path,
    *,
    verify_hashes: bool,
) -> dict[str, object]:
    _, package_format = _load_model_contract_modules()
    canonical = require_object(
        load_json(REPOSITORY_ROOT / "model" / "canonical-packages.json"),
        name="canonical-packages.json",
    )
    reference = require_object(canonical["packages"]["reference"], name="reference package")
    expected_sha = reference["manifestSha256"]
    manifest_path = package_root / "manifest.json"
    actual_sha = sha256_file(manifest_path)
    if actual_sha != expected_sha or manifest_path.stat().st_size != reference["manifestBytes"]:
        raise ValueError("browser reference package manifest is not canonical")
    manifest = package_format.verify_package(package_root, verify_hashes=verify_hashes)
    provenance = manifest["provenance"]
    if (
        manifest["profile"] != "reference"
        or provenance["referenceCommit"] != ACE_SOURCE_REVISION
        or provenance["aceSnapshot"] != ACE_MODEL_REVISION
        or provenance["plannerSnapshot"] != PLANNER_MODEL_REVISION
    ):
        raise ValueError("browser package provenance differs from pinned truth")
    return {
        "format": manifest["format"],
        "profile": manifest["profile"],
        "manifestSha256": actual_sha,
        "manifestBytes": manifest_path.stat().st_size,
        "packageBytesIncludingManifest": reference["packageBytesIncludingManifest"],
        "payloadHashesVerified": verify_hashes,
    }


def run_preflight(
    *,
    source_root: Path,
    model_cache: Path,
    browser_package: Path,
    fixture_path: Path,
    input_manifest: Path | None,
    deep_payload_hashes: bool,
) -> PreflightResult:
    golden_root = REPOSITORY_ROOT / "golden"
    golden_manifest_id = verify_golden_manifest(golden_root)
    reference_tool = reference_tool_identity(REPOSITORY_ROOT)
    fixture = load_fixture(fixture_path)
    taps = TapContract(golden_root / "taps.json")
    validate_bindings(taps, REPOSITORY_ROOT / "reference" / "upstream-bindings.json")
    source = verify_source_checkout(source_root)
    raw_models = verify_raw_model_cache(model_cache, verify_hashes=deep_payload_hashes)
    source.update(
        {
            "aceModelRevision": str(raw_models["aceSnapshot"]),
            "plannerModelRevision": str(raw_models["plannerSnapshot"]),
            "rawModelArtifactSetSha256": str(raw_models["artifactSetSha256"]),
        }
    )
    package = verify_browser_package(browser_package, verify_hashes=deep_payload_hashes)
    blockers: list[str] = []
    inputs: InputBundle | None = None
    if input_manifest is None:
        blockers.append("an externally generated, fixture-pinned input bundle is required")
    else:
        inputs = load_input_bundle(input_manifest, fixture)
    if not deep_payload_hashes:
        blockers.append("contracts-only mode does not authenticate model/package payload bytes")
    return PreflightResult(
        fixture=fixture,
        taps=taps,
        golden_manifest_id=golden_manifest_id,
        reference_tool=reference_tool,
        source=source,
        browser_package=package,
        inputs=inputs,
        deep_payload_hashes=deep_payload_hashes,
        blockers=tuple(blockers),
    )
