"""Externally generated random-input bundles for native reference capture.

The bundle carries realized initial Gaussian noise and raw planner words.  The
native process never derives these values from a PyTorch seed, which keeps the
reference independent of backend-specific RNG implementations.
"""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import INPUT_ALGORITHM_ID, INPUT_SCHEMA_VERSION
from .contracts import FixtureContract
from .jsonio import (
    canonical_json_bytes,
    load_json,
    require_exact_keys,
    require_int,
    require_object,
    require_sha256,
    sha256_bytes,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
)


@dataclass(frozen=True, slots=True)
class FloatArtifact:
    path: Path
    relative_path: str
    shape: tuple[int, ...]
    byte_length: int
    sha256: str

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    def read_values(self) -> tuple[float, ...]:
        payload = self.path.read_bytes()
        values = struct.unpack(f"<{self.element_count}f", payload)
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{self.relative_path}: contains non-finite values")
        return values


@dataclass(frozen=True, slots=True)
class WordArtifact:
    path: Path
    relative_path: str
    shape: tuple[int, ...]
    byte_length: int
    sha256: str

    def read_values(self) -> tuple[int, ...]:
        count = math.prod(self.shape)
        return struct.unpack(f"<{count}I", self.path.read_bytes())


@dataclass(frozen=True, slots=True)
class PlannerInputs:
    mapping: str
    words: WordArtifact


@dataclass(frozen=True, slots=True)
class InputBundle:
    manifest_path: Path
    manifest_sha256: str
    fixture_id: str
    fixture_contract_sha256: str
    algorithm_id: str
    gaussian_mapping: str
    initial_noise: FloatArtifact
    planner: PlannerInputs | None

    def identity(self) -> dict[str, object]:
        planner: dict[str, object] | None = None
        if self.planner is not None:
            planner = {
                "mapping": self.planner.mapping,
                "drawStart": 0,
                "wordCapacity": self.planner.words.shape[0],
                "byteLength": self.planner.words.byte_length,
                "sha256": self.planner.words.sha256,
            }
        return {
            "manifestSha256": self.manifest_sha256,
            "algorithmId": self.algorithm_id,
            "gaussianMapping": self.gaussian_mapping,
            "initialNoise": {
                "shape": list(self.initial_noise.shape),
                "byteLength": self.initial_noise.byte_length,
                "sha256": self.initial_noise.sha256,
            },
            "plannerSampling": planner,
        }


_ARTIFACT_KEYS = {"path", "dtype", "shape", "byteLength", "sha256"}


def _artifact_fields(
    bundle_root: Path,
    raw: object,
    *,
    name: str,
    expected_dtype: str,
    element_bytes: int,
) -> tuple[Path, str, tuple[int, ...], int, str]:
    value = require_object(raw, name=name)
    require_exact_keys(value, _ARTIFACT_KEYS, name=name)
    relative = value["path"]
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name}.path must be non-empty")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name}.path is unsafe")
    if value["dtype"] != expected_dtype:
        raise ValueError(f"{name}.dtype must be {expected_dtype}")
    if (
        not isinstance(value["shape"], list)
        or not value["shape"]
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in value["shape"]
        )
    ):
        raise ValueError(f"{name}.shape must contain positive integers")
    shape = tuple(value["shape"])
    byte_length = require_int(value["byteLength"], name=f"{name}.byteLength")
    if byte_length != math.prod(shape) * element_bytes:
        raise ValueError(f"{name}: byteLength does not match dtype/shape")
    expected_sha = require_sha256(value["sha256"], name=f"{name}.sha256")
    path = bundle_root / pure
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name}: missing or unsafe payload {path}")
    if path.stat().st_size != byte_length or sha256_file(path) != expected_sha:
        raise ValueError(f"{name}: payload identity mismatch")
    return path, pure.as_posix(), shape, byte_length, expected_sha


def _float_artifact(bundle_root: Path, raw: object, *, name: str) -> FloatArtifact:
    fields = _artifact_fields(
        bundle_root,
        raw,
        name=name,
        expected_dtype="float32-le",
        element_bytes=4,
    )
    artifact = FloatArtifact(*fields)
    artifact.read_values()
    return artifact


def _word_artifact(bundle_root: Path, raw: object, *, name: str) -> WordArtifact:
    return WordArtifact(
        *_artifact_fields(
            bundle_root,
            raw,
            name=name,
            expected_dtype="uint32-le",
            element_bytes=4,
        )
    )


def load_prng_oracle() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "prng_reference.py"
    spec = importlib.util.spec_from_file_location("ace_prng_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load pinned PRNG oracle: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_vectors()
    return module


def _validate_generated_inputs(bundle: InputBundle, fixture: FixtureContract) -> None:
    oracle = load_prng_oracle()
    seed = fixture.contract["random"]["userSeed"]
    for index, actual in enumerate(bundle.initial_noise.read_values()):
        word = oracle.random_word(seed, "diffusion-noise", index)
        expected = oracle.gaussian_from_word(word)
        if struct.pack("<f", actual) != struct.pack("<f", expected):
            raise ValueError(f"initial noise differs from ace-seed-v1 at element {index}")
    if bundle.planner is not None:
        for index, actual in enumerate(bundle.planner.words.read_values()):
            expected = oracle.random_word(seed, "planner-sampling", index)
            if actual != expected:
                raise ValueError(f"planner word differs from ace-seed-v1 at draw {index}")


def load_input_bundle(path: Path, fixture: FixtureContract) -> InputBundle:
    raw = require_object(load_json(path), name="reference input bundle")
    require_exact_keys(
        raw,
        {
            "schemaVersion",
            "fixtureId",
            "fixtureContractSha256",
            "algorithmId",
            "gaussianMapping",
            "initialNoise",
            "plannerSampling",
        },
        name="reference input bundle",
    )
    if raw["schemaVersion"] != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported input-bundle schema")
    if path.read_bytes() != canonical_json_bytes(raw, newline=True):
        raise ValueError("reference input bundle must use canonical JSON bytes")
    if raw["fixtureId"] != fixture.fixture_id:
        raise ValueError("input bundle belongs to another fixture")
    if raw["fixtureContractSha256"] != fixture.contract_sha256:
        raise ValueError("input bundle belongs to another fixture contract")
    if raw["algorithmId"] != INPUT_ALGORITHM_ID:
        raise ValueError("input bundle uses an unknown random algorithm")

    random_contract = fixture.contract["random"]
    gaussian_mapping = raw["gaussianMapping"]
    if (
        random_contract["gaussianMapping"] == "deferred"
        or random_contract["randomTransformCaptureStatus"]
        != "ready-for-native-capture"
    ):
        raise ValueError(
            "fixture is not authorized for capture: Gaussian mapping remains deferred"
        )
    if gaussian_mapping != random_contract["gaussianMapping"]:
        raise ValueError("input Gaussian mapping differs from fixture contract")

    root = path.parent
    initial = _float_artifact(root, raw["initialNoise"], name="initialNoise")
    expected_noise = random_contract["initialNoise"]
    if tuple(expected_noise["shape"]) != initial.shape:
        raise ValueError("initial-noise shape differs from fixture contract")
    if expected_noise["sha256"] is not None and expected_noise["sha256"] != initial.sha256:
        raise ValueError("initial-noise SHA-256 differs from fixture contract")

    planner_raw = raw["plannerSampling"]
    planner: PlannerInputs | None
    if fixture.planner_enabled:
        planner_value = require_object(planner_raw, name="plannerSampling")
        require_exact_keys(
            planner_value,
            {"mapping", "drawStart", "words"},
            name="plannerSampling",
        )
        categorical_mapping = random_contract["categoricalMapping"]
        if categorical_mapping == "deferred":
            raise ValueError(
                "fixture is not authorized for capture: categorical mapping remains deferred"
            )
        if planner_value["mapping"] != categorical_mapping:
            raise ValueError("planner mapping differs from fixture contract")
        if planner_value["drawStart"] != 0:
            raise ValueError("planner word bundle must begin at global draw ordinal zero")
        words = _word_artifact(root, planner_value["words"], name="plannerWords")
        if len(words.shape) != 1 or words.shape[0] < fixture.expected["semanticCodes"]:
            raise ValueError("planner word capacity cannot cover semantic generation")
        planner = PlannerInputs(str(planner_value["mapping"]), words)
    else:
        if planner_raw is not None:
            raise ValueError("direct fixture must not carry planner draws")
        planner = None

    bundle = InputBundle(
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        fixture_id=fixture.fixture_id,
        fixture_contract_sha256=fixture.contract_sha256,
        algorithm_id=INPUT_ALGORITHM_ID,
        gaussian_mapping=str(gaussian_mapping),
        initial_noise=initial,
        planner=planner,
    )
    _validate_generated_inputs(bundle, fixture)
    return bundle


def expected_input_bundle_identity(
    fixture: FixtureContract,
    *,
    planner_word_capacity: int | None = None,
) -> dict[str, object]:
    """Derive the one canonical input identity accepted for replay.

    Planner capacity is an explicit capture parameter; every payload byte and
    the canonical manifest hash are nevertheless derived from the committed
    fixture and independent PRNG oracle.
    """

    manifest, noise_payload, planner_payload = _build_input_material(
        fixture,
        planner_word_capacity=planner_word_capacity,
    )
    planner = manifest["plannerSampling"]
    planner_identity: dict[str, object] | None = None
    if isinstance(planner, dict):
        words = require_object(planner["words"], name="plannerSampling.words")
        planner_identity = {
            "mapping": planner["mapping"],
            "drawStart": planner["drawStart"],
            "wordCapacity": words["shape"][0],
            "byteLength": len(planner_payload or b""),
            "sha256": words["sha256"],
        }
    return {
        "manifestSha256": sha256_bytes(canonical_json_bytes(manifest, newline=True)),
        "algorithmId": manifest["algorithmId"],
        "gaussianMapping": manifest["gaussianMapping"],
        "initialNoise": {
            "shape": manifest["initialNoise"]["shape"],
            "byteLength": len(noise_payload),
            "sha256": manifest["initialNoise"]["sha256"],
        },
        "plannerSampling": planner_identity,
    }


def _build_input_material(
    fixture: FixtureContract,
    *,
    planner_word_capacity: int | None,
) -> tuple[dict[str, object], bytes, bytes | None]:
    random_contract = fixture.contract["random"]
    if random_contract["randomTransformCaptureStatus"] != "ready-for-native-capture":
        raise ValueError("fixture random transforms are not ready for native capture")
    if random_contract["gaussianMapping"] == "deferred":
        raise ValueError("fixture Gaussian mapping remains deferred")
    if fixture.planner_enabled and random_contract["categoricalMapping"] == "deferred":
        raise ValueError("fixture categorical mapping remains deferred")
    if fixture.planner_enabled:
        if (
            not isinstance(planner_word_capacity, int)
            or isinstance(planner_word_capacity, bool)
            or planner_word_capacity < fixture.expected["semanticCodes"]
            or planner_word_capacity > 1_000_000
        ):
            raise ValueError("planner word capacity is outside the accepted range")
    elif planner_word_capacity is not None:
        raise ValueError("direct fixture cannot declare planner word capacity")

    oracle = load_prng_oracle()
    seed = random_contract["userSeed"]
    noise_count = math.prod(random_contract["initialNoise"]["shape"])
    noise_payload = b"".join(
        struct.pack(
            "<f",
            oracle.gaussian_from_word(
                oracle.random_word(seed, "diffusion-noise", index)
            ),
        )
        for index in range(noise_count)
    )
    planner_payload: bytes | None = None
    planner_record: dict[str, object] | None = None
    if fixture.planner_enabled:
        assert planner_word_capacity is not None
        planner_payload = b"".join(
            struct.pack(
                "<I", oracle.random_word(seed, "planner-sampling", index)
            )
            for index in range(planner_word_capacity)
        )
        planner_record = {
            "mapping": random_contract["categoricalMapping"],
            "drawStart": 0,
            "words": {
                "path": "planner-words.u32le",
                "dtype": "uint32-le",
                "shape": [planner_word_capacity],
                "byteLength": len(planner_payload),
                "sha256": sha256_bytes(planner_payload),
            },
        }
    manifest = {
        "schemaVersion": INPUT_SCHEMA_VERSION,
        "fixtureId": fixture.fixture_id,
        "fixtureContractSha256": fixture.contract_sha256,
        "algorithmId": INPUT_ALGORITHM_ID,
        "gaussianMapping": random_contract["gaussianMapping"],
        "initialNoise": {
            "path": "initial-noise.f32le",
            "dtype": "float32-le",
            "shape": random_contract["initialNoise"]["shape"],
            "byteLength": len(noise_payload),
            "sha256": sha256_bytes(noise_payload),
        },
        "plannerSampling": planner_record,
    }
    return manifest, noise_payload, planner_payload


def prepare_input_bundle(
    fixture: FixtureContract,
    output_root: Path,
    *,
    planner_word_capacity: int = 4096,
) -> Path:
    """Materialize deterministic browser-defined inputs in ignored storage."""

    effective_capacity = planner_word_capacity if fixture.planner_enabled else None
    manifest, noise_payload, planner_payload = _build_input_material(
        fixture,
        planner_word_capacity=effective_capacity,
    )
    if output_root.exists():
        raise ValueError(f"input bundle destination already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", suffix=".partial", dir=output_root.parent)
    )
    try:
        write_bytes_atomic(partial / "initial-noise.f32le", noise_payload)
        if planner_payload is not None:
            write_bytes_atomic(partial / "planner-words.u32le", planner_payload)
        write_json_atomic(partial / "inputs.json", manifest)
        load_input_bundle(partial / "inputs.json", fixture)
        os.replace(partial, output_root)
        return output_root / "inputs.json"
    finally:
        shutil.rmtree(partial, ignore_errors=True)
