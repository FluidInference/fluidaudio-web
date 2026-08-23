from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from reference.artifacts import (
    CaptureReplayContract,
    CaptureWriter,
    compare_captures,
    load_environment_receipt,
    verify_capture,
    write_environment_receipt,
)
from reference.constants import ACE_SOURCE_REVISION
from reference.contracts import (
    FixtureContract,
    TapContract,
    validate_bindings,
    verify_golden_manifest,
)
from reference.inputs import load_input_bundle, prepare_input_bundle
from reference.instrumentation import (
    pinned_tiled_decode_kwargs,
    requires_nonchunked_vae_capture,
)
from reference.jsonio import load_json, sha256_bytes, write_json_atomic
from reference.native import stable_descending_token_order, top_p_keep_mask
from reference import preflight


ROOT = Path(__file__).resolve().parents[2]


def _fixture(root: Path) -> FixtureContract:
    contract = {
        "planner": {"enabled": False},
        "request": {"durationSeconds": 1},
        "expected": {
            "latentFrames": 1,
            "patchedTokens": 1,
            "semanticCodes": 0,
            "audioSamplesPerChannel": 1,
        },
        "random": {
            "algorithmId": "ace-seed-v1",
            "gaussianMapping": "box-muller-v1",
            "categoricalMapping": "inverse-cdf-v1",
            "randomTransformCaptureStatus": "ready-for-native-capture",
            "initialNoise": {"shape": [1, 1, 1], "sha256": "0" * 64},
        },
    }
    return FixtureContract(root / "fixture.json", "fixture", "1" * 64, contract)


def _taps(root: Path) -> TapContract:
    path = root / "taps.json"
    write_json_atomic(
        path,
        {
            "schemaVersion": 1,
            "layout": "contiguous-little-endian",
            "symbols": {},
            "captureRules": {},
            "taps": [
                {
                    "id": "diffusion.initial-noise",
                    "stage": "dit",
                    "required": "all",
                    "dtype": "float32",
                    "shape": ["B", "T", 1],
                    "capture": "full",
                },
                {
                    "id": "audio.final-output",
                    "stage": "audio",
                    "required": "all",
                    "dtype": "float32",
                    "shape": ["B", 2, "S"],
                    "capture": "full",
                },
            ],
        },
    )
    return TapContract(path)


def _make_capture(root: Path, capture_id: str) -> Path:
    fixture = _fixture(root)
    taps = _taps(root)
    writer = CaptureWriter(
        output_root=root / "out",
        capture_id=capture_id,
        fixture=fixture,
        taps=taps,
    )
    writer.add_tensor(
        "diffusion.initial-noise",
        struct.pack("<f", 0.25),
        dtype="float32",
        logical_shape=[1, 1, 1],
    )
    audio = struct.pack("<ff", 0.5, -0.5)
    writer.add_tensor(
        "audio.final-output",
        audio,
        dtype="float32",
        logical_shape=[1, 2, 1],
    )
    writer.add_float32_stereo_output(audio, samples_per_channel=1)
    replay = _replay(fixture, taps)
    return writer.finalize(
        replay=replay,
        random_injection={
            "algorithmId": "ace-seed-v1",
            "gaussianMapping": "box-muller-v1",
            "initialNoiseInjected": True,
            "initialNoiseSha256": sha256_bytes(struct.pack("<f", 0.25)),
            "plannerWordsInjected": False,
        },
        environment=_environment(),
    )


def _environment() -> dict[str, object]:
    return {
        "backend": "pytorch-eager",
        "attentionImplementation": "eager",
        "dcwBackend": "pytorch-wavelets-haar",
        "vaeComputeDtype": "float32",
        "vaeChunkSize": 256,
        "vaeOverlap": 64,
        "vaeOffloadWavToCpu": True,
        "device": "cuda",
        "python": "3.12.11",
        "platform": "test-platform",
        "accelerator": {
            "backend": "cuda",
            "deviceIndex": 0,
            "name": "test-gpu",
            "capability": "9.0",
            "runtimeVersion": "12.8",
            "driverVersion": "test-driver",
            "totalMemoryBytes": 1,
        },
        "packages": {
            "torch": "2.10.0+cu128",
            "pytorchWavelets": "1.3.0",
            "pyWavelets": "1.9.0",
            "transformers": "4.57.6",
            "diffusers": "0.37.1",
            "numpy": "2.3.5",
            "einops": "0.8.2",
        },
        "deterministicAlgorithms": True,
        "tf32": False,
    }


def _replay(fixture: FixtureContract, taps: TapContract) -> CaptureReplayContract:
    noise_sha = sha256_bytes(struct.pack("<f", 0.25))
    fixture.contract["random"]["initialNoise"]["sha256"] = noise_sha
    return CaptureReplayContract(
        fixture=fixture,
        taps=taps,
        golden_manifest_id="ace-golden-contract-" + "2" * 64,
        reference_tool={"version": 2, "fileCount": 1, "fileSetSha256": "3" * 64},
        source={
            "repository": "https://github.com/ace-step/ACE-Step-1.5.git",
            "commit": ACE_SOURCE_REVISION,
            "checkoutRemote": "test",
            "aceModelRevision": "4" * 40,
            "plannerModelRevision": "5" * 40,
            "rawModelArtifactSetSha256": "8" * 64,
        },
        browser_package={
            "format": "ace-step-webgpu-v1",
            "profile": "reference",
            "manifestSha256": "6" * 64,
            "manifestBytes": 1,
            "packageBytesIncludingManifest": 2,
            "payloadHashesVerified": True,
        },
        input_bundle={
            "manifestSha256": "7" * 64,
            "algorithmId": "ace-seed-v1",
            "gaussianMapping": "box-muller-v1",
            "initialNoise": {
                "shape": [1, 1, 1],
                "byteLength": 4,
                "sha256": noise_sha,
            },
            "plannerSampling": None,
        },
        environment=_environment(),
    )


def _rehash_capture(path: Path, mutate) -> None:
    value = load_json(path)
    mutate(value)
    identity = {
        key: item
        for key, item in value.items()
        if key not in {"captureId", "captureIdentitySha256"}
    }
    value["captureIdentitySha256"] = sha256_bytes(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    )
    write_json_atomic(path, value)


class ReferenceContractTests(unittest.TestCase):
    def test_committed_golden_manifest_and_bindings_are_exact(self) -> None:
        manifest_id = verify_golden_manifest(ROOT / "golden")
        self.assertTrue(manifest_id.startswith("ace-golden-contract-"))
        taps = TapContract(ROOT / "golden" / "taps.json")
        validate_bindings(taps, ROOT / "reference" / "upstream-bindings.json")
        self.assertIn("dit.step.7.layer.23.output", taps.specs)

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_json(path)

    def test_deferred_mapping_cannot_authorize_input_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture(root)
            fixture.contract["random"]["gaussianMapping"] = "deferred"
            fixture.contract["random"]["randomTransformCaptureStatus"] = (
                "blocked-until-mappings-land"
            )
            payload = struct.pack("<f", 0.0)
            (root / "noise.bin").write_bytes(payload)
            write_json_atomic(
                root / "inputs.json",
                {
                    "schemaVersion": 1,
                    "fixtureId": "fixture",
                    "fixtureContractSha256": "1" * 64,
                    "algorithmId": "ace-seed-v1",
                    "gaussianMapping": "box-muller-v1",
                    "initialNoise": {
                        "path": "noise.bin",
                        "dtype": "float32-le",
                        "shape": [1, 1, 1],
                        "byteLength": 4,
                        "sha256": sha256_bytes(payload),
                    },
                    "plannerSampling": None,
                },
            )
            with self.assertRaisesRegex(ValueError, "Gaussian mapping remains deferred"):
                load_input_bundle(root / "inputs.json", fixture)

    def test_prepare_and_validate_direct_input_bundle(self) -> None:
        fixture = FixtureContract(
            ROOT / "golden" / "fixtures" / "direct-instrumental-short.json",
            "direct-instrumental-short",
            load_json(
                ROOT / "golden" / "fixtures" / "direct-instrumental-short.json"
            )["contractSha256"],
            load_json(
                ROOT / "golden" / "fixtures" / "direct-instrumental-short.json"
            )["contract"],
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_input_bundle(fixture, Path(directory) / "inputs")
            bundle = load_input_bundle(manifest, fixture)
            self.assertEqual(
                bundle.initial_noise.shape,
                tuple(fixture.contract["random"]["initialNoise"]["shape"]),
            )
            self.assertIsNone(bundle.planner)

    def test_prepare_and_validate_planner_word_bundle(self) -> None:
        raw = load_json(ROOT / "golden" / "fixtures" / "planner-lyrics-short.json")
        fixture = FixtureContract(
            ROOT / "golden" / "fixtures" / "planner-lyrics-short.json",
            raw["fixtureId"],
            raw["contractSha256"],
            raw["contract"],
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_input_bundle(
                fixture,
                Path(directory) / "inputs",
                planner_word_capacity=128,
            )
            bundle = load_input_bundle(manifest, fixture)
            self.assertIsNotNone(bundle.planner)
            assert bundle.planner is not None
            self.assertEqual(bundle.planner.words.shape, (128,))
            self.assertEqual(bundle.planner.mapping, "u32-midpoint-binary64-cdf-v1")

            payload = bytearray(bundle.planner.words.path.read_bytes())
            payload[0] ^= 1
            bundle.planner.words.path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "payload identity mismatch"):
                load_input_bundle(manifest, fixture)

    def test_capture_roundtrip_and_fresh_process_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _make_capture(root, "run-01")
            second = _make_capture(root, "run-02")
            fixture = _fixture(root)
            taps = _taps(root)
            verified = verify_capture(first, replay=_replay(fixture, taps))
            self.assertEqual(verified["fixtureId"], "fixture")
            compare_captures(first, second, replay=_replay(fixture, taps))

    def test_environment_receipt_is_external_and_capture_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _make_capture(root, "run-01")
            receipt = write_environment_receipt(capture)
            self.assertEqual(receipt.parent, capture.parent)
            identity = load_json(capture / "capture.json")["captureIdentitySha256"]
            self.assertEqual(
                load_environment_receipt(
                    receipt,
                    capture_identity_sha256=identity,
                ),
                _environment(),
            )
            with self.assertRaisesRegex(ValueError, "another capture identity"):
                load_environment_receipt(
                    receipt,
                    capture_identity_sha256="f" * 64,
                )

    def test_capture_replay_rejects_tampered_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _make_capture(root, "run-01")
            tensor = capture / "tensors" / "audio.final-output.bin"
            tensor.write_bytes(b"\0" * tensor.stat().st_size)
            with self.assertRaisesRegex(ValueError, "artifact identity mismatch"):
                verify_capture(capture, replay=_replay(_fixture(root), _taps(root)))

    def test_capture_replay_rejects_coordinated_provenance_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _make_capture(root, "run-01")
            _rehash_capture(
                capture / "capture.json",
                lambda value: value["source"].__setitem__("commit", "f" * 40),
            )
            with self.assertRaisesRegex(ValueError, "authenticated provenance"):
                verify_capture(capture, replay=_replay(_fixture(root), _taps(root)))

    def test_capture_replay_rejects_coordinated_environment_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _make_capture(root, "run-01")
            _rehash_capture(
                capture / "capture.json",
                lambda value: value["environment"]["accelerator"].__setitem__(
                    "driverVersion", "forged-driver"
                ),
            )
            with self.assertRaisesRegex(ValueError, "authenticated provenance"):
                verify_capture(capture, replay=_replay(_fixture(root), _taps(root)))

    def test_capture_replay_rejects_rehashed_malformed_tensor_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = _make_capture(root, "run-01")

            def mutate(value) -> None:
                target = next(
                    record
                    for record in value["artifacts"]
                    if record["tapId"] == "diffusion.initial-noise"
                )
                target["storedShape"] = [1, 2, 1]
                material = {
                    "artifacts": value["artifacts"],
                    "auxiliary": value["auxiliary"],
                }
                value["artifactSetSha256"] = sha256_bytes(
                    json.dumps(
                        material,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                )

            _rehash_capture(capture / "capture.json", mutate)
            with self.assertRaisesRegex(ValueError, "byteLength differs"):
                verify_capture(capture, replay=_replay(_fixture(root), _taps(root)))

    def test_planner_stable_tie_and_top_p_boundary_semantics(self) -> None:
        self.assertEqual(stable_descending_token_order([2.0, 3.0, 3.0, 1.0]), (1, 2, 0, 3))
        self.assertEqual(top_p_keep_mask([0.0, 0.0, 0.0, 0.0], 0.5), (True, True, True, False))
        self.assertEqual(top_p_keep_mask([0.0, 0.0, 0.0], 1.0 / 3.0), (True, True, False))

    def test_committed_lyric_tap_dimensions_are_explicit(self) -> None:
        taps = TapContract(ROOT / "golden" / "taps.json")
        self.assertEqual(taps.specs["lyric.embedding.input"].shape, ("B", "LL", 1024))
        self.assertEqual(taps.specs["condition.lyric.layer.0"].shape, ("B", "LL", 2048))
        self.assertEqual(taps.specs["condition.lyric.final"].shape, ("B", "LL", 2048))

    def test_nonchunked_vae_capture_follows_required_inventory(self) -> None:
        class Writer:
            pass

        class Capture:
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _fixture(root)
            taps = _taps(root)
            # Inject a synthetic tap with the production short-fixture rule.
            taps.specs["vae.nonchunked.output"] = type(next(iter(taps.specs.values())))(
                tap_id="vae.nonchunked.output",
                stage="vae",
                required="short-fixtures",
                dtype="float32",
                shape=("B", 2, "S"),
                capture="full",
            )
            writer = Writer()
            writer.fixture = fixture
            writer.taps = taps
            capture = Capture()
            capture.writer = writer
            self.assertTrue(requires_nonchunked_vae_capture(capture))
            fixture.contract["request"]["durationSeconds"] = 31
            self.assertFalse(requires_nonchunked_vae_capture(capture))

    def test_tiled_vae_decode_controls_are_pinned(self) -> None:
        self.assertEqual(
            pinned_tiled_decode_kwargs({"source": "fixture"}),
            {
                "source": "fixture",
                "chunk_size": 256,
                "overlap": 64,
                "offload_wav_to_cpu": True,
            },
        )
        for key in ("chunk_size", "overlap", "offload_wav_to_cpu"):
            with self.assertRaisesRegex(ValueError, "adaptive tiled-decode"):
                pinned_tiled_decode_kwargs({key: 1})

    def test_authenticated_source_artifacts_reject_symlinks(self) -> None:
        class Artifact:
            key = "one"

            @staticmethod
            def cache_path(root: Path) -> Path:
                return root / "one.bin"

        class SourceContract:
            SOURCE_ARTIFACTS = (Artifact(),)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"payload")
            (root / "one.bin").symlink_to(target)
            original = preflight._load_model_contract_modules
            preflight._load_model_contract_modules = lambda: (SourceContract, object())
            try:
                with self.assertRaisesRegex(ValueError, "contains a symlink"):
                    preflight.authenticated_source_artifacts(root)
            finally:
                preflight._load_model_contract_modules = original

    def test_shape_contract_rejects_wrong_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = CaptureWriter(
                output_root=root / "out",
                capture_id="run-01",
                fixture=_fixture(root),
                taps=_taps(root),
            )
            with self.assertRaisesRegex(ValueError, "expected T=1"):
                writer.add_tensor(
                    "diffusion.initial-noise",
                    struct.pack("<ff", 0.0, 0.0),
                    dtype="float32",
                    logical_shape=[1, 2, 1],
                )
            writer.abort()


if __name__ == "__main__":
    unittest.main()
