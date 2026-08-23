from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from collections import Counter


MODEL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODEL_ROOT.parent
sys.path.insert(0, str(MODEL_ROOT))

from convert import verify_cached_sources  # noqa: E402
from conversion_plan import (  # noqa: E402
    ACE_ENCODE_ONLY_PARAMETER_COUNT,
    ACE_ENCODE_ONLY_PREFIX_CONTRACT,
    ACE_ENCODE_ONLY_SOURCE_BYTES,
    ACE_ENCODE_ONLY_TENSOR_COUNT,
    DIT_GEMM_TILE_LAYOUT,
    DIT_GEMM_TILE_TRANSFORMATION,
    DIT_GEMM_WEIGHT_NAMES,
    VAE_CHANNEL_VECTOR_LAYOUT,
    VAE_CHANNEL_VECTOR_TRANSFORMATION,
    VAE_CONV1D_LAYOUT,
    VAE_CONV1D_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_LAYOUT,
    VAE_CONV_TRANSPOSE1D_TRANSFORMATION,
)
from package_format import sha256_file, verify_package  # noqa: E402
from safetensors_mmap import SafetensorsFile  # noqa: E402
from source_contract import ARTIFACT_BY_KEY  # noqa: E402


CANONICAL = MODEL_ROOT / "canonical-packages.json"


class CanonicalPackageContractTests(unittest.TestCase):
    def test_canonical_identity_record_is_closed_and_complete(self) -> None:
        record = json.loads(CANONICAL.read_text(encoding="utf-8"))
        self.assertEqual(
            set(record),
            {
                "schemaVersion",
                "packageFormat",
                "converterRevision",
                "sourceTensorCount",
                "logicalTensorCount",
                "tensorRecordCount",
                "fileRecordCount",
                "weightShardFileCount",
                "packages",
                "productionPackages",
                "replacementPackages",
                "verification",
            },
        )
        self.assertEqual(record["schemaVersion"], 3)
        self.assertEqual(record["packageFormat"], "ace-step-webgpu-v1")
        self.assertEqual(record["converterRevision"], 4)
        self.assertEqual(record["sourceTensorCount"], 1_662)
        self.assertEqual(record["logicalTensorCount"], 1_412)
        self.assertEqual(record["tensorRecordCount"], 1_420)
        self.assertEqual(record["fileRecordCount"], 163)
        self.assertEqual(record["weightShardFileCount"], 138)
        self.assertEqual(set(record["packages"]), {"reference", "fp16"})
        for package in record["packages"].values():
            self.assertEqual(
                set(package),
                {
                    "directory",
                    "manifestSha256",
                    "manifestBytes",
                    "listedFileBytes",
                    "packageBytesIncludingManifest",
                },
            )
            self.assertRegex(package["manifestSha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                package["manifestBytes"] + package["listedFileBytes"],
                package["packageBytesIncludingManifest"],
            )
        self.assertEqual(
            record["productionPackages"],
            {
                "reference": {
                    "profile": "reference",
                    "converterRevision": 4,
                    "directory": "files-reference",
                    "manifestSha256": (
                        "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6"
                    ),
                    "manifestBytes": 759_692,
                    "listedFileBytes": 7_500_043_294,
                    "packageBytesIncludingManifest": 7_500_802_986,
                },
                "dit": {
                    "profile": "fp16-dit-dense-experimental",
                    "converterRevision": 7,
                    "directory": "files-fp16-dit-rev7-oracle",
                    "manifestSha256": (
                        "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f"
                    ),
                    "manifestBytes": 254_357,
                    "listedFileBytes": 3_087_910_671,
                    "packageBytesIncludingManifest": 3_088_165_028,
                },
                "vae": {
                    "profile": "fp16-vae-experimental",
                    "converterRevision": 7,
                    "directory": "files-fp16-vae-revision7-experimental",
                    "manifestSha256": (
                        "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7"
                    ),
                    "manifestBytes": 716_185,
                    "listedFileBytes": 7_331_254_814,
                    "packageBytesIncludingManifest": 7_331_970_999,
                },
            },
        )
        self.assertEqual(
            record["replacementPackages"],
            {
                "fp16-vae-experimental": {
                    "directory": "files-fp16-vae-experimental",
                    "manifestSha256": (
                        "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949"
                    ),
                    "manifestBytes": 715_301,
                    "listedFileBytes": 7_331_253_998,
                    "packageBytesIncludingManifest": 7_331_969_299,
                },
                "fp16-dit-dense-experimental": {
                    "directory": "files-fp16-dit-layer-mixed-experimental",
                    "manifestSha256": (
                        "a2f70c123fb7c4dbc3b51be68b4b494107c13b575ad2bed68c639791c93574d1"
                    ),
                    "manifestBytes": 257_789,
                    "listedFileBytes": 3_087_913_263,
                    "packageBytesIncludingManifest": 3_088_171_052,
                },
            },
        )
        self.assertEqual(
            record["verification"],
            {
                "fullAuditEnvironment": "ACE_STEP_VERIFY_FULL_PACKAGE=1",
                "reproducedFromVerifiedOfflineCache": True,
                "consecutiveManifestIdentityMatches": 2,
            },
        )
        demo_source = (REPOSITORY_ROOT / "demo/main.ts").read_text(
            encoding="utf-8"
        )
        for package in record["productionPackages"].values():
            self.assertIn(package["manifestSha256"], demo_source)
            self.assertIn(
                f'/model/{package["directory"]}/manifest.json',
                demo_source,
            )


@unittest.skipUnless(
    os.environ.get("ACE_STEP_VERIFY_FULL_PACKAGE") == "1",
    "set ACE_STEP_VERIFY_FULL_PACKAGE=1 after generating both packages",
)
class FullPackageAuditTests(unittest.TestCase):
    def test_complete_source_cache_and_packages(self) -> None:
        verify_cached_sources(MODEL_ROOT / "cache")
        canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
        ace_artifact = ARTIFACT_BY_KEY["ace-turbo-weights"]
        with SafetensorsFile(
            ace_artifact.cache_path(MODEL_ROOT / "cache")
        ) as ace_checkpoint:
            encode_only = {
                name: tensor
                for name, tensor in ace_checkpoint.tensors.items()
                if name.startswith(tuple(ACE_ENCODE_ONLY_PREFIX_CONTRACT))
            }
            self.assertEqual(len(encode_only), ACE_ENCODE_ONLY_TENSOR_COUNT)
            self.assertEqual(
                sum(tensor.byte_length for tensor in encode_only.values()),
                ACE_ENCODE_ONLY_SOURCE_BYTES,
            )
            self.assertEqual(
                sum(tensor.parameter_count for tensor in encode_only.values()),
                ACE_ENCODE_ONLY_PARAMETER_COUNT,
            )
            for prefix, (expected_count, expected_bytes) in (
                ACE_ENCODE_ONLY_PREFIX_CONTRACT.items()
            ):
                matching = [
                    tensor
                    for name, tensor in encode_only.items()
                    if name.startswith(prefix)
                ]
                self.assertEqual(len(matching), expected_count)
                self.assertEqual(
                    sum(tensor.byte_length for tensor in matching),
                    expected_bytes,
                )
        for name, profile in (
            ("files-reference", "reference"),
            ("files-fp16", "fp16"),
        ):
            package_root = MODEL_ROOT / name
            manifest_path = package_root / "manifest.json"
            manifest = verify_package(package_root, verify_hashes=True)
            expected = canonical["packages"][profile]
            self.assertEqual(manifest["profile"], profile)
            self.assertEqual(expected["directory"], name)
            self.assertEqual(
                sha256_file(manifest_path), expected["manifestSha256"]
            )
            self.assertEqual(
                manifest_path.stat().st_size, expected["manifestBytes"]
            )
            listed_bytes = sum(item["byteLength"] for item in manifest["files"])
            self.assertEqual(listed_bytes, expected["listedFileBytes"])
            self.assertEqual(
                listed_bytes + manifest_path.stat().st_size,
                expected["packageBytesIncludingManifest"],
            )
            self.assertEqual(
                manifest["accounting"]["sourceTensors"],
                canonical["sourceTensorCount"],
            )
            self.assertEqual(
                len(manifest["tensors"]), canonical["tensorRecordCount"]
            )
            self.assertEqual(
                len({item["logicalTensor"] for item in manifest["tensors"].values()}),
                canonical["logicalTensorCount"],
            )
            self.assertEqual(
                len(manifest["files"]), canonical["fileRecordCount"]
            )
            self.assertEqual(
                sum(item["kind"] == "weights" for item in manifest["files"]),
                canonical["weightShardFileCount"],
            )
            self.assertEqual(
                manifest["accounting"],
                {
                    "sourceTensors": 1_662,
                    "directlyIncluded": 1_411,
                    "consumedByTransform": 37,
                    "excluded": 214,
                    "outputTensorsBeforeRowSharding": 1_411,
                    "constantTensors": 1,
                    "outputTensorsAfterRowSharding": 1_420,
                },
            )
            plan = json.loads((package_root / "conversion-plan.json").read_bytes())
            self.assertEqual(plan["schema"], "ace-step-conversion-plan-v2")
            excluded = {
                decision["tensor"]
                for decision in plan["decisions"]
                if decision["source"] == "ace-turbo-weights"
                and decision["disposition"] == "excluded"
                and decision["tensor"].startswith(
                    tuple(ACE_ENCODE_ONLY_PREFIX_CONTRACT)
                )
            }
            self.assertEqual(excluded, set(encode_only))
            output_names = {output["output"] for output in plan["outputs"]}
            self.assertTrue("ace.detokenizer.norm.weight" in output_names)
            self.assertTrue(
                "ace.tokenizer.quantizer.project_out.weight" in output_names
            )
            self.assertFalse(
                any(
                    output.startswith("ace.")
                    and output.removeprefix("ace.") in encode_only
                    for output in output_names
                )
            )

            expected_dit_gemm_logical = {
                f"ace.{source_name}" for source_name in DIT_GEMM_WEIGHT_NAMES
            }
            planned_by_output = {
                output["output"]: output for output in plan["outputs"]
            }
            self.assertEqual(len(expected_dit_gemm_logical), 271)
            for logical_name in expected_dit_gemm_logical:
                output = planned_by_output[logical_name]
                self.assertEqual(
                    output["transformation"],
                    DIT_GEMM_TILE_TRANSFORMATION,
                )
                self.assertEqual(output["runtimeLayout"], DIT_GEMM_TILE_LAYOUT)

            parts_by_logical = {}
            for tensor in manifest["tensors"].values():
                parts_by_logical.setdefault(
                    tensor["logicalTensor"], []
                ).append(tensor)
            tiled_logical = {
                logical_name
                for logical_name, parts in parts_by_logical.items()
                if parts[0]["layout"] == DIT_GEMM_TILE_LAYOUT
            }
            self.assertEqual(tiled_logical, expected_dit_gemm_logical)
            expected_dit_transformation = (
                "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1"
                if profile == "reference"
                else "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1"
            )
            for logical_name in expected_dit_gemm_logical:
                parts = parts_by_logical[logical_name]
                self.assertEqual(len(parts), 1)
                tensor = parts[0]
                self.assertEqual(tensor["partStart"], 0)
                self.assertEqual(tensor["partEnd"], tensor["logicalShape"][0])
                self.assertEqual(
                    tensor["transformation"], expected_dit_transformation
                )
                self.assertEqual(tensor["layout"], DIT_GEMM_TILE_LAYOUT)
                self.assertEqual(len(tensor["logicalShape"]), 2)
                self.assertEqual(tensor["logicalShape"][0] % 128, 0)
                self.assertEqual(tensor["logicalShape"][1] % 32, 0)

            logical_records = {}
            for tensor in manifest["tensors"].values():
                logical_records.setdefault(tensor["logicalTensor"], tensor)
            transformations = Counter(
                tensor["transformation"] for tensor in logical_records.values()
            )
            self.assertEqual(transformations[VAE_CONV1D_TRANSFORMATION], 32)
            self.assertEqual(
                transformations[VAE_CONV_TRANSPOSE1D_TRANSFORMATION],
                5,
            )
            self.assertEqual(
                transformations[VAE_CHANNEL_VECTOR_TRANSFORMATION],
                72,
            )
            samples = {
                "vae.decoder.conv1.weight": (
                    [2048, 7, 64],
                    VAE_CONV1D_LAYOUT,
                ),
                "vae.decoder.block.0.conv_t1.weight": (
                    [1024, 20, 2048],
                    VAE_CONV_TRANSPOSE1D_LAYOUT,
                ),
                "vae.decoder.snake1.alpha": (
                    [128],
                    VAE_CHANNEL_VECTOR_LAYOUT,
                ),
            }
            for logical_name, (shape, layout) in samples.items():
                tensor = logical_records[logical_name]
                self.assertEqual(tensor["logicalShape"], shape)
                self.assertEqual(tensor["layout"], layout)


@unittest.skipUnless(
    os.environ.get("ACE_STEP_VERIFY_PRODUCTION_PACKAGES") == "1",
    "set ACE_STEP_VERIFY_PRODUCTION_PACKAGES=1 after --profile production",
)
class ProductionPackageAuditTests(unittest.TestCase):
    def test_complete_production_tuple(self) -> None:
        verify_cached_sources(MODEL_ROOT / "cache")
        canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
        for component, expected in canonical["productionPackages"].items():
            with self.subTest(component=component):
                package_root = MODEL_ROOT / expected["directory"]
                manifest_path = package_root / "manifest.json"
                manifest = verify_package(package_root, verify_hashes=True)
                self.assertEqual(manifest["profile"], expected["profile"])
                self.assertEqual(
                    manifest["provenance"]["converterRevision"],
                    expected["converterRevision"],
                )
                self.assertEqual(
                    sha256_file(manifest_path),
                    expected["manifestSha256"],
                )
                self.assertEqual(
                    manifest_path.stat().st_size,
                    expected["manifestBytes"],
                )
                listed_bytes = sum(
                    item["byteLength"] for item in manifest["files"]
                )
                self.assertEqual(listed_bytes, expected["listedFileBytes"])
                self.assertEqual(
                    listed_bytes + manifest_path.stat().st_size,
                    expected["packageBytesIncludingManifest"],
                )


if __name__ == "__main__":
    unittest.main()
