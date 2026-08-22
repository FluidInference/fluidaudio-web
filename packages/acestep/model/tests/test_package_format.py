from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversion_plan import (  # noqa: E402
    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE,
    VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS,
    revision7_vae_runtime_weight_layout,
)
from package_format import (  # noqa: E402
    ALIGNMENT,
    EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT,
    EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
    EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES,
    EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS,
    EXPERIMENTAL_DIT_DENSE_PROFILE,
    EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    EXPERIMENTAL_VAE_PARAMETER_BYTES,
    EXPERIMENTAL_VAE_PARAMETER_ELEMENTS,
    EXPERIMENTAL_VAE_PROFILE,
    EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
    PACKAGE_CONVERTER_REVISION,
    PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
    REQUIRED_LICENSE_FILES,
    SILENCE_LATENT_BYTES,
    SILENCE_LATENT_SHAPE,
    SILENCE_LATENT_SHARD,
    SILENCE_LATENT_SOURCE,
    SILENCE_LATENT_TENSOR_NAME,
    ShardWriter,
    TensorRecord,
    build_manifest,
    file_record,
    install_staged_directory,
    package_converter_revision,
    sha256_file,
    supported_package_converter_revisions,
    validate_native_tensor_contract,
    validate_experimental_dit_dense_payload,
    validate_experimental_vae_payload,
    verify_package,
    write_json_atomic,
)


def build_fixture(
    root: Path,
    marker: bytes = b"abcd",
    *,
    tiled_dit: bool = False,
) -> bytes:
    source_key = "ace-turbo-weights" if tiled_dit else "fixture"
    a_source_tensor = (
        "decoder.condition_embedder.weight" if tiled_dit else "a"
    )
    a_payload = b"\0" * 8_192 if tiled_dit else marker
    shard_name = (
        "weights/dit/shared-00.bin"
        if tiled_dit
        else "weights/text/shared-00.bin"
    )
    records = {}
    with ShardWriter(
        root,
        shard_name,
        records,
        max_bytes=16_384 if tiled_dit else 1024,
    ) as writer:
        writer.add_bytes(
            "a",
            [a_payload],
            byte_length=len(a_payload),
            dtype="uint32-bf16-pairs",
            logical_shape=[128, 32] if tiled_dit else [1],
            storage_shape=[2_048] if tiled_dit else [1],
            layout=(
                "dit-gemm-n128-k32-tile-major-v1"
                if tiled_dit
                else "source-row-major-bf16-pairs-lsb-u32"
            ),
            phase="dit" if tiled_dit else "text",
            lifetime="dit" if tiled_dit else "text",
            source=f"{source_key}:{a_source_tensor}",
            transformation=(
                "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1"
                if tiled_dit
                else "preserve-bf16-bits-pack-u32-pairs"
            ),
            logicalTensor="a",
            partAxis=0,
            partStart=0,
            partEnd=128 if tiled_dit else 1,
        )
        writer.add_bytes(
            "b",
            [b"efgh"],
            byte_length=4,
            dtype="uint32-bf16-pairs",
            logical_shape=[1],
            storage_shape=[1],
            layout="source-row-major-bf16-pairs-lsb-u32",
            phase="dit" if tiled_dit else "text",
            lifetime="dit" if tiled_dit else "text",
            source=f"{source_key}:b",
            transformation="preserve-bf16-bits-pack-u32-pairs",
            logicalTensor="b",
            partAxis=0,
            partStart=0,
            partEnd=1,
        )
    payload = file_record(root, shard_name, kind="weights")
    license_files = []
    model_root = Path(__file__).resolve().parents[1]
    for relative_name in (
        "licenses/ACE-Step-LICENSE",
        "licenses/Apache-2.0-LICENSE",
        "licenses/Qwen-NOTICE.txt",
    ):
        destination = root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((model_root / relative_name).read_bytes())
        license_files.append(file_record(root, relative_name, kind="license"))
    summary = {
        "sourceTensors": 2,
        "directlyIncluded": 2,
        "consumedByTransform": 0,
        "excluded": 0,
        "outputTensorsBeforeRowSharding": 2,
    }
    write_json_atomic(
        root / "conversion-plan.json",
        {
            "schema": "ace-step-conversion-plan-v2",
            "summary": summary,
            "decisions": [
                {
                    "source": source_key,
                    "tensor": a_source_tensor,
                    "disposition": "included",
                    "reason": "fixture tensor",
                    "output": "a",
                },
                {
                    "source": source_key,
                    "tensor": "b",
                    "disposition": "included",
                    "reason": "fixture tensor",
                    "output": "b",
                },
            ],
            "outputs": [
                {
                    "source": source_key,
                    "sourceTensor": a_source_tensor,
                    "output": "a",
                    "phase": "dit" if tiled_dit else "text",
                    "lifetime": "dit" if tiled_dit else "text",
                    "group": "dit/shared" if tiled_dit else "text/shared",
                    "transformation": (
                        "profile-float-dit-gemm-n128-k32-tile-major-v1"
                        if tiled_dit
                        else "profile-float-storage"
                    ),
                    "outputDtype": "profile-float",
                    "runtimeLayout": (
                        "dit-gemm-n128-k32-tile-major-v1"
                        if tiled_dit
                        else "source-row-major"
                    ),
                },
                {
                    "source": source_key,
                    "sourceTensor": "b",
                    "output": "b",
                    "phase": "dit" if tiled_dit else "text",
                    "lifetime": "dit" if tiled_dit else "text",
                    "group": "dit/shared" if tiled_dit else "text/shared",
                    "transformation": "profile-float-storage",
                    "outputDtype": "profile-float",
                    "runtimeLayout": "source-row-major",
                },
            ],
        },
    )
    plan_file = file_record(root, "conversion-plan.json", kind="conversion-plan")
    manifest = build_manifest(
        profile="reference",
        source=[
            {
                "key": source_key,
                "component": "test",
                "repository": "test/fixture",
                "revision": "0" * 40,
                "path": "model.safetensors",
                "byteLength": len(a_payload) + 4,
                "sha256": "0" * 64,
                "tensorCount": 2,
                "parameterCount": (128 * 32 + 1) if tiled_dit else 2,
                "headerLength": 100,
                "headerSha256": "1" * 64,
                "inventorySha256": "2" * 64,
            }
        ],
        files=[plan_file, payload, *license_files],
        tensors=records,
        accounting={
            **summary,
            "constantTensors": 0,
            "outputTensorsAfterRowSharding": 2,
        },
        licenses=[
            {
                "component": "fixture",
                "spdx": "MIT",
                "notice": "fixture only",
                "source": "https://example.invalid/fixture",
            },
            {
                "component": "fixture qwen",
                "spdx": "Apache-2.0",
                "notice": "fixture only",
                "source": "https://example.invalid/qwen",
            },
        ],
        provenance={
            "converterRevision": PACKAGE_CONVERTER_REVISION,
            "aceSnapshot": "0" * 40,
            "plannerSnapshot": "0" * 40,
            "referenceRepository": "https://example.invalid/source.git",
            "referenceCommit": "0" * 40,
            "referenceLicenseGitBlob": "0" * 40,
            "referenceLicenseSha256": "0" * 64,
            "determinism": "fixture",
        },
    )
    write_json_atomic(root / "manifest.json", manifest)
    return (root / "manifest.json").read_bytes()


def mutate_conversion_plan(
    root: Path,
    mutation,
) -> None:
    plan_path = root / "conversion-plan.json"
    plan = json.loads(plan_path.read_bytes())
    mutation(plan)
    write_json_atomic(plan_path, plan)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for record in manifest["files"]:
        if record["name"] == "conversion-plan.json":
            refreshed = file_record(root, "conversion-plan.json", kind="conversion-plan")
            record.update(
                {
                    "byteLength": refreshed.byteLength,
                    "sha256": refreshed.sha256,
                }
            )
            break
    write_json_atomic(manifest_path, manifest)


def experimental_vae_records() -> dict[str, TensorRecord]:
    records: dict[str, TensorRecord] = {}
    for source_tensor, logical_shape_tuple in (
        VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.items()
    ):
        logical_shape = list(logical_shape_tuple)
        if source_tensor.endswith(".weight_v"):
            transformation, layout = revision7_vae_runtime_weight_layout(
                source_tensor,
                logical_shape_tuple,
            )
        elif source_tensor.endswith(".bias"):
            transformation = "bf16-to-fp32-to-ieee-fp16-v1"
            layout = "source-row-major"
        else:
            transformation = (
                "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1"
            )
            layout = "channel-vector-f16-v1"
        storage_shape = logical_shape.copy()
        if transformation == VAE_K7_ROW_REUSE_FP16_TRANSFORMATION:
            output_channels, kernel, input_channels = logical_shape
            storage_shape = [
                kernel,
                input_channels // 4,
                output_channels // 64,
                32,
                2,
                4,
            ]
        elif transformation == (
            "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-"
            "tile-major-ieee-fp16-v1"
        ):
            output_channels, _, input_channels = logical_shape
            storage_shape = [
                output_channels // 128,
                input_channels // 32,
                32,
                128,
            ]
        elif transformation == (
            "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-"
            "output-ieee-fp16-v1"
        ):
            output_channels, kernel, input_channels = logical_shape
            storage_shape = [
                kernel // 2,
                2,
                input_channels,
                output_channels,
            ]
        elif transformation == VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION:
            output_channels, kernel, input_channels = logical_shape
            reuse_axis = VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS[source_tensor][1]
            outputs_per_lane = 8 if reuse_axis == "channel" else 4
            storage_shape = [
                kernel // 2,
                2,
                input_channels // 4,
                output_channels // (32 * outputs_per_lane),
                32,
                outputs_per_lane,
                4,
            ]
        name = (
            f"vae.{source_tensor.removesuffix('.weight_v')}.weight"
            if source_tensor.endswith(".weight_v")
            else f"vae.{source_tensor}"
        )
        records[name] = TensorRecord(
            shard="weights/vae/shared-00.bin",
            byteOffset=0,
            byteLength=math.prod(logical_shape) * 2,
            dtype="float16",
            logicalShape=logical_shape,
            storageShape=storage_shape,
            layout=layout,
            phase="vae",
            lifetime="vae",
            source=f"vae-weights:{source_tensor}",
            transformation=transformation,
            logicalTensor=name,
            partAxis=0,
            partStart=0,
            partEnd=logical_shape[0],
        )
    records[SILENCE_LATENT_TENSOR_NAME] = TensorRecord(
        shard=SILENCE_LATENT_SHARD,
        byteOffset=0,
        byteLength=SILENCE_LATENT_BYTES,
        dtype="float32",
        logicalShape=SILENCE_LATENT_SHAPE.copy(),
        storageShape=SILENCE_LATENT_SHAPE.copy(),
        layout="contiguous-nct-f32",
        phase="constants",
        lifetime="initial-latent",
        source=SILENCE_LATENT_SOURCE,
        transformation="validated-pytorch-zip-storage-extraction",
        logicalTensor=SILENCE_LATENT_TENSOR_NAME,
        partAxis=0,
        partStart=0,
        partEnd=1,
    )
    return records


def experimental_dit_dense_records(
    *,
    converter_revision: int = 8,
) -> dict[str, TensorRecord]:
    shapes = {
        "scale_shift_table": [1, 6, 2_048],
        "self_attn_norm.weight": [2_048],
        "self_attn.q_proj.weight": [2_048, 2_048],
        "self_attn.k_proj.weight": [1_024, 2_048],
        "self_attn.v_proj.weight": [1_024, 2_048],
        "self_attn.q_norm.weight": [128],
        "self_attn.k_norm.weight": [128],
        "self_attn.o_proj.weight": [2_048, 2_048],
        "cross_attn_norm.weight": [2_048],
        "cross_attn.q_proj.weight": [2_048, 2_048],
        "cross_attn.k_proj.weight": [1_024, 2_048],
        "cross_attn.v_proj.weight": [1_024, 2_048],
        "cross_attn.q_norm.weight": [128],
        "cross_attn.k_norm.weight": [128],
        "cross_attn.o_proj.weight": [2_048, 2_048],
        "mlp_norm.weight": [2_048],
        "mlp.gate_proj.weight": [6_144, 2_048],
        "mlp.up_proj.weight": [6_144, 2_048],
        "mlp.down_proj.weight": [2_048, 6_144],
    }
    dense_suffixes = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    cross_cache_suffixes = {
        "cross_attn.k_proj.weight",
        "cross_attn.v_proj.weight",
    }
    records: dict[str, TensorRecord] = {}
    for layer in range(24):
        for suffix, logical_shape in shapes.items():
            name = f"ace.decoder.layers.{layer}.{suffix}"
            elements = math.prod(logical_shape)
            if suffix in dense_suffixes:
                columns, inner = logical_shape
                dtype = "float16"
                if converter_revision == 7:
                    storage_shape = logical_shape
                    layout = "dit-gemm-n256-k32-tile-major-v1"
                    transformation = (
                        "bf16-to-ieee-fp16-dit-gemm-n256-k32-tile-major-v1"
                    )
                elif converter_revision == 8:
                    storage_shape = [columns // 128, inner // 4, 4, 32, 4]
                    layout = "dit-gemm-n128-k4-output4-lane32-k4-v1"
                    transformation = (
                        "bf16-to-ieee-fp16-dit-gemm-"
                        "n128-k4-output4-lane32-k4-v1"
                    )
                else:
                    raise ValueError("fixture converter revision must be 7 or 8")
            elif suffix in cross_cache_suffixes:
                dtype = "uint32-bf16-pairs"
                storage_shape = [elements // 2]
                layout = "dit-gemm-n128-k32-tile-major-v1"
                transformation = (
                    "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1"
                )
            else:
                dtype = "uint32-bf16-pairs"
                storage_shape = [elements // 2]
                layout = "source-row-major-bf16-pairs-lsb-u32"
                transformation = "preserve-bf16-bits-pack-u32-pairs"
            records[name] = TensorRecord(
                shard=f"weights/dit/layer-{layer:02d}-00.bin",
                byteOffset=0,
                byteLength=elements * 2,
                dtype=dtype,
                logicalShape=logical_shape,
                storageShape=storage_shape,
                layout=layout,
                phase="dit",
                lifetime="dit",
                source=f"ace-turbo-weights:{name.removeprefix('ace.')}",
                transformation=transformation,
                logicalTensor=name,
                partAxis=0,
                partStart=0,
                partEnd=logical_shape[0],
            )
    records[SILENCE_LATENT_TENSOR_NAME] = TensorRecord(
        shard=SILENCE_LATENT_SHARD,
        byteOffset=0,
        byteLength=SILENCE_LATENT_BYTES,
        dtype="float32",
        logicalShape=SILENCE_LATENT_SHAPE.copy(),
        storageShape=SILENCE_LATENT_SHAPE.copy(),
        layout="contiguous-nct-f32",
        phase="constants",
        lifetime="initial-latent",
        source=SILENCE_LATENT_SOURCE,
        transformation="validated-pytorch-zip-storage-extraction",
        logicalTensor=SILENCE_LATENT_TENSOR_NAME,
        partAxis=0,
        partStart=0,
        partEnd=1,
    )
    return records


class PackageTests(unittest.TestCase):
    def test_converter_revision_is_profile_specific_without_advancing_stable_profiles(self) -> None:
        self.assertEqual(package_converter_revision("reference"), 4)
        self.assertEqual(package_converter_revision("fp16"), 4)
        self.assertEqual(PACKAGE_CONVERTER_REVISION, 4)
        self.assertEqual(
            package_converter_revision(EXPERIMENTAL_VAE_PROFILE),
            EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
        )
        self.assertEqual(EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION, 7)
        self.assertEqual(
            package_converter_revision(EXPERIMENTAL_DIT_DENSE_PROFILE),
            EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
        )
        self.assertEqual(EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION, 8)
        self.assertEqual(PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION, 7)
        self.assertEqual(
            supported_package_converter_revisions(
                EXPERIMENTAL_DIT_DENSE_PROFILE
            ),
            frozenset({7, 8}),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported profile"):
            package_converter_revision("unknown")

    def test_revision_8_experimental_dit_payload_replaces_all_216_dense_layouts(
        self,
    ) -> None:
        records = experimental_dit_dense_records()
        validate_experimental_dit_dense_payload(records)
        dit_records = {
            name: record
            for name, record in records.items()
            if record.phase == "dit"
        }
        dense_records = {
            name: record
            for name, record in dit_records.items()
            if record.dtype == "float16"
        }
        self.assertEqual(len(dit_records), EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT)
        self.assertEqual(len(dense_records), 24 * 9)
        self.assertEqual(
            sum(math.prod(record.logicalShape) for record in dit_records.values()),
            EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS,
        )
        self.assertEqual(
            sum(record.byteLength for record in dit_records.values()),
            EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES,
        )
        self.assertEqual(
            {record.layout for record in dense_records.values()},
            {"dit-gemm-n128-k4-output4-lane32-k4-v1"},
        )

        name = "ace.decoder.layers.0.self_attn.q_proj.weight"
        dense = records[name]
        mutations = (
            {
                **records,
                name: replace(
                    dense,
                    layout="dit-gemm-n256-k32-tile-major-v1",
                    transformation=(
                        "bf16-to-ieee-fp16-dit-gemm-n256-k32-tile-major-v1"
                    ),
                    storageShape=dense.logicalShape,
                ),
            },
            {
                **records,
                name: replace(dense, storageShape=dense.logicalShape),
            },
            {key: value for key, value in records.items() if key != name},
        )
        for mutation in mutations:
            with self.assertRaisesRegex(
                ValueError,
                "storage contract|exactly the 24 repeated layers|mixes dense",
            ):
                validate_experimental_dit_dense_payload(mutation)

    def test_revision_7_production_dit_payload_retains_exact_native_layout(
        self,
    ) -> None:
        records = experimental_dit_dense_records(converter_revision=7)
        validate_experimental_dit_dense_payload(
            records,
            converter_revision=7,
        )
        dense_records = {
            name: record
            for name, record in records.items()
            if record.phase == "dit" and record.dtype == "float16"
        }
        self.assertEqual(len(dense_records), 24 * 9)
        self.assertEqual(
            {record.layout for record in dense_records.values()},
            {"dit-gemm-n256-k32-tile-major-v1"},
        )
        self.assertTrue(
            all(
                record.storageShape == record.logicalShape
                for record in dense_records.values()
            )
        )

        name = "ace.decoder.layers.0.self_attn.q_proj.weight"
        mixed = {
            **records,
            name: experimental_dit_dense_records(converter_revision=8)[name],
        }
        with self.assertRaisesRegex(ValueError, "storage contract|mixes dense"):
            validate_experimental_dit_dense_payload(
                mixed,
                converter_revision=7,
            )

    def test_revision_8_k4_transform_is_bound_to_repeated_dense_shapes(self) -> None:
        transformation = (
            "bf16-to-ieee-fp16-dit-gemm-n128-k4-output4-lane32-k4-v1"
        )
        source = "ace-turbo-weights:decoder.layers.23.mlp.down_proj.weight"
        validate_native_tensor_contract(
            transformation,
            source,
            [2_048, 6_144],
            "dit",
            "fixture",
        )
        for bad_source, shape, phase in (
            (
                "ace-turbo-weights:decoder.layers.0.cross_attn.k_proj.weight",
                [1_024, 2_048],
                "dit",
            ),
            (source, [127, 4], "dit"),
            (source, [128, 6], "dit"),
            (source, [128, 4], "vae"),
        ):
            with self.assertRaisesRegex(ValueError, "DiT GEMM tile-major"):
                validate_native_tensor_contract(
                    transformation,
                    bad_source,
                    shape,
                    phase,
                    "fixture",
                )

    def test_experimental_vae_payload_is_exact_ieee_fp16_and_unsharded(self) -> None:
        records = experimental_vae_records()
        validate_experimental_vae_payload(records)
        vae_records = {
            name: record
            for name, record in records.items()
            if record.phase == "vae"
        }
        self.assertEqual(len(vae_records), EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT)
        self.assertEqual(
            len({record.logicalTensor for record in vae_records.values()}),
            EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
        )
        self.assertEqual(
            sum(record.byteLength for record in vae_records.values()),
            EXPERIMENTAL_VAE_PARAMETER_BYTES,
        )
        self.assertEqual(
            sum(
                record.transformation == VAE_K7_ROW_REUSE_FP16_TRANSFORMATION
                for record in vae_records.values()
            ),
            12,
        )
        self.assertEqual(
            sum(
                record.transformation
                == VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION
                for record in vae_records.values()
            ),
            4,
        )
        self.assertEqual(
            records["vae.decoder.block.0.res_unit1.conv1.weight"].storageShape,
            [7, 256, 16, 32, 2, 4],
        )
        self.assertEqual(
            records["vae.decoder.block.3.conv_t1.weight"].storageShape,
            [4, 2, 64, 1, 32, 4, 4],
        )

        first_name = next(iter(vae_records))
        first = records[first_name]
        mutations = (
            {name: record for name, record in records.items() if name != first_name},
            {**records, first_name: replace(first, partEnd=first.partEnd - 1)},
            {
                **records,
                first_name: replace(
                    first,
                    transformation="bf16-to-fp32",
                ),
            },
        )
        for mutation in mutations:
            with self.assertRaisesRegex(
                ValueError,
                "logical tensors.*unsharded records|exact revision-7",
            ):
                validate_experimental_vae_payload(mutation)

        silence = records[SILENCE_LATENT_TENSOR_NAME]
        constant_mutations = (
            {
                name: record
                for name, record in records.items()
                if name != SILENCE_LATENT_TENSOR_NAME
            },
            {
                **records,
                "constants.extra": replace(
                    silence,
                    shard="constants/extra-f32.bin",
                    logicalTensor="constants.extra",
                ),
            },
            {
                **{
                    name: record
                    for name, record in records.items()
                    if name != SILENCE_LATENT_TENSOR_NAME
                },
                "constants.renamed": replace(
                    silence,
                    logicalTensor="constants.renamed",
                ),
            },
            {
                **records,
                SILENCE_LATENT_TENSOR_NAME: replace(
                    silence,
                    logicalShape=[1, 64, 14_999],
                ),
            },
        )
        for mutation in constant_mutations:
            with self.assertRaisesRegex(
                ValueError,
                "canonical constants.silence_latent",
            ):
                validate_experimental_vae_payload(mutation)

    def test_vae_native_transforms_are_bound_to_exact_sources_in_both_directions(self) -> None:
        contracts = (
            (
                "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
                "vae-weights:decoder.block.4.res_unit3.conv2.weight_v",
                [128, 7, 128],
            ),
            (
                "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
                "vae-weights:decoder.block.0.conv_t1.weight_v",
                [1024, 20, 2048],
            ),
            (
                "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1",
                "vae-weights:decoder.block.2.res_unit1.snake2.alpha",
                [512],
            ),
        )
        for transformation, source, shape in contracts:
            validate_native_tensor_contract(
                transformation,
                source,
                shape,
                "vae",
                "fixture",
            )
            with self.assertRaisesRegex(ValueError, "native-layout|channel-vector"):
                validate_native_tensor_contract(
                    transformation,
                    "vae-weights:decoder.unrelated.weight_v",
                    shape,
                    "vae",
                    "fixture",
                )
            with self.assertRaisesRegex(ValueError, "native-layout|channel-vector"):
                validate_native_tensor_contract(
                    "bf16-to-fp32-to-ieee-fp16-v1",
                    source,
                    shape,
                    "vae",
                    "fixture",
                )

        bias_source = "vae-weights:decoder.block.4.res_unit3.conv2.bias"
        for transformation in (
            "bf16-to-fp32",
            "bf16-to-fp32-to-ieee-fp16-v1",
        ):
            validate_native_tensor_contract(
                transformation,
                bias_source,
                [128],
                "vae",
                "fixture",
            )
            for source, shape, phase in (
                ("vae-weights:decoder.unrelated.bias", [128], "vae"),
                (bias_source, [128, 1], "vae"),
                (bias_source, [128], "dit"),
            ):
                with self.assertRaisesRegex(ValueError, "VAE bias contract"):
                    validate_native_tensor_contract(
                        transformation,
                        source,
                        shape,
                        phase,
                        "fixture",
                    )
        with self.assertRaisesRegex(ValueError, "VAE bias contract"):
            validate_native_tensor_contract(
                "profile-float-storage",
                bias_source,
                [128],
                "vae",
                "fixture",
            )

    def test_reference_package_accepts_bound_tile_major_dit_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root, tiled_dit=True)
            manifest = verify_package(root)
        tensor = manifest["tensors"]["a"]
        self.assertEqual(tensor["layout"], "dit-gemm-n128-k32-tile-major-v1")
        self.assertEqual(tensor["storageShape"], [2_048])
        self.assertEqual(
            tensor["transformation"],
            "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1",
        )

    def test_dit_gemm_tile_layout_binds_exact_sources_shapes_and_phase(self) -> None:
        source = "ace-turbo-weights:decoder.layers.23.mlp.down_proj.weight"
        for transformation in (
            "profile-float-dit-gemm-n128-k32-tile-major-v1",
            "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1",
            "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1",
        ):
            validate_native_tensor_contract(
                transformation,
                source,
                None if transformation.startswith("profile-float-") else [2048, 6144],
                "dit",
                "fixture",
            )
        for bad_source, shape, phase in (
            (
                "ace-turbo-weights:decoder.layers.24.mlp.down_proj.weight",
                [2048, 6144],
                "dit",
            ),
            (
                "ace-turbo-weights:decoder.layers.0.self_attn.q_proj.bias",
                [2048],
                "dit",
            ),
            (source, [2048, 6143], "dit"),
            (source, [2048, 6144], "vae"),
        ):
            with self.assertRaisesRegex(ValueError, "DiT GEMM tile-major"):
                validate_native_tensor_contract(
                    "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1",
                    bad_source,
                    shape,
                    phase,
                    "fixture",
                )

        for ordinary_transformation, shape in (
            ("profile-float-storage", None),
            ("preserve-bf16-bits-pack-u32-pairs", [2048, 6144]),
            ("bf16-to-ieee-fp16", [2048, 6144]),
        ):
            with self.assertRaisesRegex(ValueError, "DiT GEMM tile-major"):
                validate_native_tensor_contract(
                    ordinary_transformation,
                    source,
                    shape,
                    "dit",
                    "fixture",
                )

    def test_rehashed_exact_dit_gemm_source_cannot_be_relabelled_row_major(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root, tiled_dit=True)
            mutate_conversion_plan(
                root,
                lambda plan: plan["outputs"][0].update(
                    transformation="profile-float-storage",
                    runtimeLayout="source-row-major",
                ),
            )
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            tensor = manifest["tensors"]["a"]
            tensor["transformation"] = "preserve-bf16-bits-pack-u32-pairs"
            tensor["layout"] = "source-row-major-bf16-pairs-lsb-u32"
            write_json_atomic(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "DiT GEMM tile-major"):
                verify_package(root)

    def test_tile_major_dit_gemm_tensor_cannot_be_row_sharded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root, tiled_dit=True)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["tensors"]["a"]["partEnd"] = 64
            write_json_atomic(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "complete tile-major DiT GEMM"):
                verify_package(root)

    def test_native_layout_identifiers_bind_exact_vae_sources_and_ranks(self) -> None:
        validate_native_tensor_contract(
            "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
            "vae-weights:decoder.block.0.res_unit1.conv1.weight_v",
            [128, 7, 128],
            "vae",
            "fixture",
        )
        validate_native_tensor_contract(
            "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
            "vae-weights:decoder.block.0.res_unit1.conv2.weight_v",
            [128, 1, 128],
            "vae",
            "fixture",
        )
        validate_native_tensor_contract(
            "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
            "vae-weights:decoder.block.0.conv_t1.weight_v",
            [1024, 20, 2048],
            "vae",
            "fixture",
        )
        for source, shape, phase in (
            ("vae-weights:decoder.block.0.conv_t1.weight_v", [128, 7, 128], "vae"),
            ("vae-weights:decoder.conv1.weight_v", [128], "vae"),
            ("vae-weights:decoder.conv1.weight_v", [128, 7, 128], "dit"),
        ):
            with self.assertRaisesRegex(ValueError, "Conv1d native-layout"):
                validate_native_tensor_contract(
                    "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
                    source,
                    shape,
                    phase,
                    "fixture",
                )

    def test_required_license_identities_match_committed_payloads(self) -> None:
        model_root = Path(__file__).resolve().parents[1]
        for relative_name, (expected_bytes, expected_sha256) in (
            REQUIRED_LICENSE_FILES.items()
        ):
            path = model_root / relative_name
            self.assertEqual(path.stat().st_size, expected_bytes)
            self.assertEqual(sha256_file(path), expected_sha256)

    def test_alignment_determinism_and_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            self.assertEqual(build_fixture(first), build_fixture(second))
            manifest = verify_package(first)
            self.assertEqual(manifest["tensors"]["a"]["byteOffset"], 0)
            self.assertEqual(manifest["tensors"]["b"]["byteOffset"], ALIGNMENT)
            payload = first / "weights/text/shared-00.bin"
            payload.write_bytes(b"X" + payload.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_package(first)

    def test_unknown_format_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.mkdir(exist_ok=True)
            (root / "manifest.json").write_text('{"format":"future-v99"}')
            with self.assertRaisesRegex(ValueError, "Unknown"):
                verify_package(root)

    def test_duplicate_manifest_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            manifest = (root / "manifest.json").read_text()
            manifest = manifest.replace(
                '"profile":"reference"',
                '"profile":"reference","profile":"reference"',
            )
            (root / "manifest.json").write_text(manifest)
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                verify_package(root)

    def test_row_parts_must_reconstruct_logical_axis_without_gap_or_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            tensor = manifest["tensors"]["a"]
            tensor["logicalShape"] = [2]
            tensor["partStart"] = 1
            tensor["partEnd"] = 2
            write_json_atomic(manifest_path, manifest)
            with self.assertRaisesRegex(
                ValueError,
                "gapped, or overlapping|storage shape, bytes, or layout",
            ):
                verify_package(root)

    def test_storage_shape_and_layout_must_derive_from_logical_part_extent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            manifest_path = root / "manifest.json"
            baseline = json.loads(manifest_path.read_bytes())
            mutations = (
                {"storageShape": [2]},
                {"layout": "row-shard-axis0-bf16-pairs-lsb-u32"},
                {"partEnd": 1, "logicalShape": [2]},
            )
            for mutation in mutations:
                manifest = json.loads(json.dumps(baseline))
                manifest["tensors"]["a"].update(mutation)
                write_json_atomic(manifest_path, manifest)
                with self.assertRaisesRegex(
                    ValueError,
                    "storage shape, bytes, or layout|incomplete|Malformed tensor",
                ):
                    verify_package(root)

    def test_rehashed_plan_output_rename_cannot_detach_manifest_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            mutate_conversion_plan(
                root,
                lambda plan: plan["outputs"][0].update(output="renamed"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "outputs do not exactly match included source tensors|logical tensors",
            ):
                verify_package(root)

    def test_rehashed_coordinated_plan_output_rename_still_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)

            def rename_both(plan):
                plan["outputs"][0]["output"] = "renamed"
                plan["decisions"][0]["output"] = "renamed"

            mutate_conversion_plan(root, rename_both)
            with self.assertRaisesRegex(ValueError, "logical tensors"):
                verify_package(root)

    def test_rehashed_plan_source_phase_lifetime_transform_and_dtype_mutations_fail(self) -> None:
        mutations = (
            ("sourceTensor", "wrong", "included source tensors"),
            ("phase", "planner", "Malformed conversion-plan output"),
            ("lifetime", "planner", "source, phase, lifetime"),
            ("group", "text/layer-99", "source, phase, lifetime"),
            (
                "transformation",
                "bf16-to-fp32",
                "Malformed conversion-plan output|VAE bias contract",
            ),
            ("outputDtype", "float32", "Malformed conversion-plan output"),
            (
                "runtimeLayout",
                "conv1d-output-kernel-input-f32-v1",
                "Malformed conversion-plan output",
            ),
        )
        for field, value, pattern in mutations:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    build_fixture(root)
                    mutate_conversion_plan(
                        root,
                        lambda plan, field=field, value=value: plan["outputs"][0].update(
                            {field: value}
                        ),
                    )
                    with self.assertRaisesRegex(ValueError, pattern):
                        verify_package(root)

    def test_rehashed_plan_record_schema_and_decision_relation_mutations_fail(self) -> None:
        mutations = (
            lambda plan: plan["outputs"][0].update(extra="not allowed"),
            lambda plan: plan["decisions"][0].update(output="wrong"),
            lambda plan: plan["decisions"][0].update(source="unknown"),
            lambda plan: plan["decisions"][0].update(reason=3),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    build_fixture(root)
                    mutate_conversion_plan(root, mutation)
                    with self.assertRaises(ValueError):
                        verify_package(root)

    def test_rehashed_package_cannot_omit_or_relabel_required_license_payload(self) -> None:
        for mutation in ("remove", "relabel"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    build_fixture(root)
                    manifest_path = root / "manifest.json"
                    manifest = json.loads(manifest_path.read_bytes())
                    target = "licenses/Apache-2.0-LICENSE"
                    if mutation == "remove":
                        manifest["files"] = [
                            record
                            for record in manifest["files"]
                            if record["name"] != target
                        ]
                        (root / target).unlink()
                    else:
                        next(
                            record
                            for record in manifest["files"]
                            if record["name"] == target
                        )["kind"] = "upstream-asset"
                    write_json_atomic(manifest_path, manifest)
                    with self.assertRaisesRegex(ValueError, "required license payload"):
                        verify_package(root)

    def test_nested_source_license_and_provenance_schemas_are_closed(self) -> None:
        mutations = (
            lambda manifest: manifest["source"][0].update(extra=True),
            lambda manifest: manifest["licenses"][0].update(extra=True),
            lambda manifest: manifest["provenance"].update(extra=True),
            lambda manifest: manifest["provenance"].update(
                converterRevision=PACKAGE_CONVERTER_REVISION + 1
            ),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    build_fixture(root)
                    manifest_path = root / "manifest.json"
                    manifest = json.loads(manifest_path.read_bytes())
                    mutation(manifest)
                    write_json_atomic(manifest_path, manifest)
                    with self.assertRaises(ValueError):
                        verify_package(root)

    def test_refuses_unrecognized_nonempty_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            target = root / "target"
            staging.mkdir()
            target.mkdir()
            build_fixture(staging)
            (target / "user-file").write_text("preserve me")
            with self.assertRaises(ValueError):
                install_staged_directory(staging, target)
            self.assertEqual((target / "user-file").read_text(), "preserve me")

    def test_interrupted_install_restores_previous_valid_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            target = root / "target"
            staging.mkdir()
            target.mkdir()
            build_fixture(staging, b"new!")
            old_manifest = build_fixture(target, b"old!")
            real_replace = __import__("os").replace

            def fail_install(source, destination):
                if Path(source) == staging and Path(destination) == target:
                    raise OSError("synthetic interrupted install")
                return real_replace(source, destination)

            with mock.patch("package_format.os.replace", side_effect=fail_install):
                with self.assertRaisesRegex(OSError, "synthetic"):
                    install_staged_directory(staging, target)
            self.assertEqual((target / "manifest.json").read_bytes(), old_manifest)
            verify_package(target)

    def test_atomically_upgrades_an_exact_pinned_previous_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            target = root / "target"
            staging.mkdir()
            target.mkdir()
            build_fixture(staging, b"new!")
            build_fixture(target, b"old!")
            target_manifest = target / "manifest.json"
            previous = json.loads(target_manifest.read_bytes())
            previous["provenance"]["converterRevision"] = (
                PACKAGE_CONVERTER_REVISION - 1
            )
            write_json_atomic(target_manifest, previous)
            pinned_sha256 = sha256_file(target_manifest)

            install_staged_directory(
                staging,
                target,
                expected_existing_manifest_sha256=pinned_sha256,
            )

            verify_package(target)
            self.assertFalse(staging.exists())
            self.assertFalse((root / ".target.previous").exists())

    def test_pinned_previous_revision_does_not_authorize_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            target = root / "target"
            staging.mkdir()
            target.mkdir()
            build_fixture(staging)
            build_fixture(target)
            target_manifest = target / "manifest.json"
            previous = json.loads(target_manifest.read_bytes())
            previous["provenance"]["converterRevision"] = (
                PACKAGE_CONVERTER_REVISION - 1
            )
            write_json_atomic(target_manifest, previous)
            pinned_sha256 = sha256_file(target_manifest)
            (target / "user-file").write_text("preserve me")

            with self.assertRaisesRegex(ValueError, "unrecognized files"):
                install_staged_directory(
                    staging,
                    target,
                    expected_existing_manifest_sha256=pinned_sha256,
                )
            self.assertEqual((target / "user-file").read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
