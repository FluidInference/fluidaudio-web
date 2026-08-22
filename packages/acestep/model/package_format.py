"""Deterministic package manifest, aligned shard writer, and verifier."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

from conversion_plan import (
    DIT_DENSE_FP16_REV7_TILE_LAYOUT,
    DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
    DIT_DENSE_FP16_REV8_TILE_LAYOUT,
    DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    VAE_K7_ROW_REUSE_FP16_LAYOUT,
    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE,
    VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS,
    revision7_vae_runtime_weight_layout,
)


FORMAT_VERSION = "ace-step-webgpu-v1"
PACKAGE_CONVERTER_REVISION = 4
EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION = 7
EXPERIMENTAL_VAE_PROFILE = "fp16-vae-experimental"
PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION = 7
EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION = 8
EXPERIMENTAL_DIT_DENSE_PROFILE = "fp16-dit-dense-experimental"
ALIGNMENT = 256
MAX_STORAGE_BINDING_BYTES = 128 * 1024 * 1024
MAX_SHARD_BYTES = 120 * 1024 * 1024
COPY_CHUNK_BYTES = 4 * 1024 * 1024
SUPPORTED_PROFILES = (
    "reference",
    "fp16",
    EXPERIMENTAL_VAE_PROFILE,
    EXPERIMENTAL_DIT_DENSE_PROFILE,
)
PACKAGE_CONVERTER_REVISION_BY_PROFILE = {
    "reference": PACKAGE_CONVERTER_REVISION,
    "fp16": PACKAGE_CONVERTER_REVISION,
    EXPERIMENTAL_VAE_PROFILE: EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    EXPERIMENTAL_DIT_DENSE_PROFILE: (
        EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION
    ),
}
EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT = 456
EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS = 1_510_404_096
EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES = 3_020_808_192
EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT = 145
# Revision 4 has 146 VAE records because its largest FP32 transpose weight is
# split. At FP16 that weight is 83,886,080 bytes, so all 145 logical tensors
# are one record each. The silence latent is a separate constant record.
EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT = 145
EXPERIMENTAL_VAE_PARAMETER_ELEMENTS = 84_395_776
EXPERIMENTAL_VAE_PARAMETER_BYTES = 168_791_552
SILENCE_LATENT_TENSOR_NAME = "constants.silence_latent"
SILENCE_LATENT_SHARD = "constants/silence-latent-f32.bin"
SILENCE_LATENT_SOURCE = "ace-silence-latent:silence_latent/data/0"
SILENCE_LATENT_SHAPE = [1, 64, 15_000]
SILENCE_LATENT_BYTES = 3_840_000
MANIFEST_KEYS = {
    "format",
    "profile",
    "alignment",
    "portableStorageBindingBytes",
    "source",
    "files",
    "tensors",
    "accounting",
    "licenses",
    "provenance",
}
ACCOUNTING_KEYS = {
    "sourceTensors",
    "directlyIncluded",
    "consumedByTransform",
    "excluded",
    "outputTensorsBeforeRowSharding",
    "constantTensors",
    "outputTensorsAfterRowSharding",
}
PLAN_SUMMARY_KEYS = ACCOUNTING_KEYS - {
    "constantTensors",
    "outputTensorsAfterRowSharding",
}
PLAN_KEYS = {"schema", "summary", "decisions", "outputs"}
PLAN_DECISION_KEYS = {"source", "tensor", "disposition", "reason", "output"}
PLAN_OUTPUT_KEYS = {
    "source",
    "sourceTensor",
    "output",
    "phase",
    "lifetime",
    "group",
    "transformation",
    "outputDtype",
    "runtimeLayout",
}
PLAN_DISPOSITIONS = {"included", "consumed-by-transform", "excluded"}
PLAN_TRANSFORMATIONS = {
    "profile-float-storage",
    "profile-float-dit-gemm-n128-k32-tile-major-v1",
    DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
    DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
    "bf16-to-fp32",
    "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
    "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
    "bf16-to-fp32-flatten-1-c-1-to-c-v1",
    "bf16-to-fp32-to-ieee-fp16-v1",
    "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
    "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
    "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
    "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1",
}
PLAN_OUTPUT_DTYPES = {"profile-float", "float32", "float16"}
PLAN_RUNTIME_LAYOUTS = {
    "source-row-major",
    "dit-gemm-n128-k32-tile-major-v1",
    DIT_DENSE_FP16_REV7_TILE_LAYOUT,
    DIT_DENSE_FP16_REV8_TILE_LAYOUT,
    "conv1d-output-kernel-input-f32-v1",
    "conv-transpose1d-output-kernel-input-f32-v1",
    "channel-vector-f32-v1",
    "conv1d-output-kernel-input-f16-v1",
    "conv-transpose1d-output-kernel-input-f16-v1",
    "conv1d-k1-cout128-cin32-tile-major-f16-v1",
    "conv-transpose1d-phase-tap-input-output-f16-v1",
    VAE_K7_ROW_REUSE_FP16_LAYOUT,
    VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
    "channel-vector-f16-v1",
}
FILE_KINDS = {"conversion-plan", "upstream-asset", "license", "weights", "constant"}
SOURCE_BASE_KEYS = {
    "key",
    "component",
    "repository",
    "revision",
    "path",
    "byteLength",
    "sha256",
}
SOURCE_TENSOR_KEYS = SOURCE_BASE_KEYS | {
    "tensorCount",
    "parameterCount",
    "headerLength",
    "headerSha256",
    "inventorySha256",
}
LICENSE_RECORD_KEYS = {"component", "spdx", "notice", "source"}
REQUIRED_LICENSE_FILES = {
    "licenses/ACE-Step-LICENSE": (
        1_064,
        "05a6bce42a62636d2cfb24139cc008b6b899754e244175814bb5dd2f4a485357",
    ),
    "licenses/Apache-2.0-LICENSE": (
        11_358,
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    "licenses/Qwen-NOTICE.txt": (
        439,
        "c57cecae352eb5793befd1f28f44f351e148c9a28044d855b8c361c562195f0b",
    ),
}
TENSOR_DTYPES = {
    "float16": 2,
    "float32": 4,
    "uint32": 4,
    "uint32-bf16-pairs": 4,
}
TENSOR_PHASES = {"planner", "text", "conditioner", "semantic", "dit", "vae", "constants"}
TENSOR_LIFETIMES = TENSOR_PHASES | {"initial-latent"}
TENSOR_LAYOUTS = {
    "source-row-major",
    "source-row-major-bf16-pairs-lsb-u32",
    "row-shard-axis0",
    "row-shard-axis0-bf16-pairs-lsb-u32",
    "contiguous-nct-f32",
    "conv1d-output-kernel-input-f32-v1",
    "conv-transpose1d-output-kernel-input-f32-v1",
    "channel-vector-f32-v1",
    "conv1d-output-kernel-input-f16-v1",
    "conv-transpose1d-output-kernel-input-f16-v1",
    "conv1d-k1-cout128-cin32-tile-major-f16-v1",
    "conv-transpose1d-phase-tap-input-output-f16-v1",
    VAE_K7_ROW_REUSE_FP16_LAYOUT,
    VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
    "channel-vector-f16-v1",
    "dit-gemm-n128-k32-tile-major-v1",
    DIT_DENSE_FP16_REV7_TILE_LAYOUT,
    DIT_DENSE_FP16_REV8_TILE_LAYOUT,
}
TENSOR_TRANSFORMATIONS = {
    "preserve-bf16-bits-pack-u32-pairs",
    "bf16-to-ieee-fp16",
    "bf16-to-fp32",
    "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
    "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
    "bf16-to-fp32-flatten-1-c-1-to-c-v1",
    "bf16-to-fp32-to-ieee-fp16-v1",
    "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
    "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
    "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
    "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1",
    "validated-pytorch-zip-storage-extraction",
    "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1",
    "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1",
    DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
    DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
}
NATIVE_LAYOUT_BY_TRANSFORMATION = {
    "profile-float-dit-gemm-n128-k32-tile-major-v1": (
        "dit-gemm-n128-k32-tile-major-v1"
    ),
    "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1": (
        "dit-gemm-n128-k32-tile-major-v1"
    ),
    "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1": (
        "dit-gemm-n128-k32-tile-major-v1"
    ),
    DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION: DIT_DENSE_FP16_REV7_TILE_LAYOUT,
    DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION: DIT_DENSE_FP16_REV8_TILE_LAYOUT,
    "weightnorm-fused-fp32-pairwise-oik-to-oki-v1": (
        "conv1d-output-kernel-input-f32-v1"
    ),
    "weightnorm-fused-fp32-pairwise-iok-to-oki-v1": (
        "conv-transpose1d-output-kernel-input-f32-v1"
    ),
    "bf16-to-fp32-flatten-1-c-1-to-c-v1": "channel-vector-f32-v1",
    "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1": (
        "conv1d-output-kernel-input-f16-v1"
    ),
    "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1": (
        "conv-transpose1d-output-kernel-input-f16-v1"
    ),
    "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1": (
        "conv1d-k1-cout128-cin32-tile-major-f16-v1"
    ),
    "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1": (
        "conv-transpose1d-phase-tap-input-output-f16-v1"
    ),
    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION: VAE_K7_ROW_REUSE_FP16_LAYOUT,
    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION: (
        VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT
    ),
    "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1": (
        "channel-vector-f16-v1"
    ),
}
VAE_CONV1D_SOURCE = re.compile(
    r"^vae-weights:decoder\.(?:conv[12]|block\.\d+\.res_unit[123]\.conv[12])\."
    r"weight_v$"
)
VAE_CONV_TRANSPOSE1D_SOURCE = re.compile(
    r"^vae-weights:decoder\.block\.\d+\.conv_t1\.weight_v$"
)
VAE_CHANNEL_VECTOR_SOURCE = re.compile(
    r"^vae-weights:decoder\.(?:snake1|block\.\d+\."
    r"(?:snake1|res_unit[123]\.snake[12]))\.(?:alpha|beta)$"
)
VAE_BIAS_SOURCE = re.compile(
    r"^vae-weights:decoder\.(?:conv1|block\.[0-4]\."
    r"(?:conv_t1|res_unit[123]\.conv[12]))\.bias$"
)
DIT_GEMM_SOURCE = re.compile(
    r"^ace-turbo-weights:decoder\.(?:condition_embedder\.weight|"
    r"time_embed(?:_r)?\.(?:linear_[12]|time_proj)\.weight|"
    r"layers\.(?:[0-9]|1[0-9]|2[0-3])\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"cross_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight)$"
)
DIT_REPEATED_DENSE_SOURCE = re.compile(
    r"^ace-turbo-weights:decoder\.layers\.(?:[0-9]|1[0-9]|2[0-3])\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"cross_attn\.(?:q_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


def validate_native_tensor_contract(
    transformation: str,
    source: str,
    logical_shape: list[int] | None,
    phase: str,
    label: str,
) -> None:
    dit_gemm_transform = transformation in {
        "profile-float-dit-gemm-n128-k32-tile-major-v1",
        "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1",
        "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1",
    }
    dit_dense_fp16_transform = transformation in {
        DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
        DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
    }
    dit_dense_rev7_transform = (
        transformation == DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION
    )
    dit_gemm_source = DIT_GEMM_SOURCE.fullmatch(source) is not None
    dit_dense_fp16_source = DIT_REPEATED_DENSE_SOURCE.fullmatch(source) is not None
    if dit_gemm_transform or dit_dense_fp16_transform or dit_gemm_source:
        if (
            not (dit_gemm_transform or dit_dense_fp16_transform)
            or not dit_gemm_source
            or phase != "dit"
            or (dit_dense_fp16_transform and not dit_dense_fp16_source)
            or (
                logical_shape is not None
                and (
                    len(logical_shape) != 2
                    or logical_shape[0]
                    % (256 if dit_dense_rev7_transform else 128)
                    != 0
                    or logical_shape[1]
                    % (
                        32
                        if dit_dense_rev7_transform or not dit_dense_fp16_transform
                        else 4
                    )
                    != 0
                )
            )
        ):
            raise ValueError(f"{label}: invalid DiT GEMM tile-major contract")
    conv1d_transform = transformation in {
        "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
        "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
        "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
        VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    }
    conv1d_source = VAE_CONV1D_SOURCE.fullmatch(source) is not None
    conv_transpose_transform = transformation in {
        "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
        "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
        "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
        VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    }
    conv_transpose_source = VAE_CONV_TRANSPOSE1D_SOURCE.fullmatch(source) is not None
    channel_transform = transformation in {
        "bf16-to-fp32-flatten-1-c-1-to-c-v1",
        "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1",
    }
    channel_source = VAE_CHANNEL_VECTOR_SOURCE.fullmatch(source) is not None
    bias_transform = transformation in {
        "bf16-to-fp32",
        "bf16-to-fp32-to-ieee-fp16-v1",
    }
    bias_source = VAE_BIAS_SOURCE.fullmatch(source) is not None
    if conv1d_transform or conv1d_source:
        if (
            not conv1d_transform
            or not conv1d_source
            or phase != "vae"
            or (logical_shape is not None and len(logical_shape) != 3)
        ):
            raise ValueError(f"{label}: invalid Conv1d native-layout contract")
    elif conv_transpose_transform or conv_transpose_source:
        if (
            not conv_transpose_transform
            or not conv_transpose_source
            or phase != "vae"
            or (logical_shape is not None and len(logical_shape) != 3)
        ):
            raise ValueError(
                f"{label}: invalid ConvTranspose1d native-layout contract"
            )
    elif channel_transform or channel_source:
        if (
            not channel_transform
            or not channel_source
            or phase != "vae"
            or (logical_shape is not None and len(logical_shape) != 1)
        ):
            raise ValueError(f"{label}: invalid Snake channel-vector contract")
    elif bias_transform or bias_source:
        if (
            not bias_transform
            or not bias_source
            or phase != "vae"
            or (logical_shape is not None and len(logical_shape) != 1)
        ):
            raise ValueError(f"{label}: invalid VAE bias contract")


def package_converter_revision(profile: str) -> int:
    try:
        return PACKAGE_CONVERTER_REVISION_BY_PROFILE[profile]
    except KeyError as error:
        raise ValueError(f"Unsupported profile {profile!r}") from error


def supported_package_converter_revisions(profile: str) -> frozenset[int]:
    """Return every authenticated package revision accepted for a profile."""

    current = package_converter_revision(profile)
    if profile == EXPERIMENTAL_DIT_DENSE_PROFILE:
        return frozenset(
            {PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION, current}
        )
    return frozenset({current})


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key {key!r}")
        value[key] = item
    return value


def sha256_file(path: Path, *, chunk_bytes: int = COPY_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: object) -> None:
    write_bytes_atomic(path, canonical_json_bytes(value))


@dataclass(frozen=True, slots=True)
class TensorRecord:
    shard: str
    byteOffset: int
    byteLength: int
    dtype: str
    logicalShape: list[int]
    storageShape: list[int]
    layout: str
    phase: str
    lifetime: str
    source: str
    transformation: str
    logicalTensor: str
    partAxis: int
    partStart: int
    partEnd: int


def validate_experimental_vae_payload(
    tensors: dict[str, TensorRecord],
) -> None:
    constants = {
        name: tensor
        for name, tensor in tensors.items()
        if tensor.phase == "constants"
    }
    if set(constants) != {SILENCE_LATENT_TENSOR_NAME}:
        raise ValueError(
            "Experimental FP16 VAE package must contain exactly one canonical "
            "constants.silence_latent record"
        )
    silence = constants[SILENCE_LATENT_TENSOR_NAME]
    if (
        silence.shard != SILENCE_LATENT_SHARD
        or silence.byteOffset != 0
        or silence.byteLength != SILENCE_LATENT_BYTES
        or silence.dtype != "float32"
        or silence.logicalShape != SILENCE_LATENT_SHAPE
        or silence.storageShape != SILENCE_LATENT_SHAPE
        or silence.layout != "contiguous-nct-f32"
        or silence.lifetime != "initial-latent"
        or silence.source != SILENCE_LATENT_SOURCE
        or silence.transformation
        != "validated-pytorch-zip-storage-extraction"
        or silence.logicalTensor != SILENCE_LATENT_TENSOR_NAME
        or silence.partAxis != 0
        or silence.partStart != 0
        or silence.partEnd != 1
    ):
        raise ValueError(
            "Experimental FP16 VAE package has a non-canonical "
            "constants.silence_latent record"
        )
    vae_tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if tensor.phase == "vae"
    }
    logical_names = {tensor.logicalTensor for tensor in vae_tensors.values()}
    if (
        len(logical_names) != EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT
        or len(vae_tensors) != EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT
    ):
        raise ValueError(
            "Experimental FP16 VAE must contain exactly "
            f"{EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT} logical tensors and "
            f"{EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT} unsharded records"
        )
    expected_sources = {
        f"vae-weights:{source}": shape
        for source, shape in VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.items()
    }
    if (
        len({tensor.source for tensor in vae_tensors.values()})
        != len(vae_tensors)
        or {tensor.source for tensor in vae_tensors.values()} != set(expected_sources)
    ):
        raise ValueError(
            "Experimental FP16 VAE package does not contain the exact "
            "revision-7 decoder source inventory"
        )
    for name, tensor in vae_tensors.items():
        source_tensor = tensor.source.removeprefix("vae-weights:")
        expected_shape = expected_sources[tensor.source]
        expected_name = (
            f"vae.{source_tensor.removesuffix('.weight_v')}.weight"
            if source_tensor.endswith(".weight_v")
            else f"vae.{source_tensor}"
        )
        if source_tensor.endswith(".weight_v"):
            expected_transformation, expected_layout = (
                revision7_vae_runtime_weight_layout(
                    source_tensor,
                    tuple(tensor.logicalShape),
                )
            )
        elif source_tensor.endswith(".bias"):
            expected_transformation = "bf16-to-fp32-to-ieee-fp16-v1"
            expected_layout = "source-row-major"
        else:
            expected_transformation = (
                "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1"
            )
            expected_layout = "channel-vector-f16-v1"
        expected_storage_shape = list(expected_shape)
        if expected_transformation == VAE_K7_ROW_REUSE_FP16_TRANSFORMATION:
            output_channels, kernel, input_channels = expected_shape
            expected_storage_shape = [
                kernel,
                input_channels // 4,
                output_channels // 64,
                32,
                2,
                4,
            ]
        elif expected_transformation == (
            "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-"
            "tile-major-ieee-fp16-v1"
        ):
            output_channels, _, input_channels = expected_shape
            expected_storage_shape = [
                output_channels // 128,
                input_channels // 32,
                32,
                128,
            ]
        elif expected_transformation == (
            "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-"
            "output-ieee-fp16-v1"
        ):
            output_channels, kernel, input_channels = expected_shape
            expected_storage_shape = [
                kernel // 2,
                2,
                input_channels,
                output_channels,
            ]
        elif expected_transformation == VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION:
            output_channels, kernel, input_channels = expected_shape
            reuse_axis = VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS[source_tensor][1]
            outputs_per_lane = 8 if reuse_axis == "channel" else 4
            expected_storage_shape = [
                kernel // 2,
                2,
                input_channels // 4,
                output_channels // (32 * outputs_per_lane),
                32,
                outputs_per_lane,
                4,
            ]
        if (
            tensor.dtype != "float16"
            or tensor.logicalShape != list(expected_shape)
            or tensor.storageShape != expected_storage_shape
            or tensor.byteLength != math.prod(expected_storage_shape) * 2
            or tensor.partStart != 0
            or tensor.partEnd != tensor.logicalShape[0]
            or tensor.logicalTensor != name
            or name != expected_name
            or tensor.transformation != expected_transformation
            or tensor.layout != expected_layout
        ):
            raise ValueError(
                f"Experimental FP16 VAE tensor {name!r} violates its exact "
                "revision-7 label/shape/layout contract"
            )
    parameter_elements = sum(
        math.prod(tensor.logicalShape) for tensor in vae_tensors.values()
    )
    parameter_bytes = sum(tensor.byteLength for tensor in vae_tensors.values())
    if (
        parameter_elements != EXPERIMENTAL_VAE_PARAMETER_ELEMENTS
        or parameter_bytes != EXPERIMENTAL_VAE_PARAMETER_BYTES
        or parameter_bytes != parameter_elements * TENSOR_DTYPES["float16"]
    ):
        raise ValueError(
            "Experimental FP16 VAE payload is not exactly "
            f"{EXPERIMENTAL_VAE_PARAMETER_ELEMENTS} elements/"
            f"{EXPERIMENTAL_VAE_PARAMETER_BYTES} bytes"
        )


def validate_experimental_dit_dense_payload(
    tensors: dict[str, TensorRecord],
    *,
    converter_revision: int | None = None,
) -> None:
    constants = {
        name: tensor
        for name, tensor in tensors.items()
        if tensor.phase == "constants"
    }
    if set(constants) != {SILENCE_LATENT_TENSOR_NAME}:
        raise ValueError(
            "Experimental FP16 DiT dense package must retain the canonical "
            "silence-latent constant"
        )
    dense_suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    all_layer_suffixes = (
        "scale_shift_table",
        "self_attn_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "cross_attn_norm.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.k_proj.weight",
        "cross_attn.v_proj.weight",
        "cross_attn.q_norm.weight",
        "cross_attn.k_norm.weight",
        "cross_attn.o_proj.weight",
        "mlp_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    expected_names = {
        f"ace.decoder.layers.{layer}.{suffix}"
        for layer in range(24)
        for suffix in all_layer_suffixes
    }
    dense_names = {
        f"ace.decoder.layers.{layer}.{suffix}"
        for layer in range(24)
        for suffix in dense_suffixes
    }
    dit_tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if tensor.phase == "dit"
    }
    if set(dit_tensors) != expected_names or len(dit_tensors) != 456:
        raise ValueError(
            "Experimental mixed DiT package is not exactly the 24 repeated "
            "layers"
        )
    if set(tensors) != expected_names | {SILENCE_LATENT_TENSOR_NAME}:
        raise ValueError("Experimental FP16 DiT dense package has another phase")
    cross_cache_names = {
        f"ace.decoder.layers.{layer}.cross_attn.{projection}_proj.weight"
        for layer in range(24)
        for projection in ("k", "v")
    }
    if converter_revision is None:
        observed_dense_contracts = {
            (tensor.layout, tensor.transformation)
            for name, tensor in dit_tensors.items()
            if name in dense_names
        }
        contract_revision = {
            (
                DIT_DENSE_FP16_REV7_TILE_LAYOUT,
                DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
            ): PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
            (
                DIT_DENSE_FP16_REV8_TILE_LAYOUT,
                DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
            ): EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
        }
        if len(observed_dense_contracts) != 1:
            raise ValueError(
                "Experimental mixed DiT package mixes dense storage revisions"
            )
        converter_revision = contract_revision.get(
            next(iter(observed_dense_contracts))
        )
    dense_contract = {
        PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION: (
            DIT_DENSE_FP16_REV7_TILE_LAYOUT,
            DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
        ),
        EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION: (
            DIT_DENSE_FP16_REV8_TILE_LAYOUT,
            DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
        ),
    }.get(converter_revision)
    if dense_contract is None:
        raise ValueError("Unsupported experimental mixed DiT converter revision")
    dense_layout, dense_transformation = dense_contract
    for name, tensor in dit_tensors.items():
        common_valid = (
            tensor.logicalTensor == name
            and tensor.partStart == 0
            and tensor.partEnd == tensor.logicalShape[0]
        )
        if name in dense_names:
            common_dense_valid = (
                common_valid
                and tensor.dtype == "float16"
                and tensor.layout == dense_layout
                and tensor.transformation == dense_transformation
                and len(tensor.logicalShape) == 2
            )
            valid = common_dense_valid and (
                (
                    converter_revision
                    == PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION
                    and tensor.logicalShape[0] % 256 == 0
                    and tensor.logicalShape[1] % 32 == 0
                    and tensor.storageShape == tensor.logicalShape
                )
                or (
                    converter_revision
                    == EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION
                    and tensor.logicalShape[0] % 128 == 0
                    and tensor.logicalShape[1] % 4 == 0
                    and tensor.storageShape
                    == [
                        tensor.logicalShape[0] // 128,
                        tensor.logicalShape[1] // 4,
                        4,
                        32,
                        4,
                    ]
                )
            )
        elif name in cross_cache_names:
            valid = (
                common_valid
                and tensor.dtype == "uint32-bf16-pairs"
                and tensor.layout == "dit-gemm-n128-k32-tile-major-v1"
                and tensor.transformation
                == "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1"
            )
        else:
            valid = (
                common_valid
                and tensor.dtype == "uint32-bf16-pairs"
                and tensor.layout == "source-row-major-bf16-pairs-lsb-u32"
                and tensor.transformation == "preserve-bf16-bits-pack-u32-pairs"
            )
        if not valid:
            raise ValueError(
                f"Experimental mixed DiT tensor {name!r} violates its exact "
                "storage contract"
            )
    parameter_elements = sum(
        math.prod(tensor.logicalShape) for tensor in dit_tensors.values()
    )
    parameter_bytes = sum(tensor.byteLength for tensor in dit_tensors.values())
    if (
        parameter_elements != EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS
        or parameter_bytes != EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES
    ):
        raise ValueError(
            "Experimental mixed DiT layer payload is not exactly "
            f"{EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS} elements/"
            f"{EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES} bytes"
        )


@dataclass(frozen=True, slots=True)
class FileRecord:
    name: str
    byteLength: int
    sha256: str
    kind: str


class ShardWriter:
    """Write one operation-oriented binary shard with aligned tensor spans."""

    def __init__(
        self,
        output_root: Path,
        shard_name: str,
        records: dict[str, TensorRecord],
        *,
        max_bytes: int = MAX_SHARD_BYTES,
    ) -> None:
        relative = PurePosixPath(shard_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe shard name {shard_name!r}")
        if max_bytes <= 0 or max_bytes > MAX_STORAGE_BINDING_BYTES:
            raise ValueError("max_bytes must fit the portable storage-binding limit")
        self.output_root = output_root
        self.shard_name = relative.as_posix()
        self.path = output_root / relative
        self.partial = self.path.with_name(f".{self.path.name}.partial")
        self.records = records
        self.max_bytes = max_bytes
        self._stream: BinaryIO | None = None
        self.offset = 0

    def __enter__(self) -> ShardWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.partial.open("xb")
        return self

    def add_bytes(
        self,
        name: str,
        chunks: Iterable[bytes],
        *,
        byte_length: int,
        dtype: str,
        logical_shape: Iterable[int],
        storage_shape: Iterable[int],
        layout: str,
        phase: str,
        lifetime: str,
        source: str,
        transformation: str,
        logicalTensor: str,
        partAxis: int,
        partStart: int,
        partEnd: int,
    ) -> None:
        if self._stream is None:
            raise RuntimeError("ShardWriter is not open")
        if name in self.records:
            raise ValueError(f"Duplicate package tensor {name!r}")
        self._align()
        if self.offset + byte_length > self.max_bytes:
            raise ValueError(
                f"{self.shard_name}: adding {name!r} would produce "
                f"{self.offset + byte_length} bytes, over limit {self.max_bytes}"
            )
        start = self.offset
        written = 0
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                chunk = bytes(chunk)
            self._stream.write(chunk)
            written += len(chunk)
        if written != byte_length:
            raise ValueError(
                f"{name!r}: wrote {written} bytes, expected {byte_length}"
            )
        self.offset += written
        self.records[name] = TensorRecord(
            shard=self.shard_name,
            byteOffset=start,
            byteLength=written,
            dtype=dtype,
            logicalShape=list(logical_shape),
            storageShape=list(storage_shape),
            layout=layout,
            phase=phase,
            lifetime=lifetime,
            source=source,
            transformation=transformation,
            logicalTensor=logicalTensor,
            partAxis=partAxis,
            partStart=partStart,
            partEnd=partEnd,
        )

    def _align(self) -> None:
        if self._stream is None:
            raise RuntimeError("ShardWriter is not open")
        padding = (-self.offset) % ALIGNMENT
        if padding:
            self._stream.write(b"\0" * padding)
            self.offset += padding

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            if exc_type is None:
                self._align_with_stream(stream)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            stream.close()
        if exc_type is None:
            os.replace(self.partial, self.path)
        else:
            self.partial.unlink(missing_ok=True)

    def _align_with_stream(self, stream: BinaryIO) -> None:
        padding = (-self.offset) % ALIGNMENT
        if padding:
            stream.write(b"\0" * padding)
            self.offset += padding


def file_record(root: Path, relative_name: str, *, kind: str) -> FileRecord:
    path = root / relative_name
    return FileRecord(
        name=relative_name,
        byteLength=path.stat().st_size,
        sha256=sha256_file(path),
        kind=kind,
    )


def validate_tensor_spans(
    tensors: dict[str, TensorRecord],
    files: dict[str, FileRecord],
) -> None:
    by_shard: dict[str, list[tuple[str, TensorRecord]]] = {}
    for name, tensor in tensors.items():
        if tensor.byteOffset < 0 or tensor.byteLength < 0:
            raise ValueError(f"{name!r}: negative tensor span")
        if tensor.byteOffset % ALIGNMENT != 0:
            raise ValueError(f"{name!r}: offset is not {ALIGNMENT}-byte aligned")
        by_shard.setdefault(tensor.shard, []).append((name, tensor))
    for shard, entries in by_shard.items():
        if shard not in files:
            raise ValueError(f"Tensor shard {shard!r} is absent from files")
        cursor = 0
        for name, tensor in sorted(entries, key=lambda item: item[1].byteOffset):
            if tensor.byteOffset < cursor:
                raise ValueError(f"{name!r}: tensor span overlaps a preceding tensor")
            end = tensor.byteOffset + tensor.byteLength
            if end > files[shard].byteLength:
                raise ValueError(f"{name!r}: tensor span exceeds shard bounds")
            cursor = end


def build_manifest(
    *,
    profile: str,
    source: list[dict[str, object]],
    files: Iterable[FileRecord],
    tensors: dict[str, TensorRecord],
    accounting: dict[str, object],
    licenses: list[dict[str, str]],
    provenance: dict[str, object],
) -> dict[str, object]:
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported profile {profile!r}")
    file_records = list(files)
    file_map = {record.name: record for record in file_records}
    if len(file_map) != len(file_records):
        raise ValueError("Duplicate manifest file")
    if "manifest.json" in file_map:
        raise ValueError("manifest.json cannot list or hash itself")
    validate_tensor_spans(tensors, file_map)
    if profile == EXPERIMENTAL_VAE_PROFILE:
        validate_experimental_vae_payload(tensors)
    if profile == EXPERIMENTAL_DIT_DENSE_PROFILE:
        validate_experimental_dit_dense_payload(
            tensors,
            converter_revision=provenance.get("converterRevision"),
        )
    return {
        "format": FORMAT_VERSION,
        "profile": profile,
        "alignment": ALIGNMENT,
        "portableStorageBindingBytes": MAX_STORAGE_BINDING_BYTES,
        "source": source,
        "files": [asdict(file_map[name]) for name in sorted(file_map)],
        "tensors": {name: asdict(tensors[name]) for name in sorted(tensors)},
        "accounting": accounting,
        "licenses": licenses,
        "provenance": provenance,
    }


def verify_package(root: Path, *, verify_hashes: bool = True) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(
            manifest_path.read_bytes(),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except FileNotFoundError as error:
        raise ValueError(f"Missing package manifest at {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_VERSION:
        raise ValueError("Unknown or missing package format")
    if set(manifest) != MANIFEST_KEYS:
        raise ValueError(
            f"Malformed manifest keys: got {sorted(manifest)}, "
            f"expected {sorted(MANIFEST_KEYS)}"
        )
    if manifest.get("profile") not in SUPPORTED_PROFILES:
        raise ValueError("Unknown package profile")
    if manifest.get("alignment") != ALIGNMENT:
        raise ValueError("Unsupported package alignment")
    if manifest.get("portableStorageBindingBytes") != MAX_STORAGE_BINDING_BYTES:
        raise ValueError("Unsupported portable storage-binding contract")
    raw_files = manifest.get("files")
    raw_tensors = manifest.get("tensors")
    raw_source = manifest.get("source")
    accounting = manifest.get("accounting")
    licenses = manifest.get("licenses")
    provenance = manifest.get("provenance")
    if (
        not isinstance(raw_files, list)
        or not isinstance(raw_tensors, dict)
        or not isinstance(raw_source, list)
        or not isinstance(accounting, dict)
        or not isinstance(licenses, list)
        or not isinstance(provenance, dict)
    ):
        raise ValueError("Malformed package file or tensor inventory")
    if set(accounting) != ACCOUNTING_KEYS or any(
        not isinstance(accounting[key], int)
        or isinstance(accounting[key], bool)
        or accounting[key] < 0
        for key in ACCOUNTING_KEYS
    ):
        raise ValueError("Malformed package accounting")
    if (
        accounting["directlyIncluded"]
        + accounting["consumedByTransform"]
        + accounting["excluded"]
        != accounting["sourceTensors"]
        or accounting["outputTensorsBeforeRowSharding"]
        != accounting["directlyIncluded"]
        or accounting["outputTensorsAfterRowSharding"] != len(raw_tensors)
        or accounting["outputTensorsAfterRowSharding"]
        < accounting["outputTensorsBeforeRowSharding"]
        + accounting["constantTensors"]
    ):
        raise ValueError("Inconsistent package accounting")
    source_keys: set[str] = set()
    source_tensor_counts: dict[str, int] = {}
    source_tensor_count = 0
    for source in raw_source:
        if not isinstance(source, dict):
            raise ValueError("Malformed source record")
        expected_source_keys = (
            SOURCE_TENSOR_KEYS if "tensorCount" in source else SOURCE_BASE_KEYS
        )
        if set(source) != expected_source_keys:
            raise ValueError("Malformed source-record schema")
        key = source["key"]
        revision = source["revision"]
        sha256 = source["sha256"]
        if (
            not isinstance(key, str)
            or not key
            or key in source_keys
            or any(
                not isinstance(source[field], str) or not source[field]
                for field in ("component", "repository", "path")
            )
            or not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(source["byteLength"], int)
            or isinstance(source["byteLength"], bool)
            or source["byteLength"] < 0
        ):
            raise ValueError("Invalid source identity record")
        source_keys.add(key)
        if "tensorCount" in source:
            tensor_count = source["tensorCount"]
            if (
                not isinstance(tensor_count, int)
                or isinstance(tensor_count, bool)
                or tensor_count <= 0
            ):
                raise ValueError("Invalid source tensor count")
            tensor_identity_fields = (
                "parameterCount",
                "headerLength",
                "headerSha256",
                "inventorySha256",
            )
            if not all(field in source for field in tensor_identity_fields):
                raise ValueError("Incomplete safetensor source identity")
            if (
                not isinstance(source["parameterCount"], int)
                or source["parameterCount"] <= 0
                or not isinstance(source["headerLength"], int)
                or source["headerLength"] <= 0
                or any(
                    not isinstance(source[field], str)
                    or len(source[field]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in source[field]
                    )
                    for field in ("headerSha256", "inventorySha256")
                )
            ):
                raise ValueError("Invalid safetensor source identity")
            source_tensor_counts[key] = tensor_count
            source_tensor_count += tensor_count
    if source_tensor_count != accounting["sourceTensors"]:
        raise ValueError("Source tensor totals do not match package accounting")
    if not licenses or any(
        not isinstance(item, dict)
        or set(item) != LICENSE_RECORD_KEYS
        or not all(
            isinstance(item.get(key), str) and item[key]
            for key in ("component", "spdx", "notice", "source")
        )
        for item in licenses
    ):
        raise ValueError("Malformed or absent package license metadata")
    if not {"MIT", "Apache-2.0"}.issubset(
        {item["spdx"] for item in licenses}
    ):
        raise ValueError("Required ACE-Step/Qwen license notices are absent")
    required_provenance = {
        "converterRevision",
        "aceSnapshot",
        "plannerSnapshot",
        "referenceRepository",
        "referenceCommit",
        "referenceLicenseGitBlob",
        "referenceLicenseSha256",
        "determinism",
    }
    if (
        set(provenance) != required_provenance
        or provenance.get("converterRevision")
        not in supported_package_converter_revisions(manifest["profile"])
        or isinstance(provenance.get("converterRevision"), bool)
        or any(
            not isinstance(provenance[key], str) or not provenance[key]
            for key in required_provenance - {"converterRevision"}
        )
    ):
        raise ValueError("Missing converter provenance")
    files: dict[str, FileRecord] = {}
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "byteLength",
            "sha256",
            "kind",
        }:
            raise ValueError("Malformed package file record")
        record = FileRecord(**raw)
        if not isinstance(record.name, str) or not record.name:
            raise ValueError("Malformed package filename")
        relative = PurePosixPath(record.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe package filename {record.name!r}")
        if record.name in files:
            raise ValueError(f"Duplicate package filename {record.name!r}")
        if (
            not isinstance(record.byteLength, int)
            or isinstance(record.byteLength, bool)
            or record.byteLength < 0
            or not isinstance(record.sha256, str)
            or len(record.sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in record.sha256
            )
            or record.kind not in FILE_KINDS
        ):
            raise ValueError(f"Malformed package file record {record.name!r}")
        if record.kind == "weights" and record.byteLength > MAX_SHARD_BYTES:
            raise ValueError(f"Weight shard exceeds package limit: {record.name!r}")
        if record.kind == "constant" and record.byteLength > MAX_STORAGE_BINDING_BYTES:
            raise ValueError(f"Constant exceeds portable binding limit: {record.name!r}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe package payload {record.name!r}")
        if path.stat().st_size != record.byteLength:
            raise ValueError(f"Package length mismatch for {record.name!r}")
        if verify_hashes and sha256_file(path) != record.sha256:
            raise ValueError(f"Package SHA-256 mismatch for {record.name!r}")
        files[record.name] = record
    for license_path, (expected_bytes, expected_sha256) in REQUIRED_LICENSE_FILES.items():
        record = files.get(license_path)
        if (
            record is None
            or record.kind != "license"
            or record.byteLength != expected_bytes
            or record.sha256 != expected_sha256
        ):
            raise ValueError(
                f"Missing or modified required license payload {license_path!r}"
            )
    expected_paths = set(files) | {"manifest.json"}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            f"Package file inventory mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    tensor_field_names = set(TensorRecord.__dataclass_fields__)
    if any(
        not isinstance(raw, dict) or set(raw) != tensor_field_names
        for raw in raw_tensors.values()
    ):
        raise ValueError("Malformed tensor-record schema")
    tensors = {name: TensorRecord(**raw) for name, raw in raw_tensors.items()}
    for name, tensor in tensors.items():
        if (
            not isinstance(name, str)
            or not name
            or tensor.dtype not in TENSOR_DTYPES
            or tensor.layout not in TENSOR_LAYOUTS
            or tensor.transformation not in TENSOR_TRANSFORMATIONS
            or not tensor.logicalShape
            or not tensor.storageShape
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                for dimension in tensor.logicalShape + tensor.storageShape
            )
            or math.prod(tensor.storageShape) * TENSOR_DTYPES[tensor.dtype]
            != tensor.byteLength
            or tensor.phase not in TENSOR_PHASES
            or tensor.lifetime not in TENSOR_LIFETIMES
            or not isinstance(tensor.logicalTensor, str)
            or not tensor.logicalTensor
            or not isinstance(tensor.partAxis, int)
            or isinstance(tensor.partAxis, bool)
            or tensor.partAxis != 0
            or not isinstance(tensor.partStart, int)
            or isinstance(tensor.partStart, bool)
            or not isinstance(tensor.partEnd, int)
            or isinstance(tensor.partEnd, bool)
            or tensor.partStart < 0
            or tensor.partEnd <= tensor.partStart
            or tensor.partEnd > tensor.logicalShape[0]
            or not all(
                isinstance(value, str) and value
                for value in (
                    tensor.layout,
                    tensor.phase,
                    tensor.lifetime,
                    tensor.source,
                    tensor.transformation,
                )
            )
        ):
            raise ValueError(f"Malformed tensor record {name!r}")
        validate_native_tensor_contract(
            tensor.transformation,
            tensor.source,
            tensor.logicalShape,
            tensor.phase,
            name,
        )
        part_extent = tensor.partEnd - tensor.partStart
        part_shape = [part_extent, *tensor.logicalShape[1:]]
        part_elements = math.prod(part_shape)
        row_sharded = (
            tensor.partStart != 0
            or tensor.partEnd != tensor.logicalShape[0]
        )
        if (
            tensor.transformation
            in {
                "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1",
                "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1",
                DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
                DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
                "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
                "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
                VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
                VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
            }
            and row_sharded
        ):
            raise ValueError(
                f"Tensor {name!r} must contain its complete tile-major DiT "
                "GEMM matrix"
            )
        expected_layout_base = (
            "row-shard-axis0" if row_sharded else "source-row-major"
        )
        if (
            tensor.transformation == DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION
        ):
            columns, inner = tensor.logicalShape
            if columns % 128 != 0 or inner % 4 != 0:
                raise ValueError(
                    f"Tensor {name!r} has an invalid packed DiT K4 shape"
                )
            expected_storage_shape = [
                columns // 128,
                inner // 4,
                4,
                32,
                4,
            ]
            expected_byte_length = part_elements * TENSOR_DTYPES[tensor.dtype]
            expected_layout = NATIVE_LAYOUT_BY_TRANSFORMATION[
                tensor.transformation
            ]
        elif (
            tensor.transformation
            == "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1"
        ):
            output_channels, kernel, input_channels = tensor.logicalShape
            if (
                kernel != 1
                or output_channels != input_channels
                or output_channels % 128 != 0
                or input_channels % 32 != 0
            ):
                raise ValueError(f"Tensor {name!r} has an invalid packed K1 shape")
            expected_storage_shape = [
                output_channels // 128,
                input_channels // 32,
                32,
                128,
            ]
            expected_byte_length = part_elements * TENSOR_DTYPES[tensor.dtype]
            expected_layout = NATIVE_LAYOUT_BY_TRANSFORMATION[tensor.transformation]
        elif tensor.transformation == VAE_K7_ROW_REUSE_FP16_TRANSFORMATION:
            output_channels, kernel, input_channels = tensor.logicalShape
            source_tensor = tensor.source.removeprefix("vae-weights:")
            expected_transform, _ = revision7_vae_runtime_weight_layout(
                source_tensor,
                tuple(tensor.logicalShape),
            )
            if (
                expected_transform != tensor.transformation
                or kernel != 7
                or output_channels % 128 != 0
                or input_channels % 4 != 0
            ):
                raise ValueError(
                    f"Tensor {name!r} has an invalid row-reuse K7 shape"
                )
            expected_storage_shape = [
                kernel,
                input_channels // 4,
                output_channels // 64,
                32,
                2,
                4,
            ]
            expected_byte_length = part_elements * TENSOR_DTYPES[tensor.dtype]
            expected_layout = NATIVE_LAYOUT_BY_TRANSFORMATION[tensor.transformation]
        elif (
            tensor.transformation
            == "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1"
        ):
            output_channels, kernel, input_channels = tensor.logicalShape
            if kernel % 2 != 0:
                raise ValueError(
                    f"Tensor {name!r} has an invalid polyphase transpose shape"
                )
            expected_storage_shape = [
                kernel // 2,
                2,
                input_channels,
                output_channels,
            ]
            expected_byte_length = part_elements * TENSOR_DTYPES[tensor.dtype]
            expected_layout = NATIVE_LAYOUT_BY_TRANSFORMATION[tensor.transformation]
        elif tensor.transformation == VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION:
            output_channels, kernel, input_channels = tensor.logicalShape
            source_tensor = tensor.source.removeprefix("vae-weights:")
            expected_transform, _ = revision7_vae_runtime_weight_layout(
                source_tensor,
                tuple(tensor.logicalShape),
            )
            contract = VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS.get(source_tensor)
            if (
                expected_transform != tensor.transformation
                or contract is None
                or source_tensor == "decoder.block.0.conv_t1.weight_v"
                or kernel % 2 != 0
                or input_channels % 4 != 0
            ):
                raise ValueError(
                    f"Tensor {name!r} has an invalid transpose K4 shape"
                )
            outputs_per_lane = 8 if contract[1] == "channel" else 4
            output_tile = 32 * outputs_per_lane
            if output_channels % output_tile != 0:
                raise ValueError(
                    f"Tensor {name!r} has an invalid transpose K4 output tile"
                )
            expected_storage_shape = [
                kernel // 2,
                2,
                input_channels // 4,
                output_channels // output_tile,
                32,
                outputs_per_lane,
                4,
            ]
            expected_byte_length = part_elements * TENSOR_DTYPES[tensor.dtype]
            expected_layout = NATIVE_LAYOUT_BY_TRANSFORMATION[tensor.transformation]
        elif tensor.dtype == "uint32-bf16-pairs":
            expected_storage_shape = [(part_elements + 1) // 2]
            expected_byte_length = expected_storage_shape[0] * 4
            expected_layout = NATIVE_LAYOUT_BY_TRANSFORMATION.get(
                tensor.transformation,
                expected_layout_base + "-bf16-pairs-lsb-u32",
            )
        else:
            expected_storage_shape = part_shape
            expected_byte_length = part_elements * TENSOR_DTYPES[tensor.dtype]
            expected_layout = (
                "contiguous-nct-f32"
                if tensor.phase == "constants"
                else NATIVE_LAYOUT_BY_TRANSFORMATION.get(
                    tensor.transformation,
                    expected_layout_base,
                )
            )
        if (
            tensor.storageShape != expected_storage_shape
            or tensor.byteLength != expected_byte_length
            or tensor.layout != expected_layout
        ):
            raise ValueError(
                f"Tensor {name!r} storage shape, bytes, or layout does not "
                "match its exact logical part extent"
            )
        shard = files.get(tensor.shard)
        if shard is None or shard.kind not in {"weights", "constant"}:
            raise ValueError(f"Tensor {name!r} does not reference a tensor payload")
    validate_tensor_spans(tensors, files)
    parts_by_logical: dict[str, list[tuple[str, TensorRecord]]] = {}
    for name, tensor in tensors.items():
        parts_by_logical.setdefault(tensor.logicalTensor, []).append((name, tensor))
    for logical_name, parts in parts_by_logical.items():
        ordered = sorted(parts, key=lambda item: (item[1].partStart, item[1].partEnd))
        first = ordered[0][1]
        cursor = 0
        for part_name, part in ordered:
            if (
                part.partStart != cursor
                or part.logicalShape != first.logicalShape
                or part.dtype != first.dtype
                or part.source != first.source
                or part.transformation != first.transformation
                or part.phase != first.phase
                or part.lifetime != first.lifetime
            ):
                raise ValueError(
                    f"Logical tensor {logical_name!r} has inconsistent, gapped, "
                    f"or overlapping part {part_name!r}"
                )
            cursor = part.partEnd
        if cursor != first.logicalShape[0]:
            raise ValueError(f"Logical tensor {logical_name!r} is incomplete")
    if manifest["profile"] == EXPERIMENTAL_VAE_PROFILE:
        validate_experimental_vae_payload(tensors)
    if manifest["profile"] == EXPERIMENTAL_DIT_DENSE_PROFILE:
        validate_experimental_dit_dense_payload(
            tensors,
            converter_revision=provenance["converterRevision"],
        )
    conversion_plans = [record for record in files.values() if record.kind == "conversion-plan"]
    if len(conversion_plans) != 1:
        raise ValueError("Package must contain exactly one conversion plan")
    plan_path = root / conversion_plans[0].name
    plan = json.loads(
        plan_path.read_bytes(),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    if (
        not isinstance(plan, dict)
        or set(plan) != PLAN_KEYS
        or plan.get("schema") != "ace-step-conversion-plan-v2"
    ):
        raise ValueError("Unknown conversion-plan schema")
    decisions = plan.get("decisions")
    outputs = plan.get("outputs")
    summary = plan.get("summary")
    if (
        not isinstance(decisions, list)
        or not isinstance(outputs, list)
        or not isinstance(summary, dict)
        or len(decisions) != accounting["sourceTensors"]
        or len(outputs) != accounting["outputTensorsBeforeRowSharding"]
        or set(summary) != PLAN_SUMMARY_KEYS
        or any(summary[key] != accounting[key] for key in PLAN_SUMMARY_KEYS)
    ):
        raise ValueError("Conversion plan does not match manifest accounting")
    if any(
        not isinstance(decision, dict) or set(decision) != PLAN_DECISION_KEYS
        for decision in decisions
    ):
        raise ValueError("Malformed conversion-plan decision schema")
    if any(
        not isinstance(output, dict) or set(output) != PLAN_OUTPUT_KEYS
        for output in outputs
    ):
        raise ValueError("Malformed conversion-plan output schema")

    decision_keys: set[tuple[str, str]] = set()
    decisions_by_source: dict[str, int] = {}
    included_relations: set[tuple[str, str, str]] = set()
    consumed_decisions: list[dict[str, object]] = []
    for decision in decisions:
        source = decision["source"]
        tensor = decision["tensor"]
        disposition = decision["disposition"]
        reason = decision["reason"]
        output_name = decision["output"]
        if (
            not isinstance(source, str)
            or source not in source_tensor_counts
            or not isinstance(tensor, str)
            or not tensor
            or not isinstance(disposition, str)
            or disposition not in PLAN_DISPOSITIONS
            or not isinstance(reason, str)
            or not reason
            or (source, tensor) in decision_keys
            or (
                disposition == "excluded"
                and output_name is not None
            )
            or (
                disposition != "excluded"
                and (not isinstance(output_name, str) or not output_name)
            )
        ):
            raise ValueError("Malformed conversion-plan decision record")
        decision_keys.add((source, tensor))
        decisions_by_source[source] = decisions_by_source.get(source, 0) + 1
        if disposition == "included":
            assert isinstance(output_name, str)
            included_relations.add((source, tensor, output_name))
        elif disposition == "consumed-by-transform":
            consumed_decisions.append(decision)
    if decisions_by_source != source_tensor_counts:
        raise ValueError("Conversion-plan decisions do not cover source inventories")

    planned_outputs: dict[str, dict[str, object]] = {}
    output_relations: set[tuple[str, str, str]] = set()
    for output in outputs:
        source = output["source"]
        source_tensor = output["sourceTensor"]
        output_name = output["output"]
        phase = output["phase"]
        lifetime = output["lifetime"]
        group = output["group"]
        transformation = output["transformation"]
        output_dtype = output["outputDtype"]
        runtime_layout = output["runtimeLayout"]
        expected_runtime_layout = (
            NATIVE_LAYOUT_BY_TRANSFORMATION.get(
                transformation,
                "source-row-major",
            )
            if isinstance(transformation, str)
            else None
        )
        expected_output_dtype = (
            "profile-float"
            if transformation
            in {
                "profile-float-storage",
                "profile-float-dit-gemm-n128-k32-tile-major-v1",
            }
            else "float16"
            if transformation
            in {
                DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
                DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
                "bf16-to-fp32-to-ieee-fp16-v1",
                "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
                "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
                "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
                "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
                VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
                VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
                "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1",
            }
            else "float32"
        )
        if isinstance(source, str) and isinstance(source_tensor, str):
            validate_native_tensor_contract(
                transformation if isinstance(transformation, str) else "",
                f"{source}:{source_tensor}",
                None,
                phase if isinstance(phase, str) else "",
                f"conversion-plan output {output_name!r}",
            )
        if (
            not isinstance(source, str)
            or source not in source_tensor_counts
            or not isinstance(source_tensor, str)
            or not source_tensor
            or not isinstance(output_name, str)
            or not output_name
            or output_name in planned_outputs
            or not isinstance(phase, str)
            or phase not in TENSOR_PHASES - {"constants"}
            or not isinstance(lifetime, str)
            or lifetime not in TENSOR_LIFETIMES
            or not isinstance(group, str)
            or not group
            or not group.startswith(f"{phase}/")
            or PurePosixPath(group).is_absolute()
            or ".." in PurePosixPath(group).parts
            or not isinstance(transformation, str)
            or transformation not in PLAN_TRANSFORMATIONS
            or not isinstance(output_dtype, str)
            or output_dtype not in PLAN_OUTPUT_DTYPES
            or not isinstance(runtime_layout, str)
            or runtime_layout not in PLAN_RUNTIME_LAYOUTS
            or runtime_layout != expected_runtime_layout
            or output_dtype != expected_output_dtype
        ):
            raise ValueError("Malformed conversion-plan output record")
        planned_outputs[output_name] = output
        output_relations.add((source, source_tensor, output_name))
    if output_relations != included_relations:
        raise ValueError(
            "Conversion-plan outputs do not exactly match included source tensors"
        )
    consumed_output_names: set[str] = set()
    for decision in consumed_decisions:
        output_name = decision["output"]
        assert isinstance(output_name, str)
        target = planned_outputs.get(output_name)
        consumed_tensor = decision["tensor"]
        assert isinstance(consumed_tensor, str)
        expected_companion = (
            consumed_tensor.removesuffix(".weight_g") + ".weight_v"
            if consumed_tensor.endswith(".weight_g")
            else None
        )
        if (
            target is None
            or target["source"] != decision["source"]
            or expected_companion is None
            or target["sourceTensor"] != expected_companion
            or target["transformation"]
            not in {
                "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
                "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
                "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
                "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
                "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
                "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
                VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
                VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
            }
            or target["outputDtype"]
            != (
                "float16"
                if target["transformation"]
                in {
                    "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
                    "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
                    "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
                    "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
                    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
                    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
                }
                else "float32"
            )
        ):
            raise ValueError(
                "Consumed conversion-plan tensor has no matching transform chain"
            )
        consumed_output_names.add(output_name)
    weightnorm_output_names = {
        output_name
        for output_name, output in planned_outputs.items()
        if output["transformation"]
        in {
            "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
            "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
            "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
            "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
            "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
            "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
            VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
            VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
        }
    }
    if consumed_output_names != weightnorm_output_names:
        raise ValueError(
            "Every weight-normalized output must authenticate one weight_g source"
        )
    disposition_counts = {
        disposition: sum(
            isinstance(decision, dict)
            and decision.get("disposition") == disposition
            for decision in decisions
        )
        for disposition in ("included", "consumed-by-transform", "excluded")
    }
    if (
        disposition_counts["included"] != accounting["directlyIncluded"]
        or disposition_counts["consumed-by-transform"]
        != accounting["consumedByTransform"]
        or disposition_counts["excluded"] != accounting["excluded"]
    ):
        raise ValueError("Conversion-plan dispositions or outputs are inconsistent")

    constant_logical = {
        logical_name
        for logical_name, parts in parts_by_logical.items()
        if parts[0][1].phase == "constants"
    }
    nonconstant_logical = set(parts_by_logical) - constant_logical
    if (
        len(constant_logical) != accounting["constantTensors"]
        or nonconstant_logical != set(planned_outputs)
    ):
        raise ValueError(
            "Manifest logical tensors do not exactly match conversion-plan outputs"
        )
    expected_profile_policy = {
        ("reference", "profile-float-storage", "profile-float"): (
            "uint32-bf16-pairs",
            "preserve-bf16-bits-pack-u32-pairs",
        ),
        ("fp16", "profile-float-storage", "profile-float"): (
            "float16",
            "bf16-to-ieee-fp16",
        ),
        (
            EXPERIMENTAL_VAE_PROFILE,
            "profile-float-storage",
            "profile-float",
        ): (
            "float16",
            "bf16-to-ieee-fp16",
        ),
        (
            "reference",
            "profile-float-dit-gemm-n128-k32-tile-major-v1",
            "profile-float",
        ): (
            "uint32-bf16-pairs",
            "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1",
        ),
        (
            "fp16",
            "profile-float-dit-gemm-n128-k32-tile-major-v1",
            "profile-float",
        ): (
            "float16",
            "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1",
        ),
        (
            EXPERIMENTAL_VAE_PROFILE,
            "profile-float-dit-gemm-n128-k32-tile-major-v1",
            "profile-float",
        ): (
            "float16",
            "bf16-to-ieee-fp16-dit-gemm-n128-k32-tile-major-v1",
        ),
        (
            EXPERIMENTAL_DIT_DENSE_PROFILE,
            "profile-float-storage",
            "profile-float",
        ): (
            "uint32-bf16-pairs",
            "preserve-bf16-bits-pack-u32-pairs",
        ),
        (
            EXPERIMENTAL_DIT_DENSE_PROFILE,
            "profile-float-dit-gemm-n128-k32-tile-major-v1",
            "profile-float",
        ): (
            "uint32-bf16-pairs",
            "preserve-bf16-bits-dit-gemm-n128-k32-tile-major-v1",
        ),
        (
            EXPERIMENTAL_DIT_DENSE_PROFILE,
            DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
            "float16",
        ): (
            "float16",
            DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
        ),
        (
            EXPERIMENTAL_DIT_DENSE_PROFILE,
            DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
            "float16",
        ): (
            "float16",
            DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
        ),
        ("reference", "bf16-to-fp32", "float32"): (
            "float32",
            "bf16-to-fp32",
        ),
        ("fp16", "bf16-to-fp32", "float32"): (
            "float32",
            "bf16-to-fp32",
        ),
        (
            "reference",
            "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
            "float32",
        ): (
            "float32",
            "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
        ),
        (
            "fp16",
            "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
            "float32",
        ): (
            "float32",
            "weightnorm-fused-fp32-pairwise-oik-to-oki-v1",
        ),
        (
            "reference",
            "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
            "float32",
        ): (
            "float32",
            "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
        ),
        (
            "fp16",
            "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
            "float32",
        ): (
            "float32",
            "weightnorm-fused-fp32-pairwise-iok-to-oki-v1",
        ),
        (
            "reference",
            "bf16-to-fp32-flatten-1-c-1-to-c-v1",
            "float32",
        ): (
            "float32",
            "bf16-to-fp32-flatten-1-c-1-to-c-v1",
        ),
        (
            EXPERIMENTAL_VAE_PROFILE,
            "bf16-to-fp32-to-ieee-fp16-v1",
            "float16",
        ): ("float16", "bf16-to-fp32-to-ieee-fp16-v1"),
        (
            EXPERIMENTAL_VAE_PROFILE,
            "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
            "float16",
        ): (
            "float16",
            "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
        ),
        (
            EXPERIMENTAL_VAE_PROFILE,
            "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
            "float16",
        ): (
            "float16",
            "weightnorm-fused-fp32-pairwise-iok-to-oki-ieee-fp16-v1",
        ),
        (
            EXPERIMENTAL_VAE_PROFILE,
            "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
            "float16",
        ): (
            "float16",
            "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
        ),
        (
            EXPERIMENTAL_VAE_PROFILE,
            "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
            "float16",
        ): (
            "float16",
            "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
        ),
        (
            EXPERIMENTAL_VAE_PROFILE,
            VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
            "float16",
        ): ("float16", VAE_K7_ROW_REUSE_FP16_TRANSFORMATION),
        (
            EXPERIMENTAL_VAE_PROFILE,
            VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
            "float16",
        ): ("float16", VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION),
        (
            EXPERIMENTAL_VAE_PROFILE,
            "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1",
            "float16",
        ): (
            "float16",
            "bf16-to-fp32-flatten-1-c-1-to-c-ieee-fp16-v1",
        ),
        (
            "fp16",
            "bf16-to-fp32-flatten-1-c-1-to-c-v1",
            "float32",
        ): (
            "float32",
            "bf16-to-fp32-flatten-1-c-1-to-c-v1",
        ),
    }
    profile = manifest["profile"]
    for output_name, output in planned_outputs.items():
        parts = parts_by_logical[output_name]
        first = parts[0][1]
        policy_key = (
            profile,
            output["transformation"],
            output["outputDtype"],
        )
        expected_policy = expected_profile_policy.get(policy_key)
        expected_source = f"{output['source']}:{output['sourceTensor']}"
        expected_shard_prefix = f"weights/{output['group']}-"
        if (
            expected_policy is None
            or (first.dtype, first.transformation) != expected_policy
            or first.source != expected_source
            or first.phase != output["phase"]
            or first.lifetime != output["lifetime"]
            or any(
                not part.shard.startswith(expected_shard_prefix)
                or not part.shard.endswith(".bin")
                for _, part in parts
            )
        ):
            raise ValueError(
                f"Manifest tensor {output_name!r} does not implement its exact "
                "conversion-plan source, phase, lifetime, group, dtype, and "
                "transformation policy"
            )
    return manifest


def verify_pinned_package_for_replacement(
    root: Path,
    expected_manifest_sha256: str,
) -> None:
    """Authenticate an older canonical package without accepting its schema.

    The exact committed manifest digest is the trust root. Its complete file
    inventory is still length/hash checked and must exactly cover the target,
    so a copied manifest cannot authorize deleting unrelated user files.
    """

    if (
        len(expected_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_manifest_sha256
        )
    ):
        raise ValueError("Invalid pinned replacement manifest SHA-256")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Pinned replacement target has no safe manifest")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Replacement target does not match its pinned manifest")
    manifest = json.loads(
        manifest_path.read_bytes(),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    raw_files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(raw_files, list):
        raise ValueError("Pinned replacement manifest has no file inventory")
    expected_paths = {"manifest.json"}
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "byteLength",
            "sha256",
            "kind",
        }:
            raise ValueError("Malformed pinned replacement file record")
        name = raw["name"]
        byte_length = raw["byteLength"]
        digest = raw["sha256"]
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (
            relative is None
            or not name
            or relative.is_absolute()
            or ".." in relative.parts
            or name in expected_paths
            or not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Malformed pinned replacement file identity")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != byte_length
            or sha256_file(path) != digest
        ):
            raise ValueError(f"Pinned replacement payload mismatch for {name!r}")
        expected_paths.add(name)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != expected_paths:
        raise ValueError("Pinned replacement target contains unrecognized files")


def install_staged_directory(
    staging: Path,
    target: Path,
    *,
    expected_existing_manifest_sha256: str | None = None,
) -> None:
    """Atomically install staging, restoring a recognized target on failure."""

    if staging.parent != target.parent:
        raise ValueError("Staging and target must share a parent for atomic rename")
    verify_package(staging)
    if target.is_symlink():
        raise ValueError(f"Refusing symbolic-link target {target}")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"Refusing non-directory target {target}")
        if any(target.iterdir()):
            try:
                verify_package(target)
            except ValueError:
                if expected_existing_manifest_sha256 is None:
                    raise
                verify_pinned_package_for_replacement(
                    target,
                    expected_existing_manifest_sha256,
                )
    backup = target.with_name(f".{target.name}.previous")
    if backup.exists():
        raise ValueError(f"Stale package backup requires inspection: {backup}")
    replaced = False
    try:
        if target.exists():
            os.replace(target, backup)
            replaced = True
        os.replace(staging, target)
    except BaseException:
        if replaced and not target.exists() and backup.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)
