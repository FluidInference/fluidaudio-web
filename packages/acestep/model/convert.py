#!/usr/bin/env python3
"""Download, validate, and package ACE-Step 1.5 for the WebGPU runtime.

Every remote file is pinned by repository commit, byte length, and SHA-256.
Safetensor payloads are consumed through read-only mmap views. Output is staged,
independently verified, and atomically installed only after it is complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from conversion_plan import (
    DIT_DENSE_FP16_REV7_TILE_LAYOUT,
    DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
    DIT_DENSE_FP16_REV8_TILE_LAYOUT,
    DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
    DIT_GEMM_TILE_LAYOUT,
    DIT_GEMM_TILE_TRANSFORMATION,
    DIT_REPEATED_DENSE_WEIGHT_NAMES,
    EXPERIMENTAL_DIT_DENSE_PROFILE,
    EXPERIMENTAL_VAE_PROFILE,
    SOURCE_ROW_MAJOR_LAYOUT,
    VAE_BIAS_FP16_TRANSFORMATION,
    VAE_CHANNEL_VECTOR_FP16_LAYOUT,
    VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
    VAE_CHANNEL_VECTOR_TRANSFORMATION,
    VAE_CONV1D_FP16_TRANSFORMATION,
    VAE_CONV1D_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION,
    VAE_CONV_TRANSPOSE1D_TRANSFORMATION,
    VAE_K1_FP16_TILE_TRANSFORMATION,
    VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE,
    VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS,
    ConversionPlan,
    OutputTensorPlan,
    assert_canonical_encode_only_exclusions,
    build_conversion_plan,
    revision7_vae_runtime_weight_layout,
)
from download import download_artifact, verify_source_file
from package_format import (
    ALIGNMENT,
    EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT,
    EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES,
    EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS,
    EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    EXPERIMENTAL_VAE_PARAMETER_BYTES,
    EXPERIMENTAL_VAE_PARAMETER_ELEMENTS,
    EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
    MAX_SHARD_BYTES,
    PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
    FileRecord,
    ShardWriter,
    TensorRecord,
    build_manifest,
    file_record,
    install_staged_directory,
    package_converter_revision,
    sha256_file,
    verify_package,
    write_json_atomic,
)
from safetensors_mmap import SafetensorsFile, TensorInfo
from silence_latent import (
    TENSOR_DTYPE as SILENCE_DTYPE,
    TENSOR_MEMBER as SILENCE_MEMBER,
    TENSOR_SHAPE as SILENCE_SHAPE,
    extract_silence_latent,
)
from source_contract import (
    ACE_REVISION,
    ARTIFACT_BY_KEY,
    PLANNER_REVISION,
    REFERENCE_LICENSE_GIT_BLOB,
    REFERENCE_LICENSE_SHA256,
    REFERENCE_REPOSITORY,
    REFERENCE_REVISION,
    SAFETENSOR_ARTIFACTS,
    SOURCE_ARTIFACTS,
)


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = MODEL_ROOT / "cache"
DEFAULT_OUTPUT = {
    "reference": MODEL_ROOT / "files-reference",
    "fp16": MODEL_ROOT / "files-fp16",
    EXPERIMENTAL_VAE_PROFILE: MODEL_ROOT / "files-fp16-vae-experimental",
    EXPERIMENTAL_DIT_DENSE_PROFILE: (
        MODEL_ROOT / "files-fp16-dit-layer-mixed-experimental"
    ),
}
PRODUCTION_PROFILE = "production"
PRODUCTION_DIT_PROFILE = "production-dit-rev7"
PRODUCTION_VAE_PROFILE = "production-vae-rev7"
PRODUCTION_OUTPUT = {
    "reference": MODEL_ROOT / "files-reference",
    "dit": MODEL_ROOT / "files-fp16-dit-rev7-oracle",
    "vae": MODEL_ROOT / "files-fp16-vae-revision7-experimental",
}
CANONICAL_PACKAGES = MODEL_ROOT / "canonical-packages.json"
PIECE_TARGET_BYTES = 96 * 1024 * 1024
LOCAL_LICENSE_FILES = (
    "licenses/ACE-Step-LICENSE",
    "licenses/Apache-2.0-LICENSE",
    "licenses/Qwen-NOTICE.txt",
)


def canonical_replacement_manifest_sha256(
    profile: str,
    target: Path,
) -> str | None:
    """Return the committed identity that may authorize a canonical upgrade."""

    canonical_target = DEFAULT_OUTPUT.get(profile)
    if canonical_target is None or target.resolve() != canonical_target.resolve():
        return None
    record = json.loads(CANONICAL_PACKAGES.read_bytes())
    package_key = (
        "replacementPackages"
        if profile in {
            EXPERIMENTAL_VAE_PROFILE,
            EXPERIMENTAL_DIT_DENSE_PROFILE,
        }
        else "packages"
    )
    packages = record.get(package_key) if isinstance(record, dict) else None
    package = packages.get(profile) if isinstance(packages, dict) else None
    if profile in {
        EXPERIMENTAL_VAE_PROFILE,
        EXPERIMENTAL_DIT_DENSE_PROFILE,
    } and package is None:
        return None
    digest = package.get("manifestSha256") if isinstance(package, dict) else None
    if (
        record.get("packageFormat") != "ace-step-webgpu-v1"
        or not isinstance(package, dict)
        or package.get("directory") != target.name
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Malformed canonical package replacement identity")
    return digest


@dataclass(frozen=True, slots=True)
class PackagePreparation:
    label: str
    profile: str
    target: Path
    converter_revision: int
    dit_dense_converter_revision: int = 8
    expected_manifest_sha256: str | None = None


def production_package_record(component: str) -> dict[str, object]:
    """Return one closed, committed production package identity."""

    record = json.loads(CANONICAL_PACKAGES.read_bytes())
    if not isinstance(record, dict):
        raise ValueError("Malformed canonical package identity record")
    packages = record.get("productionPackages")
    package = packages.get(component) if isinstance(packages, dict) else None
    expected_directory = PRODUCTION_OUTPUT.get(component)
    required_keys = {
        "profile",
        "converterRevision",
        "directory",
        "manifestSha256",
        "manifestBytes",
        "listedFileBytes",
        "packageBytesIncludingManifest",
    }
    digest = package.get("manifestSha256") if isinstance(package, dict) else None
    if (
        record.get("packageFormat") != "ace-step-webgpu-v1"
        or not isinstance(package, dict)
        or set(package) != required_keys
        or expected_directory is None
        or package.get("directory") != expected_directory.name
        or not isinstance(package.get("profile"), str)
        or not isinstance(package.get("converterRevision"), int)
        or isinstance(package.get("converterRevision"), bool)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or any(
            not isinstance(package.get(key), int)
            or isinstance(package.get(key), bool)
            or package[key] <= 0
            for key in (
                "manifestBytes",
                "listedFileBytes",
                "packageBytesIncludingManifest",
            )
        )
        or package["manifestBytes"] + package["listedFileBytes"]
        != package["packageBytesIncludingManifest"]
    ):
        raise ValueError(f"Malformed production package identity {component!r}")
    return package


def production_preparations() -> tuple[PackagePreparation, ...]:
    """Resolve the exact package tuple selected by the browser demo."""

    definitions = (
        ("reference", "reference", 4, 8),
        (
            "dit",
            EXPERIMENTAL_DIT_DENSE_PROFILE,
            PRODUCTION_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
            7,
        ),
        ("vae", EXPERIMENTAL_VAE_PROFILE, 7, 8),
    )
    preparations: list[PackagePreparation] = []
    for component, profile, revision, dit_revision in definitions:
        package = production_package_record(component)
        if (
            package["profile"] != profile
            or package["converterRevision"] != revision
        ):
            raise ValueError(
                f"Production package recipe {component!r} differs from its identity"
            )
        preparations.append(PackagePreparation(
            label=f"production-{component}",
            profile=profile,
            target=PRODUCTION_OUTPUT[component],
            converter_revision=revision,
            dit_dense_converter_revision=dit_revision,
            expected_manifest_sha256=str(package["manifestSha256"]),
        ))
    return tuple(preparations)


def require_manifest_identity(path: Path, expected_sha256: str | None) -> str:
    """Hash a staged manifest and enforce its external production trust root."""

    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Generated manifest identity {actual_sha256} does not match "
            f"production identity {expected_sha256}"
        )
    return actual_sha256


@dataclass(frozen=True, slots=True)
class Piece:
    plan: OutputTensorPlan
    source_info: TensorInfo
    logical_shape: tuple[int, ...]
    row_start: int
    row_end: int
    output_name: str
    output_bytes: int


def bf16_bytes_to_float32(payload: bytes) -> np.ndarray:
    bits = np.frombuffer(payload, dtype="<u2")
    wide = np.left_shift(bits.astype("<u4"), np.uint32(16))
    return wide.view("<f4")


def bf16_bytes_to_fp16(payload: bytes) -> bytes:
    with np.errstate(over="ignore", invalid="ignore"):
        values = bf16_bytes_to_float32(payload).astype("<f2")
    return values.tobytes()


def fp32_values_to_fp16_bytes(values: np.ndarray) -> bytes:
    with np.errstate(over="ignore", invalid="ignore"):
        source = np.asarray(values, dtype="<f4")
    if not np.isfinite(source).all():
        raise ValueError("IEEE FP16 conversion input must be finite FP32")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = source.astype("<f2")
    if not np.isfinite(rounded).all():
        raise ValueError("IEEE FP16 conversion overflowed to a non-finite value")
    return rounded.tobytes()


def _output_item_bytes(plan: OutputTensorPlan, profile: str) -> int:
    if plan.outputDtype == "float32":
        return 4
    if plan.outputDtype in {"profile-float", "float16"}:
        return 2
    raise ValueError(f"Unknown output dtype policy {plan.outputDtype!r}")


def runtime_shape(
    plan: OutputTensorPlan,
    source_info: TensorInfo,
) -> tuple[int, ...]:
    """Return the post-transform logical shape consumed by WebGPU kernels."""

    shape = source_info.shape
    if plan.transformation == DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION:
        if (
            len(shape) != 2
            or shape[0] % 256 != 0
            or shape[1] % 32 != 0
        ):
            raise ValueError(
                f"{plan.output}: revision-7 DiT GEMM source must be rank two "
                "with N divisible by 256 and K divisible by 32"
            )
        expected = shape
    elif plan.transformation == DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION:
        if (
            len(shape) != 2
            or shape[0] % 128 != 0
            or shape[1] % 4 != 0
        ):
            raise ValueError(
                f"{plan.output}: K4-packed DiT GEMM source must be rank two "
                "with N divisible by 128 and K divisible by 4"
            )
        expected = shape
    elif plan.transformation == DIT_GEMM_TILE_TRANSFORMATION:
        if (
            len(shape) != 2
            or shape[0] % 128 != 0
            or shape[1] % 32 != 0
        ):
            raise ValueError(
                f"{plan.output}: tile-major DiT GEMM source must be rank two "
                "with N divisible by 128 and K divisible by 32"
            )
        expected = shape
    elif plan.transformation in {
        VAE_CONV1D_TRANSFORMATION,
        VAE_CONV1D_FP16_TRANSFORMATION,
        VAE_K1_FP16_TILE_TRANSFORMATION,
        VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
    }:
        if len(shape) != 3:
            raise ValueError(f"{plan.output}: Conv1d source must have rank 3")
        output_channels, input_channels, kernel = shape
        if plan.transformation == VAE_K1_FP16_TILE_TRANSFORMATION and (
            kernel != 1
            or output_channels != input_channels
            or output_channels % 128 != 0
            or input_channels % 32 != 0
        ):
            raise ValueError(
                f"{plan.output}: packed K1 source must be square with "
                "Cout divisible by 128 and Cin divisible by 32"
            )
        expected = (output_channels, kernel, input_channels)
    elif plan.transformation in {
        VAE_CONV_TRANSPOSE1D_TRANSFORMATION,
        VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION,
        VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
        VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    }:
        if len(shape) != 3:
            raise ValueError(
                f"{plan.output}: ConvTranspose1d source must have rank 3"
            )
        input_channels, output_channels, kernel = shape
        if (
            plan.transformation
            in {
                VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
                VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
            }
            and kernel % 2 != 0
        ):
            raise ValueError(
                f"{plan.output}: polyphase ConvTranspose1d kernel must be even"
            )
        expected = (output_channels, kernel, input_channels)
    elif plan.transformation in {
        VAE_CHANNEL_VECTOR_TRANSFORMATION,
        VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
    }:
        if len(shape) != 3 or shape[0] != 1 or shape[2] != 1:
            raise ValueError(
                f"{plan.output}: Snake parameter must have source shape [1,C,1]"
            )
        expected = (shape[1],)
    elif plan.transformation in {
        "bf16-to-fp32",
        VAE_BIAS_FP16_TRANSFORMATION,
    }:
        if len(shape) != 1:
            raise ValueError(f"{plan.output}: VAE bias source must have rank 1")
        expected = shape
    else:
        expected = shape
    if not expected or any(dimension <= 0 for dimension in expected):
        raise ValueError(f"{plan.output}: runtime tensors must have positive extents")
    return expected


def validate_experimental_vae_plan_payload(
    plan: ConversionPlan,
    checkpoints: dict[str, SafetensorsFile],
) -> None:
    vae_outputs = [output for output in plan.outputs if output.phase == "vae"]
    if (
        len(vae_outputs) != len(VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE)
        or {output.sourceTensor for output in vae_outputs}
        != set(VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE)
    ):
        raise ValueError(
            "Experimental FP16 VAE plan does not contain the exact "
            "revision-7 decoder tensor inventory"
        )
    parameter_elements = 0
    parameter_bytes = 0
    tensor_records = 0
    for output in vae_outputs:
        if output.outputDtype != "float16":
            raise ValueError(
                f"{output.output}: experimental VAE output is not IEEE FP16"
            )
        info = checkpoints[output.source].tensor(output.sourceTensor)
        expected_runtime_shape = VAE_REVISION7_RUNTIME_SHAPES_BY_SOURCE.get(
            output.sourceTensor
        )
        expected_output = (
            f"vae.{output.sourceTensor.removesuffix('.weight_v')}.weight"
            if output.sourceTensor.endswith(".weight_v")
            else f"vae.{output.sourceTensor}"
        )
        shape = runtime_shape(output, info)
        if (
            expected_runtime_shape is None
            or shape != expected_runtime_shape
            or output.output != expected_output
        ):
            raise ValueError(
                f"{output.output}: experimental VAE revision-7 label/shape "
                "contract changed"
            )
        if output.sourceTensor.endswith(".weight_v"):
            expected_transformation, expected_layout = (
                revision7_vae_runtime_weight_layout(output.sourceTensor, shape)
            )
            if (
                output.transformation != expected_transformation
                or output.runtimeLayout != expected_layout
            ):
                raise ValueError(
                    f"{output.output}: experimental VAE revision-7 weight "
                    "layout changed"
                )
            g_name = output.sourceTensor.removesuffix(".weight_v") + ".weight_g"
            validate_weightnorm_pair(
                output.sourceTensor,
                info,
                g_name,
                checkpoints[output.source].tensor(g_name),
            )
        elif output.sourceTensor.endswith(".bias"):
            if (
                output.transformation != VAE_BIAS_FP16_TRANSFORMATION
                or output.runtimeLayout != SOURCE_ROW_MAJOR_LAYOUT
            ):
                raise ValueError(
                    f"{output.output}: experimental VAE revision-7 bias "
                    "layout changed"
                )
        elif (
            output.transformation != VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION
            or output.runtimeLayout != VAE_CHANNEL_VECTOR_FP16_LAYOUT
        ):
            raise ValueError(
                f"{output.output}: experimental VAE revision-7 Snake layout changed"
            )
        parameter_elements += math.prod(shape)
        pieces = split_output_plan(output, info, EXPERIMENTAL_VAE_PROFILE)
        if (
            len(pieces) != 1
            or pieces[0].row_start != 0
            or pieces[0].row_end != shape[0]
            or pieces[0].output_name != output.output
        ):
            raise ValueError(
                f"{output.output}: experimental VAE tensor must remain unsharded"
            )
        parameter_bytes += pieces[0].output_bytes
        tensor_records += len(pieces)
    if (
        len(vae_outputs) != EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT
        or tensor_records != EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT
        or parameter_elements != EXPERIMENTAL_VAE_PARAMETER_ELEMENTS
        or parameter_bytes != EXPERIMENTAL_VAE_PARAMETER_BYTES
    ):
        raise ValueError(
            "Experimental FP16 VAE plan is not exactly "
            f"{EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT} tensors/"
            f"{EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT} records/"
            f"{EXPERIMENTAL_VAE_PARAMETER_ELEMENTS} elements/"
            f"{EXPERIMENTAL_VAE_PARAMETER_BYTES} bytes"
        )


def validate_experimental_dit_dense_plan_payload(
    plan: ConversionPlan,
    checkpoints: dict[str, SafetensorsFile],
    *,
    converter_revision: int = 8,
) -> None:
    outputs = [output for output in plan.outputs if output.phase == "dit"]
    expected_dense_outputs = {
        f"ace.{name}" for name in DIT_REPEATED_DENSE_WEIGHT_NAMES
    }
    expected_outputs = {
        f"ace.decoder.layers.{layer}.{suffix}"
        for layer in range(24)
        for suffix in (
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
    }
    parameter_elements = 0
    parameter_bytes = 0
    dense_contract = {
        7: (
            DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
            DIT_DENSE_FP16_REV7_TILE_LAYOUT,
        ),
        8: (
            DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
            DIT_DENSE_FP16_REV8_TILE_LAYOUT,
        ),
    }.get(converter_revision)
    if dense_contract is None:
        raise ValueError("Experimental mixed DiT converter revision must be 7 or 8")
    dense_transformation, dense_layout = dense_contract
    for output in outputs:
        if output.output not in expected_outputs:
            raise ValueError(
                f"{output.output}: invalid experimental mixed DiT output"
            )
        if output.output in expected_dense_outputs:
            if (
                output.outputDtype != "float16"
                or output.transformation != dense_transformation
                or output.runtimeLayout != dense_layout
            ):
                raise ValueError(
                    f"{output.output}: invalid experimental DiT dense output"
                )
        elif output.outputDtype != "profile-float":
            raise ValueError(
                f"{output.output}: mixed DiT support tensor is not reference BF16"
            )
        info = checkpoints[output.source].tensor(output.sourceTensor)
        shape = runtime_shape(output, info)
        pieces = split_output_plan(output, info, EXPERIMENTAL_DIT_DENSE_PROFILE)
        if (
            len(pieces) != 1
            or pieces[0].row_start != 0
            or pieces[0].row_end != shape[0]
        ):
            raise ValueError(
                f"{output.output}: experimental mixed DiT tensor must be unsharded"
            )
        parameter_elements += math.prod(shape)
        parameter_bytes += pieces[0].output_bytes
    if (
        len(outputs) != EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT
        or {output.output for output in outputs} != expected_outputs
        or parameter_elements != EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS
        or parameter_bytes != EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES
    ):
        raise ValueError(
            "Experimental mixed DiT plan does not match the exact 24-layer payload"
        )


def split_output_plan(
    plan: OutputTensorPlan,
    source_info: TensorInfo,
    profile: str,
) -> list[Piece]:
    if source_info.dtype != "BF16":
        raise ValueError(
            f"{plan.source}:{plan.sourceTensor} is {source_info.dtype}, but the "
            "v1 conversion recipes require audited BF16 source storage"
        )
    item_bytes = _output_item_bytes(plan, profile)
    logical_shape = runtime_shape(plan, source_info)
    rows = logical_shape[0]
    row_elements = math.prod(logical_shape[1:])
    row_bytes = row_elements * item_bytes
    if row_bytes > PIECE_TARGET_BYTES:
        raise ValueError(
            f"{plan.output}: one first-axis row is {row_bytes} bytes; add a "
            "model-specific inner-axis sharding recipe"
        )
    rows_per_piece = max(1, PIECE_TARGET_BYTES // max(1, row_bytes))
    if plan.transformation in {
        DIT_GEMM_TILE_TRANSFORMATION,
        DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
        DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
        VAE_K1_FP16_TILE_TRANSFORMATION,
        VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
        VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
        VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
    }:
        # A physical N tile may not straddle independently bound shards.
        # Canonical DiT matrices are below the shard target, so reject rather
        # than inventing an unaudited inner-axis or tile-axis package split.
        if rows * row_bytes > PIECE_TARGET_BYTES:
            raise ValueError(
                f"{plan.output}: tile-major DiT GEMM exceeds one package piece"
            )
        rows_per_piece = rows
    pieces: list[Piece] = []
    for row_start in range(0, rows, rows_per_piece):
        row_end = min(rows, row_start + rows_per_piece)
        byte_length = (row_end - row_start) * row_elements * item_bytes
        if profile == "reference" and plan.outputDtype == "profile-float":
            byte_length += (-byte_length) % 4
        output_name = plan.output
        if row_start != 0 or row_end != rows:
            output_name = f"{output_name}.rows-{row_start:06d}-{row_end:06d}"
        pieces.append(
            Piece(
                plan=plan,
                source_info=source_info,
                logical_shape=logical_shape,
                row_start=row_start,
                row_end=row_end,
                output_name=output_name,
                output_bytes=byte_length,
            )
        )
    return pieces


def _source_region(piece: Piece) -> tuple[int, int]:
    row_elements = (
        math.prod(piece.source_info.shape[1:]) if piece.source_info.shape else 1
    )
    start = piece.row_start * row_elements * 2
    length = (piece.row_end - piece.row_start) * row_elements * 2
    return start, length


def _iter_bf16_profile_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
    profile: str,
    *,
    source_chunk_bytes: int = 4 * 1024 * 1024,
) -> Iterator[bytes]:
    start, length = _source_region(piece)
    cursor = 0
    while cursor < length:
        take = min(source_chunk_bytes, length - cursor)
        take -= take % 2
        payload = checkpoint.read_tensor_region(
            piece.plan.sourceTensor,
            start + cursor,
            take,
        )
        if profile == "reference":
            yield payload
        elif profile in {"fp16", EXPERIMENTAL_VAE_PROFILE}:
            yield bf16_bytes_to_fp16(payload)
        else:
            raise ValueError(f"Unsupported profile {profile!r}")
        cursor += take
    if profile == "reference" and length % 4:
        yield b"\0" * ((-length) % 4)


def _iter_dit_gemm_tile_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
    profile: str,
) -> Iterator[bytes]:
    if (
        piece.plan.runtimeLayout
        not in {
            DIT_GEMM_TILE_LAYOUT,
            DIT_DENSE_FP16_REV7_TILE_LAYOUT,
            DIT_DENSE_FP16_REV8_TILE_LAYOUT,
        }
        or piece.row_start != 0
        or piece.row_end != piece.logical_shape[0]
        or piece.source_info.shape != piece.logical_shape
        or len(piece.logical_shape) != 2
    ):
        raise ValueError(f"{piece.plan.output}: invalid tile-major DiT GEMM piece")
    columns, inner = piece.logical_shape
    dense_rev7 = piece.plan.runtimeLayout == DIT_DENSE_FP16_REV7_TILE_LAYOUT
    dense_k4 = piece.plan.runtimeLayout == DIT_DENSE_FP16_REV8_TILE_LAYOUT
    if dense_k4:
        if columns % 128 != 0 or inner % 4 != 0:
            raise ValueError(f"{piece.plan.output}: invalid K4-packed DiT GEMM shape")
    elif dense_rev7:
        if columns % 256 != 0 or inner % 32 != 0:
            raise ValueError(
                f"{piece.plan.output}: invalid revision-7 DiT GEMM shape"
            )
    elif columns % 128 != 0 or inner % 32 != 0:
        raise ValueError(f"{piece.plan.output}: invalid tile-major DiT GEMM shape")
    payload = checkpoint.read_tensor_region(
        piece.plan.sourceTensor,
        0,
        piece.source_info.byte_length,
    )
    source = np.frombuffer(payload, dtype="<u2").reshape(columns, inner)
    if dense_k4:
        # Logical source [N,K] -> OPT-0032 physical
        # [N/128,K/4,output4,lane32,K4]. Within one N tile, logical output
        # n=lane*4+output, exactly matching the fixed32 kernel's vector owner.
        tiled = source.reshape(
            columns // 128,
            32,
            4,
            inner // 4,
            4,
        ).transpose(
            0,
            3,
            2,
            1,
            4,
        ).copy(order="C")
    else:
        tile_columns = 256 if dense_rev7 else 128
        # Logical [N,K] -> physical [N/tile,K/32,K-in-tile,N-in-tile].
        tiled = source.reshape(
            columns // tile_columns,
            tile_columns,
            inner // 32,
            32,
        ).transpose(
            0,
            2,
            3,
            1,
        ).copy(order="C")
    if profile == "reference":
        yield tiled.tobytes()
    elif profile in {
        "fp16",
        EXPERIMENTAL_VAE_PROFILE,
        EXPERIMENTAL_DIT_DENSE_PROFILE,
    }:
        yield bf16_bytes_to_fp16(tiled.tobytes())
    else:
        raise ValueError(f"Unsupported profile {profile!r}")


def _iter_vae_fp32_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    start, length = _source_region(piece)
    cursor = 0
    while cursor < length:
        take = min(2 * 1024 * 1024, length - cursor)
        take -= take % 2
        payload = checkpoint.read_tensor_region(
            piece.plan.sourceTensor,
            start + cursor,
            take,
        )
        yield bf16_bytes_to_float32(payload).tobytes()
        cursor += take


def _iter_vae_fp16_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    start, length = _source_region(piece)
    cursor = 0
    while cursor < length:
        take = min(2 * 1024 * 1024, length - cursor)
        take -= take % 2
        payload = checkpoint.read_tensor_region(
            piece.plan.sourceTensor,
            start + cursor,
            take,
        )
        yield fp32_values_to_fp16_bytes(bf16_bytes_to_float32(payload))
        cursor += take


def _iter_snake_channel_vector_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    if piece.source_info.shape != (1, piece.logical_shape[0], 1):
        raise ValueError(f"{piece.plan.output}: invalid Snake source/runtime shapes")
    start = piece.row_start * 2
    length = (piece.row_end - piece.row_start) * 2
    cursor = 0
    while cursor < length:
        take = min(2 * 1024 * 1024, length - cursor)
        take -= take % 2
        payload = checkpoint.read_tensor_region(
            piece.plan.sourceTensor,
            start + cursor,
            take,
        )
        values = bf16_bytes_to_float32(payload)
        yield (
            fp32_values_to_fp16_bytes(values)
            if piece.plan.outputDtype == "float16"
            else values.tobytes()
        )
        cursor += take


def _weightnorm_pair(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> tuple[str, TensorInfo]:
    source_name = piece.plan.sourceTensor
    if not source_name.endswith(".weight_v"):
        raise ValueError(f"Expected a weight_v source, got {source_name!r}")
    g_name = source_name.removesuffix(".weight_v") + ".weight_g"
    g_info = checkpoint.tensor(g_name)
    validate_weightnorm_pair(source_name, piece.source_info, g_name, g_info)
    return g_name, g_info


def pack_vae_k7_row_reuse_u16(native_weight: np.ndarray) -> np.ndarray:
    """Bit-preserving O-K-I to OPT-0051 row-reuse physical order."""

    native = np.asarray(native_weight)
    if native.dtype != np.dtype("<u2") or native.ndim != 3:
        raise ValueError("VAE K7 row-reuse pack requires a rank-three U16 array")
    output_channels, kernel, input_channels = native.shape
    if (
        kernel != 7
        or output_channels < 128
        or output_channels % 128 != 0
        or input_channels % 4 != 0
    ):
        raise ValueError(
            "VAE K7 row-reuse pack requires K7, Cout divisible by 128, "
            "and Cin divisible by 4"
        )
    return (
        native.reshape(
            output_channels // 64,
            32,
            2,
            7,
            input_channels // 4,
            4,
        )
        .transpose(3, 4, 0, 1, 2, 5)
        .copy(order="C")
    )


def unpack_vae_k7_row_reuse_u16(packed_weight: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`pack_vae_k7_row_reuse_u16`."""

    packed = np.asarray(packed_weight)
    if (
        packed.dtype != np.dtype("<u2")
        or packed.ndim != 6
        or packed.shape[0] != 7
        or packed.shape[3:] != (32, 2, 4)
    ):
        raise ValueError("VAE K7 row-reuse inverse received an invalid shape")
    _, input_channel_groups, output_bands, _, _, _ = packed.shape
    if output_bands < 2 or output_bands % 2 != 0:
        raise ValueError("VAE K7 row-reuse inverse requires whole 128-Cout tiles")
    return (
        packed.transpose(2, 3, 4, 0, 1, 5)
        .reshape(output_bands * 64, 7, input_channel_groups * 4)
        .copy(order="C")
    )


def pack_vae_conv_transpose_k4_u16(
    polyphase_weight: np.ndarray,
    reuse_axis: str,
) -> np.ndarray:
    """Bit-preserving rev6 polyphase to OPT-0048 K4 physical order."""

    logical = np.asarray(polyphase_weight)
    if (
        logical.dtype != np.dtype("<u2")
        or logical.ndim != 4
        or logical.shape[1] != 2
    ):
        raise ValueError(
            "VAE transpose K4 pack requires [phase,2,Cin,Cout] U16 words"
        )
    phases, _, input_channels, output_channels = logical.shape
    outputs_per_lane = 8 if reuse_axis == "channel" else 4 if reuse_axis == "row" else 0
    output_tile = 32 * outputs_per_lane
    if (
        phases < 1
        or input_channels % 4 != 0
        or outputs_per_lane == 0
        or output_channels % output_tile != 0
    ):
        raise ValueError(
            "VAE transpose K4 pack has incompatible channels or reuse axis"
        )
    return (
        logical.reshape(
            phases,
            2,
            input_channels // 4,
            4,
            output_channels // output_tile,
            32,
            outputs_per_lane,
        )
        .transpose(0, 1, 2, 4, 5, 6, 3)
        .copy(order="C")
    )


def unpack_vae_conv_transpose_k4_u16(
    packed_weight: np.ndarray,
    reuse_axis: str,
) -> np.ndarray:
    """Exact inverse of :func:`pack_vae_conv_transpose_k4_u16`."""

    packed = np.asarray(packed_weight)
    outputs_per_lane = 8 if reuse_axis == "channel" else 4 if reuse_axis == "row" else 0
    if (
        packed.dtype != np.dtype("<u2")
        or packed.ndim != 7
        or packed.shape[1] != 2
        or packed.shape[4] != 32
        or packed.shape[5] != outputs_per_lane
        or packed.shape[6] != 4
        or outputs_per_lane == 0
    ):
        raise ValueError("VAE transpose K4 inverse received an invalid shape")
    phases, _, input_groups, output_tiles, _, _, _ = packed.shape
    return (
        packed.transpose(0, 1, 2, 6, 3, 4, 5)
        .reshape(
            phases,
            2,
            input_groups * 4,
            output_tiles * 32 * outputs_per_lane,
        )
        .copy(order="C")
    )


def _iter_weightnorm_conv1d_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    g_name, _ = _weightnorm_pair(checkpoint, piece)
    source_name = piece.plan.sourceTensor
    output_channels, input_channels, kernel = piece.source_info.shape
    if piece.logical_shape != (output_channels, kernel, input_channels):
        raise ValueError(f"{piece.plan.output}: invalid Conv1d runtime shape")
    row_elements = math.prod(piece.source_info.shape[1:])
    for row in range(piece.row_start, piece.row_end):
        v_payload = checkpoint.read_tensor_region(
            source_name,
            row * row_elements * 2,
            row_elements * 2,
        )
        g_payload = checkpoint.read_tensor_region(g_name, row * 2, 2)
        values = bf16_bytes_to_float32(v_payload)
        coefficient = bf16_bytes_to_float32(g_payload)[0]
        fused = fuse_weightnorm_row_fp32(coefficient, values).reshape(
            input_channels,
            kernel,
        )
        runtime_row = fused.transpose(1, 0).copy(order="C")
        yield (
            fp32_values_to_fp16_bytes(runtime_row)
            if piece.plan.outputDtype == "float16"
            else runtime_row.tobytes()
        )


def _iter_weightnorm_k7_row_reuse_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    g_name, _ = _weightnorm_pair(checkpoint, piece)
    source_name = piece.plan.sourceTensor
    output_channels, input_channels, kernel = piece.source_info.shape
    if (
        piece.logical_shape != (output_channels, 7, input_channels)
        or kernel != 7
        or piece.row_start != 0
        or piece.row_end != output_channels
    ):
        raise ValueError(f"{piece.plan.output}: invalid row-reuse K7 shape")
    native_words = np.empty(piece.logical_shape, dtype="<u2")
    row_elements = input_channels * kernel
    for output_channel in range(output_channels):
        v_payload = checkpoint.read_tensor_region(
            source_name,
            output_channel * row_elements * 2,
            row_elements * 2,
        )
        g_payload = checkpoint.read_tensor_region(
            g_name,
            output_channel * 2,
            2,
        )
        fused = fuse_weightnorm_row_fp32(
            bf16_bytes_to_float32(g_payload)[0],
            bf16_bytes_to_float32(v_payload),
        ).reshape(input_channels, kernel)
        native_words[output_channel] = np.frombuffer(
            fp32_values_to_fp16_bytes(fused.transpose(1, 0).copy(order="C")),
            dtype="<u2",
        ).reshape(7, input_channels)
    yield pack_vae_k7_row_reuse_u16(native_words).tobytes(order="C")


def _iter_weightnorm_k1_tile_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    g_name, _ = _weightnorm_pair(checkpoint, piece)
    source_name = piece.plan.sourceTensor
    output_channels, input_channels, kernel = piece.source_info.shape
    if (
        kernel != 1
        or output_channels != input_channels
        or output_channels % 128 != 0
        or input_channels % 32 != 0
        or piece.logical_shape != (output_channels, 1, input_channels)
        or piece.row_start != 0
        or piece.row_end != output_channels
    ):
        raise ValueError(f"{piece.plan.output}: invalid packed K1 shape")
    packed = np.empty(
        (output_channels // 128, input_channels // 32, 32, 128),
        dtype="<f2",
    )
    for output_channel in range(output_channels):
        v_payload = checkpoint.read_tensor_region(
            source_name,
            output_channel * input_channels * 2,
            input_channels * 2,
        )
        g_payload = checkpoint.read_tensor_region(g_name, output_channel * 2, 2)
        fused = fuse_weightnorm_row_fp32(
            bf16_bytes_to_float32(g_payload)[0],
            bf16_bytes_to_float32(v_payload),
        )
        rounded = np.frombuffer(
            fp32_values_to_fp16_bytes(fused),
            dtype="<f2",
        ).reshape(input_channels // 32, 32)
        packed[output_channel // 128, :, :, output_channel % 128] = rounded
    yield packed.tobytes(order="C")


def _iter_weightnorm_conv_transpose1d_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    g_name, _ = _weightnorm_pair(checkpoint, piece)
    source_name = piece.plan.sourceTensor
    input_channels, output_channels, kernel = piece.source_info.shape
    if piece.logical_shape != (output_channels, kernel, input_channels):
        raise ValueError(
            f"{piece.plan.output}: invalid ConvTranspose1d runtime shape"
        )
    # A piece is selected only after deriving the transformed [out,kernel,in]
    # shape. The bounded array is therefore a runtime-axis-0 shard, not a
    # source-axis shard. The largest canonical piece remains <=96 MiB.
    output = np.empty(
        (piece.row_end - piece.row_start, kernel, input_channels),
        dtype="<f4",
    )
    source_row_elements = output_channels * kernel
    for input_channel in range(input_channels):
        v_payload = checkpoint.read_tensor_region(
            source_name,
            input_channel * source_row_elements * 2,
            source_row_elements * 2,
        )
        g_payload = checkpoint.read_tensor_region(
            g_name,
            input_channel * 2,
            2,
        )
        values = bf16_bytes_to_float32(v_payload)
        coefficient = bf16_bytes_to_float32(g_payload)[0]
        fused = fuse_weightnorm_row_fp32(coefficient, values).reshape(
            output_channels,
            kernel,
        )
        output[:, :, input_channel] = fused[piece.row_start : piece.row_end, :]
    for runtime_row in output:
        yield (
            fp32_values_to_fp16_bytes(runtime_row)
            if piece.plan.outputDtype == "float16"
            else runtime_row.tobytes()
        )


def _iter_weightnorm_conv_transpose1d_polyphase_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    g_name, _ = _weightnorm_pair(checkpoint, piece)
    source_name = piece.plan.sourceTensor
    input_channels, output_channels, kernel = piece.source_info.shape
    if (
        kernel % 2 != 0
        or piece.logical_shape != (output_channels, kernel, input_channels)
        or piece.row_start != 0
        or piece.row_end != output_channels
    ):
        raise ValueError(f"{piece.plan.output}: invalid polyphase transpose shape")
    stride = kernel // 2
    packed = np.empty((stride, 2, input_channels, output_channels), dtype="<f2")
    source_row_elements = output_channels * kernel
    for input_channel in range(input_channels):
        v_payload = checkpoint.read_tensor_region(
            source_name,
            input_channel * source_row_elements * 2,
            source_row_elements * 2,
        )
        g_payload = checkpoint.read_tensor_region(g_name, input_channel * 2, 2)
        fused = fuse_weightnorm_row_fp32(
            bf16_bytes_to_float32(g_payload)[0],
            bf16_bytes_to_float32(v_payload),
        ).reshape(output_channels, kernel)
        rounded = np.frombuffer(
            fp32_values_to_fp16_bytes(fused),
            dtype="<f2",
        ).reshape(output_channels, kernel)
        for phase in range(stride):
            packed[phase, 0, input_channel, :] = rounded[:, phase]
            packed[phase, 1, input_channel, :] = rounded[:, phase + stride]
    yield packed.tobytes(order="C")


def _iter_weightnorm_conv_transpose1d_k4_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
) -> Iterator[bytes]:
    g_name, _ = _weightnorm_pair(checkpoint, piece)
    source_name = piece.plan.sourceTensor
    input_channels, output_channels, kernel = piece.source_info.shape
    contract = VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS.get(source_name)
    if (
        contract is None
        or contract[0] != tuple(piece.source_info.shape)
        or source_name == "decoder.block.0.conv_t1.weight_v"
        or kernel % 2 != 0
        or piece.logical_shape != (output_channels, kernel, input_channels)
        or piece.row_start != 0
        or piece.row_end != output_channels
    ):
        raise ValueError(f"{piece.plan.output}: invalid revision-7 transpose K4 shape")
    stride = kernel // 2
    polyphase_words = np.empty(
        (stride, 2, input_channels, output_channels),
        dtype="<u2",
    )
    source_row_elements = output_channels * kernel
    for input_channel in range(input_channels):
        v_payload = checkpoint.read_tensor_region(
            source_name,
            input_channel * source_row_elements * 2,
            source_row_elements * 2,
        )
        g_payload = checkpoint.read_tensor_region(g_name, input_channel * 2, 2)
        fused = fuse_weightnorm_row_fp32(
            bf16_bytes_to_float32(g_payload)[0],
            bf16_bytes_to_float32(v_payload),
        ).reshape(output_channels, kernel)
        rounded = np.frombuffer(
            fp32_values_to_fp16_bytes(fused),
            dtype="<u2",
        ).reshape(output_channels, kernel)
        for phase in range(stride):
            polyphase_words[phase, 0, input_channel, :] = rounded[:, phase]
            polyphase_words[phase, 1, input_channel, :] = rounded[
                :,
                phase + stride,
            ]
    yield pack_vae_conv_transpose_k4_u16(
        polyphase_words,
        contract[1],
    ).tobytes(order="C")


def deterministic_fp32_norm(values: np.ndarray) -> np.float32:
    """Compute a norm with a fixed adjacent-pair FP32 reduction tree.

    Upstream converts ``g`` and ``v`` to FP32 before evaluating
    ``g * v / (norm(v) + 1e-9)``. NumPy/BLAS norm implementations may choose
    host-specific reduction trees, so package conversion specifies this tree:
    square in FP32, add adjacent pairs in FP32, carry an odd final value, and
    repeat until one value remains. The final square root is FP32.
    """

    flat = np.asarray(values, dtype="<f4").reshape(-1)
    if flat.size == 0:
        return np.float32(0.0)
    work = np.multiply(flat, flat, dtype=np.float32)
    while work.size > 1:
        pair_count = work.size // 2
        reduced = np.empty(pair_count + (work.size % 2), dtype="<f4")
        if pair_count:
            np.add(
                work[: pair_count * 2 : 2],
                work[1 : pair_count * 2 : 2],
                out=reduced[:pair_count],
                dtype=np.float32,
            )
        if work.size % 2:
            reduced[-1] = work[-1]
        work = reduced
    return np.sqrt(work[0], dtype=np.float32)


def fuse_weightnorm_row_fp32(
    coefficient: np.float32 | float,
    values: np.ndarray,
) -> np.ndarray:
    """Fuse one dim-0 weight-normalization row using only specified FP32 ops."""

    row = np.asarray(values, dtype="<f4").reshape(-1)
    g = np.float32(coefficient)
    norm = deterministic_fp32_norm(row)
    denominator = np.add(norm, np.float32(1e-9), dtype=np.float32)
    numerator = np.multiply(g, row, dtype=np.float32)
    return np.divide(numerator, denominator, dtype=np.float32).astype(
        "<f4",
        copy=False,
    )


def validate_weightnorm_pair(
    v_name: str,
    v_info: TensorInfo,
    g_name: str,
    g_info: TensorInfo,
) -> None:
    """Require PyTorch default dim=0 weight-normalization geometry."""

    if not v_info.shape:
        raise ValueError(f"Weight normalization source {v_name!r} is scalar")
    expected_g_shape = (v_info.shape[0],) + (1,) * (len(v_info.shape) - 1)
    if (
        v_info.dtype != "BF16"
        or g_info.dtype != "BF16"
        or g_info.shape != expected_g_shape
    ):
        raise ValueError(
            f"Incompatible weight normalization pair {v_name!r}/{g_name!r}: "
            f"v={v_info.dtype}{v_info.shape}, g={g_info.dtype}{g_info.shape}, "
            f"expected g=BF16{expected_g_shape}"
        )


def _piece_bytes(
    checkpoint: SafetensorsFile,
    piece: Piece,
    profile: str,
) -> Iterable[bytes]:
    effective_profile = (
        "reference"
        if profile == EXPERIMENTAL_DIT_DENSE_PROFILE
        and piece.plan.outputDtype == "profile-float"
        else profile
    )
    if piece.plan.transformation in {
        DIT_GEMM_TILE_TRANSFORMATION,
        DIT_DENSE_FP16_REV7_TILE_TRANSFORMATION,
        DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION,
    }:
        return _iter_dit_gemm_tile_bytes(checkpoint, piece, effective_profile)
    if piece.plan.transformation in {
        VAE_CONV1D_TRANSFORMATION,
        VAE_CONV1D_FP16_TRANSFORMATION,
    }:
        return _iter_weightnorm_conv1d_bytes(checkpoint, piece)
    if piece.plan.transformation == VAE_K7_ROW_REUSE_FP16_TRANSFORMATION:
        return _iter_weightnorm_k7_row_reuse_bytes(checkpoint, piece)
    if piece.plan.transformation == VAE_K1_FP16_TILE_TRANSFORMATION:
        return _iter_weightnorm_k1_tile_bytes(checkpoint, piece)
    if piece.plan.transformation in {
        VAE_CONV_TRANSPOSE1D_TRANSFORMATION,
        VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION,
    }:
        return _iter_weightnorm_conv_transpose1d_bytes(checkpoint, piece)
    if (
        piece.plan.transformation
        == VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
    ):
        return _iter_weightnorm_conv_transpose1d_polyphase_bytes(checkpoint, piece)
    if piece.plan.transformation == VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION:
        return _iter_weightnorm_conv_transpose1d_k4_bytes(checkpoint, piece)
    if piece.plan.transformation in {
        VAE_CHANNEL_VECTOR_TRANSFORMATION,
        VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
    }:
        return _iter_snake_channel_vector_bytes(checkpoint, piece)
    if piece.plan.outputDtype == "float32":
        return _iter_vae_fp32_bytes(checkpoint, piece)
    if (
        piece.plan.outputDtype == "float16"
        and piece.plan.transformation == VAE_BIAS_FP16_TRANSFORMATION
    ):
        return _iter_vae_fp16_bytes(checkpoint, piece)
    return _iter_bf16_profile_bytes(checkpoint, piece, effective_profile)


def _copy_verified_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _source_records() -> list[dict[str, object]]:
    return [
        {
            "key": artifact.key,
            "component": artifact.component,
            "repository": artifact.repository,
            "revision": artifact.revision,
            "path": artifact.path,
            "byteLength": artifact.byte_length,
            "sha256": artifact.sha256,
            **(
                {
                    "tensorCount": artifact.safetensors.tensor_count,
                    "parameterCount": artifact.safetensors.parameter_count,
                    "headerLength": artifact.safetensors.header_length,
                    "headerSha256": artifact.safetensors.header_sha256,
                    "inventorySha256": artifact.safetensors.inventory_sha256,
                }
                if artifact.safetensors is not None
                else {}
            ),
        }
        for artifact in SOURCE_ARTIFACTS
    ]


def prepare_profile(
    profile: str,
    *,
    cache_root: Path,
    target: Path,
    offline: bool,
    converter_revision: int | None = None,
    dit_dense_converter_revision: int = 8,
    expected_manifest_sha256: str | None = None,
) -> Path:
    effective_converter_revision = (
        package_converter_revision(profile)
        if converter_revision is None
        else converter_revision
    )
    if profile == EXPERIMENTAL_DIT_DENSE_PROFILE:
        if (
            dit_dense_converter_revision not in {7, 8}
            or effective_converter_revision != dit_dense_converter_revision
        ):
            raise ValueError(
                "Experimental mixed DiT recipe and converter revision must match"
            )
    elif converter_revision is not None and (
        effective_converter_revision != package_converter_revision(profile)
    ):
        raise ValueError(f"Unsupported converter revision for profile {profile!r}")
    source_paths: dict[str, Path] = {}
    for index, artifact in enumerate(SOURCE_ARTIFACTS, start=1):
        print(
            f"[{index:02d}/{len(SOURCE_ARTIFACTS):02d}] "
            f"verify/download {artifact.key}",
            flush=True,
        )
        source_paths[artifact.key] = download_artifact(
            artifact,
            cache_root,
            offline=offline,
        )

    checkpoints: dict[str, SafetensorsFile] = {}
    try:
        for artifact in SAFETENSOR_ARTIFACTS:
            checkpoint = SafetensorsFile(source_paths[artifact.key])
            assert artifact.safetensors is not None
            checkpoint.assert_contract(artifact.safetensors)
            unexpected_dtypes = sorted(
                {
                    tensor.dtype
                    for tensor in checkpoint.tensors.values()
                    if tensor.dtype != "BF16"
                }
            )
            if unexpected_dtypes:
                raise ValueError(
                    f"{artifact.key}: v1 recipes require all-BF16 sources; "
                    f"found {unexpected_dtypes}"
                )
            checkpoints[artifact.key] = checkpoint
        plan = build_conversion_plan(
            checkpoints,
            profile=profile,
            dit_dense_converter_revision=dit_dense_converter_revision,
        )
        assert_canonical_encode_only_exclusions(
            checkpoints["ace-turbo-weights"],
            plan,
        )
        if profile == EXPERIMENTAL_VAE_PROFILE:
            validate_experimental_vae_plan_payload(plan, checkpoints)
        if profile == EXPERIMENTAL_DIT_DENSE_PROFILE:
            validate_experimental_dit_dense_plan_payload(
                plan,
                checkpoints,
                converter_revision=dit_dense_converter_revision,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        try:
            write_json_atomic(staging / "conversion-plan.json", plan.as_json())
            payload_files: list[FileRecord] = [
                file_record(staging, "conversion-plan.json", kind="conversion-plan")
            ]
            for artifact in SOURCE_ARTIFACTS:
                if artifact.package_path is None:
                    continue
                _copy_verified_asset(
                    source_paths[artifact.key],
                    staging / artifact.package_path,
                )
                payload_files.append(
                    file_record(staging, artifact.package_path, kind="upstream-asset")
                )
            for relative_name in LOCAL_LICENSE_FILES:
                source_license = MODEL_ROOT / relative_name
                _copy_verified_asset(source_license, staging / relative_name)
                payload_files.append(
                    file_record(staging, relative_name, kind="license")
                )

            tensor_records: dict[str, TensorRecord] = {}
            silence_name = "constants/silence-latent-f32.bin"
            extract_silence_latent(
                source_paths["ace-silence-latent"],
                staging / silence_name,
            )
            silence_file = file_record(staging, silence_name, kind="constant")
            payload_files.append(silence_file)
            tensor_records["constants.silence_latent"] = TensorRecord(
                shard=silence_name,
                byteOffset=0,
                byteLength=silence_file.byteLength,
                dtype=SILENCE_DTYPE,
                logicalShape=list(SILENCE_SHAPE),
                storageShape=list(SILENCE_SHAPE),
                layout="contiguous-nct-f32",
                phase="constants",
                lifetime="initial-latent",
                source=f"ace-silence-latent:{SILENCE_MEMBER}",
                transformation="validated-pytorch-zip-storage-extraction",
                logicalTensor="constants.silence_latent",
                partAxis=0,
                partStart=0,
                partEnd=1,
            )

            grouped: dict[str, list[Piece]] = defaultdict(list)
            for output in plan.outputs:
                info = checkpoints[output.source].tensor(output.sourceTensor)
                grouped[output.group].extend(split_output_plan(output, info, profile))
            for group in sorted(grouped):
                part = 0
                writer: ShardWriter | None = None
                try:
                    for piece in grouped[group]:
                        if writer is None or (
                            ((writer.offset + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
                            + piece.output_bytes
                            > MAX_SHARD_BYTES
                        ):
                            if writer is not None:
                                writer.__exit__(None, None, None)
                                payload_files.append(
                                    file_record(
                                        staging,
                                        writer.shard_name,
                                        kind="weights",
                                    )
                                )
                            shard_name = f"weights/{group}-{part:02d}.bin"
                            part += 1
                            writer = ShardWriter(staging, shard_name, tensor_records)
                            writer.__enter__()
                        logical_shape = piece.logical_shape
                        storage_shape = list(logical_shape)
                        storage_shape[0] = piece.row_end - piece.row_start
                        if (
                            piece.plan.transformation
                            == DIT_DENSE_FP16_REV8_TILE_TRANSFORMATION
                        ):
                            columns, inner = logical_shape
                            storage_shape = [
                                columns // 128,
                                inner // 4,
                                4,
                                32,
                                4,
                            ]
                        elif (
                            piece.plan.transformation
                            == VAE_K1_FP16_TILE_TRANSFORMATION
                        ):
                            output_channels, _, input_channels = logical_shape
                            storage_shape = [
                                output_channels // 128,
                                input_channels // 32,
                                32,
                                128,
                            ]
                        elif (
                            piece.plan.transformation
                            == VAE_K7_ROW_REUSE_FP16_TRANSFORMATION
                        ):
                            output_channels, kernel, input_channels = logical_shape
                            storage_shape = [
                                kernel,
                                input_channels // 4,
                                output_channels // 64,
                                32,
                                2,
                                4,
                            ]
                        elif (
                            piece.plan.transformation
                            == VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
                        ):
                            output_channels, kernel, input_channels = logical_shape
                            storage_shape = [
                                kernel // 2,
                                2,
                                input_channels,
                                output_channels,
                            ]
                        elif (
                            piece.plan.transformation
                            == VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION
                        ):
                            output_channels, kernel, input_channels = logical_shape
                            contract = VAE_REVISION7_TRANSPOSE_SOURCE_CONTRACTS[
                                piece.plan.sourceTensor
                            ]
                            outputs_per_lane = 8 if contract[1] == "channel" else 4
                            storage_shape = [
                                kernel // 2,
                                2,
                                input_channels // 4,
                                output_channels // (32 * outputs_per_lane),
                                32,
                                outputs_per_lane,
                                4,
                            ]
                        reference_storage = profile == "reference" or (
                            profile == EXPERIMENTAL_DIT_DENSE_PROFILE
                            and piece.plan.outputDtype == "profile-float"
                        )
                        output_dtype = (
                            piece.plan.outputDtype
                            if piece.plan.outputDtype in {"float32", "float16"}
                            else "uint32-bf16-pairs"
                            if reference_storage
                            else "float16"
                        )
                        if output_dtype == "uint32-bf16-pairs":
                            storage_shape = [piece.output_bytes // 4]
                        row_sharded = bool(
                            piece.row_start
                            or piece.row_end
                            != logical_shape[0]
                        )
                        layout = piece.plan.runtimeLayout
                        transformation = piece.plan.transformation
                        if piece.plan.outputDtype == "profile-float":
                            if piece.plan.transformation == DIT_GEMM_TILE_TRANSFORMATION:
                                if (
                                    layout != DIT_GEMM_TILE_LAYOUT
                                    or row_sharded
                                    or len(logical_shape) != 2
                                ):
                                    raise ValueError(
                                        f"{piece.plan.output}: invalid tile-major GEMM "
                                        "package piece"
                                    )
                                transformation = (
                                    "preserve-bf16-bits-dit-gemm-"
                                    "n128-k32-tile-major-v1"
                                    if reference_storage
                                    else "bf16-to-ieee-fp16-dit-gemm-"
                                    "n128-k32-tile-major-v1"
                                )
                            elif layout != SOURCE_ROW_MAJOR_LAYOUT:
                                raise ValueError(
                                    f"{piece.plan.output}: profile-float tensor has "
                                    "a non-source layout"
                                )
                            else:
                                layout = (
                                    "row-shard-axis0"
                                    if row_sharded
                                    else SOURCE_ROW_MAJOR_LAYOUT
                                )
                                if reference_storage:
                                    layout += "-bf16-pairs-lsb-u32"
                                    transformation = (
                                        "preserve-bf16-bits-pack-u32-pairs"
                                    )
                                else:
                                    transformation = "bf16-to-ieee-fp16"
                        elif layout == SOURCE_ROW_MAJOR_LAYOUT and row_sharded:
                            layout = "row-shard-axis0"
                        writer.add_bytes(
                            piece.output_name,
                            _piece_bytes(checkpoints[piece.plan.source], piece, profile),
                            byte_length=piece.output_bytes,
                            dtype=output_dtype,
                            logical_shape=logical_shape,
                            storage_shape=storage_shape,
                            layout=layout,
                            phase=piece.plan.phase,
                            lifetime=piece.plan.lifetime,
                            source=f"{piece.plan.source}:{piece.plan.sourceTensor}",
                            transformation=transformation,
                            logicalTensor=piece.plan.output,
                            partAxis=0,
                            partStart=piece.row_start,
                            partEnd=piece.row_end,
                        )
                    if writer is not None:
                        writer.__exit__(None, None, None)
                        payload_files.append(
                            file_record(staging, writer.shard_name, kind="weights")
                        )
                        writer = None
                except BaseException as error:
                    if writer is not None:
                        writer.__exit__(type(error), error, error.__traceback__)
                    raise

            licenses = [
                {
                    "component": "ACE-Step source and model snapshots",
                    "spdx": "MIT",
                    "notice": (
                        "Copyright (c) 2026 ACEStep; preserve the MIT notice. "
                        "The pinned model cards are included under licenses/."
                    ),
                    "source": REFERENCE_REPOSITORY,
                },
                {
                    "component": "Qwen3 architecture, tokenizer, and derived weights",
                    "spdx": "Apache-2.0",
                    "notice": (
                        "Qwen3 is provided by the Qwen Team under Apache-2.0; "
                        "preserve applicable copyright, attribution, NOTICE, and "
                        "license terms when redistributing derived packages."
                    ),
                    "source": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
                },
            ]
            accounting = dict(plan.as_json()["summary"])
            accounting["constantTensors"] = 1
            accounting["outputTensorsAfterRowSharding"] = len(tensor_records)
            manifest = build_manifest(
                profile=profile,
                source=_source_records(),
                files=payload_files,
                tensors=tensor_records,
                accounting=accounting,
                licenses=licenses,
                provenance={
                    "converterRevision": effective_converter_revision,
                    "aceSnapshot": ACE_REVISION,
                    "plannerSnapshot": PLANNER_REVISION,
                    "referenceRepository": REFERENCE_REPOSITORY,
                    "referenceCommit": REFERENCE_REVISION,
                    "referenceLicenseGitBlob": REFERENCE_LICENSE_GIT_BLOB,
                    "referenceLicenseSha256": REFERENCE_LICENSE_SHA256,
                    "determinism": (
                        "sorted source/output inventories, fixed transforms, canonical JSON"
                    ),
                },
            )
            write_json_atomic(staging / "manifest.json", manifest)
            verify_package(staging)
            require_manifest_identity(
                staging / "manifest.json",
                expected_manifest_sha256,
            )
            install_staged_directory(
                staging,
                target,
                expected_existing_manifest_sha256=(
                    canonical_replacement_manifest_sha256(profile, target)
                ),
            )
            return target
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    finally:
        for checkpoint in checkpoints.values():
            checkpoint.close()


def verify_cached_sources(cache_root: Path) -> None:
    for artifact in SOURCE_ARTIFACTS:
        verify_source_file(artifact.cache_path(cache_root), artifact)
        if artifact.safetensors is not None:
            with SafetensorsFile(artifact.cache_path(cache_root)) as checkpoint:
                checkpoint.assert_contract(artifact.safetensors)


def preparations_for_selector(
    selector: str,
    *,
    output_dir: Path | None = None,
) -> tuple[PackagePreparation, ...]:
    if selector == PRODUCTION_PROFILE:
        if output_dir is not None:
            raise ValueError(
                "--output-dir cannot be combined with --profile production"
            )
        return production_preparations()
    if selector in {PRODUCTION_DIT_PROFILE, PRODUCTION_VAE_PROFILE}:
        component = "dit" if selector == PRODUCTION_DIT_PROFILE else "vae"
        preparation = next(
            item
            for item in production_preparations()
            if item.label == f"production-{component}"
        )
        if output_dir is not None:
            preparation = replace(preparation, target=output_dir)
        return (preparation,)
    if selector == "all":
        if output_dir is not None:
            raise ValueError("--output-dir cannot be combined with --profile all")
        profiles = ("reference", "fp16")
    else:
        profiles = (selector,)
    return tuple(
        PackagePreparation(
            label=profile,
            profile=profile,
            target=output_dir if output_dir is not None else DEFAULT_OUTPUT[profile],
            converter_revision=package_converter_revision(profile),
        )
        for profile in profiles
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "reference",
            "fp16",
            EXPERIMENTAL_VAE_PROFILE,
            EXPERIMENTAL_DIT_DENSE_PROFILE,
            PRODUCTION_PROFILE,
            PRODUCTION_DIT_PROFILE,
            PRODUCTION_VAE_PROFILE,
            "all",
        ),
        default="reference",
        help=(
            "browser package profile to generate; production emits the exact "
            "three-package browser tuple, while all emits the two revision-4 "
            "profiles (default: reference)"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"source cache (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override one output directory; unavailable with multi-package profiles",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="refuse network access and require a complete verified cache",
    )
    parser.add_argument(
        "--verify-cache-only",
        action="store_true",
        help="verify all cached source files and inventories, then exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = args.cache_dir.expanduser().resolve()
    if args.verify_cache_only:
        if args.output_dir is not None:
            raise SystemExit("--output-dir has no meaning with --verify-cache-only")
        verify_cached_sources(cache_root)
        print("All pinned source files and safetensor inventories verified.")
        return 0
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else None
    )
    try:
        preparations = preparations_for_selector(
            args.profile,
            output_dir=output_dir,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for preparation in preparations:
        result = prepare_profile(
            preparation.profile,
            cache_root=cache_root,
            target=preparation.target,
            offline=args.offline,
            converter_revision=preparation.converter_revision,
            dit_dense_converter_revision=(
                preparation.dit_dense_converter_revision
            ),
            expected_manifest_sha256=preparation.expected_manifest_sha256,
        )
        print(f"Installed verified {preparation.label} package at {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
