#!/usr/bin/env python3
"""Download and convert the released DiCoSe checkpoints for WebGPU.

With no checkpoint arguments, the exporter downloads the two pinned Hugging Face
release files.  It verifies their exact bytes *before* unpickling, exports only
the two inference networks, and writes a transactional f16 package:

    <output>/manifest.json
    <output>/weights.f16.bin

Run explicitly from the repository root:

    pnpm model:prepare

The manifest names tensors as ``det.<key>`` and ``cd.<key>``.  All payloads live
in one f16 file and begin at 256-byte aligned offsets.  Dense matrices whose
dimensions fit the production subgroup GEMM are stored in converter-native
N128/N256 × K32 tiles; smaller tail shapes stay ``[in_features, out_features]``.
Conv2d tensors remain OIHW.
Repeated Torch storage views (notably RoPE frequency buffers) share one payload
and are marked with ``aliasOf`` in the manifest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALIGNMENT_BYTES = 256
LINEAR_TILE_INNER = 32
LINEAR_TILE_COLUMNS = (256, 128)
OUTPUT_MANIFEST = "manifest.json"
OUTPUT_WEIGHTS = "weights.f16.bin"
PACKAGE_SCHEMA = "dicose-wgsl-package-v1"
MODEL_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_DIRECTORY.parent
UPSTREAM_REPOSITORY = "karchkha/DiCoSe"
UPSTREAM_REVISION = "b3e44147b96e55b08eea2dd0b6b4e017748a87a9"
CANONICAL_WEIGHTS_BYTES = 623_246_848
CANONICAL_WEIGHTS_SHA256 = (
    "96f65545dd3ef7aa3189a353c51cdb30c4e592c356260c87f06459f092eeb0fe"
)
CANONICAL_MANIFEST_SHA256 = (
    "a280ddc377f1effac71698f85d5e12547f8e992aaceeb5efff4d07b6c5913a94"
)

# These are the byte-level LFS objects published at
# https://huggingface.co/karchkha/DiCoSe, revision `UPSTREAM_REVISION`.
# Checking them before torch.load() both verifies the model identity and avoids
# unpickling an arbitrary local file by accident.
@dataclass(frozen=True)
class SourceSpec:
    role: str
    repository_path: str
    expected_sha256: str
    expected_bytes: int


DETERMINISTIC_SOURCE = SourceSpec(
    role="deterministic",
    repository_path="Deterministic_model_MSST_bs_roformer/model.ckpt",
    expected_sha256="8087fbdcbc63f11f3ee305ef042cf42a42a5802e8a76678997f6448cb45256f5",
    expected_bytes=527_434_267,
)
CD_SOURCE = SourceSpec(
    role="consistency_distilled",
    repository_path="CD_MSST_bs_roformer/model.ckpt",
    expected_sha256="d25035bed7294a227fcb0f1ea691a0d1b8452ef76bde0e411c2b75536acf13da",
    expected_bytes=4_129_571_657,
)


BANDS = [2] * 24 + [4] * 12 + [12] * 8 + [24] * 8 + [48] * 8 + [128, 129]
assert len(BANDS) == 62 and sum(BANDS) == 1025

# Fixed architecture values from configs/consistency_model/bsrf_eval.yaml and
# configs/deterministic_model/bsrf_eval.yaml in the official source repository.
MODEL_CONFIG: dict[str, Any] = {
    "name": "DiCoSe BS-RoFormer + CD",
    "sampleRate": 44_100,
    "nFft": 2_048,
    "hopLength": 441,
    "winLength": 2_048,
    "stftNormalized": False,
    "stftCenter": True,
    "stftWindow": "hann_periodic",
    "zeroDc": True,
    "stereo": True,
    "numStems": 4,
    "stems": ["drums", "bass", "other", "vocals"],
    "dim": 384,
    "depth": 8,
    "heads": 8,
    "dimHead": 64,
    "timeTransformerDepth": 1,
    "freqTransformerDepth": 1,
    "linearTransformerDepth": 0,
    "maskEstimatorDepth": 2,
    "mlpExpansionFactor": 2,
    "freqsPerBands": BANDS,
    "deterministic": {
        "modelType": "bs_roformer",
        "useContextTime": False,
    },
    "consistencyDistilled": {
        "modelType": "bs_roformer_stems_in_out_stem_cond_random_stem",
        "useContextTime": True,
        "timeEmbedding": "Positional",
        "useMixtureFeatureConditioning": True,
        "stftAdapterType": "conv2d",
        "stftAdapterHidden": 128,
        "diffusionSigmaData": 0.06,
        "sampler": "cm_multistep_cd",
        "oneStepSigmaMax": 0.003934,
        "sigmaMin": 0.0001,
        "rho": 9,
    },
}


@dataclass(frozen=True)
class ComponentSpec:
    id: str
    state_prefix: str
    namespace: str
    expected_tensor_count: int
    expected_numel: int


DETERMINISTIC_COMPONENT = ComponentSpec(
    id="deterministic",
    state_prefix="model.unet.",
    namespace="det",
    expected_tensor_count=1_355,
    expected_numel=131_704_612,
)
CD_COMPONENT = ComponentSpec(
    id="consistency_distilled",
    state_prefix="net.model.diffusion.net.",
    namespace="cd",
    expected_tensor_count=1_502,
    expected_numel=179_866_024,
)
CD_EMBEDDED_DETERMINISTIC_PREFIX = "pre_trained_mixture_feature_extractor_model.model.unet."
# Model1d registers its underlying separator both as ``model.unet`` and as the
# internal Diffusion object's ``model.diffusion.net``. The latter is canonical
# for this exporter; the former is a state-dict alias.
CD_DUPLICATE_STUDENT_PREFIX = "net.model.unet."


class ConversionError(RuntimeError):
    """An input or output failed an intentional package invariant."""


class IgnoredLightningMetadata:
    """Inert replacement for unavailable Lightning/Hydra metadata classes.

    The release checkpoints pickle dataset/config objects in metadata that the
    exporter never reads. Their tensor ``state_dict`` is still reconstructed by
    Torch's standard unpickler.  Keeping these inert avoids installing the
    upstream training stack just to unpack an inference state dictionary.
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        self.state = state


class LightningMetadataUnpickler(pickle.Unpickler):
    """Resolve unavailable metadata globals without executing upstream imports."""

    def find_class(self, module: str, name: str) -> Any:
        # These are training-only values in the released checkpoint's
        # hyperparameters. Do not import the upstream project or its optional
        # training dependencies merely to deserialize them.
        if module == "main" or module.startswith("main."):
            return IgnoredLightningMetadata
        if module == "ml_collections" or module.startswith("ml_collections."):
            return IgnoredLightningMetadata
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, ImportError, AttributeError):
            # State-dict tensor rebuilding uses Torch globals, which resolve
            # normally. Any remaining unavailable global can only belong to
            # unused checkpoint metadata because the source SHA-256 was
            # verified before this loader is entered.
            return IgnoredLightningMetadata


class LightningCheckpointPickleModule:
    """pickle-module facade accepted by torch.load()."""

    Unpickler = LightningMetadataUnpickler
    load = pickle.load
    dump = pickle.dump
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


@dataclass
class SelectedTensor:
    component: ComponentSpec
    suffix: str
    source_key: str
    name: str
    tensor: Any


@dataclass(frozen=True)
class StorageView:
    storage_id: int
    storage_nbytes: int
    storage_offset: int
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass(frozen=True)
class Payload:
    name: str
    offset: int
    byte_length: int
    sha256: str
    numel: int
    shape: tuple[int, ...]
    layout: str


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def verify_source(path: Path, spec: SourceSpec) -> dict[str, Any]:
    if not path.is_file():
        raise ConversionError(f"{spec.role} checkpoint is not a regular file: {path}")

    sha256, byte_count = sha256_file(path)
    if byte_count != spec.expected_bytes:
        raise ConversionError(
            f"{spec.role} checkpoint has {byte_count:,} bytes; expected "
            f"{spec.expected_bytes:,} for the released DiCoSe checkpoint."
        )
    if sha256 != spec.expected_sha256:
        raise ConversionError(
            f"{spec.role} checkpoint SHA-256 does not match the released DiCoSe object.\n"
            f"expected: {spec.expected_sha256}\nactual:   {sha256}"
        )

    return {
        "role": spec.role,
        "fileName": Path(spec.repository_path).name,
        "byteLength": byte_count,
        "sha256": sha256,
    }


def download_source_checkpoints(
    cache_dir: Path,
    snapshot_download: Any | None = None,
) -> tuple[Path, Path]:
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hugging_face_snapshot_download

        snapshot_download = hugging_face_snapshot_download

    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading pinned DiCoSe checkpoints from {UPSTREAM_REPOSITORY} "
        f"at {UPSTREAM_REVISION}...",
        flush=True,
    )
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=UPSTREAM_REPOSITORY,
                revision=UPSTREAM_REVISION,
                cache_dir=cache_dir / "huggingface",
                local_dir=cache_dir / "official",
                allow_patterns=[
                    DETERMINISTIC_SOURCE.repository_path,
                    CD_SOURCE.repository_path,
                ],
            )
        )
    except Exception as exc:
        raise ConversionError(
            "unable to download pinned DiCoSe checkpoints "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    deterministic = snapshot / DETERMINISTIC_SOURCE.repository_path
    cd = snapshot / CD_SOURCE.repository_path
    for path, spec in ((deterministic, DETERMINISTIC_SOURCE), (cd, CD_SOURCE)):
        if not path.is_file():
            raise ConversionError(
                f"Hugging Face download omitted {spec.role} checkpoint: {path}"
            )
    return deterministic, cd


def resolve_source_paths(
    args: argparse.Namespace,
    snapshot_download: Any | None = None,
) -> tuple[Path, Path]:
    deterministic = args.deterministic
    cd = args.cd
    if deterministic is None and cd is None:
        return download_source_checkpoints(args.cache_dir, snapshot_download)
    if deterministic is None or cd is None:
        raise ConversionError(
            "provide both --deterministic and --cd, or omit both to download the pinned release"
        )
    return deterministic, cd


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on caller env
        raise ConversionError(
            "PyTorch is required. Run this script with `uv run --project model ...` "
            "so model/pyproject.toml supplies the pinned runtime."
        ) from exc
    return torch


def load_lightning_state_dict(torch: Any, path: Path, role: str) -> Mapping[str, Any]:
    # mmap keeps the enormous CD checkpoint storage-backed instead of eagerly
    # copying its 4.1 GB archive into Python heap memory.
    try:
        checkpoint = torch.load(
            str(path),
            map_location="cpu",
            mmap=True,
            weights_only=False,
            pickle_module=LightningCheckpointPickleModule,
        )
    except Exception as exc:  # torch errors vary by release
        raise ConversionError(
            f"unable to load {role} Lightning checkpoint: {path} "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    if not isinstance(checkpoint, Mapping):
        raise ConversionError(f"{role} checkpoint root must be a mapping")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ConversionError(f"{role} checkpoint is missing mapping key `state_dict`")
    return state_dict


def validate_tensor(torch: Any, tensor: Any, source_key: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ConversionError(f"{source_key} is not a Tensor")
    if tensor.device.type != "cpu":
        raise ConversionError(f"{source_key} did not load onto CPU")
    if tensor.dtype != torch.float32:
        raise ConversionError(
            f"{source_key} has dtype {tensor.dtype}; released DiCoSe inference tensors must be float32"
        )
    if tensor.layout != torch.strided:
        raise ConversionError(f"{source_key} has unsupported layout {tensor.layout}")
    if tensor.numel() == 0:
        raise ConversionError(f"{source_key} is empty")


def select_component(
    torch: Any, state_dict: Mapping[str, Any], spec: ComponentSpec
) -> list[SelectedTensor]:
    selected: list[SelectedTensor] = []
    for source_key in sorted(state_dict):
        if not source_key.startswith(spec.state_prefix):
            continue
        suffix = source_key.removeprefix(spec.state_prefix)
        if not suffix:
            raise ConversionError(f"empty tensor key below {spec.state_prefix!r}")
        tensor = state_dict[source_key]
        validate_tensor(torch, tensor, source_key)
        selected.append(
            SelectedTensor(
                component=spec,
                suffix=suffix,
                source_key=source_key,
                name=f"{spec.namespace}.{suffix}",
                tensor=tensor,
            )
        )

    names = [item.name for item in selected]
    if len(names) != len(set(names)):
        raise ConversionError(f"{spec.id} selection contains duplicate output tensor names")
    numel = sum(item.tensor.numel() for item in selected)
    if len(selected) != spec.expected_tensor_count or numel != spec.expected_numel:
        raise ConversionError(
            f"{spec.id} selection does not match the released architecture: "
            f"got {len(selected):,} tensors / {numel:,} elements, expected "
            f"{spec.expected_tensor_count:,} / {spec.expected_numel:,}."
        )
    return selected


def storage_view(tensor: Any) -> StorageView:
    storage = tensor.untyped_storage()
    # _cdata is the native storage identity, unlike a Tensor's data_ptr which
    # incorporates a possible view offset. It remains stable for this load.
    storage_id = int(storage._cdata)
    return StorageView(
        storage_id=storage_id,
        storage_nbytes=int(storage.nbytes()),
        storage_offset=int(tensor.storage_offset()),
        dtype=str(tensor.dtype),
        shape=tuple(int(value) for value in tensor.shape),
        stride=tuple(int(value) for value in tensor.stride()),
    )


def verify_cd_duplicate_student_aliases(
    torch: Any, state_dict: Mapping[str, Any], selected_student: Sequence[SelectedTensor]
) -> None:
    """Ensure the excluded Lightning alias really aliases the canonical student."""

    duplicate: dict[str, Any] = {
        key.removeprefix(CD_DUPLICATE_STUDENT_PREFIX): value
        for key, value in state_dict.items()
        if key.startswith(CD_DUPLICATE_STUDENT_PREFIX)
    }
    canonical = {item.suffix: item.tensor for item in selected_student}
    if set(duplicate) != set(canonical):
        raise ConversionError(
            "CD checkpoint's `net.model.unet.` alias does not exactly mirror "
            "`net.model.diffusion.net.`. Refusing an ambiguous student selection."
        )
    for suffix, tensor in canonical.items():
        other = duplicate[suffix]
        validate_tensor(torch, other, f"{CD_DUPLICATE_STUDENT_PREFIX}{suffix}")
        if storage_view(tensor) != storage_view(other):
            raise ConversionError(
                f"CD duplicated student storage is not an exact alias for {suffix!r}"
            )


def verify_embedded_deterministic(
    torch: Any,
    deterministic: Sequence[SelectedTensor],
    cd_state_dict: Mapping[str, Any],
) -> None:
    """Fail if the CD checkpoint was conditioned on different deterministic weights."""

    embedded: dict[str, Any] = {
        key.removeprefix(CD_EMBEDDED_DETERMINISTIC_PREFIX): value
        for key, value in cd_state_dict.items()
        if key.startswith(CD_EMBEDDED_DETERMINISTIC_PREFIX)
    }
    expected = {item.suffix: item.tensor for item in deterministic}
    if set(embedded) != set(expected):
        raise ConversionError(
            "CD checkpoint's embedded frozen deterministic extractor does not match "
            "the expected DiCoSe BS-RoFormer tensor namespace."
        )

    for suffix, tensor in expected.items():
        other = embedded[suffix]
        validate_tensor(torch, other, f"{CD_EMBEDDED_DETERMINISTIC_PREFIX}{suffix}")
        if tuple(tensor.shape) != tuple(other.shape) or not torch.equal(tensor, other):
            raise ConversionError(
                "The supplied deterministic checkpoint is not bit-identical to the "
                f"extractor embedded in the CD checkpoint at tensor {suffix!r}."
            )


def packed_layout_and_shape(tensor: Any, source_key: str) -> tuple[str, tuple[int, ...]]:
    """Return the WebGPU layout and shape for one released inference tensor.

    PyTorch ``nn.Linear`` persists its matrix as ``[out_features, in_features]``.
    The WGSL dense kernels index weights as ``[in_features, out_features]`` so
    all released two-dimensional ``*.weight`` tensors are transposed at export,
    except the student-conditioning lookup table.  The checkpoint hash makes
    this intentionally narrow classification safe: the only non-Linear 2-D
    weight in these two inference namespaces is ``stem_embedding.weight``.
    """

    shape = tuple(int(value) for value in tensor.shape)
    if tensor.ndim == 4 and source_key.endswith(".weight"):
        # PyTorch Conv2d tensors are [out_channels, in_channels, height, width]
        # and WGSL uses that same OIHW ordering.
        return "conv-oihw", shape
    if tensor.ndim == 2 and source_key.endswith(".weight"):
        if source_key.endswith(".stem_embedding.weight"):
            return "row-major", shape
        packed_shape = (shape[1], shape[0])
        inner, columns = packed_shape
        if inner % LINEAR_TILE_INNER == 0:
            for tile_columns in LINEAR_TILE_COLUMNS:
                if columns % tile_columns == 0:
                    return f"linear-tile-n{tile_columns}-k{LINEAR_TILE_INNER}", packed_shape
        return "linear-in-out", packed_shape
    return "row-major", shape


def f16_bytes(torch: Any, tensor: Any, source_key: str, layout: str) -> bytes:
    source = tensor.detach()
    if layout == "linear-in-out" or layout.startswith("linear-tile-"):
        if source.ndim != 2:
            raise ConversionError(f"linear layout requested for non-matrix {source_key}")
        source = source.transpose(0, 1)
    if layout.startswith("linear-tile-"):
        inner, columns = (int(value) for value in source.shape)
        tile_columns = int(layout.removeprefix("linear-tile-n").split("-", 1)[0])
        if inner % LINEAR_TILE_INNER or columns % tile_columns:
            raise ConversionError(f"invalid tiled linear shape for {source_key}: {source.shape}")
        source = (
            source.reshape(
                inner // LINEAR_TILE_INNER,
                LINEAR_TILE_INNER,
                columns // tile_columns,
                tile_columns,
            )
            .permute(2, 0, 1, 3)
        )
    converted = source.to(device="cpu", dtype=torch.float16).contiguous()
    try:
        raw = converted.view(torch.uint8).numpy().tobytes(order="C")
    finally:
        del converted
    expected = tensor.numel() * 2
    if len(raw) != expected:
        raise ConversionError(
            f"f16 conversion for {source_key} produced {len(raw)} bytes, expected {expected}"
        )
    return raw


def align_file(handle: Any, digest: Any, offset: int) -> int:
    padding = (-offset) % ALIGNMENT_BYTES
    if padding:
        zeros = b"\0" * padding
        handle.write(zeros)
        digest.update(zeros)
    return offset + padding


def write_weight_blob(
    torch: Any, stage_dir: Path, tensors: Sequence[SelectedTensor]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write a single aligned blob and return its descriptor plus tensor entries."""

    blob_path = stage_dir / OUTPUT_WEIGHTS
    aliases: dict[StorageView, Payload] = {}
    entries: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    offset = 0
    logical_elements = 0

    with blob_path.open("xb") as handle:
        for item in tensors:
            logical_elements += item.tensor.numel()
            view = storage_view(item.tensor)
            layout, packed_shape = packed_layout_and_shape(item.tensor, item.source_key)
            existing = aliases.get(view)
            if existing is None:
                offset = align_file(handle, digest, offset)
                raw = f16_bytes(torch, item.tensor, item.source_key, layout)
                payload = Payload(
                    name=item.name,
                    offset=offset,
                    byte_length=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    numel=item.tensor.numel(),
                    shape=packed_shape,
                    layout=layout,
                )
                handle.write(raw)
                digest.update(raw)
                offset += len(raw)
                aliases[view] = payload
                alias_of: str | None = None
            else:
                payload = existing
                alias_of = existing.name
                if (
                    payload.numel != item.tensor.numel()
                    or payload.shape != packed_shape
                    or payload.layout != layout
                ):
                    raise ConversionError(
                        f"storage alias packing mismatch for {item.name}; refusing to "
                        "reuse a payload with a different WebGPU interpretation"
                    )

            entry: dict[str, Any] = {
                "name": item.name,
                "sourceKey": item.source_key,
                "sourceShape": list(item.tensor.shape),
                "shape": list(packed_shape),
                "dtype": "f16",
                "layout": layout,
                "offset": payload.offset,
                "byteLength": payload.byte_length,
                "sha256": payload.sha256,
            }
            if alias_of is not None:
                entry["aliasOf"] = alias_of
            entries.append(entry)

        handle.flush()
        os.fsync(handle.fileno())

    file_sha256, byte_count = sha256_file(blob_path)
    expected_sha256 = digest.hexdigest()
    if file_sha256 != expected_sha256 or byte_count != offset:
        raise ConversionError("weights.f16.bin changed while it was being written")

    canonical_entries = [entry for entry in entries if "aliasOf" not in entry]
    return (
        {
            "file": OUTPUT_WEIGHTS,
            "byteLength": byte_count,
            "sha256": file_sha256,
            "dtype": "f16",
            "endianness": "little",
            "alignment": ALIGNMENT_BYTES,
            "logicalTensorCount": len(entries),
            "uniqueTensorCount": len(canonical_entries),
            "logicalElementCount": logical_elements,
            "uniquePayloadBytes": sum(entry["byteLength"] for entry in canonical_entries),
        },
        entries,
    )


def validate_package_layout(weights: Mapping[str, Any], tensors: Sequence[Mapping[str, Any]]) -> None:
    by_name = {entry["name"]: entry for entry in tensors}
    if len(by_name) != len(tensors):
        raise ConversionError("manifest would contain duplicate tensor names")

    occupied: list[tuple[int, int, str]] = []
    for entry in tensors:
        offset = entry["offset"]
        byte_length = entry["byteLength"]
        expected_length = 2
        for dimension in entry["shape"]:
            expected_length *= dimension
        if byte_length != expected_length:
            raise ConversionError(f"invalid packed length for {entry['name']}")
        if offset % ALIGNMENT_BYTES:
            raise ConversionError(f"unaligned packed offset for {entry['name']}")
        alias_of = entry.get("aliasOf")
        if alias_of is not None:
            target = by_name.get(alias_of)
            if target is None:
                raise ConversionError(f"alias target missing for {entry['name']}")
            for key in ("offset", "byteLength", "sha256", "shape", "layout"):
                if entry[key] != target[key]:
                    raise ConversionError(f"alias payload mismatch for {entry['name']}")
            continue
        occupied.append((offset, offset + byte_length, entry["name"]))

    previous_end = 0
    for start, end, name in sorted(occupied):
        if start < previous_end or end > weights["byteLength"]:
            raise ConversionError(f"overlapping or out-of-range payload for {name}")
        previous_end = end


def validate_canonical_package(package_dir: Path) -> None:
    weights_path = package_dir / OUTPUT_WEIGHTS
    manifest_path = package_dir / OUTPUT_MANIFEST
    weights_sha256, weights_bytes = sha256_file(weights_path)
    if (
        weights_bytes != CANONICAL_WEIGHTS_BYTES
        or weights_sha256 != CANONICAL_WEIGHTS_SHA256
    ):
        raise ConversionError(
            "generated weights do not match the canonical production package\n"
            f"expected: {CANONICAL_WEIGHTS_BYTES} bytes / {CANONICAL_WEIGHTS_SHA256}\n"
            f"actual:   {weights_bytes} bytes / {weights_sha256}"
        )
    manifest_sha256, _ = sha256_file(manifest_path)
    if manifest_sha256 != CANONICAL_MANIFEST_SHA256:
        raise ConversionError(
            "generated manifest does not match the canonical production package\n"
            f"expected: {CANONICAL_MANIFEST_SHA256}\nactual:   {manifest_sha256}"
        )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Best-effort directory durability on POSIX filesystems."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def ensure_safe_output(output: Path, sources: Sequence[Path], overwrite: bool) -> Path:
    output_parent = output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    resolved = (output_parent / output.name).resolve()

    for source in sources:
        resolved_source = source.resolve()
        if resolved_source == resolved or resolved_source.is_relative_to(resolved):
            raise ConversionError(
                "refusing an output directory that would contain or replace a source checkpoint"
            )

    if resolved.exists():
        if not resolved.is_dir():
            raise ConversionError(f"output exists and is not a directory: {resolved}")
        if not overwrite:
            raise ConversionError(
                f"output already exists: {resolved}. Re-run with --overwrite to replace it."
            )
    return resolved


def publish_transaction(stage: Path, output: Path, overwrite: bool) -> None:
    backup: Path | None = None
    try:
        if output.exists():
            if not overwrite:  # protected earlier, retained as a race-safe guard
                raise ConversionError(f"output appeared during conversion: {output}")
            backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
            os.replace(output, backup)
        os.replace(stage, output)
        fsync_directory(output.parent)
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    else:
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                # The new package was published successfully. Keeping the prior
                # directory is safer than reporting a failed conversion or
                # deleting it through a less reliable recovery path.
                print(f"warning: retained previous output at {backup}: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and strictly export the released DiCoSe BS-RoFormer "
            "deterministic and consistency-distilled checkpoints into one "
            "aligned WebGPU f16 package."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--deterministic",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "local deterministic checkpoint; omit with --cd to download both "
            "pinned files"
        ),
    )
    parser.add_argument(
        "--cd",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "local CD checkpoint; omit with --deterministic to download both "
            "pinned files"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=MODEL_DIRECTORY / "cache",
        metavar="DIR",
        help="ignored directory for resumable Hugging Face downloads",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "public/model",
        metavar="DIR",
        help="directory to publish manifest.json and weights.f16.bin",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="transactionally replace an existing output directory",
    )
    return parser


def convert(args: argparse.Namespace) -> dict[str, Any]:
    if sys.byteorder != "little":
        raise ConversionError("this f16 package format requires a little-endian host")

    deterministic_path = args.deterministic.resolve()
    cd_path = args.cd.resolve()
    output = ensure_safe_output(args.output, [deterministic_path, cd_path], args.overwrite)

    # Hash first: `torch.load(..., weights_only=False)` is necessary for the
    # Lightning metadata, so only known published bytes are accepted.
    source_manifest = [
        verify_source(deterministic_path, DETERMINISTIC_SOURCE),
        verify_source(cd_path, CD_SOURCE),
    ]
    source_by_role = {entry["role"]: entry for entry in source_manifest}

    torch = require_torch()
    deterministic_state = load_lightning_state_dict(torch, deterministic_path, "deterministic")
    deterministic = select_component(torch, deterministic_state, DETERMINISTIC_COMPONENT)

    cd_state = load_lightning_state_dict(torch, cd_path, "consistency-distilled")
    cd_student = select_component(torch, cd_state, CD_COMPONENT)
    verify_cd_duplicate_student_aliases(torch, cd_state, cd_student)
    verify_embedded_deterministic(torch, deterministic, cd_state)

    selected = [*deterministic, *cd_student]
    stage = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    try:
        stage.mkdir(parents=False, exist_ok=False)
        weights, tensor_entries = write_weight_blob(torch, stage, selected)
        validate_package_layout(weights, tensor_entries)
        manifest: dict[str, Any] = {
            "schema": PACKAGE_SCHEMA,
            "source": {
                "upstreamRevision": UPSTREAM_REVISION,
                "deterministicCheckpointSha256": source_by_role["deterministic"]["sha256"],
                "cdCheckpointSha256": source_by_role["consistency_distilled"]["sha256"],
            },
            "config": MODEL_CONFIG,
            # Retain the audited source-file details beyond the small runtime
            # source object, including their checked byte lengths.
            "sources": source_manifest,
            "components": [
                {
                    "id": DETERMINISTIC_COMPONENT.id,
                    "namespace": DETERMINISTIC_COMPONENT.namespace,
                    "stateDictPrefix": DETERMINISTIC_COMPONENT.state_prefix,
                    "expectedTensorCount": DETERMINISTIC_COMPONENT.expected_tensor_count,
                    "expectedElementCount": DETERMINISTIC_COMPONENT.expected_numel,
                },
                {
                    "id": CD_COMPONENT.id,
                    "namespace": CD_COMPONENT.namespace,
                    "stateDictPrefix": CD_COMPONENT.state_prefix,
                    "expectedTensorCount": CD_COMPONENT.expected_tensor_count,
                    "expectedElementCount": CD_COMPONENT.expected_numel,
                },
            ],
            "weights": weights,
            "tensors": tensor_entries,
        }
        write_json(stage / OUTPUT_MANIFEST, manifest)
        validate_canonical_package(stage)
        fsync_directory(stage)
        publish_transaction(stage, output, args.overwrite)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    finally:
        # Release the mmap-backed state dictionaries before returning to a caller
        # that may immediately invoke a browser benchmark in the same shell.
        del selected, deterministic, cd_student, deterministic_state, cd_state
        gc.collect()

    return {
        "output": output,
        "weights": weights,
        "tensor_count": len(tensor_entries),
        "unique_tensor_count": weights["uniqueTensorCount"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.deterministic, args.cd = resolve_source_paths(args)
        result = convert(args)
    except ConversionError as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 2

    print(
        "wrote "
        f"{result['output'] / OUTPUT_MANIFEST} and {result['output'] / OUTPUT_WEIGHTS} "
        f"({result['tensor_count']:,} logical tensors; "
        f"{result['unique_tensor_count']:,} unique f16 payloads; "
        f"{result['weights']['byteLength']:,} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
