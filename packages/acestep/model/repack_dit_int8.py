"""Create the authenticated OPT-0091 packed-INT8 DiT runtime package.

The input is the exact revision-7 mixed DiT package. Only the 216 repeated
dense matrices are repacked; cross-attention K/V and support tensors retain
their authenticated BF16 storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SOURCE_MANIFEST_SHA256 = (
    "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f"
)
SOURCE_LAYOUT = "dit-gemm-n256-k32-tile-major-v1"
SOURCE_TRANSFORMATION = "bf16-to-ieee-fp16-dit-gemm-n256-k32-tile-major-v1"
PACKED_LAYOUT = "dit-gemm-n256-k32-int8-fp16-scale-tile-major-v1"
PACKED_TRANSFORMATION = (
    "bf16-to-symmetric-int8-fp16-scale-n256-k32-tile-major-v1"
)
PACKED_DTYPE = "uint32-int8-fp16-blocks"
PACKED_WORDS_PER_TILE = 2_176
EXPECTED_DENSE_TENSORS = 216
EXPECTED_WEIGHT_BYTES = 1_699_602_432


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verify_runtime_files(directory: Path, manifest: dict[str, object]) -> None:
    for record in manifest["files"]:
        if record["kind"] != "weights":
            continue
        path = directory / record["name"]
        payload = path.read_bytes()
        if len(payload) != record["byteLength"] or _sha256(payload) != record["sha256"]:
            raise ValueError(f"packed shard identity changed: {record['name']}")


def pack_dense_tensor(payload: bytes, shape: list[int]) -> tuple[bytes, list[int]]:
    """Pack one rev7 [N,K] FP16 tile-major matrix deterministically."""

    if len(shape) != 2:
        raise ValueError("packed INT8 dense tensor must be rank two")
    columns, inner = shape
    if columns % 256 != 0 or inner % 32 != 0:
        raise ValueError("packed INT8 dense dimensions must divide N256/K32")
    expected_bytes = columns * inner * 2
    if len(payload) != expected_bytes:
        raise ValueError("packed INT8 source tensor byte length changed")

    tiles = np.frombuffer(payload, dtype="<f2").reshape(
        columns // 256, inner // 32, 32, 256
    )
    output = bytearray()
    for column_tile in range(columns // 256):
        for inner_tile in range(inner // 32):
            values = tiles[column_tile, inner_tile].astype(np.float32)
            scales = (np.max(np.abs(values), axis=0) / np.float32(127.0)).astype(
                "<f2"
            )
            divisors = scales.astype(np.float32)
            divisors = np.where(divisors == 0.0, np.float32(1.0), divisors)
            quantized = np.clip(
                np.rint(values / divisors[None, :]), -127, 127
            ).astype(np.int8)
            output.extend(quantized.tobytes(order="C"))
            output.extend(scales.tobytes(order="C"))

    storage_shape = [columns // 256, inner // 32, PACKED_WORDS_PER_TILE]
    if len(output) != np.prod(storage_shape, dtype=np.int64) * 4:
        raise ValueError("packed INT8 tensor storage shape changed")
    return bytes(output), storage_shape


def repack(source: Path, output_root: Path) -> tuple[Path, dict[str, int | str]]:
    source_manifest_bytes = (source / "manifest.json").read_bytes()
    if _sha256(source_manifest_bytes) != SOURCE_MANIFEST_SHA256:
        raise ValueError("OPT-0091 source manifest identity changed")
    manifest = json.loads(source_manifest_bytes)
    if (
        manifest.get("profile") != "fp16-dit-dense-experimental"
        or manifest.get("provenance", {}).get("converterRevision") != 7
    ):
        raise ValueError("OPT-0091 requires the authenticated revision-7 package")

    tensors_by_shard: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for name, tensor in manifest["tensors"].items():
        tensors_by_shard.setdefault(tensor["shard"], []).append((name, tensor))

    dense_count = 0
    source_dense_bytes = 0
    packed_dense_bytes = 0
    new_files: list[dict[str, object]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".opt-0091-", dir=output_root) as temporary:
        stage = Path(temporary)
        for file_record in manifest["files"]:
            if file_record["kind"] != "weights":
                new_files.append(dict(file_record))
                continue
            source_path = source / file_record["name"]
            source_bytes = source_path.read_bytes()
            if (
                len(source_bytes) != file_record["byteLength"]
                or _sha256(source_bytes) != file_record["sha256"]
            ):
                raise ValueError(f"source shard identity changed: {file_record['name']}")
            output = bytearray()
            tensors = sorted(
                tensors_by_shard[file_record["name"]],
                key=lambda item: item[1]["byteOffset"],
            )
            for name, tensor in tensors:
                output.extend(b"\0" * (-len(output) % 256))
                start = tensor["byteOffset"]
                payload = source_bytes[start : start + tensor["byteLength"]]
                tensor["byteOffset"] = len(output)
                if tensor["layout"] == SOURCE_LAYOUT:
                    if (
                        tensor["dtype"] != "float16"
                        or tensor["transformation"] != SOURCE_TRANSFORMATION
                    ):
                        raise ValueError(f"dense source contract changed: {name}")
                    source_dense_bytes += len(payload)
                    payload, storage_shape = pack_dense_tensor(
                        payload, tensor["logicalShape"]
                    )
                    tensor.update(
                        dtype=PACKED_DTYPE,
                        layout=PACKED_LAYOUT,
                        transformation=PACKED_TRANSFORMATION,
                        storageShape=storage_shape,
                        byteLength=len(payload),
                    )
                    dense_count += 1
                    packed_dense_bytes += len(payload)
                output.extend(payload)
            output.extend(b"\0" * (-len(output) % 256))
            destination = stage / file_record["name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(output)
            new_files.append(
                {
                    **file_record,
                    "byteLength": len(output),
                    "sha256": _sha256(output),
                }
            )

        weight_bytes = sum(
            record["byteLength"] for record in new_files if record["kind"] == "weights"
        )
        if dense_count != EXPECTED_DENSE_TENSORS or weight_bytes != EXPECTED_WEIGHT_BYTES:
            raise ValueError("OPT-0091 packed inventory changed")
        manifest["profile"] = "int8-dit-dense-experimental"
        manifest["files"] = new_files
        manifest["provenance"]["converterRevision"] = 9
        manifest_bytes = _canonical_json(manifest)
        manifest_sha256 = _sha256(manifest_bytes)
        (stage / "manifest.json").write_bytes(manifest_bytes)
        destination = output_root / manifest_sha256
        if destination.is_symlink():
            raise ValueError("OPT-0091 output must not be a symlink")
        if destination.exists():
            if (destination / "manifest.json").read_bytes() != manifest_bytes:
                raise ValueError("existing OPT-0091 output does not match its digest")
            _verify_runtime_files(destination, manifest)
        else:
            os.rename(stage, destination)
            _verify_runtime_files(destination, manifest)

    metrics: dict[str, int | str] = {
        "manifestSha256": manifest_sha256,
        "manifestBytes": len(manifest_bytes),
        "denseTensorCount": dense_count,
        "sourceDenseBytes": source_dense_bytes,
        "packedDenseBytes": packed_dense_bytes,
        "weightBytes": weight_bytes,
    }
    return destination, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    destination, metrics = repack(args.source.resolve(), args.output_root.resolve())
    print(json.dumps({**metrics, "directory": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
