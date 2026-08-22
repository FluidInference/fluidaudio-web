"""Strict, read-only, memory-mapped safetensors inspection.

The implementation intentionally parses only the small JSON header eagerly.
Tensor payloads remain memory mapped and are exposed as bounded byte chunks;
opening a multi-gigabyte checkpoint does not allocate a multi-gigabyte Python
object.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from source_contract import SafetensorsContract


MAX_HEADER_BYTES = 128 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024

DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class SafetensorsFormatError(ValueError):
    """The file is not a safe, internally consistent safetensors container."""


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def byte_length(self) -> int:
        return self.data_end - self.data_start

    @property
    def parameter_count(self) -> int:
        return math.prod(self.shape)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SafetensorsFormatError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _inventory_payload(tensors: dict[str, TensorInfo]) -> bytes:
    inventory = {
        name: {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [tensor.data_start, tensor.data_end],
        }
        for name, tensor in sorted(tensors.items())
    }
    return json.dumps(
        inventory,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class SafetensorsFile:
    """A strict mmap view over one safetensors file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            self.file_size = os.fstat(self._stream.fileno()).st_size
            if self.file_size < 10:
                raise SafetensorsFormatError(f"{path}: file is too short")
            prefix = self._stream.read(8)
            if len(prefix) != 8:
                raise SafetensorsFormatError(f"{path}: truncated header length")
            self.header_length = struct.unpack("<Q", prefix)[0]
            if self.header_length < 2 or self.header_length > MAX_HEADER_BYTES:
                raise SafetensorsFormatError(
                    f"{path}: invalid header length {self.header_length}"
                )
            self.data_offset = 8 + self.header_length
            if self.data_offset > self.file_size:
                raise SafetensorsFormatError(f"{path}: truncated JSON header")
            header = self._stream.read(self.header_length)
            if len(header) != self.header_length:
                raise SafetensorsFormatError(f"{path}: truncated JSON header")
            self.header_sha256 = hashlib.sha256(prefix + header).hexdigest()
            try:
                decoded = json.loads(
                    header,
                    object_pairs_hook=_object_without_duplicate_keys,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SafetensorsFormatError(
                    f"{path}: invalid UTF-8 JSON header: {error}"
                ) from error
            if not isinstance(decoded, dict):
                raise SafetensorsFormatError(f"{path}: header must be an object")
            metadata = decoded.pop("__metadata__", None)
            if metadata is not None and not isinstance(metadata, dict):
                raise SafetensorsFormatError(
                    f"{path}: __metadata__ must be an object"
                )
            self.metadata = metadata or {}
            self.tensors = self._parse_tensors(decoded)
            self._validate_spans()
            self.inventory_sha256 = hashlib.sha256(
                _inventory_payload(self.tensors)
            ).hexdigest()
            self.parameter_count = sum(
                tensor.parameter_count for tensor in self.tensors.values()
            )
            self._mapping = mmap.mmap(
                self._stream.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
        except BaseException:
            self._stream.close()
            raise

    def _parse_tensors(self, decoded: dict[str, Any]) -> dict[str, TensorInfo]:
        tensors: dict[str, TensorInfo] = {}
        for name, raw in decoded.items():
            if not isinstance(name, str) or not name or name == "__metadata__":
                raise SafetensorsFormatError(
                    f"{self.path}: invalid tensor name {name!r}"
                )
            if not isinstance(raw, dict) or set(raw) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise SafetensorsFormatError(
                    f"{self.path}: {name!r} has an invalid tensor record"
                )
            dtype = raw["dtype"]
            shape = raw["shape"]
            offsets = raw["data_offsets"]
            if dtype not in DTYPE_BYTES:
                raise SafetensorsFormatError(
                    f"{self.path}: {name!r} uses unsupported dtype {dtype!r}"
                )
            if not isinstance(shape, list) or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                for dimension in shape
            ):
                raise SafetensorsFormatError(
                    f"{self.path}: {name!r} has invalid shape {shape!r}"
                )
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    not isinstance(offset, int)
                    or isinstance(offset, bool)
                    or offset < 0
                    for offset in offsets
                )
                or offsets[1] < offsets[0]
            ):
                raise SafetensorsFormatError(
                    f"{self.path}: {name!r} has invalid offsets {offsets!r}"
                )
            expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
            actual_bytes = offsets[1] - offsets[0]
            if actual_bytes != expected_bytes:
                raise SafetensorsFormatError(
                    f"{self.path}: {name!r} has {actual_bytes} payload bytes; "
                    f"shape and dtype require {expected_bytes}"
                )
            tensors[name] = TensorInfo(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                data_start=offsets[0],
                data_end=offsets[1],
            )
        if not tensors:
            raise SafetensorsFormatError(f"{self.path}: no tensors")
        return tensors

    def _validate_spans(self) -> None:
        data_bytes = self.file_size - self.data_offset
        cursor = 0
        for tensor in sorted(
            self.tensors.values(),
            key=lambda item: (item.data_start, item.data_end, item.name),
        ):
            if tensor.data_start != cursor:
                relation = "overlaps prior data" if tensor.data_start < cursor else "has a gap"
                raise SafetensorsFormatError(
                    f"{self.path}: tensor {tensor.name!r} {relation} at "
                    f"{tensor.data_start}; expected {cursor}"
                )
            cursor = tensor.data_end
        if cursor != data_bytes:
            raise SafetensorsFormatError(
                f"{self.path}: tensor payload accounts for {cursor} bytes, "
                f"but file has {data_bytes} data bytes"
            )

    def assert_contract(self, contract: SafetensorsContract) -> None:
        checks = {
            "tensor count": (len(self.tensors), contract.tensor_count),
            "parameter count": (self.parameter_count, contract.parameter_count),
            "header length": (self.header_length, contract.header_length),
            "header SHA-256": (self.header_sha256, contract.header_sha256),
            "inventory SHA-256": (
                self.inventory_sha256,
                contract.inventory_sha256,
            ),
        }
        failures = [
            f"{label}: got {actual!r}, expected {expected!r}"
            for label, (actual, expected) in checks.items()
            if actual != expected
        ]
        if failures:
            raise SafetensorsFormatError(
                f"{self.path}: source inventory mismatch:\n  "
                + "\n  ".join(failures)
            )

    def tensor(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"{self.path}: no tensor named {name!r}") from error

    def iter_tensor_chunks(
        self,
        name: str,
        *,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> Iterator[bytes]:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if self._mapping is None:
            raise RuntimeError("Safetensors file is closed")
        tensor = self.tensor(name)
        absolute_start = self.data_offset + tensor.data_start
        absolute_end = self.data_offset + tensor.data_end
        for start in range(absolute_start, absolute_end, chunk_bytes):
            end = min(start + chunk_bytes, absolute_end)
            yield self._mapping[start:end]

    def read_tensor_region(
        self,
        name: str,
        byte_offset: int,
        byte_length: int,
    ) -> bytes:
        if self._mapping is None:
            raise RuntimeError("Safetensors file is closed")
        tensor = self.tensor(name)
        if (
            byte_offset < 0
            or byte_length < 0
            or byte_offset + byte_length > tensor.byte_length
        ):
            raise ValueError(f"{name!r}: requested region is out of bounds")
        start = self.data_offset + tensor.data_start + byte_offset
        return self._mapping[start : start + byte_length]

    def read_tensor_bytes(self, name: str, *, limit: int = 64 * 1024 * 1024) -> bytes:
        tensor = self.tensor(name)
        if tensor.byte_length > limit:
            raise ValueError(
                f"{name!r} is {tensor.byte_length} bytes; explicit limit is {limit}"
            )
        return b"".join(self.iter_tensor_chunks(name, chunk_bytes=limit))

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self._stream.close()

    def __enter__(self) -> SafetensorsFile:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
