"""Restricted extraction of the one pinned PyTorch silence-latent archive."""

from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MemberContract:
    byte_length: int
    sha256: str


MEMBERS = {
    "silence_latent/data.pkl": MemberContract(
        166,
        "0cf4dfccdf90059def80535ecbc518694c65583c720447cd0f4e463eb6a87749",
    ),
    "silence_latent/byteorder": MemberContract(
        6,
        "180ca01b95f0dfdd36fbb600e51cf6e46c8ef468de56b017847886fefaf7b6f9",
    ),
    "silence_latent/data/0": MemberContract(
        3_840_000,
        "1491511c30d62238eb9b55ef0a01e220a3a664c2679444c44b3cdabd8cbbd29f",
    ),
    "silence_latent/version": MemberContract(
        2,
        "1121cfccd5913f0a63fec40a6ffd44ea64f9dc135c66634ba001d10bcf4302a2",
    ),
    "silence_latent/.data/serialization_id": MemberContract(
        40,
        "f84e805faa29263a0231be575b3f5b19d708571aa1d711f0eccc866d36de6033",
    ),
}
TENSOR_MEMBER = "silence_latent/data/0"
TENSOR_SHAPE = (1, 64, 15_000)
TENSOR_DTYPE = "float32"


def extract_silence_latent(source: Path, destination: Path) -> None:
    """Validate the exact archive without unpickling and extract raw F32 data."""

    extract_validated_zip_member(source, destination, MEMBERS, TENSOR_MEMBER)


def extract_validated_zip_member(
    source: Path,
    destination: Path,
    members: dict[str, MemberContract],
    tensor_member: str,
) -> None:
    """Validate an exact stored ZIP inventory and extract one member."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    try:
        with zipfile.ZipFile(source, "r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != list(members):
                raise ValueError("Silence latent ZIP inventory/order mismatch")
            for info in infos:
                contract = members[info.filename]
                if (
                    info.is_dir()
                    or info.file_size != contract.byte_length
                    or info.compress_type != zipfile.ZIP_STORED
                ):
                    raise ValueError(
                        f"Silence latent member contract mismatch: {info.filename}"
                    )
                digest = hashlib.sha256()
                output = partial.open("xb") if info.filename == tensor_member else None
                try:
                    with archive.open(info, "r") as reader:
                        while chunk := reader.read(1024 * 1024):
                            digest.update(chunk)
                            if output is not None:
                                output.write(chunk)
                    if output is not None:
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    if output is not None:
                        output.close()
                if digest.hexdigest() != contract.sha256:
                    raise ValueError(
                        f"Silence latent member SHA-256 mismatch: {info.filename}"
                    )
        if partial.stat().st_size != members[tensor_member].byte_length:
            raise ValueError("Extracted silence latent has the wrong length")
        os.replace(partial, destination)
    except (zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError(f"Invalid silence latent PyTorch ZIP: {error}") from error
    finally:
        partial.unlink(missing_ok=True)
