"""Bounded, resumable downloader for exact immutable source artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests

from package_format import COPY_CHUNK_BYTES, sha256_file
from source_contract import SourceArtifact


class SourceIdentityError(ValueError):
    """Downloaded or cached bytes do not match their immutable contract."""


_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _validate_response_extent(
    response: requests.Response,
    artifact: SourceArtifact,
    *,
    offset: int,
) -> None:
    expected_response_bytes = artifact.byte_length - offset
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise SourceIdentityError(
                f"{artifact.key}: invalid Content-Length {content_length!r}"
            ) from error
        if declared_length != expected_response_bytes:
            raise SourceIdentityError(
                f"{artifact.key}: response declares {declared_length} bytes, "
                f"expected {expected_response_bytes}"
            )
    if offset:
        content_range = response.headers.get("Content-Range", "")
        match = _CONTENT_RANGE.fullmatch(content_range)
        if match is None:
            raise SourceIdentityError(
                f"{artifact.key}: invalid resume Content-Range {content_range!r}"
            )
        start, end, total = (int(value) for value in match.groups())
        if (
            start != offset
            or end != artifact.byte_length - 1
            or total != artifact.byte_length
        ):
            raise SourceIdentityError(
                f"{artifact.key}: resume Content-Range {content_range!r} "
                f"does not describe bytes {offset}-{artifact.byte_length - 1}/"
                f"{artifact.byte_length}"
            )


def verify_source_file(path: Path, artifact: SourceArtifact) -> None:
    if path.is_symlink() or not path.is_file():
        raise SourceIdentityError(f"{artifact.key}: missing or unsafe file {path}")
    actual_length = path.stat().st_size
    if actual_length != artifact.byte_length:
        raise SourceIdentityError(
            f"{artifact.key}: got {actual_length} bytes, expected {artifact.byte_length}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != artifact.sha256:
        raise SourceIdentityError(
            f"{artifact.key}: got SHA-256 {actual_sha256}, expected {artifact.sha256}"
        )


def download_artifact(
    artifact: SourceArtifact,
    cache_root: Path,
    *,
    offline: bool = False,
    session: requests.Session | None = None,
) -> Path:
    destination = artifact.cache_path(cache_root)
    if destination.exists():
        verify_source_file(destination, artifact)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.is_symlink():
        raise SourceIdentityError(f"{artifact.key}: unsafe partial path {partial}")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > artifact.byte_length:
        partial.unlink()
        offset = 0
    elif offset == artifact.byte_length:
        try:
            verify_source_file(partial, artifact)
        except SourceIdentityError:
            partial.unlink()
            offset = 0
        else:
            os.replace(partial, destination)
            return destination
    if offline:
        raise FileNotFoundError(
            f"{artifact.key}: not cached at {destination}; offline mode forbids download"
        )
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    client = session or requests.Session()
    close_client = session is None
    try:
        with client.get(
            artifact.resolve_url,
            headers=headers,
            stream=True,
            timeout=(30, 120),
            allow_redirects=True,
        ) as response:
            if offset and response.status_code == 200:
                partial.unlink(missing_ok=True)
                offset = 0
            elif offset and response.status_code != 206:
                raise SourceIdentityError(
                    f"{artifact.key}: resume returned HTTP {response.status_code}"
                )
            elif not offset and response.status_code != 200:
                raise SourceIdentityError(
                    f"{artifact.key}: download returned HTTP {response.status_code}"
                )
            response.raise_for_status()
            _validate_response_extent(response, artifact, offset=offset)
            mode = "ab" if offset else "wb"
            with partial.open(mode) as stream:
                for chunk in response.iter_content(chunk_size=COPY_CHUNK_BYTES):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    if stream.tell() > artifact.byte_length:
                        raise SourceIdentityError(
                            f"{artifact.key}: server sent more than expected "
                            f"{artifact.byte_length} bytes"
                        )
                stream.flush()
                os.fsync(stream.fileno())
        verify_source_file(partial, artifact)
        os.replace(partial, destination)
        return destination
    except BaseException:
        # Keep an in-range partial for a later Range request. Corrupt complete
        # downloads are removed so they can never masquerade as resumable data.
        if partial.exists() and partial.stat().st_size >= artifact.byte_length:
            partial.unlink()
        raise
    finally:
        if close_client:
            client.close()
