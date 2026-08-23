from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from download import (  # noqa: E402
    SourceIdentityError,
    _validate_response_extent,
    download_artifact,
)
from source_contract import SourceArtifact  # noqa: E402


def artifact_for(payload: bytes) -> SourceArtifact:
    return SourceArtifact(
        key="fixture",
        component="fixture",
        repository="test/fixture",
        revision="0" * 40,
        path="model.bin",
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


class DownloadTests(unittest.TestCase):
    def test_complete_verified_partial_is_promoted_without_network(self) -> None:
        payload = b"complete pinned payload"
        artifact = artifact_for(payload)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            destination = artifact.cache_path(cache)
            destination.parent.mkdir(parents=True)
            partial = destination.with_name(destination.name + ".partial")
            partial.write_bytes(payload)
            result = download_artifact(artifact, cache, offline=True)
            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_resume_extent_requires_exact_content_range_and_length(self) -> None:
        artifact = artifact_for(b"0123456789")
        response = SimpleNamespace(
            headers={"Content-Length": "6", "Content-Range": "bytes 4-9/10"}
        )
        _validate_response_extent(response, artifact, offset=4)
        for content_range in ("", "bytes 3-9/10", "bytes 4-8/10", "bytes 4-9/11"):
            bad = SimpleNamespace(
                headers={"Content-Length": "6", "Content-Range": content_range}
            )
            with self.assertRaises(SourceIdentityError):
                _validate_response_extent(bad, artifact, offset=4)
        bad_length = SimpleNamespace(
            headers={"Content-Length": "5", "Content-Range": "bytes 4-9/10"}
        )
        with self.assertRaises(SourceIdentityError):
            _validate_response_extent(bad_length, artifact, offset=4)


if __name__ == "__main__":
    unittest.main()
