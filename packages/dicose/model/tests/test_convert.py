from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert  # noqa: E402


class ModelPreparationTests(unittest.TestCase):
    def test_default_download_is_pinned_and_selective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            calls: list[dict[str, object]] = []

            def snapshot_download(**kwargs: object) -> str:
                calls.append(kwargs)
                local_dir = Path(kwargs["local_dir"])
                for spec in (convert.DETERMINISTIC_SOURCE, convert.CD_SOURCE):
                    path = local_dir / spec.repository_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                return str(local_dir)

            deterministic, cd = convert.download_source_checkpoints(
                cache_dir,
                snapshot_download,
            )

            self.assertEqual(len(calls), 1)
            call = calls[0]
            self.assertEqual(call["repo_id"], convert.UPSTREAM_REPOSITORY)
            self.assertEqual(call["revision"], convert.UPSTREAM_REVISION)
            self.assertEqual(
                call["allow_patterns"],
                [
                    convert.DETERMINISTIC_SOURCE.repository_path,
                    convert.CD_SOURCE.repository_path,
                ],
            )
            self.assertEqual(
                deterministic,
                Path(call["local_dir"])
                / convert.DETERMINISTIC_SOURCE.repository_path,
            )
            self.assertEqual(
                cd,
                Path(call["local_dir"]) / convert.CD_SOURCE.repository_path,
            )

    def test_local_overrides_require_both_checkpoints(self) -> None:
        args = argparse.Namespace(
            deterministic=Path("det.ckpt"),
            cd=None,
            cache_dir=Path("cache"),
        )
        with self.assertRaisesRegex(convert.ConversionError, "provide both"):
            convert.resolve_source_paths(args)

    def test_download_failure_has_a_concise_conversion_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:

            def fail_download(**_kwargs: object) -> str:
                raise RuntimeError("network unavailable")

            with self.assertRaisesRegex(
                convert.ConversionError,
                "unable to download pinned DiCoSe checkpoints",
            ):
                convert.download_source_checkpoints(
                    Path(temporary) / "cache",
                    fail_download,
                )

    def test_local_overrides_bypass_download(self) -> None:
        deterministic = Path("det.ckpt")
        cd = Path("cd.ckpt")
        args = argparse.Namespace(
            deterministic=deterministic,
            cd=cd,
            cache_dir=Path("cache"),
        )
        downloader = mock.Mock(side_effect=AssertionError("unexpected download"))
        self.assertEqual(
            convert.resolve_source_paths(args, downloader),
            (deterministic, cd),
        )
        downloader.assert_not_called()

    def test_default_paths_do_not_depend_on_the_calling_directory(self) -> None:
        args = convert.build_parser().parse_args([])
        self.assertEqual(args.cache_dir, convert.MODEL_DIRECTORY / "cache")
        self.assertEqual(args.output, convert.REPOSITORY_ROOT / "public/model")

    def test_canonical_package_gate_checks_independent_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            weights = b"weights"
            manifest = b"manifest"
            (package / convert.OUTPUT_WEIGHTS).write_bytes(weights)
            (package / convert.OUTPUT_MANIFEST).write_bytes(manifest)
            with (
                mock.patch.object(convert, "CANONICAL_WEIGHTS_BYTES", len(weights)),
                mock.patch.object(
                    convert,
                    "CANONICAL_WEIGHTS_SHA256",
                    hashlib.sha256(weights).hexdigest(),
                ),
                mock.patch.object(
                    convert,
                    "CANONICAL_MANIFEST_SHA256",
                    hashlib.sha256(manifest).hexdigest(),
                ),
            ):
                convert.validate_canonical_package(package)
                (package / convert.OUTPUT_WEIGHTS).write_bytes(b"changed")
                with self.assertRaisesRegex(
                    convert.ConversionError,
                    "canonical production",
                ):
                    convert.validate_canonical_package(package)


if __name__ == "__main__":
    unittest.main()
