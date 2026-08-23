#!/usr/bin/env python3
"""CLI for pinned ACE native-reference preflight, capture, and replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import (
    CaptureReplayContract,
    compare_captures,
    load_environment_receipt,
    verify_capture,
)
from .contracts import TapContract, load_fixture, reference_tool_identity, verify_golden_manifest
from .inputs import expected_input_bundle_identity, prepare_input_bundle
from .preflight import (
    REPOSITORY_ROOT,
    raw_model_contract_identity,
    run_preflight,
    verify_browser_package,
    verify_source_checkout,
)


DEFAULT_SOURCE = Path("/tmp/ace-step-1.5-source")
DEFAULT_CACHE = REPOSITORY_ROOT / "model" / "cache"
DEFAULT_PACKAGE = REPOSITORY_ROOT / "model" / "files-reference"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "golden-local"


def _fixture_path(value: str) -> Path:
    supplied = Path(value)
    if supplied.suffix == ".json" or supplied.parent != Path("."):
        raise ValueError("fixtures must be selected by committed fixture id")
    return (REPOSITORY_ROOT / "golden" / "fixtures" / f"{value}.json").resolve()


def _add_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--browser-package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--fixture", required=True, help="committed fixture id")
    parser.add_argument("--inputs", type=Path, help="external random-input manifest")
    parser.add_argument(
        "--deep-payload-hashes",
        action="store_true",
        help="hash every raw model and browser-package payload (required for capture)",
    )


def _preflight(args: argparse.Namespace):
    return run_preflight(
        source_root=args.source.resolve(),
        model_cache=args.model_cache.resolve(),
        browser_package=args.browser_package.resolve(),
        fixture_path=_fixture_path(args.fixture),
        input_manifest=args.inputs.resolve() if args.inputs else None,
        deep_payload_hashes=args.deep_payload_hashes,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="authenticate contracts and inputs")
    _add_preflight_arguments(preflight)

    prepare = commands.add_parser(
        "prepare-inputs",
        help="materialize browser-defined noise and planner words in ignored storage",
    )
    prepare.add_argument("--fixture", required=True, help="committed fixture id")
    prepare.add_argument(
        "--output-root",
        type=Path,
        help="destination directory (default: golden-local/inputs/<fixture-id>)",
    )
    prepare.add_argument(
        "--planner-word-capacity",
        type=int,
        default=4096,
        help="planner-only raw-word capacity spanning CoT, codes, and stop tokens",
    )

    run = commands.add_parser("run", help="execute one atomic upstream capture")
    _add_preflight_arguments(run)
    run.add_argument("--capture-id", required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--device", choices=("cuda", "xpu"), required=True)

    verify = commands.add_parser("verify", help="replay capture hashes and inventory")
    verify.add_argument("capture", type=Path)
    verify.add_argument("--fixture", required=True, help="committed fixture id")
    verify.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    verify.add_argument("--browser-package", type=Path, default=DEFAULT_PACKAGE)
    verify.add_argument("--environment-contract", type=Path, required=True)

    compare = commands.add_parser(
        "compare", help="require two fresh-process captures to have identical artifacts"
    )
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--fixture", required=True, help="committed fixture id")
    compare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    compare.add_argument("--browser-package", type=Path, default=DEFAULT_PACKAGE)
    compare.add_argument("--environment-contract", type=Path, required=True)
    return parser


def _replay_contract(args: argparse.Namespace, capture_root: Path) -> CaptureReplayContract:
    fixture = load_fixture(_fixture_path(args.fixture))
    taps = TapContract(REPOSITORY_ROOT / "golden" / "taps.json")
    from .jsonio import load_json, require_object

    capture_json = require_object(load_json(capture_root / "capture.json"), name="capture.json")
    input_identity = require_object(capture_json.get("inputBundle"), name="inputBundle")
    planner = input_identity.get("plannerSampling")
    capacity = planner.get("wordCapacity") if isinstance(planner, dict) else None
    source = verify_source_checkout(args.source.resolve())
    source.update(
        {
            "aceModelRevision": fixture.contract["source"]["mainModelRevision"],
            "plannerModelRevision": fixture.contract["source"]["plannerModelRevision"],
            "rawModelArtifactSetSha256": raw_model_contract_identity(),
        }
    )
    capture_identity = capture_json.get("captureIdentitySha256")
    if not isinstance(capture_identity, str):
        raise ValueError("capture has no valid identity for environment-contract binding")
    environment = load_environment_receipt(
        args.environment_contract.resolve(),
        capture_identity_sha256=capture_identity,
    )
    return CaptureReplayContract(
        fixture=fixture,
        taps=taps,
        golden_manifest_id=verify_golden_manifest(REPOSITORY_ROOT / "golden"),
        reference_tool=reference_tool_identity(REPOSITORY_ROOT),
        source=source,
        browser_package=verify_browser_package(
            args.browser_package.resolve(),
            verify_hashes=True,
        ),
        input_bundle=expected_input_bundle_identity(
            fixture,
            planner_word_capacity=capacity,
        ),
        environment=environment,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare-inputs":
            if sys.version_info[:2] != (3, 13):
                raise ValueError("prepare-inputs must run in the frozen reference Python 3.13 environment")
            fixture = load_fixture(_fixture_path(args.fixture))
            output_root = (
                args.output_root.resolve()
                if args.output_root is not None
                else (DEFAULT_OUTPUT / "inputs" / fixture.fixture_id).resolve()
            )
            manifest = prepare_input_bundle(
                fixture,
                output_root,
                planner_word_capacity=args.planner_word_capacity,
            )
            print(manifest)
            return 0
        if args.command == "preflight":
            result = _preflight(args)
            print(json.dumps(result.report(), indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            if sys.version_info[:2] != (3, 12):
                raise ValueError(
                    "native run must use ACE's locked Python 3.12 environment; "
                    "run `uv run --frozen --project <ACE-source> --python 3.12 "
                    "python3 -m reference.capture run ...`"
                )
            if not args.deep_payload_hashes:
                raise ValueError("run requires --deep-payload-hashes")
            result = _preflight(args)
            if not result.capture_authorized:
                raise ValueError(f"capture preflight is blocked: {list(result.blockers)}")
            from .runner import execute_capture

            capture_root = execute_capture(
                result,
                source_root=args.source.resolve(),
                model_cache=args.model_cache.resolve(),
                output_root=args.output_root.resolve(),
                capture_id=args.capture_id,
                device=args.device,
            )
            print(capture_root)
            return 0
        if args.command == "verify":
            root = args.capture.resolve()
            capture = verify_capture(root, replay=_replay_contract(args, root))
            print(
                json.dumps(
                    {
                        "captureId": capture["captureId"],
                        "fixtureId": capture["fixtureId"],
                        "artifactSetSha256": capture["artifactSetSha256"],
                        "status": "verified",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "compare":
            first = args.first.resolve()
            replay = _replay_contract(args, first)
            compare_captures(first, args.second.resolve(), replay=replay)
            print("capture artifact identities match")
            return 0
        raise AssertionError(args.command)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"reference capture error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
