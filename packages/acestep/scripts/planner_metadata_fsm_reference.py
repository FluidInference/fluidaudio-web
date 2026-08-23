#!/usr/bin/env python3
"""Capture/check metadata-FSM vectors from the exact pinned upstream class.

This script never downloads a model. It imports the checked-out ACE source and
loads the already authenticated local planner tokenizer with
``local_files_only=True``. Reproduce the committed vectors with CPython 3.13:

    uv run --python 3.13 \
      --with torch==2.10.0 \
      --with transformers==4.57.6 \
      --with loguru==0.7.3 \
      scripts/planner_metadata_fsm_reference.py \
      --ace-source /tmp/ACE-Step-1.5

The ACE checkout must be detached at the revision pinned below. ``--emit``
prints canonical JSON for reviewed fixture updates; it does not edit the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = ROOT / "test/planner-metadata-fsm-vectors.json"
TOKENIZER_ROOT = ROOT / "model/files-reference/assets/planner"
SOURCE_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0"
SOURCE_SHA256 = "84cf84ad894130397ba53a4cbd8666961bf578c77295b79599b288a3825faa32"
CONSTANTS_SHA256 = "7b8d4ce49649c819d1b3be87a434be2d90768308768d4620639328f906209b22"
TOKENIZER_SHA256 = "35af56c3f5cb3ea2cc578aa28a8937770981d504f183ac5c8c38baf4bbd4af4d"
VOCABULARY_SIZE = 217_204


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def u32le_sha256(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(struct.pack("<I", value))
    return digest.hexdigest()


def verify_source(source: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"ACE source is {revision}, expected {SOURCE_REVISION}")
    checks = (
        (source / "acestep/constrained_logits_processor.py", SOURCE_SHA256),
        (source / "acestep/constants.py", CONSTANTS_SHA256),
        (TOKENIZER_ROOT / "tokenizer.json", TOKENIZER_SHA256),
    )
    for path, expected in checks:
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"{path} SHA-256 is {actual}, expected {expected}")


def capture(source: Path) -> dict[str, Any]:
    verify_source(source)
    sys.path.insert(0, str(source))

    import torch
    import transformers
    from loguru import logger
    from transformers import AutoTokenizer

    logger.disable("acestep")
    from acestep.constrained_logits_processor import (  # noqa: PLC0415
        FSMState,
        MetadataConstrainedLogitsProcessor,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ROOT,
        local_files_only=True,
        use_fast=True,
    )
    if len(tokenizer) != VOCABULARY_SIZE:
        raise RuntimeError("Pinned planner tokenizer has the wrong vocabulary size")

    def processor(
        user: dict[str, str] | None = None,
        *,
        skip_caption: bool = False,
        skip_language: bool = False,
    ) -> Any:
        value = MetadataConstrainedLogitsProcessor(
            tokenizer,
            genres_vocab_path="/nonexistent/ace-genres-vocabulary",
            max_duration=240,
        )
        value.reset()
        value.set_user_metadata(user)
        value.set_stop_at_reasoning(True)
        value.set_skip_genres(True)
        value.set_skip_caption(skip_caption)
        value.set_skip_language(skip_language)
        value.set_generation_phase("cot")
        return value

    def allowed(value: Any, overrides: dict[int, float] | None = None) -> list[int]:
        scores = torch.zeros((1, len(tokenizer)), dtype=torch.float32)
        for token_id, logit in (overrides or {}).items():
            scores[0, token_id] = logit
        result = value(torch.tensor([[1, 2, 3]], dtype=torch.long), scores)
        return torch.isfinite(result[0]).nonzero().flatten().tolist()

    def drive_to(value: Any, state: Any) -> list[int]:
        emitted: list[int] = []
        while value.state != state:
            candidates = allowed(value)
            if len(candidates) != 1:
                raise RuntimeError(
                    f"{value.state} exposed {len(candidates)} candidates before {state}"
                )
            value.update_state(candidates[0])
            emitted.append(candidates[0])
        return emitted

    def prefix_vector(
        state: Any,
        user: dict[str, str] | None = None,
        *,
        skip_caption: bool = False,
        skip_language: bool = False,
        extra: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        value = processor(
            user,
            skip_caption=skip_caption,
            skip_language=skip_language,
        )
        emitted = drive_to(value, state)
        for token_id in extra:
            candidates = allowed(value)
            if token_id not in candidates:
                raise RuntimeError(f"Reference path does not admit token {token_id}")
            value.update_state(token_id)
            emitted.append(token_id)
        return {
            "emittedTokenIds": emitted,
            "allowedTokenIds": allowed(value),
        }

    metadata = {
        "bpm": "120",
        "caption": "Crisp drums.",
        "duration": "12",
        "keyscale": "C major",
        "language": "en",
        "timesignature": "4",
    }
    injected = processor(metadata)
    emitted: list[int] = []
    while injected.state != FSMState.COMPLETED:
        candidates = allowed(injected)
        if len(candidates) != 1:
            raise RuntimeError(f"Injected trace exposed {len(candidates)} candidates")
        injected.update_state(candidates[0])
        emitted.append(candidates[0])

    caption = processor({"bpm": "120"})
    caption_prefix = drive_to(caption, FSMState.CAPTION_VALUE)
    caption_allowed = allowed(caption)

    base_through_duration = {"bpm": "120", "duration": "12"}
    base_through_keyscale = {
        "bpm": "120",
        "duration": "12",
        "keyscale": "C major",
    }
    return {
        "schemaVersion": 1,
        "aceSourceRevision": SOURCE_REVISION,
        "sourceFile": "acestep/constrained_logits_processor.py",
        "sourceFileSha256": SOURCE_SHA256,
        "constantsFileSha256": CONSTANTS_SHA256,
        "sourceClass": "MetadataConstrainedLogitsProcessor",
        "referencePython": f"{sys.version_info.major}.{sys.version_info.minor}",
        "referenceTorch": torch.__version__,
        "referenceTransformers": transformers.__version__,
        "tokenizerSha256": TOKENIZER_SHA256,
        "profile": {
            "maxDuration": 240,
            "skipGenres": True,
            "stopAtReasoning": True,
            "generationPhase": "cot",
        },
        "injectedTrace": {
            "metadata": metadata,
            "emittedTokenIds": emitted,
            "decoded": tokenizer.decode(emitted),
        },
        "prefixes": {
            "bpmRoot": prefix_vector(FSMState.BPM_VALUE),
            "bpmAfterSpace": prefix_vector(FSMState.BPM_VALUE, extra=(220,)),
            "durationRoot": prefix_vector(
                FSMState.DURATION_VALUE,
                {"bpm": "120"},
                skip_caption=True,
                skip_language=True,
            ),
            "durationAfterSpace": prefix_vector(
                FSMState.DURATION_VALUE,
                {"bpm": "120"},
                skip_caption=True,
                skip_language=True,
                extra=(220,),
            ),
            "duration240Complete": prefix_vector(
                FSMState.DURATION_VALUE,
                {"bpm": "120"},
                skip_caption=True,
                skip_language=True,
                extra=(220, 17, 19, 15),
            ),
            "keyscaleRoot": prefix_vector(
                FSMState.KEYSCALE_VALUE,
                base_through_duration,
                skip_caption=True,
                skip_language=True,
            ),
            "keyscaleCMajorComplete": prefix_vector(
                FSMState.KEYSCALE_VALUE,
                base_through_duration,
                skip_caption=True,
                skip_language=True,
                extra=(356, 3598),
            ),
            "timesignatureAfterSpace": prefix_vector(
                FSMState.TIMESIG_VALUE,
                base_through_keyscale,
                skip_caption=True,
                skip_language=True,
                extra=(220,),
            ),
        },
        "languageEqualLogits": {
            **prefix_vector(
                FSMState.LANGUAGE_VALUE,
                base_through_keyscale,
                skip_caption=True,
            ),
            "decoded": " ko",
        },
        "captionMask": {
            "emittedTokenIds": caption_prefix,
            "allowedTokenCount": len(caption_allowed),
            "sortedU32LeSha256": u32le_sha256(caption_allowed),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace-source", type=Path, required=True)
    parser.add_argument("--emit", action="store_true")
    args = parser.parse_args()

    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Metadata FSM reference requires CPython 3.13")
    actual = capture(args.ace_source.resolve())
    if args.emit:
        print(json.dumps(actual, indent=2, ensure_ascii=False))
        return
    expected = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    if actual != expected:
        raise AssertionError(
            "Pinned upstream metadata-FSM capture differs from the committed vectors; "
            "rerun with --emit and review the diff"
        )
    print("planner metadata FSM vectors: OK")


if __name__ == "__main__":
    main()
