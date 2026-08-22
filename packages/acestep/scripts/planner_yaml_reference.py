#!/usr/bin/env python3
"""Verify committed planner CoT vectors with pinned PyYAML 6.0.3.

Run with:
  uv run --python 3.13 --with pyyaml==6.0.3 \
    python3 scripts/planner_yaml_reference.py --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "test" / "planner-yaml-vectors.json"
KEYS = ("bpm", "caption", "duration", "keyscale", "language", "timesignature")


def format_metadata(metadata: dict[str, Any]) -> str:
    items: dict[str, Any] = {}
    for key in KEYS:
        if key not in metadata or metadata[key] is None:
            continue
        value = metadata[key]
        if key == "timesignature" and isinstance(value, str) and value.endswith("/4"):
            value = value.split("/")[0]
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        items[key] = value
    content = yaml.dump(items, allow_unicode=True, sort_keys=True).strip() if items else ""
    return f"<think>\n{content}\n</think>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    if yaml.__version__ != "6.0.3":
        raise SystemExit(f"PyYAML 6.0.3 required, found {yaml.__version__}")
    payload = json.loads(VECTORS.read_text(encoding="utf-8"))
    if payload.get("pyyamlVersion") != yaml.__version__:
        raise SystemExit("vector PyYAML version does not match runtime")
    for vector in payload["cases"]:
        actual = format_metadata(vector["metadata"])
        if actual != vector["expected"]:
            raise SystemExit(f"planner YAML vector {vector['id']} differs")
    print(f"verified {len(payload['cases'])} planner YAML vectors with PyYAML {yaml.__version__}")


if __name__ == "__main__":
    main()
