#!/usr/bin/env python3
"""Verify the three released qwen7b checkpoint files against their manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "checkpoints/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("reference_model") != "qwen7b":
        raise SystemExit("Checkpoint manifest does not select qwen7b")
    entries = manifest.get("checkpoints")
    if not isinstance(entries, list) or {item.get("benchmark") for item in entries} != {
        "asb", "dojo", "injectagent"
    }:
        raise SystemExit("Checkpoint manifest must contain exactly the three released benchmarks")
    for item in entries:
        relative = Path(item["path"])
        config_path = root / f"configs/{item['benchmark']}/evaluation.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("reference_model", {}).get("id") != "qwen7b":
            raise SystemExit(f"Configuration does not select qwen7b: {config_path.relative_to(root)}")
        if config.get("checkpoint") != relative.as_posix():
            raise SystemExit(f"Configuration checkpoint mismatch: {config_path.relative_to(root)}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(f"Checkpoint escapes artifact root: {relative}") from exc
        if not path.is_file():
            raise SystemExit(f"Checkpoint is missing: {relative}")
        if path.name != item["filename"]:
            raise SystemExit(f"Checkpoint filename mismatch: {relative}")
        if path.stat().st_size != item["size_bytes"]:
            raise SystemExit(f"Checkpoint size mismatch: {relative}")
        actual_hash = sha256(path)
        if actual_hash != item["sha256"]:
            raise SystemExit(f"Checkpoint SHA-256 mismatch: {relative}")
        print(f"{item['benchmark']}: {relative} {actual_hash}")
    print("Checkpoint integrity verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
