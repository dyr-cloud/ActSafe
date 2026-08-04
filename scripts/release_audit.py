#!/usr/bin/env python3
"""Check release scope, forbidden paths, credentials, and first-party language."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FIRST_PARTY_SUFFIXES = {".py", ".sh", ".md", ".toml", ".yaml", ".yml"}
FORBIDDEN_CODE_TERMS = re.compile(r"\bmelon\b|\bpromptarmor\b|\bf-secure\b|\bfsecure\b|\bbipia\b|\bmulturn\b|\bincon\b|\bace\b", re.I)
FORBIDDEN_PATH_TERMS = re.compile(r"train|fine.?tun|split_data|dataset_(?:build|construct)|checkpoint", re.I)
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b", re.I)
WINDOWS_PATH = re.compile(r"[A-Za-z]:\\Users\\")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    problems: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        asb_training = (
            relative.parts[:3] == ("benchmarks", "asb", "training")
            or relative.as_posix() == "requirements-training.txt"
            or relative.as_posix() == "scripts/verify_checkpoints.py"
        )
        if path.is_dir():
            if path.name != "checkpoints" and FORBIDDEN_PATH_TERMS.search(path.name) and not asb_training:
                problems.append(f"forbidden directory name: {path.relative_to(root)}")
            continue
        if FORBIDDEN_PATH_TERMS.search(path.name) and not asb_training:
            problems.append(f"forbidden file name: {relative}")
        first_party_json = (
            path.suffix.lower() == ".json"
            and (
                relative.parts[0] == "configs"
                or path.name.endswith(".meta.json")
                or relative.as_posix() == "checkpoints/MANIFEST.json"
            )
        )
        if (
            path.suffix.lower() not in FIRST_PARTY_SUFFIXES
            and path.name not in {".env.example", ".gitignore"}
            and not first_party_json
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if CJK.search(text):
            problems.append(f"Chinese text: {relative}")
        if SECRET.search(text) and path.name != ".env.example":
            problems.append(f"possible credential: {relative}")
        if WINDOWS_PATH.search(text):
            problems.append(f"machine-specific path: {relative}")
        executable_scope = relative.parts[0] in {"benchmarks", "configs", "scripts", "src", "tests", "examples"}
        scope_test = path.name == "test_release_scope.py"
        if executable_scope and path.name != "release_audit.py" and not scope_test and FORBIDDEN_CODE_TERMS.search(text):
            problems.append(f"comparison-defense reference in executable scope: {relative}")
    for problem in problems:
        print(problem)
    if problems:
        print(f"Release audit failed with {len(problems)} issue(s).")
        return 1
    print("Release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
