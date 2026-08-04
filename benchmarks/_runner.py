"""Shared test-only runner utilities for the released benchmarks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return config


def require_reference_model(config: dict[str, Any], model: str) -> str:
    reference = config.get("reference_model", {})
    if model != reference.get("id"):
        raise ValueError(f"Only the released reference model {reference.get('id')!r} is supported")
    return str(reference["endpoint_name"])


def require_files(env: dict[str, str], names: list[str]) -> None:
    missing_variables = [name for name in names if not env.get(name)]
    if missing_variables:
        raise ValueError("Missing required environment variables: " + ", ".join(missing_variables))
    missing_files = [f"{name}={env[name]}" for name in names if not Path(env[name]).is_file()]
    if missing_files:
        raise ValueError("Required files do not exist: " + ", ".join(missing_files))


def execute(runner: Path, env: dict[str, str], dry_run: bool) -> int:
    if not runner.is_file():
        raise ValueError(f"Runner is not included: {runner}")
    command = [sys.executable, runner.name]
    print(f"Runner: {runner.relative_to(ARTIFACT_ROOT)}")
    print(f"Working directory: {runner.parent.relative_to(ARTIFACT_ROOT)}")
    print("Command: " + " ".join(command))
    if dry_run:
        return 0
    if not env.get("OPENAI_BASE_URL"):
        raise ValueError("OPENAI_BASE_URL is required")
    if not env.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required; use EMPTY only for an authorized local endpoint")
    return subprocess.run(command, cwd=runner.parent, env=env, check=False).returncode
