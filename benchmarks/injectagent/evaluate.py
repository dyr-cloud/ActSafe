#!/usr/bin/env python3
"""Validate and print locally generated InjectAgent metrics without model calls."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
raise SystemExit(subprocess.run([sys.executable, str(ROOT / "scripts/summarize_results.py"), "--benchmark", "injectagent"], cwd=ROOT).returncode)
