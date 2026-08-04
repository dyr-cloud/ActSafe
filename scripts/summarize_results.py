#!/usr/bin/env python3
"""Validate and print stored metrics for one released benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_tee_agent_artifact.results import ResultValidationError, summarize_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("asb", "dojo", "injectagent"))
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    candidates = args.paths or sorted((ROOT / "results" / args.benchmark).rglob("*.json"))
    reports = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable result {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
            continue
        try:
            reports.append(summarize_result(path))
        except ResultValidationError as exc:
            print(f"Result validation failed: {exc}", file=sys.stderr)
            return 1
    if not reports:
        print(f"No structured {args.benchmark} result summaries were found.", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=True))
    else:
        for report in reports:
            summary = report["summary"]
            metrics = " ".join(
                f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in summary.items()
                if key.endswith(("_percent", "_asr", "_bu", "_coverage", "_fpr"))
            )
            print(f"{report['path']}: total={report['total']} {metrics}".rstrip())
        print(f"Validated {len(reports)} structured {args.benchmark} result files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
