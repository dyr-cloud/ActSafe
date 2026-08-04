"""Read and validate frozen structured experiment results without model calls."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ResultValidationError(ValueError):
    """Raised when stored metric counts and percentages disagree."""


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a result object and require the release's summary/results schema."""
    result_path = Path(path)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultValidationError(f"Result file does not exist: {result_path}") from exc
    except json.JSONDecodeError as exc:
        raise ResultValidationError(f"Invalid JSON in {result_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
        raise ResultValidationError(
            f"Result file lacks a top-level summary object: {result_path}"
        )
    return payload


def _validate_ratio(summary: dict[str, Any], count_key: str, percent_key: str, total: int) -> None:
    if count_key not in summary or percent_key not in summary:
        return
    count = summary[count_key]
    percent = summary[percent_key]
    if not isinstance(count, int) or not isinstance(percent, (int, float)):
        raise ResultValidationError(f"Invalid types for {count_key}/{percent_key}")
    expected = 0.0 if total == 0 else count * 100.0 / total
    if not math.isclose(float(percent), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ResultValidationError(
            f"Stored {percent_key}={percent} disagrees with {count_key}/{total}={expected}"
        )


def summarize_result(path: str | Path) -> dict[str, Any]:
    """Return stored metrics after validating any available count/ratio pairs.

    This function intentionally does not derive new success labels from trajectories.
    The experiment runners remain authoritative for action matching, ordering,
    argument handling, repeat limits, false positives, and irreversible success.
    """
    payload = load_result(path)
    summary = payload["summary"]
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ResultValidationError("Top-level results must be a list")
    total = summary.get(
        "total_attack_sessions",
        summary.get("total_benign_tasks", summary.get("total", len(results))),
    )
    if not isinstance(total, int) or total < 0:
        raise ResultValidationError("Summary total must be a non-negative integer")
    if results and len(results) != total:
        raise ResultValidationError(
            f"Stored total {total} disagrees with {len(results)} result records"
        )
    for count_key, percent_key in (
        ("no_defense_bu_successes", "no_defense_bu_percent"),
        ("ours_bu_successes", "ours_bu_percent"),
        ("no_defense_ua_successes", "no_defense_ua_percent"),
        ("no_defense_asr_successes", "no_defense_asr_percent"),
        ("ours_ua_successes", "ours_ua_percent"),
        ("ours_asr_successes", "ours_asr_percent"),
        ("ua_successes", "ua_percent"),
        ("asr_successes", "asr_percent"),
    ):
        _validate_ratio(summary, count_key, percent_key, total)
    return {"path": str(Path(path)), "total": total, "summary": summary}
