"""Lightweight tests for frozen-result validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_tee_agent_artifact.results import ResultValidationError, summarize_result


class ResultSummaryTests(unittest.TestCase):
    """Exercise metric consistency checks without external dependencies."""

    def _write(self, payload: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
        with handle:
            json.dump(payload, handle)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_accepts_consistent_bu_summary(self) -> None:
        path = self._write(
            {
                "summary": {
                    "total_benign_tasks": 4,
                    "ours_bu_successes": 3,
                    "ours_bu_percent": 75.0,
                },
                "results": [{}, {}, {}, {}],
            }
        )
        self.assertEqual(summarize_result(path)["total"], 4)

    def test_rejects_inconsistent_percentage(self) -> None:
        path = self._write(
            {
                "summary": {
                    "total_attack_sessions": 4,
                    "asr_successes": 1,
                    "asr_percent": 50.0,
                },
                "results": [{}, {}, {}, {}],
            }
        )
        with self.assertRaises(ResultValidationError):
            summarize_result(path)


if __name__ == "__main__":
    unittest.main()

