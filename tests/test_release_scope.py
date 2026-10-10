"""Structural tests for the narrowed public release scope."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseScopeTests(unittest.TestCase):
    def test_injectagent_uses_full_configured_test_set(self) -> None:
        config = json.loads((ROOT / "configs/injectagent/evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(config["scope"], "full configured test set using the fixed zero-shot intent split")
        self.assertNotIn("selected_task_ids", config)
        self.assertNotIn("selected_task_intents", config)

    def test_reference_checkpoint_paths(self) -> None:
        expected = {
            "asb": "checkpoints/asb/checker_tinybert_17b.pth",
            "dojo": "checkpoints/dojo/checker_tinybert_17b.pth",
            "injectagent": "checkpoints/injectagent/checker_tinybert_injecagent.pth",
        }
        for benchmark, relative in expected.items():
            config = json.loads((ROOT / f"configs/{benchmark}/evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(config["checkpoint"], relative)
            self.assertTrue((ROOT / relative).is_file())

    def test_documented_checkpoint_exports_match_preparation_defaults(self) -> None:
        expected = {
            "ASB_AUDITOR_WEIGHTS": "checkpoints/asb/checker_tinybert_17b.pth",
            "DOJO_AUDITOR_WEIGHTS": "checkpoints/dojo/checker_tinybert_17b.pth",
            "INJECAGENT_AUDITOR_WEIGHTS": "checkpoints/injectagent/checker_tinybert_injecagent.pth",
        }
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        environment = (ROOT / ".env.example").read_text(encoding="utf-8")
        preparation = (ROOT / "scripts/prepare_occlum_image.sh").read_text(encoding="utf-8")
        for variable, relative in expected.items():
            self.assertIn(f'export {variable}="$PWD/{relative}"', readme)
            self.assertIn(f'{variable}="$PWD/{relative}"', environment)
            self.assertIn(f'${{ARTIFACT_ROOT}}/{relative}', preparation)

    def test_no_excluded_directories(self) -> None:
        excluded = re.compile(r"^(?:fsecure|promptarmor|melon|multurn|incon|ace)$", re.I)
        offenders = [path for path in ROOT.rglob("*") if path.is_dir() and excluded.match(path.name)]
        self.assertEqual(offenders, [])

    def test_asb_training_and_test_data_are_scoped(self) -> None:
        data_files = sorted(path.name for path in (ROOT / "benchmarks/asb").glob("*.json"))
        self.assertEqual(data_files, ["asb_official_test_split.json"])
        training_data = sorted(path.name for path in (ROOT / "benchmarks/asb/training").glob("*.json"))
        self.assertEqual(training_data, ["asb_auditor_test.json", "asb_train.json"])

    def test_training_exists_only_for_asb(self) -> None:
        training_scripts = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*train*.py"))
        self.assertEqual(training_scripts, ["benchmarks/asb/training/train_auditor.py"])
        self.assertFalse((ROOT / "benchmarks/dojo/training").exists())
        self.assertFalse((ROOT / "benchmarks/injectagent/training").exists())

    def test_only_reference_model_is_configured(self) -> None:
        for benchmark in ("asb", "dojo", "injectagent"):
            config = json.loads((ROOT / f"configs/{benchmark}/evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(config["reference_model"], {"id": "qwen7b", "endpoint_name": "qwen2.5-7b"})
            self.assertNotIn("models", config)

    def test_no_model_specific_subdirectories(self) -> None:
        for relative in ("benchmarks/asb", "benchmarks/dojo", "benchmarks/injectagent"):
            subdirectories = [path.name for path in (ROOT / relative).iterdir() if path.is_dir() and path.name != "training"]
            self.assertEqual(subdirectories, [])

    def test_precomputed_results_are_not_released(self) -> None:
        self.assertEqual(list(ROOT.rglob("frozen_*.json")), [])
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/results/", gitignore)


if __name__ == "__main__":
    unittest.main()
