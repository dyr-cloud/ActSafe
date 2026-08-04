#!/usr/bin/env python3
"""Run the released method on the full configured InjecAgent test set."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import ARTIFACT_ROOT, execute, load_config, require_files, require_reference_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ARTIFACT_ROOT / "configs/injectagent/evaluation.json")
    parser.add_argument("--model", default="qwen7b")
    parser.add_argument("--phase", choices=("attack", "enhanced", "benign"), default="attack")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    env = os.environ.copy()
    env["MODEL_NAME"] = require_reference_model(config, args.model)
    runners = {
        "attack": "test_checker_privacy_base.py",
        "enhanced": "test_checker_privacy_en.py",
        "benign": "test_checker_utility.py",
    }
    runner = Path(__file__).resolve().parent / runners[args.phase]
    result_dir = ARTIFACT_ROOT / "results/injectagent"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_variable = "INJECAGENT_UTILITY_RESULT_FILE" if args.phase == "benign" else "INJECAGENT_OURS_RESULT_FILE"
    env[result_variable] = str(result_dir / f"generated_{args.phase}.json")
    required = ["INJECAGENT_INPUT_FILE", "INJECAGENT_TOOLS_FILE"]
    if not args.dry_run:
        require_files(env, required)
        image_root = Path(env.get("OCCLUM_IMAGE_ROOT", "/host/image"))
        if not (image_root / "opt/llm_tee_auditor/entrypoint.py").is_file():
            raise ValueError("prepared Occlum auditor image is required")
    return execute(runner, env, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
