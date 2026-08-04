#!/usr/bin/env python3
"""Run the released method on the immutable ASB test split."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _runner import ARTIFACT_ROOT, execute, load_config, require_files, require_reference_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ARTIFACT_ROOT / "configs/asb/evaluation.json")
    parser.add_argument("--model", default="qwen7b")
    parser.add_argument("--phase", choices=("attack", "benign"), default="attack")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    model_name = require_reference_model(config, args.model)
    runner_name = "test_privacy.py" if args.phase == "attack" else "test_utility.py"
    runner = Path(__file__).resolve().parent / runner_name
    result_dir = ARTIFACT_ROOT / "results/asb"
    result_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ASB_MODEL_NAME"] = model_name
    env["ASB_OFFICIAL_TEST_FILE"] = str(ARTIFACT_ROOT / config["test_file"])
    env["ASB_OURS_RESULT_FILE" if args.phase == "attack" else "ASB_OURS_BU_RESULT_FILE"] = str(
        result_dir / f"generated_{args.phase}.json"
    )
    if not args.dry_run:
        require_files(env, ["ASB_OFFICIAL_TEST_FILE"])
        image_root = Path(env.get("OCCLUM_IMAGE_ROOT", "/host/image"))
        if not (image_root / "opt/llm_tee_auditor/entrypoint.py").is_file():
            raise ValueError("prepared Occlum auditor image is required")
    return execute(runner, env, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
