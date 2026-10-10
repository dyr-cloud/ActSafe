# Released Benchmarks

This directory contains the released `qwen7b` reference pipeline only.

- `asb/` contains the official evaluation input, complete auditor training and test data, the training and auditor-test programs, and attack and benign evaluation runners.
- `dojo/` contains attack and benign test-time runners. Training data and training code are intentionally outside the release scope.
- `injectagent/` contains attack, enhanced-attack, and benign test-time runners for the full configured zero-shot test split. Training data and training code are intentionally outside the release scope.

Each `run_test.py` writes a fresh `generated_*.json` file under the matching `results/<benchmark>/` directory. The corresponding `evaluate.py` validates locally generated result files. The entire `results/` directory is ignored by version control, and no precomputed metrics are included in this repository.
