#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
python benchmarks/asb/evaluate.py
python benchmarks/dojo/evaluate.py
python benchmarks/injectagent/evaluate.py
