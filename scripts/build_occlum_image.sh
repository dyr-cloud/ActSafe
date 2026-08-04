#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${OCCLUM_IMAGE_ROOT:-/host/image}"
INSTANCE_ROOT="$(dirname -- "${IMAGE_ROOT}")"
command -v occlum >/dev/null 2>&1 || { echo "Occlum is required." >&2; exit 1; }
[[ -f "${INSTANCE_ROOT}/Occlum.json" ]] || { echo "Occlum manifest is missing." >&2; exit 1; }
[[ -f "${IMAGE_ROOT}/opt/llm_tee_auditor/entrypoint.py" ]] || { echo "Run prepare_occlum_image.sh first." >&2; exit 1; }
cd -- "${INSTANCE_ROOT}"
occlum build
echo "Built the Occlum protected auditor image."
