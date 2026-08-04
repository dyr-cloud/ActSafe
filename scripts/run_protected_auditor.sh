#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${OCCLUM_IMAGE_ROOT:-/host/image}"
INSTANCE_ROOT="$(dirname -- "${IMAGE_ROOT}")"
command -v occlum >/dev/null 2>&1 || { echo "Occlum is required." >&2; exit 1; }
[[ -f "${IMAGE_ROOT}/opt/llm_tee_auditor/entrypoint.py" ]] || { echo "Protected entry point is missing." >&2; exit 1; }
cd -- "${INSTANCE_ROOT}"
exec occlum run /usr/bin/python3 /opt/llm_tee_auditor/entrypoint.py
