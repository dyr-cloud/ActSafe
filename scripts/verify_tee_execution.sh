#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUEST='{"version":1,"type":"health"}'
RESPONSE="$(printf '%s\n' "${REQUEST}" | "${SCRIPT_DIR}/run_protected_auditor.sh")"
python3 -c 'import json,sys; value=json.loads(sys.argv[1]); assert value == {"version":1,"decision":"ready","reason_code":"OCCLUM_ENTRYPOINT_ACTIVE"}' "${RESPONSE}"
python3 "${SCRIPT_DIR}/verify_tee_boundary.py"
echo "Verified Occlum entry point and fail-closed host boundary."
