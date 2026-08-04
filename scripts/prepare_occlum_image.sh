#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE_ROOT="${OCCLUM_IMAGE_ROOT:-/host/image}"
INSTANCE_ROOT="$(dirname -- "${IMAGE_ROOT}")"
RUNTIME_ROOT="${IMAGE_ROOT}/opt/llm_tee_auditor"
MODEL_SOURCE="${AUDITOR_MODEL_SOURCE:?Set AUDITOR_MODEL_SOURCE to the redistributable TinyBERT model directory}"
SPACY_SOURCE="${SPACY_MODEL_SOURCE:?Set SPACY_MODEL_SOURCE to the en_core_web_sm package directory}"
ASB_WEIGHTS="${ASB_AUDITOR_WEIGHTS:-${ARTIFACT_ROOT}/checkpoints/asb/checker_tinybert_17b.pth}"
DOJO_WEIGHTS="${DOJO_AUDITOR_WEIGHTS:-${ARTIFACT_ROOT}/checkpoints/dojo/checker_tinybert_17b.pth}"
INJECT_WEIGHTS="${INJECAGENT_AUDITOR_WEIGHTS:-${ARTIFACT_ROOT}/checkpoints/injectagent/checker_tinybert_injecagent.pth}"

command -v occlum >/dev/null 2>&1 || { echo "Occlum is required." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required." >&2; exit 1; }
python3 "${ARTIFACT_ROOT}/scripts/verify_checkpoints.py" --root "${ARTIFACT_ROOT}"
[[ -d "${IMAGE_ROOT}" && -f "${INSTANCE_ROOT}/Occlum.json" ]] || {
  echo "Initialize the Occlum instance at ${INSTANCE_ROOT} before preparation." >&2
  exit 1
}
[[ -x "${IMAGE_ROOT}/usr/bin/python3" ]] || {
  echo "The Occlum image must be initialized from a Python base image." >&2
  exit 1
}
for required in "${MODEL_SOURCE}" "${SPACY_SOURCE}"; do
  [[ -d "${required}" ]] || { echo "Required directory is missing: ${required}" >&2; exit 1; }
done
for required in "${ASB_WEIGHTS}" "${DOJO_WEIGHTS}" "${INJECT_WEIGHTS}"; do
  [[ -s "${required}" ]] || { echo "Required checkpoint file is missing or empty: ${required}" >&2; exit 1; }
done

install -d "${RUNTIME_ROOT}/src" "${RUNTIME_ROOT}/python" "${RUNTIME_ROOT}/config"
install -d "${RUNTIME_ROOT}/models/llm_tee_data/models--prajjwal1--bert-tiny/snapshots"
install -d "${RUNTIME_ROOT}/models/weights/asb" "${RUNTIME_ROOT}/models/weights/dojo" "${RUNTIME_ROOT}/models/weights/injectagent"
cp -a "${ARTIFACT_ROOT}/src/llm_tee_agent_artifact" "${RUNTIME_ROOT}/src/"
install -m 0644 "${ARTIFACT_ROOT}/scripts/occlum_entrypoint.py" "${RUNTIME_ROOT}/entrypoint.py"
install -m 0644 "${ARTIFACT_ROOT}/configs/occlum/empty.json" "${RUNTIME_ROOT}/config/empty.json"
cp -a "${MODEL_SOURCE}" "${RUNTIME_ROOT}/models/llm_tee_data/models--prajjwal1--bert-tiny/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
cp -a "${SPACY_SOURCE}" "${RUNTIME_ROOT}/python/en_core_web_sm"

install -d "${RUNTIME_ROOT}/policies/asb" "${RUNTIME_ROOT}/policies/dojo" "${RUNTIME_ROOT}/policies/injectagent"
install -m 0644 "${ARTIFACT_ROOT}/src/llm_tee_agent_artifact/tee/policies/asb/policy.py" "${RUNTIME_ROOT}/policies/asb/policy.py"
install -m 0644 "${ARTIFACT_ROOT}/src/llm_tee_agent_artifact/tee/policies/asb/checker_tinybert_17b.meta.json" "${RUNTIME_ROOT}/policies/asb/checker_tinybert_17b.meta.json"
install -m 0644 "${ARTIFACT_ROOT}/src/llm_tee_agent_artifact/tee/policies/dojo/policy.py" "${RUNTIME_ROOT}/policies/dojo/policy.py"
install -m 0644 "${ARTIFACT_ROOT}/src/llm_tee_agent_artifact/tee/policies/injectagent/policy.py" "${RUNTIME_ROOT}/policies/injectagent/policy.py"
install -m 0600 "${ASB_WEIGHTS}" "${RUNTIME_ROOT}/models/weights/asb/checker_tinybert_17b.pth"
install -m 0600 "${DOJO_WEIGHTS}" "${RUNTIME_ROOT}/models/weights/dojo/checker_tinybert_17b.pth"
install -m 0600 "${INJECT_WEIGHTS}" "${RUNTIME_ROOT}/models/weights/injectagent/checker_tinybert_injecagent.pth"

python3 -m pip install --no-compile --target "${RUNTIME_ROOT}/python" -r "${ARTIFACT_ROOT}/requirements-tee.txt"
install -m 0644 "${ARTIFACT_ROOT}/configs/occlum/Occlum.json" "${INSTANCE_ROOT}/Occlum.json"
find "${RUNTIME_ROOT}" -type f ! -name tee-image-manifest.sha256 -exec sha256sum {} + \
  | sort > "${RUNTIME_ROOT}/tee-image-manifest.sha256"
echo "Prepared protected auditor files under ${IMAGE_ROOT}."
