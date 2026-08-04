"""Single source of truth for the host-visible Occlum image layout."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


OCCLUM_IMAGE_ROOT = Path(os.environ.get("OCCLUM_IMAGE_ROOT", "/host/image"))
OCCLUM_INSTANCE_ROOT = OCCLUM_IMAGE_ROOT.parent
PROTECTED_INSTALL_ROOT = OCCLUM_IMAGE_ROOT / "opt" / "llm_tee_auditor"
PROTECTED_ENTRYPOINT = PurePosixPath("/opt/llm_tee_auditor/entrypoint.py")
PROTECTED_CONFIG_ROOT = PurePosixPath("/opt/llm_tee_auditor/config")
PROTECTED_MODEL_ROOT = PurePosixPath("/opt/llm_tee_auditor/models")
