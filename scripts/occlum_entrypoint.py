"""Installed as /opt/llm_tee_auditor/entrypoint.py inside the Occlum image."""

import asyncio
import sys

sys.path.insert(0, "/opt/llm_tee_auditor/src")

from llm_tee_agent_artifact.tee.entrypoint import serve


if __name__ == "__main__":
    raise SystemExit(asyncio.run(serve()))
