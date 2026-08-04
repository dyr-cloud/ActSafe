"""Newline-delimited JSON entry point executed by ``occlum run``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys

from .protocol import PROTOCOL_VERSION, ProtocolError, validate_request
from .service import ProtectedAuditEngine


def response(decision: str, reason: str) -> dict[str, object]:
    return {"version": PROTOCOL_VERSION, "decision": decision, "reason_code": reason}


async def serve() -> int:
    if os.environ.get("LLM_TEE_INSIDE_OCCLUM") != "1":
        print(json.dumps(response("block", "NOT_INSIDE_OCCLUM")), flush=True)
        return 78
    protocol_stdout = sys.stdout
    engine = ProtectedAuditEngine()
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return 0
        try:
            request = validate_request(json.loads(line))
            with contextlib.redirect_stdout(sys.stderr):
                if request["type"] == "initialize":
                    result = await engine.initialize(request)
                elif request["type"] == "audit":
                    result = engine.audit(request)
                elif request["type"] == "commit":
                    result = engine.commit(request["decision_token"])
                elif request["type"] == "health":
                    result = response("ready", "OCCLUM_ENTRYPOINT_ACTIVE")
                else:
                    result = response("closed", "SESSION_CLOSED")
        except (json.JSONDecodeError, ProtocolError):
            result = response("block", "MALFORMED_REQUEST")
        except Exception as exc:
            print(f"Protected auditor error: {type(exc).__name__}: {exc}", file=sys.stderr)
            result = response("block", "PROTECTED_AUDITOR_FAILURE")
        protocol_stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
        protocol_stdout.flush()
        if result["decision"] == "closed":
            return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(serve()))
