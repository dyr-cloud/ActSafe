"""Fail-closed host controller for a persistent auditor inside Occlum."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import OCCLUM_INSTANCE_ROOT, PROTECTED_ENTRYPOINT
from .protocol import PROTOCOL_VERSION, ProtocolError, validate_response


class ProtectedAuditorError(RuntimeError):
    """The protected auditor could not provide a valid decision."""


@dataclass(frozen=True)
class AuditDecision:
    decision: str
    reason_code: str
    decision_token: str = ""
    confirmation_fields: tuple[str, ...] = ()

    @property
    def permits_execution(self) -> bool:
        return self.decision == "allow"


class ProtectedAuditorSession:
    """One measured Occlum process containing all mutable audit state."""

    def __init__(self, benchmark: str, variant: str, timeout: float | None = None):
        self.benchmark = benchmark
        self.variant = variant
        self.timeout = timeout or float(os.environ.get("TEE_AUDITOR_TIMEOUT", "120"))
        self._process: asyncio.subprocess.Process | None = None
        self._closed = False

    @staticmethod
    def command() -> tuple[str, ...]:
        executable = os.environ.get("OCCLUM_BIN", "occlum")
        if Path(executable).name not in {"occlum", "occlum.exe"}:
            raise ProtectedAuditorError("OCCLUM_BIN must name the Occlum launcher")
        return executable, "run", "/usr/bin/python3", str(PROTECTED_ENTRYPOINT)

    async def start(self, user_instruction: str, audit_context: dict[str, Any]) -> AuditDecision:
        if self._process is not None or self._closed:
            raise ProtectedAuditorError("protected auditor session cannot be started")
        if not OCCLUM_INSTANCE_ROOT.is_dir():
            raise ProtectedAuditorError(f"Occlum instance is unavailable: {OCCLUM_INSTANCE_ROOT}")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command(),
                cwd=str(OCCLUM_INSTANCE_ROOT),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            decision = await self._exchange({
                "version": PROTOCOL_VERSION,
                "type": "initialize",
                "benchmark": self.benchmark,
                "variant": self.variant,
                "user_instruction": user_instruction,
                "audit_context": audit_context,
            })
        except Exception as exc:
            await self._terminate()
            if isinstance(exc, ProtectedAuditorError):
                raise
            raise ProtectedAuditorError(f"Occlum auditor startup failed: {exc}") from exc
        if decision.decision != "ready":
            await self._terminate()
            raise ProtectedAuditorError(f"Occlum auditor initialization blocked: {decision.reason_code}")
        return decision

    async def audit(self, operation: str, arguments: dict[str, Any], raw_action: str = "", audit_context: dict[str, Any] | None = None) -> AuditDecision:
        decision = await self._exchange({
            "version": PROTOCOL_VERSION,
            "type": "audit",
            "proposed_action": {"operation": operation, "arguments": arguments, "raw_action": raw_action},
            "audit_context": audit_context or {},
        })
        if decision.decision not in {"allow", "block", "confirm"}:
            raise ProtectedAuditorError("protected auditor returned a non-final audit decision")
        return decision

    async def commit(self, decision_token: str) -> None:
        result = await self._exchange({"version": PROTOCOL_VERSION, "type": "commit", "decision_token": decision_token})
        if result.decision != "committed":
            raise ProtectedAuditorError(f"protected commit failed: {result.reason_code}")

    async def close(self) -> None:
        if self._process is not None and self._process.returncode is None:
            try:
                await self._exchange({"version": PROTOCOL_VERSION, "type": "close"})
            except ProtectedAuditorError:
                pass
        await self._terminate()
        self._closed = True

    async def __aenter__(self) -> "ProtectedAuditorSession":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _exchange(self, request: dict[str, Any]) -> AuditDecision:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None or process.returncode is not None:
            raise ProtectedAuditorError("protected auditor is unavailable; action is blocked")
        try:
            process.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"))
            await asyncio.wait_for(process.stdin.drain(), timeout=self.timeout)
            line = await asyncio.wait_for(process.stdout.readline(), timeout=self.timeout)
            if not line:
                stderr = b""
                if process.stderr is not None:
                    stderr = await process.stderr.read()
                raise ProtectedAuditorError("protected auditor exited without a decision: " + stderr.decode("utf-8", "replace")[-500:])
            payload = validate_response(json.loads(line.decode("utf-8")))
        except (asyncio.TimeoutError, json.JSONDecodeError, ProtocolError) as exc:
            await self._terminate()
            raise ProtectedAuditorError(f"invalid or timed-out protected decision; action is blocked: {exc}") from exc
        return AuditDecision(
            payload["decision"], payload["reason_code"], payload.get("decision_token", ""), tuple(payload.get("confirmation_fields", []))
        )

    async def _terminate(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
