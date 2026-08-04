import asyncio
import json
import os
import unittest
from unittest.mock import patch

from llm_tee_agent_artifact.tee.client import ProtectedAuditorError, ProtectedAuditorSession
from llm_tee_agent_artifact.tee.protocol import ProtocolError, validate_request, validate_response


class ProtocolTests(unittest.TestCase):
    def test_rejects_missing_action_arguments(self):
        with self.assertRaises(ProtocolError):
            validate_request({
                "version": 1,
                "type": "audit",
                "proposed_action": {"operation": "send_email"},
                "audit_context": {},
            })

    def test_rejects_unknown_boundary_fields(self):
        with self.assertRaises(ProtocolError):
            validate_request({"version": 1, "type": "health", "approve": True})

    def test_response_has_closed_decision_vocabulary(self):
        with self.assertRaises(ProtocolError):
            validate_response({"version": 1, "decision": "maybe", "reason_code": "X"})


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def test_command_always_uses_occlum(self):
        with patch.dict(os.environ, {"OCCLUM_BIN": "occlum"}):
            self.assertEqual(
                ProtectedAuditorSession.command(),
                ("occlum", "run", "/usr/bin/python3", "/opt/llm_tee_auditor/entrypoint.py"),
            )

    def test_rejects_non_occlum_launcher(self):
        with patch.dict(os.environ, {"OCCLUM_BIN": "python3"}):
            with self.assertRaises(ProtectedAuditorError):
                ProtectedAuditorSession.command()

    async def test_unstarted_session_fails_closed(self):
        session = ProtectedAuditorSession("asb", "qwen7b")
        with self.assertRaisesRegex(ProtectedAuditorError, "action is blocked"):
            await session.audit("send_email", {}, "send_email()")


if __name__ == "__main__":
    unittest.main()
