"""Protected policy adapter. This module is executed only by the Occlum entry point."""

from __future__ import annotations

import importlib.util
import json
import os
import secrets
from pathlib import Path
from types import ModuleType
from typing import Any


RUNTIME_ROOT = Path(os.environ.get("TEE_RUNTIME_ROOT", "/opt/llm_tee_auditor"))
POLICY_ROOT = RUNTIME_ROOT / "policies"
MODEL_ROOT = RUNTIME_ROOT / "models"
CONFIG_ROOT = RUNTIME_ROOT / "config"
POLICY_FILES = {"asb": "policy.py", "dojo": "policy.py", "injectagent": "policy.py"}
CHECKPOINT_FILES = {
    "asb": "checker_tinybert_17b.pth",
    "dojo": "checker_tinybert_17b.pth",
    "injectagent": "checker_tinybert_injecagent.pth",
}
REFERENCE_VARIANT = "qwen7b"


class ProtectedPolicyError(RuntimeError):
    pass


def configure_policy_environment(benchmark: str, variant: str) -> None:
    model_data = MODEL_ROOT / "llm_tee_data"
    weights = MODEL_ROOT / "weights" / benchmark / CHECKPOINT_FILES[benchmark]
    os.environ["LLM_TEE_INSIDE_OCCLUM"] = "1"
    os.environ["LLM_TEE_DATA_ROOT"] = str(model_data)
    if benchmark == "asb":
        os.environ["ASB_OFFICIAL_TEST_FILE"] = str(CONFIG_ROOT / "empty.json")
        os.environ["ASB_CHECKER_WEIGHT_PATH"] = str(weights)
    elif benchmark == "dojo":
        os.environ["DOJO_TOOLS_FILE"] = str(CONFIG_ROOT / "empty.json")
        os.environ["DOJO_ENVIRONMENTS_FILE"] = str(CONFIG_ROOT / "empty.json")
        os.environ["DOJO_INJECTION_VECTORS_FILE"] = str(CONFIG_ROOT / "empty.json")
        os.environ["DOJO_CHECKER_WEIGHT_PATH"] = str(weights)
    else:
        os.environ["INJECAGENT_INPUT_FILE"] = str(CONFIG_ROOT / "empty.json")
        os.environ["INJECAGENT_TOOLS_FILE"] = str(CONFIG_ROOT / "empty.json")
        os.environ["INJECAGENT_AUDITOR_MODEL"] = str(
            model_data / "models--prajjwal1--bert-tiny" / "snapshots" /
            "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
        )
        os.environ["INJECAGENT_AUDITOR_WEIGHTS"] = str(weights)


def load_policy(benchmark: str, variant: str) -> ModuleType:
    if variant != REFERENCE_VARIANT:
        raise ProtectedPolicyError("only the released qwen7b reference policy is available")
    configure_policy_environment(benchmark, variant)
    path = POLICY_ROOT / benchmark / POLICY_FILES[benchmark]
    if not path.is_file():
        raise ProtectedPolicyError(f"protected policy is missing: {path}")
    spec = importlib.util.spec_from_file_location(f"protected_{benchmark}_{variant}", path)
    if spec is None or spec.loader is None:
        raise ProtectedPolicyError("protected policy cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtectedAuditEngine:
    """Owns trusted references and execution counters for one request stream."""

    def __init__(self) -> None:
        self.benchmark = ""
        self.variant = ""
        self.user_instruction = ""
        self.module: ModuleType | None = None
        self.distilled: Any = None
        self.guard: Any = None
        self.pending: dict[str, tuple[Any, str, str]] = {}

    async def initialize(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.module is not None:
            raise ProtectedPolicyError("session is already initialized")
        self.benchmark = request["benchmark"]
        self.variant = request["variant"]
        self.user_instruction = request["user_instruction"]
        context = request["audit_context"]
        module = self.module = load_policy(self.benchmark, self.variant)
        client = module.AsyncOpenAI(api_key=module.API_KEY, base_url=module.BASE_URL)

        if self.benchmark == "asb":
            tools = context.get("trusted_tools")
            if not isinstance(tools, list) or not tools:
                raise ProtectedPolicyError("ASB trusted_tools are required")
            self.distilled = await module.distill_intent(client, self.user_instruction)
            reference = await module.v2_extract_task_reference(client, self.user_instruction, tools)
            if not reference.get("bindings"):
                raise ProtectedPolicyError("ASB trusted reference extraction failed")
            self.guard = module.V2BindingGuard(reference)
        elif self.benchmark == "dojo":
            tools_json = context.get("available_tools_json")
            menu = context.get("available_tools_menu")
            model_name = context.get("model_name", module.MODEL_NAME)
            if not isinstance(tools_json, str) or not tools_json or not isinstance(menu, str):
                raise ProtectedPolicyError("Dojo tool schema and menu are required")
            self.distilled = await module.distill_intent(client, self.user_instruction, menu, model_name)
            reference = await module.extract_operation_argument_bindings(client, self.user_instruction, tools_json, model_name)
            self.guard = module.OperationArgumentGuard(reference)
        else:
            tools = context.get("available_tools")
            if not isinstance(tools, list) or not tools:
                raise ProtectedPolicyError("InjecAgent available_tools are required")
            self.distilled = await module.distill_intent(client, self.user_instruction)
            reference = await module.extract_operation_argument_bindings(client, self.user_instruction, tools)
            self.guard = module.OperationArgumentGuard(reference)
        return self._response("ready", "SESSION_INITIALIZED")

    def audit(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.module is None or self.guard is None:
            raise ProtectedPolicyError("session is not initialized")
        module = self.module
        action = request["proposed_action"]
        operation = action["operation"]
        raw_action = action.get("raw_action") or self._render_action(operation, action["arguments"])
        context = request["audit_context"]
        binding_id = None

        if self.benchmark == "asb":
            allowed, _, _, _ = module.is_safe_action_dual_stream(self.distilled, operation)
            if allowed:
                allowed, _, binding_id = self.guard.check(operation, raw_action)
            confirmation_fields = module.v2_human_fields(raw_action) if allowed else []
        elif self.benchmark == "dojo":
            clean_environment = context.get("clean_environment_text", "")
            if not isinstance(clean_environment, str):
                raise ProtectedPolicyError("clean_environment_text must be a string")
            allowed, _, _, _ = module.audit_dapa_action(
                self.distilled, self.user_instruction, clean_environment, operation, raw_action
            )
            if allowed:
                allowed, _, binding_id, _ = self.guard.check(
                    operation, raw_action, module.auditor.is_write_action(operation)
                )
            confirmation_fields = module.auditor.get_human_reverification_fields(raw_action) if allowed else []
        else:
            allowed, _, _, _ = module.is_safe_action_dual_stream(self.distilled, operation)
            if allowed and module.is_write_action(operation):
                allowed, _ = module.audit_structured_entities(self.user_instruction, operation, raw_action)
            if allowed:
                allowed, _, binding_id, _ = self.guard.check(
                    operation, raw_action, module.is_write_action(operation)
                )
            confirmation_fields = module.has_nonempty_free_text(raw_action) if allowed else []

        if not allowed:
            return self._response("block", "POLICY_BLOCK")
        token = secrets.token_urlsafe(24)
        self.pending = {token: (binding_id, operation, raw_action)}
        decision = "confirm" if confirmation_fields else "allow"
        return self._response(decision, "CONFIRMATION_REQUIRED" if confirmation_fields else "CONSISTENT", token, confirmation_fields)

    def commit(self, token: str) -> dict[str, Any]:
        pending = self.pending.pop(token, None)
        if pending is None:
            raise ProtectedPolicyError("unknown or expired decision token")
        binding_id, operation, raw_action = pending
        self.guard.commit(binding_id, operation, raw_action)
        return self._response("committed", "STATE_COMMITTED")

    @staticmethod
    def _render_action(operation: str, arguments: dict[str, Any]) -> str:
        rendered = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
        return f"{operation}({rendered})"

    @staticmethod
    def _response(decision: str, reason: str, token: str = "", fields: list[str] | None = None) -> dict[str, Any]:
        response: dict[str, Any] = {"version": 1, "decision": decision, "reason_code": reason}
        if token:
            response["decision_token"] = token
        if fields:
            response["confirmation_fields"] = sorted(set(fields))
        return response
