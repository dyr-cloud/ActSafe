"""Strict newline-delimited JSON protocol for the TEE boundary."""

from __future__ import annotations

from typing import Any


PROTOCOL_VERSION = 1
BENCHMARKS = frozenset({"asb", "dojo", "injectagent"})
DECISIONS = frozenset({"ready", "allow", "block", "confirm", "committed", "closed"})


class ProtocolError(ValueError):
    """A boundary message is malformed or contains unsupported fields."""


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} must be a JSON object")
    return value


def _exact_keys(value: dict[str, Any], required: set[str], optional: set[str] = set()) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        raise ProtocolError(f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ProtocolError(f"unknown fields: {', '.join(sorted(extra))}")


def validate_request(value: Any) -> dict[str, Any]:
    request = _object(value, "request")
    _exact_keys(request, {"version", "type"}, {"benchmark", "variant", "user_instruction", "audit_context", "proposed_action", "decision_token"})
    if request["version"] != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")
    request_type = request["type"]
    if request_type == "initialize":
        _exact_keys(request, {"version", "type", "benchmark", "variant", "user_instruction", "audit_context"})
        if request["benchmark"] not in BENCHMARKS:
            raise ProtocolError("unsupported benchmark")
        if not isinstance(request["variant"], str) or not request["variant"].strip():
            raise ProtocolError("variant must be a non-empty string")
        if not isinstance(request["user_instruction"], str) or not request["user_instruction"].strip():
            raise ProtocolError("user_instruction must be a non-empty string")
        _object(request["audit_context"], "audit_context")
    elif request_type == "audit":
        _exact_keys(request, {"version", "type", "proposed_action", "audit_context"})
        action = _object(request["proposed_action"], "proposed_action")
        _exact_keys(action, {"operation", "arguments"}, {"raw_action"})
        if not isinstance(action["operation"], str) or not action["operation"].strip():
            raise ProtocolError("operation must be a non-empty string")
        _object(action["arguments"], "proposed_action.arguments")
        if "raw_action" in action and not isinstance(action["raw_action"], str):
            raise ProtocolError("raw_action must be a string")
        _object(request["audit_context"], "audit_context")
    elif request_type == "commit":
        _exact_keys(request, {"version", "type", "decision_token"})
        if not isinstance(request["decision_token"], str) or not request["decision_token"]:
            raise ProtocolError("decision_token must be a non-empty string")
    elif request_type in {"health", "close"}:
        _exact_keys(request, {"version", "type"})
    else:
        raise ProtocolError("unsupported request type")
    return request


def validate_response(value: Any) -> dict[str, Any]:
    response = _object(value, "response")
    _exact_keys(response, {"version", "decision", "reason_code"}, {"decision_token", "confirmation_fields"})
    if response["version"] != PROTOCOL_VERSION:
        raise ProtocolError("unsupported response version")
    if response["decision"] not in DECISIONS:
        raise ProtocolError("unsupported decision")
    if not isinstance(response["reason_code"], str) or not response["reason_code"]:
        raise ProtocolError("reason_code must be a non-empty string")
    if "decision_token" in response and not isinstance(response["decision_token"], str):
        raise ProtocolError("decision_token must be a string")
    fields = response.get("confirmation_fields", [])
    if not isinstance(fields, list) or any(not isinstance(item, str) for item in fields):
        raise ProtocolError("confirmation_fields must be a string list")
    return response
