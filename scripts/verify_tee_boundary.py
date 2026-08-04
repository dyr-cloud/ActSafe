#!/usr/bin/env python3
"""Static release check: benchmark hosts may use only the Occlum client."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "distill_intent", "is_safe_action_dual_stream", "v2_extract_task_reference",
    "V2BindingGuard", "OperationArgumentGuard", "audit_dapa_action",
    "audit_structured_entities",
}
SECURITY_DEFINITIONS = FORBIDDEN | {"FineGrainedAuditor"}


def active_host_runners():
    for relative in (
        "benchmarks/asb/test_privacy.py",
        "benchmarks/asb/test_utility.py",
        "benchmarks/dojo/test_privacy_order7.py",
        "benchmarks/dojo/test_checker_utility7.py",
        "benchmarks/injectagent/test_checker_privacy_base.py",
        "benchmarks/injectagent/test_checker_privacy_en.py",
        "benchmarks/injectagent/test_checker_utility.py",
    ):
        yield ROOT / relative


def main() -> int:
    failures = []
    for path in active_host_runners():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in SECURITY_DEFINITIONS:
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                name = child.func.id if isinstance(child.func, ast.Name) else ""
                if name in FORBIDDEN:
                    failures.append(f"{path.relative_to(ROOT)}:{child.lineno}:{name}")
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                imported = FORBIDDEN & {alias.name for alias in node.names}
                if imported:
                    failures.append(f"{path.relative_to(ROOT)}:{node.lineno}:imports {sorted(imported)}")
    if failures:
        raise SystemExit("Direct host auditor calls remain:\n" + "\n".join(failures))
    client = (ROOT / "src/llm_tee_agent_artifact/tee/client.py").read_text(encoding="utf-8")
    required = ("occlum", '"run"', "PROTECTED_ENTRYPOINT")
    if not all(value in client for value in required):
        raise SystemExit("Occlum-only client invariant is not satisfied")
    print("TEE boundary static audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
