# Security and Privacy

All proposed defended actions cross a strict newline-delimited JSON boundary before simulated tool execution. Initialization accepts the benchmark, the fixed `qwen7b` reference identifier, trusted user instruction, and the minimum tool schema context. Audit messages accept an operation, argument object, raw action, and benchmark-specific untrusted context. Unknown fields, missing values, invalid types, malformed JSON, unsupported decisions, timeouts, process exits, missing images, and unavailable Occlum fail closed.

The response vocabulary is `allow`, `block`, or `confirm`, with a stable reason code. An allowed or confirmed proposal includes a random opaque one-use token. Only the protected process can commit that token and advance execution counters. A confirmation result is never reinterpreted as approval; the benchmark's existing simulated `yes` event is recorded before the token is committed. Blocked proposals are not executed.

The host command is fixed to:

```text
occlum run /usr/bin/python3 /opt/llm_tee_auditor/entrypoint.py
```

`OCCLUM_BIN` may select the installed Occlum binary path, but its basename must remain `occlum`. There is no direct-Python mode and no host fallback. `LLM_TEE_INSIDE_OCCLUM=1` is supplied by the measured Occlum manifest, and the entry point refuses to serve without it.

The prior arrangement defined and invoked decomposition, TinyBERT classification, fine-grained argument checks, and binding guards in each host benchmark module. Active host call sites now instantiate only `ProtectedAuditorSession`; the protected policy copies, models, tokenizer, prompts, reference, pending decision, and execution counters reside under `/host/image/opt/llm_tee_auditor` and execute through the protected entry point.

This defensive artifact contains adversarial prompts. Use only authorized endpoints and test data. It contains no credentials, private data, production account access, agent-model weights, or live tool integrations. It includes only the three released qwen7b auditor checkpoints recorded in `checkpoints/MANIFEST.json`. The rights-reserved `LICENSE` must be replaced before public redistribution.
