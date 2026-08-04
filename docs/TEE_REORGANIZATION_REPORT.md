# Reference Occlum Deployment

The protected package is `src/llm_tee_agent_artifact/tee`. Exactly three policy files are released:

- `tee/policies/asb/policy.py`, plus its threshold metadata;
- `tee/policies/dojo/policy.py`;
- `tee/policies/injectagent/policy.py`.

`prepare_occlum_image.sh` verifies the released checkpoint manifest and accepts exactly one ASB, one AgentDojo, and one InjecAgent checkpoint input. It installs each file under `models/weights/<benchmark>/` without changing its original filename. It contains no model loop, menu, or non-reference destination.

The host launches:

```text
occlum run /usr/bin/python3 /opt/llm_tee_auditor/entrypoint.py
```

The protected process constructs the trusted reference, performs decomposition, operation/argument/count checks, classifies confirmation requirements, retains mutable state, and returns a minimal decision. Startup, protocol, timeout, and commit failures remain fail-closed. Agent reasoning and approved simulated execution remain outside Occlum.
