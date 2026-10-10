# Artifact Scope

This artifact provides the code and configuration for one reference LLM setup. It includes the complete ASB auditor training and evaluation workflow and the AgentDojo and InjecAgent test-time evaluation workflows for that configuration. Experiments conducted with other LLMs are outside the scope of this artifact.

No precomputed experiment results or reported metrics are included. Locally generated results are written under the Git-ignored `results/` directory.

Included:

- one `qwen7b` endpoint configuration;
- ASB auditor training data, training code, auditor test data, test-time attack/benign runners, and metric computation;
- AgentDojo attack/benign test-time runners and metric computation, with no training or split-generation code;
- InjecAgent attack/enhanced/benign test-time runners for the full configured zero-shot test split and metric computation, with no training or split-generation code;
- one protected policy and one released reference checkpoint per benchmark, one ASB threshold file, the Occlum workflow, and verification tests.

Excluded:

- every non-reference model directory, mapping, policy, threshold, runner, and configuration;
- all precomputed result files and reported metric outputs;
- all comparison-defense implementations;
- AgentDojo and InjecAgent training, fine-tuning, dataset construction, split generation, and task sampling;
- credentials, caches, logs, private data, agent-model weights, and benchmark datasets not redistributable with this snapshot.

The three released qwen7b auditor checkpoints and their SHA-256 manifest are under `checkpoints/`. The ASB training procedure writes a separate newly generated checkpoint under `benchmarks/asb/training/generated/`.
