# Reproducibility

The artifact reproduces one reference configuration only. ASB includes training, checkpoint testing, attack testing, benign testing, and metric validation. AgentDojo and InjecAgent are test-only workflows.

ASB training:

```bash
python -m pip install -r requirements-training.txt
LLM_TEE_DATA_ROOT=/path/to/external/model/root python benchmarks/asb/training/train_auditor.py
LLM_TEE_DATA_ROOT=/path/to/external/model/root python benchmarks/asb/training/test_auditor.py
```

The training script preserves seed 42, 30 epochs, batch size 16, maximum length 128, focal loss, optimizer groups, held-out task split, threshold search, and early stopping. Outputs go to `benchmarks/asb/training/generated/`.

Test commands are listed in the root `README.md`. The model flag is accepted only when it equals `qwen7b`. Frozen results remain under `results/<benchmark>/frozen_*.json`; generated results use a separate ignored prefix.

AgentDojo requires its fixed public test assets. InjecAgent requires its public input/tool registry and evaluates the full configured zero-shot test split. Neither workflow contains training or split construction.
