# ActSafe

This repository provides the code and configuration for one reference LLM setup. It includes the complete ASB auditor training and evaluation workflow and the AgentDojo and InjecAgent test-time evaluation workflows for that configuration. Experiments conducted with other LLMs are outside the scope of this repository.

Precomputed experiment results and reported metrics are not included. Results produced by local runs are written under `results/`, which is ignored by Git.

The selected reference is `qwen7b`, using the endpoint model name `qwen2.5-7b`. The `--model` argument is retained for command compatibility but rejects every value except `qwen7b`.

## Install

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

## ASB auditor training

The released training and auditor test data, random seed 42, optimizer settings, threshold selection procedure, and early stopping settings are provided under `benchmarks/asb/training`. Training writes only to its ignored `generated/` directory.

```bash
python -m pip install -r requirements-training.txt
export LLM_TEE_DATA_ROOT=/path/to/external/model/root

python benchmarks/asb/training/train_auditor.py
python benchmarks/asb/training/test_auditor.py
```

The training workflow produces:

```text
benchmarks/asb/training/generated/checker_tinybert_17b.pth
```

The reference checkpoints required by the ASB, AgentDojo, and InjecAgent evaluation workflows are included in this artifact. The ASB training workflow can be used to regenerate `checker_tinybert_17b.pth`.

A newly trained checkpoint may not be bit for bit identical to the included reference checkpoint because of differences in hardware, dependency versions, and nondeterministic numerical operations.

The auditor test uses the newly generated ASB checkpoint by default. To test the included reference checkpoint instead, set:

```bash
export ASB_AUDITOR_WEIGHTS="$PWD/checkpoints/asb/checker_tinybert_17b.pth"
python benchmarks/asb/training/test_auditor.py
```

## Prepare the protected auditor

Run on an SGX capable Linux host with an Occlum Python instance at `/host`.

```bash
export OCCLUM_IMAGE_ROOT=/host/image
export AUDITOR_MODEL_SOURCE=/path/to/bert-tiny-snapshot
export SPACY_MODEL_SOURCE=/path/to/en_core_web_sm
export ASB_AUDITOR_WEIGHTS="$PWD/checkpoints/asb/checker_tinybert_17b.pth"
export DOJO_AUDITOR_WEIGHTS="$PWD/checkpoints/dojo/checker_tinybert_17b.pth"
export INJECAGENT_AUDITOR_WEIGHTS="$PWD/checkpoints/injectagent/checker_tinybert_injecagent.pth"
bash scripts/prepare_occlum_image.sh
bash scripts/build_occlum_image.sh
bash scripts/verify_tee_execution.sh
```

Only the three reference policies, ASB threshold metadata, three included reference checkpoints, protected package, and required runtime assets are copied into `/host/image`.

## Test and evaluate

```bash
python benchmarks/asb/run_test.py --model qwen7b --phase attack
python benchmarks/asb/run_test.py --model qwen7b --phase benign
python benchmarks/asb/evaluate.py

python benchmarks/dojo/run_test.py --model qwen7b --phase attack
python benchmarks/dojo/run_test.py --model qwen7b --phase benign
python benchmarks/dojo/evaluate.py

python benchmarks/injectagent/run_test.py --model qwen7b --phase attack
python benchmarks/injectagent/run_test.py --model qwen7b --phase enhanced
python benchmarks/injectagent/run_test.py --model qwen7b --phase benign
python benchmarks/injectagent/evaluate.py
```

Each run writes a `generated_*.json` file under `results/<benchmark>/`. These locally generated outputs are ignored by Git and should not be committed.

## Verify

```bash
python -m unittest discover -s tests -v
python scripts/verify_tee_boundary.py
python scripts/verify_checkpoints.py
python scripts/release_audit.py --root .
```

The complete auditor remains inside Occlum. Agent reasoning, external benchmark data, orchestration, result collection, and approved simulated tool execution remain outside. There is no unprotected fallback.
