# Public Benchmark Data

Raw third-party benchmark files must remain unchanged. The only released training material is the existing ASB reference-auditor training and auditor-test data. No data-construction utility is included.

## ASB

Official source: <https://github.com/agiresearch/ASB>

The required immutable pipeline test input is redistributed as `benchmarks/asb/asb_official_test_split.json` (1,063,829 bytes, SHA-256 `1525bcb1e051a42239269e808200f34541f49ed8012ee5d965604d8f3fde5a4a`). The existing reference-auditor data are released unchanged as `benchmarks/asb/training/asb_train.json` and `benchmarks/asb/training/asb_auditor_test.json`. No split-generation code is included.

## Dojo

Official public source: <https://github.com/ethz-spylab/agentdojo>

This is a test/evaluation-only release for AgentDojo. The artifact does not redistribute the prepared evaluation snapshot. Export these paths to unchanged files supplied by the public benchmark/test environment:

| Variable | Expected file | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `DOJO_TEST_FILE` | `dojo_test.json` | 109,428 | `6a7472b76f8981366ce667e4ea646335c329ad73f65dee407507ade0ed153ac4` |
| `DOJO_TOOLS_FILE` | `dojo_all_tools.json` | 83,347 | `421ea134bb68b5a50ec9898d18d5fe70b3a87a62de346ad526b6f03537e15639` |
| `DOJO_ENVIRONMENTS_FILE` | `dojo_all_environments.json` | 112,866 | `4d54571f7f57b71fae496e103b737de45a56866449bc931ca6e56422d1a5afdf` |
| `DOJO_INJECTION_VECTORS_FILE` | `dojo_all_injection_vectors.json` | 7,622 | `b6bed2760d76ffc2a6bcd8ffcd39566942d3c0fc1bb64147746b66f829c0816f` |

These hashes identify the already prepared paper test snapshot. Do not regenerate a split. If the official distribution does not provide that exact snapshot, obtain it from the artifact authors after redistribution review.

## InjectAgent

Official public source: <https://github.com/uiuc-kang-lab/InjecAgent>

This is a test/evaluation-only release for InjecAgent. The official repository documents 1,054 attack cases across 17 user tools and 62 attacker tools. The paper environment used an immutable normalized evaluation input containing those attack records plus 17 benign records. It is not redistributed here because it is a prepared local representation. Expected files:

| Variable | Expected file | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `INJECAGENT_INPUT_FILE` | `injecagent_official_1054.json` | 571,534 | `20c73a9c187cb3045862cc75cf4835da7cc4b8d4fb718f1c880b313c42186251` |
| `INJECAGENT_TOOLS_FILE` | `formatted_tools.json` | 254,158 | `4c27d51e77182dcf50a31bd0d14353c3bf838b31f3b0381b4be175a25c722067` |

The released workflow evaluates the full configured test set using the original fixed seed-42 zero-shot intent split. No task resampling option or preprocessing script is included.
