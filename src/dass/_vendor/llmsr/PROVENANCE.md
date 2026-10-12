# Vendored LLM-SR modules

Source: [LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench),
`methods/llmsr/` at commit `6fdbe409b75ea749c0a93c69b6861637d804dec4`.
The upstream README declares the repository MIT-licensed. The MIT terms are
included in `LICENSE-MIT`.

Five imported modules (`buffer.py`, `code_manipulation.py`, `config.py`,
`evaluator.py`, and `sampler.py`) retain their DeepMind FunSearch copyright and
Apache-2.0 source headers. The full terms are in `LICENSE-APACHE-2.0`.

The selected upstream modules are placed under `dass._vendor.llmsr` so an
installation does not rely on a separate source checkout. The adapter adds
this installed directory to `sys.path`; the upstream `from llmsr import ...`
statements remain in the vendored files. The adapter in `dass.solvers.llmsr`
provides search-stage control, request handling, candidate scoring, and DASS
guidance.

Changes to the imported source files:

| File | Change |
|---|---|
| `buffer.py` | `absl.logging` import replaced by standard-library `logging`. |
| `code_manipulation.py` | Same logging import replacement. |
| `sampler.py` | Retains upstream `LLM`, `Sampler`, and `_extract_body`; removes unused local-model and request-client code and its dependencies. |
| `config.py`, `evaluator.py`, `evaluator_accelerate.py` | Copied without behavioral edits. |
| `specs.py` | Extracts the equation and optimization specifications required by this adapter from upstream `searcher.py` and `sampler.py`. |

The changed Apache-licensed files carry modified-file notices directly after
their original copyright and license headers. Upstream command-line and model
server implementations are not included.
