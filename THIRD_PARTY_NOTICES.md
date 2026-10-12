# Third-party notices

The MIT license in [LICENSE](LICENSE) applies to the original DASS code in this
repository. Vendored LLM-SR code retains its upstream terms and source headers.

- The files in `src/dass/_vendor/llmsr/` originate from the
  [LLM-SRBench repository](https://github.com/deep-symbolic-mathematics/llm-srbench)
  at commit `6fdbe409b75ea749c0a93c69b6861637d804dec4`. Its repository README
  declares the project MIT-licensed. The MIT terms are reproduced in
  `src/dass/_vendor/llmsr/LICENSE-MIT`.
- `buffer.py`, `code_manipulation.py`, `config.py`, `evaluator.py`, and
  `sampler.py` retain copyright and Apache-2.0 notices from
  [DeepMind FunSearch](https://github.com/google-deepmind/funsearch). The full
  Apache-2.0 terms are in `src/dass/_vendor/llmsr/LICENSE-APACHE-2.0`.

The vendor-specific file history and local adaptations are recorded in
`src/dass/_vendor/llmsr/PROVENANCE.md`.
