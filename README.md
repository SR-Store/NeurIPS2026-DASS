# DASS

DASS is a dynamic basis-guidance layer for symbolic regression. This package
supports PySR and LLM-SR in two modes: `base` runs the solver alone, while
`dass` updates basis-term weights, guides each search stage, and uses solver
feedback to reshape the basis. For LLM-SR, `dass` also fits basis combinations
on training data and places selected programs in the experience buffer.

## Install

Python 3.10+ on Linux or macOS is required. Install from this source directory:

```sh
python3 -m pip install "/path/to/dass[test]"
```

For PySR runs, install the optional PySR dependency and its Julia requirements:

```sh
python3 -m pip install "/path/to/dass[pysr]"
```

LLM-SR uses an OpenAI-compatible chat server, such as a local vLLM service.
The server and model weights are supplied separately; installation does not
start a model service.

## Data

Supply an SR-Scientist-format Parquet file with one equation per row. Required
columns are `dataset_identifier`, `equation_idx`, `symbols`, `symbol_descs`, and
`samples`; `expression` is optional metadata. Each `samples` value has `train`,
`test`, and `ood_test` arrays; each row is `[target, input_1, ...]` in `symbols`
order. `test` is the in-domain split. Download the matching
[SR-Scientist `test.parquet`](https://huggingface.co/datasets/GAIR/SR-Scientist/resolve/main/test.parquet?download=true)
and save it as `/path/to/data/test.parquet`. Pass that file with `--data` or set
`DASS_DATA`. Add `--official` to validate the full 129-task, four-domain
collection.

The four included basis files provide initial terms. Searches and injected
programs are fitted on `train`; `test` and `ood_test` are used only for final
scoring. The file's `expression` field is metadata and is not used in search.

## Run

Use the same output directory for both variants; records are stored by solver,
backbone, variant, task, and seed. Existing records are skipped. The default
seeds are `0,1,2`; `--tasks` selects equation IDs.

```sh
dass-run --data /path/to/test.parquet --solver pysr --variant base --tasks PO19 --seeds 0 --out /path/to/results
dass-run --data /path/to/test.parquet --solver pysr --variant dass --tasks PO19 --seeds 0 --out /path/to/results
```

For LLM-SR, point `--endpoints` to one or more running server roots (without
`/v1`). `qwen` and `gemma` use the configured served-name aliases
`qwen35-9b` and `gemma4-31b`; use `--model` if your server exposes another
name. Set `DASS_API_KEY` only if your server requires authentication.

```sh
dass-run --data /path/to/test.parquet --solver llmsr --backbone qwen --endpoints http://127.0.0.1:8000 --variant base --tasks PO19 --seeds 0 --out /path/to/results
dass-run --data /path/to/test.parquet --solver llmsr --backbone qwen --endpoints http://127.0.0.1:8000 --variant dass --tasks PO19 --seeds 0 --out /path/to/results
```

LLM-SR generates 100 model samples per run in both modes. Fitted programs
inserted by DASS are additional candidates and do not count toward those 100
samples. PySR uses 65 outer iterations in both modes. A result JSON records
the final program, ID/OOD NMSE and Acc@0.01, elapsed search time, and the
DASS portion of that time. LLM-SR also saves a `.trace.jsonl` file beside each
result JSON, including injected programs and model replies. Summarize completed
records with:

```sh
python3 /path/to/dass/scripts/make_table.py /path/to/results --out /path/to/summary.md
```

NMSE is mean squared error divided by target variance. Acc@0.01 passes when
the worst 5% of pointwise errors are discarded and every remaining relative
error is at most 0.01; targets close to zero use the implementation's
near-zero rule in `dass.metrics`. Nonfinite predictions fail both scores.

## Test

The unit tests and tiny local example need neither a model server nor Julia:

```sh
python3 -m pytest /path/to/dass/tests -m "not integration"
python3 /path/to/dass/examples/tiny_offline.py --data /path/to/dass/tests/fixtures/tiny_lsr_synth.parquet
```

Environment-specific integration tests run separately when their dependencies
are available. The example reads a tiny synthetic dataset, performs one small
DASS calibration step, and scores a fixed candidate without a solver or model
call.

## License

Original project code is MIT-licensed; see [LICENSE](LICENSE). Vendored
LLM-SR code retains its own MIT and Apache-2.0 terms; see
[third-party notices](THIRD_PARTY_NOTICES.md).
