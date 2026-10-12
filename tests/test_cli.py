"""Exercise the public runner and summarizer together without a model service."""

import json
import subprocess
import sys
import time
from pathlib import Path

from dass import config, runner
from dass.interfaces import Candidate


SOURCE = "def equation(x, t, v, params):\n    return -0.8*x - 0.5*x**3 - 0.3*v"


class TraceAdapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_dass = False

    def start(self, task, seed, workdir):
        self.workdir = workdir
        workdir.mkdir(parents=True, exist_ok=True)

    def run_stage(self, stage, n_stages, guidance):
        self.is_dass |= guidance is not None
        if stage == 0:
            (self.workdir / "trace.jsonl").write_text(json.dumps({
                "sample": -1 if self.is_dass else 1,
                "injected": self.is_dass,
                "body": SOURCE if self.is_dass else "model reply",
            }) + "\n")
        time.sleep(0.005)  # ensure the serialized base time is nonzero for ratios
        return [self.final()]

    def set_basis(self, basis):
        pass

    def final(self):
        return Candidate("-0.8*x - 0.5*x**3 - 0.3*v" if self.is_dass else "x",
                         0.0 if self.is_dass else 1.0, source=SOURCE if self.is_dass else "model reply")

    def cost(self):
        return {"calls": 0, "samples": 0, "inject_s": 0.0}


def test_cli_records_and_summarizes_tiny_llmsr_run(monkeypatch, tiny_parquet, tmp_path):
    monkeypatch.setattr(config, "M_SAMPLES", 8)
    monkeypatch.setitem(config.ROUNDS, "llmsr", 2)
    created = []

    def make(_name, **kwargs):
        adapter = TraceAdapter(**kwargs)
        created.append(adapter)
        return adapter

    monkeypatch.setattr(runner, "make", make)
    output = tmp_path / "results"
    workdir = tmp_path / "work"
    common = ["--data", str(tiny_parquet), "--solver", "llmsr", "--backbone", "qwen",
              "--model", "tiny-served-model", "--endpoints", "http://127.0.0.1:9",
              "--tasks", "SYN1", "--seeds", "0", "--out", str(output), "--workdir", str(workdir)]

    assert runner.main([*common, "--variant", "base"]) == 0
    assert runner.main([*common, "--variant", "dass"]) == 0
    assert runner.main([*common, "--variant", "base"]) == 0  # completed item is skipped
    assert len(created) == 2
    assert all(a.kwargs["model"] == "tiny-served-model" for a in created)

    for variant in ("base", "dass"):
        result = output / "llmsr" / "qwen" / variant / "SYN1_s0.json"
        trace = result.with_suffix(".trace.jsonl")
        work_trace = workdir / "llmsr" / "qwen" / variant / "SYN1_s0" / "trace.jsonl"
        record = json.loads(result.read_text())
        assert record["model"] == "tiny-served-model"
        assert record["started_at"].endswith("+08:00")
        assert record["finished_at"].endswith("+08:00")
        assert record["time_s"] > 0
        assert json.loads(trace.read_text()) == json.loads(work_trace.read_text())
        assert bool(json.loads(trace.read_text())["injected"]) == (variant == "dass")

    table_path = tmp_path / "summary.md"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_table.py"
    subprocess.run([sys.executable, str(script), str(output), "--out", str(table_path)],
                   check=True, capture_output=True, text=True)
    table = table_path.read_text()
    assert "| llmsr | qwen | Phys | 1/1 |" in table
    assert "| qwen | Phys | 1 | 1 (100%) | 0 (0%) |" in table
    assert "| qwen | All | 1 | 1 (100%) | 0 (0%) |" in table
