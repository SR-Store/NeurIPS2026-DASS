"""LSR-Synth tasks (LLM-SRBench, 129 tasks) read from the SR-Scientist parquet.

Each row of ``samples.{train,test,ood_test}`` is ``[y, x_1, ..., x_k]`` in the order of
``symbols``; ``symbols[0]`` is the output.  ``test`` is the in-domain (ID) split.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DOMAIN_COUNTS = {"bio_pop_growth": 24, "chem_react": 36, "matsci": 25, "phys_osc": 44}


@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str  # phys_osc | matsci | chem_react | bio_pop_growth
    output: str  # e.g. "dv_dt"
    inputs: tuple[str, ...]  # e.g. ("x", "t", "v")
    output_desc: str
    input_descs: tuple[str, ...]
    X_train: np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)
    X_id: np.ndarray = field(repr=False)
    y_id: np.ndarray = field(repr=False)
    X_ood: np.ndarray = field(repr=False)
    y_ood: np.ndarray = field(repr=False)
    truth: str = field(repr=False, default="")  # ground truth, for reports only

    @property
    def time_like(self) -> str:
        """Variable the extrapolation environment sorts on: time if present, else the first input."""
        return "t" if "t" in self.inputs else self.inputs[0]


def _split(rows) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray([np.asarray(r, dtype=float) for r in rows], dtype=float)
    return a[:, 1:], a[:, 0]


def load_tasks(path: Path | str | None = None, domains: set[str] | None = None,
               official: bool = False) -> list[Task]:
    """Load tasks from a supplied Parquet file or ``DASS_DATA``.

    ``official=True`` checks the published 129-task domain counts. Small local data
    sets can use the same schema without claiming to be the benchmark.
    """
    import pandas as pd

    if path is None:
        path = os.environ.get("DASS_DATA")
    if not path:
        raise ValueError("dataset path required: pass --data or set DASS_DATA")
    frame = pd.read_parquet(path)
    tasks = []
    for row in frame.itertuples(index=False):
        domain = str(row.dataset_identifier).split("/")[-1]
        if domains and domain not in domains:
            continue
        symbols = [str(s) for s in row.symbols]
        descs = [str(s) for s in row.symbol_descs]
        Xtr, ytr = _split(row.samples["train"])
        Xid, yid = _split(row.samples["test"])
        Xood, yood = _split(row.samples["ood_test"])
        tasks.append(
            Task(
                task_id=str(row.equation_idx),
                domain=domain,
                output=symbols[0],
                inputs=tuple(symbols[1:]),
                output_desc=descs[0],
                input_descs=tuple(descs[1:]),
                X_train=Xtr, y_train=ytr, X_id=Xid, y_id=yid, X_ood=Xood, y_ood=yood,
                truth=str(getattr(row, "expression", "")),
            )
        )
    if official:
        counts = dict(Counter(t.domain for t in tasks))
        expected = {d: n for d, n in DOMAIN_COUNTS.items() if domains is None or d in domains}
        if counts != expected:
            raise AssertionError(f"LSR-Synth domain counts {counts} != {expected}")
    return tasks
