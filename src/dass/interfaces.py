"""Contract between the DASS driver and a base solver adapter.

The driver (``dass.runner``) runs one task as T stages.  Before each stage the DASS
variant calibrates the auxiliary subspace and hands the adapter the top-k% basis terms
as guidance; the base variant runs the same stages with ``guidance=None``, so the only
difference between the two is DASS.  After each stage the adapter returns the
high-quality expressions it found in that stage, which DASS uses to reshape the basis.

    adapter.start(task, seed, workdir)
    for t in range(T):
        guidance = dass.calibrate_and_guide() if dass else None
        candidates = adapter.run_stage(t, T, guidance)
        if dass: dass.reshape(candidates)
    final = adapter.final()

An adapter may also define ``set_basis(basis)``; the +DASS driver then hands it the
whole basis B by omega before each stage, next to the top-k% guidance (LLM-SR fits
combinations over all of B).

Expressions crossing this boundary are sympy-parsable strings over the task's input
names (``task.inputs``) with numeric constants inline, e.g. ``"-0.52*x - 1.3*x**3"``.
Guidance terms use coefficient placeholders ``c0, c1, ...`` instead of numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .data import Task


@dataclass
class Candidate:
    """One expression a solver found."""

    expr: str  # sympy-parsable, task input names, numeric constants inline
    train_nmse: float  # the solver's own error on (a subset of) the training data
    # Exact predictor when ``expr`` is only an approximation of what the solver holds
    # (e.g. an LLM-SR program with fitted params).  None: the driver lambdifies ``expr``.
    predict: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)
    source: str = field(default="", repr=False)  # solver-native form (program text, tree, ...)


@dataclass(frozen=True)
class GuidanceTerm:
    """One top-k% basis term handed to a solver before a stage."""

    term: str  # canonical, placeholders c0, c1, ...: e.g. "x*sin(c0*t)"
    weight: float  # normalized potential score omega over the whole basis
    display: str  # readable form with Greek coefficient names: "x*sin(omega*t)"


class SolverAdapter(Protocol):
    name: str

    def start(self, task: Task, seed: int, workdir: Path) -> None:
        """Set up the solver for one task; nothing is searched yet."""

    def run_stage(self, stage: int, n_stages: int, guidance: list[GuidanceTerm] | None) -> list[Candidate]:
        """Spend this stage's share of the budget (the adapter splits its total budget
        into ``n_stages`` shares) and return the best expressions found so far, best
        first (at most ~50).  ``guidance`` is None for the base variant."""

    def final(self) -> Candidate:
        """The solver's own final answer under its native selection rule."""

    def cost(self) -> dict:
        """LLM calls, prompt and completion tokens, and anything else worth logging."""
