"""Base solver adapters (see ``dass.interfaces.SolverAdapter``).

One module per solver, registered here by name; imports are lazy so the driver only
loads the solver a run uses.
"""

from __future__ import annotations

from importlib import import_module

from ..interfaces import SolverAdapter

# name -> (module, class)
REGISTRY = {
    "llmsr": ("dass.solvers.llmsr", "LLMSRAdapter"),
    "pysr": ("dass.solvers.pysr_solver", "PySRAdapter"),
}
LLM_SOLVERS = {"llmsr"}


def make(name: str, **kwargs) -> SolverAdapter:
    """``kwargs`` for LLM solvers: ``endpoints`` (base URLs) and ``model`` (served name)."""
    module, cls = REGISTRY[name]
    return getattr(import_module(module), cls)(**kwargs)
