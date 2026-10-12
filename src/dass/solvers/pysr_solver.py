"""PySR search with DASS guidance passed as guesses.

One ``PySRRegressor`` per run with ``warm_start=True``: the 65 outer iterations are
split into ``n_stages`` consecutive fits of the same regressor (65 over 3 stages: 22,
22, 21), and every fit continues from the populations and hall of fame the previous
one left.  Before a DASS stage the guidance terms become that fit's ``guesses``; the
base variant runs the same fits without them.

Guesses: every guidance term alone, then additive combinations of two or more terms
in decreasing total omega, each as ``sum_i b_i * s_i``.  A guess must stay within
``maxsize`` (PySR complexity: one per operator, variable and constant) and use only the
configured operators; a term with any other function (``tanh``, ``sign``, ...), too
large alone, or not finite on the data is left out and logged.  SymbolicRegression.jl
optimizes the constants of every guess, puts it in the hall of fame, and migrates the
guesses into the populations at its default ``fraction_replaced_guesses``; from then on
they are ordinary individuals.

``dass.fitting.combinations`` fits each term's placeholders and the leading constants
``b_i`` on training rows before passing the guesses to PySR.  Each stage supplies at
most ``config.PYSR["max_guesses"]`` guesses.  Rendering uses Julia syntax and checks
the configured operator and complexity limits.
"""

from __future__ import annotations

import ast
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import sympy as sp

from .. import config
from ..data import Task
from ..fitting import combinations
from ..interfaces import Candidate, GuidanceTerm
from ..julia_env import precompile_once, scratch_dir
from ..terms import canonical_term, parse

MAX_GUESSES = config.PYSR["max_guesses"]
_warm = False  # the search kernels are compiled in this process


def share(total: int, stage: int, n_stages: int) -> int:
    """Stage ``stage``'s part of a budget of ``total`` split into ``n_stages``."""
    return total // n_stages + (1 if stage < total % n_stages else 0)


# --- guesses in PySR's syntax -----------------------------------------------------
_TOKEN = re.compile(
    r"(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"|(?P<name>[A-Za-z_]\w*)"
    r"|(?P<op>[-+*/^])"
    r"|(?P<other>\S)"
)


def _tokens(text: str) -> list[tuple[str, str, int]]:
    return [(str(m.lastgroup), m.group(), m.start()) for m in _TOKEN.finditer(text)]


def _minus_is_sign(tokens, i: int) -> bool:
    """A ``-`` that opens a term: Julia reads it as a unary call, which is not in the
    operator set, unless it folds into a numeric literal."""
    if i == 0:
        return True
    kind, text, _ = tokens[i - 1]
    return kind not in ("num", "name") and text != ")"


def _expand_unary_minus(text: str) -> str:
    """``-x*y`` -> ``-1.0*x*y`` and ``exp(-x)`` -> ``exp(-1.0*x)``; a negative literal
    stays, unless a ``^`` follows it (``-2^x`` is ``-(2^x)``)."""
    toks = _tokens(text)
    out, cut = [], 0
    for i, (kind, tok, off) in enumerate(toks):
        if kind != "op" or tok != "-" or not _minus_is_sign(toks, i):
            continue
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        after = toks[i + 2] if i + 2 < len(toks) else None
        if nxt is not None and nxt[0] == "num" and not (after is not None and after[1] == "^"):
            continue
        out.append(text[cut:off])
        out.append("-1.0*")
        cut = off + 1
    out.append(text[cut:])
    return "".join(out)


def _float_literals(e: sp.Expr) -> sp.Expr:
    """Non-integer rationals as floats (``x/2`` would need ``/``), except in exponents,
    where ``1/2`` is how sympy spells ``sqrt``."""
    if isinstance(e, sp.Pow):
        return sp.Pow(_float_literals(e.base), e.exp)
    if not e.args:
        return sp.Float(e) if isinstance(e, sp.Rational) and not e.is_Integer else e
    return e.func(*(_float_literals(a) for a in e.args))


def _fold_constants(e: sp.Expr) -> sp.Expr:
    """``pi`` and ``E`` as literals; PySR would read the names as unknown features."""
    consts = {c: sp.Float(c.evalf(15)) for c in e.atoms(sp.NumberSymbol)}
    return e.xreplace(consts) if consts else e


def render(e: sp.Expr) -> str:
    """The string SymbolicRegression.jl parses: ``^`` for powers, ``abs`` for ``Abs``,
    literal constants, no bare unary minus."""
    text = sp.sstr(_float_literals(_fold_constants(e)))
    text = re.sub(r"\bAbs\(", "abs(", text)
    return _expand_unary_minus(text.replace("**", "^"))


def operators(rendered: str) -> tuple[set[str], set[str]]:
    """``(binary, unary)`` operators a Julia parse of ``rendered`` looks up."""
    binary, unary = set(), set()
    toks = _tokens(rendered)
    for i, (kind, tok, _) in enumerate(toks):
        if kind == "op":
            if not (tok == "-" and _minus_is_sign(toks, i)):
                binary.add(tok)
        elif kind == "name" and i + 1 < len(toks) and toks[i + 1][1] == "(":
            unary.add(tok)
    return binary, unary


def complexity(rendered: str) -> int:
    """PySR's default complexity of a rendered guess: one per operator, variable and
    constant (a negative literal is one constant)."""

    def count(n) -> int:
        if isinstance(n, ast.BinOp):
            return 1 + count(n.left) + count(n.right)
        if isinstance(n, ast.UnaryOp):
            return count(n.operand) if isinstance(n.operand, ast.Constant) else 1 + count(n.operand)
        if isinstance(n, ast.Call):
            return 1 + sum(count(a) for a in n.args)
        return 1

    return count(ast.parse(rendered.replace("^", "**"), mode="eval").body)


def build_guesses(guidance: list[GuidanceTerm], inputs, X: np.ndarray, y: np.ndarray) -> tuple[list[str], list[dict]]:
    """Rendered guesses for one stage, and the guidance terms that could not be used."""
    binary_ok = set(config.PYSR["binary_operators"])
    unary_ok = set(config.PYSR["unary_operators"])
    maxsize = config.PYSR["maxsize"]
    probe = sp.Float(0.7)  # stands in for fitted constants: no special form (1, 0.5, 2) arises
    terms, weights, names, skipped = [], [], [], []
    for g in guidance:
        e = parse(g.term, inputs)
        t = canonical_term(e, inputs) if e is not None else None
        if t is None:
            skipped.append({"term": g.display, "reason": "parse"})
            continue
        b, u = operators(render(probe * t.at([0.7] * t.n_params)))
        if not (b <= binary_ok and u <= unary_ok):
            skipped.append({"term": g.display, "reason": "operator " + ",".join(sorted((b - binary_ok) | (u - unary_ok)))})
            continue
        terms.append(t)
        weights.append(g.weight)
        names.append(g.display)

    def admissible(c) -> bool:
        return complexity(render(sp.Add(*[probe * terms[i].at([0.7] * terms[i].n_params) for i in c]))) <= maxsize

    guesses, seen = [], set()
    for c, pairs, reason in combinations(terms, weights, X, y, admissible):
        if pairs is None:
            if len(c) == 1:
                skipped.append({"term": names[c[0]], "reason": "complexity" if reason == "structure" else reason})
            continue
        text = render(sp.Add(*[sp.Float(b) * e for b, e in pairs]))
        if text in seen or complexity(text) > maxsize:
            continue
        seen.add(text)
        guesses.append(text)
        if len(guesses) == MAX_GUESSES:
            break
    return guesses, skipped


# --- the adapter ---------------------------------------------------------------------
def _settings() -> dict:
    p = config.PYSR
    return dict(populations=p["populations"], population_size=p["population_size"],
                ncycles_per_iteration=p["ncycles_per_iteration"], maxsize=p["maxsize"],
                binary_operators=list(p["binary_operators"]), unary_operators=list(p["unary_operators"]),
                parallelism="multithreading", progress=False, verbosity=0)


class PySRAdapter:
    name = "pysr"

    def __init__(self, threads: int = 8):
        self.threads = threads  # Julia threads = the cores given to the run

    def start(self, task: Task, seed: int, workdir: Path) -> None:
        global _warm
        self.task = task
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        # An explicitly configured Julia environment can be precompiled once outside
        # juliacall.  In a regular PySR installation, let PySR manage its own Julia setup.
        julia_env = tuple(os.environ.get(key) for key in
                          ("PYTHON_JULIAPKG_EXE", "PYTHON_JULIAPKG_PROJECT", "JULIA_DEPOT_PATH"))
        self.precompile_s = None
        if all(julia_env):
            self.precompile_s = precompile_once("pysr", *julia_env, scratch_dir(self.workdir))
            os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"
        os.environ["PYTHON_JULIACALL_THREADS"] = os.environ["JULIA_NUM_THREADS"] = str(self.threads)
        from pysr import PySRRegressor
        from pysr.julia_import import jl

        var = float(np.var(task.y_train))
        self.var_y = var if var > 0 else 1.0
        self.fit_kwargs = {"variable_names": list(task.inputs)}
        rows = np.random.default_rng(seed).choice(len(task.y_train), min(config.FIT_ROWS, len(task.y_train)),
                                                  replace=False)
        self.X_fit, self.y_fit = task.X_train[rows], task.y_train[rows]
        if not _warm:  # compile the search, batching and guess parsing outside the timed stages
            PySRRegressor(**{**_settings(), "populations": 2, "ncycles_per_iteration": 10}, niterations=1,
                          guesses=[task.inputs[0]], output_directory=str(self.workdir / "warmup"),
                          random_state=seed).fit(task.X_train, task.y_train, **self.fit_kwargs)
            _warm = True
        self.model = PySRRegressor(**_settings(), warm_start=True, random_state=np.random.RandomState(seed),
                                   output_directory=str(self.workdir / "pysr"), run_id=f"{task.task_id}_s{seed}")
        self.julia_threads = int(jl.Threads.nthreads())
        self.stages: list[dict] = []
        self.inject_s = 0.0

    def run_stage(self, stage: int, n_stages: int, guidance: list[GuidanceTerm] | None) -> list[Candidate]:
        k = share(config.PYSR["niterations"], stage, n_stages)
        t0 = time.time()
        guesses, skipped = build_guesses(guidance, self.task.inputs, self.X_fit, self.y_fit) if guidance else ([], [])
        guess_s = time.time() - t0 if guidance else 0.0
        self.inject_s += guess_s
        self.model.set_params(niterations=k, guesses=guesses or None)
        t0 = time.time()
        self.model.fit(self.task.X_train, self.task.y_train, **self.fit_kwargs)
        cands = self._hall_of_fame()
        self.stages.append({"niterations": k, "guess_s": round(guess_s, 3), "fit_s": round(time.time() - t0, 3),
                            "n_guesses": len(guesses), "guesses": guesses, "skipped_terms": skipped,
                            "hall_of_fame": len(cands)})
        return cands

    def _hall_of_fame(self) -> list[Candidate]:
        eqs = self.model.equations_
        inputs = set(self.task.inputs)
        out = []
        for r in ([] if eqs is None else eqs.to_dict("records")):
            e, loss = r.get("sympy_format"), float(r.get("loss", math.nan))
            if e is None or not math.isfinite(loss) or not {s.name for s in e.free_symbols} & inputs:
                continue  # a constant or a failed row
            out.append(Candidate(expr=str(e), train_nmse=loss / self.var_y, source=str(r["equation"])))
        return sorted(out, key=lambda c: c.train_nmse)

    def final(self) -> Candidate:
        r = self.model.get_best()  # PySR's own model_selection ("best")
        return Candidate(expr=str(r["sympy_format"]), train_nmse=float(r["loss"]) / self.var_y,
                         source=str(r["equation"]))

    def cost(self) -> dict:
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "julia_threads": self.julia_threads, "precompile_s": self.precompile_s,
                "inject_s": round(self.inject_s, 3), "stages": self.stages}
