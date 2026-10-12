"""Basis terms: parsing, canonical form, and splitting an expression into terms.

A basis term is a single building block over the task's input variables whose free
coefficients are placeholders ``c0, c1, ...``.  Every term enters a sampled expression
with its own linear coefficient, so a leading multiplicative constant is dropped from
the canonical form: ``alpha*x``, ``3.1*x`` and ``x`` are the same term.  Constants
inside a term are placeholders, except simple rational exponents (``x**2``, ``A**0.5``)
which are structure. Terms differing only in how coefficients are named merge.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs, "tanh": sp.tanh, "sinh": sp.sinh,
    "cosh": sp.cosh, "arctan": sp.atan, "atan": sp.atan, "arcsin": sp.asin, "asin": sp.asin,
    "arccos": sp.acos, "acos": sp.acos, "sign": sp.sign, "pi": sp.pi,
    "Max": sp.Max, "Min": sp.Min, "maximum": sp.Max, "minimum": sp.Min,  # np.clip, np.maximum in programs
}
GREEK = ("alpha", "beta", "gamma", "delta", "omega", "kappa", "lambda", "mu", "nu", "rho",
         "sigma", "tau", "phi", "psi", "theta", "zeta", "eta", "xi", "chi", "epsilon_c")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_MAX_EXPANDED = 20


def _placeholder(i: int) -> sp.Symbol:
    return sp.Symbol(f"c{i}", real=True)


def input_symbols(inputs) -> dict[str, sp.Symbol]:
    return {name: sp.Symbol(name, real=True) for name in inputs}


def parse(text: str, inputs) -> sp.Expr | None:
    """Parse a term or expression written by an LLM or a solver.

    ``x(t)`` becomes ``x``; ``np.`` prefixes and ``**``/``^`` both work; every other
    identifier that is not a known function is a coefficient.
    """
    s = text.strip().rstrip(",;.")
    s = s.replace("np.", "").replace("numpy.", "").replace("math.", "").replace("^", "**")
    s = re.sub(r"\blambda\b", "lambda_", s)
    for name in inputs:
        s = re.sub(rf"\b{re.escape(name)}\s*\(\s*t\s*\)", name, s)
    syms = input_symbols(inputs)
    local = dict(FUNCS)
    local.update(syms)
    for ident in set(_IDENT.findall(s)):
        if ident not in local:
            local[ident] = sp.Symbol(f"k_{ident}", real=True)
    try:
        expr = parse_expr(s, local_dict=local, evaluate=True)
    except Exception:
        return None
    if not isinstance(expr, sp.Expr) or expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I):
        return None
    return expr


def _simple_rational(value: float) -> sp.Rational | None:
    if not math.isfinite(value):  # a SymPy Float beyond double range
        return None
    frac = Fraction(value).limit_denominator(3)
    if abs(float(frac) - value) < 1e-3 * max(1.0, abs(value)):
        return sp.Rational(frac.numerator, frac.denominator)
    return None


class _Canon:
    """Rebuild an expression with every free constant replaced by one placeholder."""

    def __init__(self, inputs):
        self.inputs = set(input_symbols(inputs).values())
        self.n = 0

    def new(self) -> sp.Symbol:
        self.n += 1
        return sp.Symbol(f"_p{self.n}", real=True)

    def is_param(self, e) -> bool:
        return e.free_symbols.isdisjoint(self.inputs)

    def __call__(self, e):
        if e in self.inputs:
            return e
        if self.is_param(e):  # constant subtree (numbers or coefficients)
            if e.is_Integer or e.is_Rational or e in (sp.pi, sp.E):
                return e
            if e.is_Float:
                r = _simple_rational(float(e))
                return r if r is not None and r.q == 1 and abs(r.p) <= 3 else self.new()
            return self.new()
        if e.is_Pow:
            base, exp = e.args
            if self.is_param(exp):
                if exp.is_Rational:
                    return sp.Pow(self(base), exp)
                if exp.is_Float:
                    r = _simple_rational(float(exp))
                    if r is not None:
                        return sp.Pow(self(base), r)
                return sp.Pow(self(base), self.new())
            return sp.Pow(self(base), self(exp))
        if e.is_Mul or e.is_Add:
            args = [self(a) for a in e.args]
            params = [a for a in args if self.is_param(a)]
            rest = [a for a in args if not self.is_param(a)]
            if any(not a.is_number for a in params):  # a placeholder absorbs the other constants
                params = [self.new()]
            return (sp.Mul if e.is_Mul else sp.Add)(*params, *rest)
        return e.func(*[self(a) for a in e.args])


def _drop_leading_constant(e, inputs_set):
    if e.is_Mul:
        kept = [a for a in e.args if not a.free_symbols.isdisjoint(inputs_set)]
        return sp.Mul(*kept) if kept else sp.Integer(1)
    return e


def _number_placeholders(e) -> tuple[sp.Expr, int]:
    """Rename the internal placeholders to c0, c1, ... in a deterministic order."""
    olds = sorted((s for s in e.free_symbols if s.name.startswith("_p")), key=lambda s: int(s.name[2:]))
    if not olds:
        return e, 0
    # Order by first appearance in the printed form of the expression with them unified.
    tmp = e.subs({s: sp.Symbol(f"_q{i:03d}", real=True) for i, s in enumerate(olds)})
    order = sorted(range(len(olds)), key=lambda i: str(tmp).find(f"_q{i:03d}"))
    mapping = {olds[i]: _placeholder(j) for j, i in enumerate(order)}
    return e.subs(mapping), len(olds)


@dataclass(frozen=True)
class Term:
    key: str  # canonical string; equal keys are the same term
    expr: sp.Expr  # canonical sympy form with placeholders c0..c{n_params-1}
    n_params: int
    inputs: tuple[str, ...]

    @cached_property
    def display(self) -> str:
        names = {_placeholder(i): sp.Symbol(GREEK[i % len(GREEK)]) for i in range(self.n_params)}
        return str(self.expr.subs(names))

    def __getstate__(self):
        """Pickle without the compiled caches (``fn`` is a lambdified closure); workers rebuild them."""
        return {k: v for k, v in self.__dict__.items() if k not in ("fn", "display")}

    def at(self, theta) -> sp.Expr:
        """The term with its placeholders set to ``theta``."""
        return self.expr.xreplace({_placeholder(j): sp.Float(float(v)) for j, v in enumerate(theta)})

    @cached_property
    def fn(self):
        """numpy function f(X, theta) -> column; X columns in ``inputs`` order."""
        syms = [sp.Symbol(n, real=True) for n in self.inputs]
        params = [_placeholder(i) for i in range(self.n_params)]
        f = sp.lambdify(syms + params, self.expr, modules="numpy")
        k = len(syms)

        def call(X, theta=()):
            out = f(*[X[:, i] for i in range(k)], *theta)
            return np.broadcast_to(np.asarray(out, dtype=float), (X.shape[0],))

        return call


def canonical_term(expr: sp.Expr, inputs) -> Term | None:
    """Canonical Term for a single building block, or None for a constant."""
    syms = set(input_symbols(inputs).values())
    if expr.free_symbols.isdisjoint(syms):
        return None
    c = _Canon(inputs)
    e = _drop_leading_constant(c(expr), syms)
    e, n = _number_placeholders(e)
    if e.free_symbols.isdisjoint(syms) or e.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        return None  # canonicalising can create a division by zero
    key = str(e.subs({_placeholder(i): sp.Symbol("c") for i in range(n)}))
    return Term(key=key, expr=e, n_params=n, inputs=tuple(inputs))


def split_terms(expr: sp.Expr, inputs) -> list[Term]:
    """Split an expression into canonical terms at its top-level sum.

    The top level is distributed first (``c*P*(1 - P/K)`` gives ``P`` and ``P**2``);
    function arguments are left alone.  Duplicate terms are merged; constants dropped.
    """
    expanded = sp.expand(expr, deep=False, power_exp=False, power_base=False, log=False)
    args = sp.Add.make_args(expanded)
    if len(args) > _MAX_EXPANDED:
        args = sp.Add.make_args(expr)
    out, seen = [], set()
    for a in args:
        t = canonical_term(a, inputs)
        if t is not None and t.key not in seen:
            seen.add(t.key)
            out.append(t)
    return out


def term_from_text(text: str, inputs) -> Term | None:
    """A basis term from LLM text.  Only a top-level sum of two or more non-constant parts
    is not a single term; a product containing a sum (``r*P*(1 - P/K)``) is one canonical
    mechanism form and is kept whole, not distributed."""
    e = parse(text, inputs)
    if e is None:
        return None
    parts = [canonical_term(a, inputs) for a in sp.Add.make_args(e)]
    parts = [t for t in parts if t is not None]
    return parts[0] if len(parts) == 1 else None
