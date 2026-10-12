"""Fit expressions built from basis terms in three evaluation environments.

An expression is ``a_0 + sum_i a_i * s_i(X; theta_i)``: every term has a linear
coefficient and an intercept is always present. Linear coefficients are
solved by least squares for given nonlinear ``theta`` (variable projection); ``theta``
is optimized with BFGS from all-ones, as LLM-SR does for its params.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize

from . import config
from .terms import Term

_BIG = 1e30


def _design(terms: list[Term], X: np.ndarray, thetas: list[np.ndarray]) -> np.ndarray:
    cols = [np.ones(X.shape[0])]
    with np.errstate(all="ignore"):
        for t, th in zip(terms, thetas):
            cols.append(t.fn(X, th))
    return np.column_stack(cols)


def _split_theta(terms: list[Term], flat: np.ndarray) -> list[np.ndarray]:
    out, i = [], 0
    for t in terms:
        out.append(flat[i:i + t.n_params])
        i += t.n_params
    return out


def _lstsq(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    if not np.all(np.isfinite(A)):
        return np.zeros(A.shape[1]), _BIG
    try:
        a, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(A.shape[1]), _BIG
    r = A @ a - y
    return a, float(np.mean(r * r))


@dataclass
class Fit:
    terms: list[Term]
    thetas: list[np.ndarray]
    coefs: np.ndarray  # intercept first

    def predict(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            return _design(self.terms, X, self.thetas) @ self.coefs


def fit(terms: list[Term], X: np.ndarray, y: np.ndarray, maxiter: int = 60) -> Fit:
    n_theta = sum(t.n_params for t in terms)
    if n_theta == 0:
        coefs, _ = _lstsq(_design(terms, X, [np.zeros(0)] * len(terms)), y)
        return Fit(terms, [np.zeros(0)] * len(terms), coefs)

    def loss(flat):
        _, mse = _lstsq(_design(terms, X, _split_theta(terms, flat)), y)
        return mse if math.isfinite(mse) else _BIG

    best = None
    for start in (np.ones(n_theta), -np.ones(n_theta)):
        with np.errstate(all="ignore"):
            res = minimize(loss, start, method="BFGS", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    thetas = _split_theta(terms, best.x)
    coefs, _ = _lstsq(_design(terms, X, thetas), y)
    return Fit(terms, thetas, coefs)


def combinations(terms: list[Term], weights: list[float], X: np.ndarray, y: np.ndarray, admissible):
    """Guidance terms as seeds for a solver's search (PySR guesses).

    Each term alone, then every combination of two or more in decreasing total weight,
    as ``sum_i b_i * s_i``: the placeholders of the ``s_i`` fitted by ``fit`` and the
    ``b_i`` by least squares without intercept on (X, y).  Yields ``(indices, pairs,
    reason)`` with ``pairs`` = [(b_i, s_i with its constants inlined)], or ``pairs`` None
    and ``reason`` "structure" when ``admissible(indices)`` rejects the combination
    (checked before fitting) or "not finite on the data".  Consumers stop when they have
    enough.
    """
    subsets = [(i,) for i in range(len(terms))]
    combos = [c for r in range(2, len(terms) + 1) for c in itertools.combinations(range(len(terms)), r)]
    subsets += sorted(combos, key=lambda c: (-sum(weights[i] for i in c), len(c)))
    for c in subsets:
        if not admissible(c):
            yield c, None, "structure"
            continue
        ts = [terms[i] for i in c]
        f = fit(ts, X, y)
        with np.errstate(all="ignore"):
            A = np.column_stack([t.fn(X, th) for t, th in zip(ts, f.thetas)])
        if not np.all(np.isfinite(A)):
            yield c, None, "not finite on the data"
            continue
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        yield c, [(float(bi), t.at(th)) for bi, t, th in zip(b, ts, f.thetas)], ""


def nmse(pred: np.ndarray, y: np.ndarray) -> float:
    if not np.all(np.isfinite(pred)):
        return math.inf
    var = float(np.var(y))
    mse = float(np.mean((pred - y) ** 2))
    return mse / var if var > 0 else (0.0 if mse == 0 else math.inf)


@dataclass
class Environment:
    """One evaluation environment of a DASS round: where to fit and where to score."""

    name: str
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    _memo: dict = field(default_factory=dict, repr=False)  # term keys -> fitness

    def fitness(self, terms: list[Term]) -> float:
        """Fitness = 1 / (1 + L), L the NMSE on the evaluation part.  A round samples the
        same term sets many times once omega concentrates; each set is fitted once."""
        key = tuple(t.key for t in terms)
        if key not in self._memo:
            try:
                f = fit(terms, self.X_fit, self.y_fit)
                with np.errstate(all="ignore"):
                    loss = nmse(f.predict(self.X_eval), self.y_eval)
            except (ArithmeticError, ValueError, TypeError, KeyError):  # a term numpy cannot evaluate
                loss = math.inf
            self._memo[key] = 1.0 / (1.0 + loss) if math.isfinite(loss) else 0.0
        return self._memo[key]

    def fitness_many(self, term_lists: list[list[Term]], pool=None, workers: int = 1) -> list[float]:
        """``fitness`` of each term list; the sets not yet scored here are fitted in parallel
        on ``pool`` (a process pool of ``workers``), one chunk per worker."""
        keys = [tuple(t.key for t in ts) for ts in term_lists]
        todo = {}
        for k, ts in zip(keys, term_lists):
            if k not in self._memo:
                todo.setdefault(k, ts)
        if pool is not None and workers > 1 and len(todo) > 1:
            items = list(todo.items())
            chunks = [c for c in (items[i::workers] for i in range(workers)) if c]
            light = replace(self, _memo={})
            for chunk, scores in zip(chunks, pool.map(_score_chunk, [(light, [ts for _, ts in c]) for c in chunks])):
                self._memo.update((k, f) for (k, _), f in zip(chunk, scores))
        else:
            for ts in todo.values():
                self.fitness(ts)
        return [self._memo[k] for k in keys]


def _score_chunk(args) -> list[float]:
    env, term_lists = args
    return [env.fitness(ts) for ts in term_lists]


def make_environment(name: str, X: np.ndarray, y: np.ndarray, sort_col: int,
                     rng: np.random.Generator) -> Environment:
    """extrap: fit the lower half along the time-like variable, score the upper half.
    sparse: fit a random 10% subset, score the rest.  noisy: add Gaussian noise
    (5% of std y) to all targets, fit one random half, score the other.
    NMSE divides by the variance of the scored part."""
    n = len(y)
    if n > config.FIT_ROWS:
        idx = rng.choice(n, config.FIT_ROWS, replace=False)
        X, y, n = X[idx], y[idx], config.FIT_ROWS
    if name == "extrap":
        order = np.argsort(X[:, sort_col], kind="stable")
        a, b = order[: n // 2], order[n // 2:]
    elif name == "sparse":
        perm = rng.permutation(n)
        k = max(int(round(config.SPARSE_FRACTION * n)), 10)
        a, b = perm[:k], perm[k:]
    elif name == "noisy":
        y = y + rng.normal(0.0, config.NOISE_REL_STD * float(np.std(y)), size=n)
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
    else:
        raise ValueError(name)
    return Environment(name, X[a], y[a], X[b], y[b])
