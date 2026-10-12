"""The DASS state for one task: calibration, guidance, and reshaping.

    dass = DASS(task, init_terms, seed)
    for t in range(T):
        guidance = dass.step()          # calibrate omega in one environment, return top-k%
        candidates = solver.run_stage(t, T, guidance)
        dass.reshape(candidates)        # replace low-utility terms by new high-utility ones

Utilities are smoothed within each environment, averaged across environments, and
discounted when their minimum environment score is low. The resulting utility updates
term weights before solver feedback can reshape the basis. Implementation choices:

* environments are visited round-robin;
* before the first step, one batch of sampled term sets is scored in every
  environment to initialize the stability discount;
* a newly accepted term receives its current-environment utility in all environment
  slots, even though it has only been evaluated in the current environment;
* its initial weight is omega0 * exp(beta * u) / |B|, scaling against a uniform
  basis weight before all weights are normalized again.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import sys
import weakref
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from . import config
from .data import Task
from .fitting import Environment, make_environment
from .interfaces import Candidate, GuidanceTerm
from .terms import Term, parse, split_terms


@dataclass
class RoundLog:
    round: int
    env: str
    elite_min_fitness: float
    best_fitness: float
    omega_top: list[tuple[str, float]]
    replaced: list[tuple[str, str]] = field(default_factory=list)  # (old key, new key)
    n_feedback: int = 0
    n_retained: int = 0


def admissible_new_term(t: Term) -> bool:
    """A solver term may enter B only within B(0)'s building-block size (config)."""
    return t.n_params <= config.MAX_NEW_TERM_PARAMS and sp.count_ops(t.expr) <= config.MAX_NEW_TERM_OPS


class DASS:
    def __init__(self, task: Task, init_terms: list[tuple[Term, float]], seed: int, workers: int | None = None):
        if not init_terms:
            raise ValueError("empty initial basis")
        self.task = task
        # Independent fits use the CPUs available to this process.
        if workers is None:
            workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
        self.workers = workers
        self._executor = None
        self.rng = np.random.default_rng(seed)
        self.terms: list[Term] = [t for t, _ in init_terms][: config.BASIS_SIZE]
        w = np.array([max(float(c), 0.0) for _, c in init_terms][: config.BASIS_SIZE])
        self.omega = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        n = len(self.terms)
        self.u_env = {e: np.zeros(n) for e in config.ENVIRONMENTS}
        self.u = np.zeros(n)  # final utility u^(t) of the last round
        self.t = 0
        self.sort_col = task.inputs.index(task.time_like)
        self.env: Environment | None = None
        self.elite_min = 0.0
        self.log: list[RoundLog] = []
        self._warm_start()

    # -- sampling and scoring ------------------------------------------------
    def _sample(self) -> list[tuple[int, ...]]:
        n = len(self.terms)
        out = []
        for _ in range(config.M_SAMPLES):
            k = int(self.rng.integers(1, min(config.MAX_TERMS_PER_SAMPLE, n) + 1))
            idx = self.rng.choice(n, size=k, replace=False, p=self.omega)
            out.append(tuple(sorted(int(i) for i in idx)))
        return out

    def _env(self, name: str) -> Environment:
        return make_environment(name, self.task.X_train, self.task.y_train, self.sort_col, self.rng)

    def _pool(self):
        main = sys.modules.get("__main__")
        if getattr(main, "__spec__", None) is None and not os.path.isfile(getattr(main, "__file__", None) or ""):
            self.workers = 1  # a script read from stdin cannot be re-imported by worker processes
        if self._executor is None and self.workers > 1:
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")  # read by the workers' BLAS; this process's is set up already
            # Preload fitting dependencies in the forkserver before workers are forked.
            ctx = multiprocessing.get_context("forkserver")
            ctx.set_forkserver_preload(["dass.fitting"])
            self._executor = ProcessPoolExecutor(self.workers, mp_context=ctx)
            weakref.finalize(self, self._executor.shutdown, wait=False, cancel_futures=True)
        return self._executor

    def _score(self, env: Environment, samples) -> np.ndarray:
        return np.array(env.fitness_many([[self.terms[i] for i in s] for s in samples], self._pool(), self.workers))

    def _obtained_utility(self, samples, fitness) -> tuple[np.ndarray, float]:
        """Mean Fitness(f)/|f| over elite expressions containing each term."""
        k = max(1, int(math.ceil(config.ELITE_RATIO * len(samples))))
        elite = np.argsort(-fitness, kind="stable")[:k]
        total = np.zeros(len(self.terms))
        count = np.zeros(len(self.terms))
        for j in elite:
            for i in samples[j]:
                total[i] += fitness[j] / len(samples[j])
                count[i] += 1
        u = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        return u, float(fitness[elite].min())

    def _warm_start(self):
        samples = self._sample()
        for name in config.ENVIRONMENTS:
            u, _ = self._obtained_utility(samples, self._score(self._env(name), samples))
            self.u_env[name] = u

    # -- one DASS round --------------------------------------------------------
    def step(self) -> list[GuidanceTerm]:
        """Calibrate omega in this round's environment and return the top-k% terms."""
        name = config.ENVIRONMENTS[self.t % len(config.ENVIRONMENTS)]
        self.env = self._env(name)
        samples = self._sample()
        fitness = self._score(self.env, samples)
        u_obt, self.elite_min = self._obtained_utility(samples, fitness)
        self.u_env[name] = (1 - config.ETA) * self.u_env[name] + config.ETA * u_obt  # smooth current environment
        u_com = np.mean([self.u_env[e] for e in config.ENVIRONMENTS], axis=0)  # cross-environment mean
        gamma = np.min([self.u_env[e] for e in config.ENVIRONMENTS], axis=0) / (u_com + config.EPS)
        self.u = u_com * gamma
        w = self.omega * np.exp(config.BETA * self.u)  # Gibbs posterior
        self.omega = w / w.sum()
        self.t += 1
        self.log.append(RoundLog(self.t, name, self.elite_min, float(fitness.max()),
                                 [(self.terms[i].key, float(self.omega[i])) for i in np.argsort(-self.omega)[:10]]))
        return self.guidance()

    def guidance(self) -> list[GuidanceTerm]:
        k = max(1, int(round(config.GUIDE_RATIO * len(self.terms))))
        top = np.argsort(-self.omega, kind="stable")[:k]
        return [GuidanceTerm(term=str(self.terms[i].expr), weight=float(self.omega[i]),
                             display=self.terms[i].display) for i in top]

    def ranked_basis(self) -> list[GuidanceTerm]:
        """The whole basis B by decreasing omega, for a solver that fits over all of it."""
        return [GuidanceTerm(term=str(self.terms[i].expr), weight=float(self.omega[i]),
                             display=self.terms[i].display) for i in np.argsort(-self.omega, kind="stable")]

    # -- reshaping from solver feedback ---------------------------------------
    def reshape(self, candidates: list[Candidate]) -> None:
        """Keep solver expressions at least as fit as the elite set's minimum
        (scored like sampled expressions, in this round's environment) and replace the
        lowest-utility basis terms by new terms of higher utility."""
        if self.env is None or not candidates:
            return
        split = []
        for c in candidates:
            e = parse(c.expr, self.task.inputs) if c.expr else None
            parts = split_terms(e, self.task.inputs) if e is not None else []
            if parts:
                split.append(parts)
        scores = self.env.fitness_many(split, self._pool(), self.workers)
        retained = [(parts, f) for parts, f in zip(split, scores) if f >= self.elite_min]
        log = self.log[-1] if self.log else None
        if log is not None:
            log.n_feedback, log.n_retained = len(candidates), len(retained)
        known = {t.key for t in self.terms}
        total: dict[str, float] = {}
        count: dict[str, int] = {}
        new_terms: dict[str, Term] = {}
        for parts, f in retained:
            for t in parts:
                if t.key in known or not admissible_new_term(t):
                    continue
                new_terms[t.key] = t
                total[t.key] = total.get(t.key, 0.0) + f / len(parts)
                count[t.key] = count.get(t.key, 0) + 1
        ranked_new = sorted(new_terms, key=lambda k: -total[k] / count[k])
        order_old = list(np.argsort(self.u, kind="stable"))
        uniform = 1.0 / len(self.terms)
        for key in ranked_new[: config.MAX_REPLACE_PER_ROUND]:
            u_new = total[key] / count[key]
            i = order_old.pop(0) if order_old else None
            if i is None or u_new <= self.u[i]:
                break
            if log is not None:
                log.replaced.append((self.terms[i].key, key))
            self.terms[i] = new_terms[key]
            self.omega[i] = config.OMEGA0 * uniform * math.exp(config.BETA * u_new)
            self.u[i] = u_new
            for e in config.ENVIRONMENTS:
                self.u_env[e][i] = u_new
        self.omega = self.omega / self.omega.sum()

    def state(self) -> dict:
        return {
            "round": self.t,
            "terms": [{"key": t.key, "display": t.display, "omega": float(w), "u": float(u)}
                      for t, w, u in zip(self.terms, self.omega, self.u)],
            "log": [vars(r) for r in self.log],
        }
