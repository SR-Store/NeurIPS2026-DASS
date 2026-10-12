"""LLM-SR search with DASS guidance and fitted program injection.

The search uses LLM-SRBench components bundled in ``dass._vendor.llmsr``: the task specification
built from the symbol descriptions as ``LLMSRSearcher.discover`` builds it, the
``ExperienceBuffer`` with its islands and clusters, the ``Evaluator`` with
``LocalSandbox`` subprocess execution and BFGS fitting of MAX_NPARAMS=10 params,
``_extract_body``, and ``LocalLLM``'s request (its instruction, then the island
prompt, as one user message).  This module replaces only upstream's endless
``Sampler.sample`` loop by one that stops at a sample count, so the budget is spent
in T stages and the buffer is kept from one stage to the next:

* stage t of T spends samples [N*t//T, N*(t+1)//T) in prompts of 4; the last prompt
  of a stage is shorter when the share is not a multiple of 4 (N=100, T=9: 11 or 12
  samples a stage, prompts of 4, 4, 3 or 4, 4, 4);
* with guidance, every prompt of the stage carries the Reference Terms block between
  the problem preface and the first example program;
  without, the prompt is upstream's byte for byte;
* with guidance, the stage first puts DASS's best fitted combinations into the buffer
  as programs (``build_programs``), on every island, the way PySR's +DASS gets them as
  guesses: combinations of the guidance terms, and the forward-selection path over the
  whole basis B that DASS hands over through ``set_basis``; they are DASS's, not
  samples, and do not count against the sample budget;
* the samples of a prompt are requested concurrently.  ``prompts_in_flight`` > 1
  draws the next prompts from the buffer before earlier ones are registered, as
  upstream's several-samplers mode would; the default 1 is LLM-SRBench's single
  sampler.  Samples are registered in the order they were requested, so a run depends
  only on the seed and on what the model returns.

Candidates are the best distinct programs by upstream score.  Their params come from
upstream's ``optimization_spec`` (the BFGS from all-ones that LLM-SRBench reruns on
its final program; deterministic, so the same params the score was computed with),
run in-process on programs the sandbox already ran.  ``predict`` is the program with
those params; ``expr`` transcribes straight-line numpy code into sympy and is kept
only when it reproduces the program on training rows, else it is "".
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import re
import signal
import sys
import threading
import time
import warnings
from collections import Counter, deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from .. import config
from ..data import Task
from ..fitting import fit, nmse
from ..interfaces import Candidate, GuidanceTerm
from ..llm import LLMClient
from ..terms import canonical_term, input_symbols, parse

VENDOR = Path(__file__).resolve().parents[1] / "_vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from llmsr import buffer, code_manipulation, evaluator, sampler, specs  # noqa: E402
from llmsr import config as llmsr_config  # noqa: E402

EVOLVE, RUN = "equation", "evaluate"  # the two decorated functions of the specification
# LLM-SRBench's instruction plus a request to reason briefly before completing the function.
INSTRUCTION = ("You are a helpful assistant tasked with discovering mathematical function structures for "
               "scientific systems. Complete the `equation` function below, considering the physical meaning "
               "and relationships of inputs. Explain your reasoning briefly before completing the function. "
               "Let's think step by step.\n\n")
_VERSION_DEF = re.compile(rf"^def {EVOLVE}_v\d+\(", re.M)


def extract_body(text: str, cfg) -> str:
    """Upstream ``sampler._extract_body`` applied from the reply's last ``equation_vN`` definition.

    Upstream takes the body of the first ``def`` in a reply.  Some replies echo the
    prompt's initial ``equation_v0`` before defining a new ``equation_v1``.  Taking the
    last versioned definition avoids treating that echo as the proposed program.  A
    reply with at most one version definition is handled exactly as upstream."""
    defs = list(_VERSION_DEF.finditer(text))
    return sampler._extract_body(text[defs[-1].start():] if len(defs) > 1 else text, cfg)


MAX_CANDIDATES = 50
CHECK_ROWS = 256  # training rows on which a transcribed expr must reproduce the program


# --- prompt -------------------------------------------------------------------
def specification(task: Task) -> tuple[str, str]:
    """``(specification, equation_specification)`` as ``LLMSRSearcher.discover`` writes them.

    LSR-Synth symbols are plain identifiers, so upstream's LaTeX clean-up of the
    names is a no-op and the names are ``task.inputs`` as they are.
    """
    var_name = ["output", *task.inputs]
    var_desc = [task.output_desc, *task.input_descs]
    if len(var_desc) > 2:
        input_desc = ", ".join(var_desc[1:-1]) + ", and " + var_desc[-1]
    else:
        input_desc = var_desc[-1]
    spec = ('"""\n' + f"Find the mathematical function skeleton that represents {var_desc[0]}, "
            f"given data on {input_desc}.\n" + '"""\n' + specs.eval_spec)
    inputs = var_name[1:]
    equation = (
        "@equation.evolve\n"
        + "def equation(" + ", ".join(f"{n}: np.ndarray" for n in inputs) + ", params: np.ndarray) -> np.ndarray:\n"
        + f'    """ Mathematical function for {var_desc[0]}\n\n'
        + "    Args:\n"
        + "\n".join(f"        {n}: A numpy array representing observations of {d}."
                    for n, d in zip(inputs, var_desc[1:])) + "\n"
        + "        params: Array of numeric constants or parameters to be optimized\n\n"
        + "    Return:\n"
        + f"        A numpy array representing {var_desc[0]} as the result of applying the mathematical function "
        "to the inputs.\n"
        + '    """\n'
        + f"    {var_name[0]} = " + " + ".join(f"params[{i}] * {n}" for i, n in enumerate(inputs))
        + f" + params[{len(inputs)}]\n"
        + f"    return {var_name[0]}"
    )
    return spec + equation, equation


REFERENCE_HEADER = "# ===== Reference Terms (DASS) ====="


def reference_terms(guidance: list[GuidanceTerm]) -> str:
    """The Reference Terms (DASS) prompt block."""
    width = max(16, max((len(g.display) for g in guidance), default=0))
    lines = [REFERENCE_HEADER,
             f"# Top {len(guidance)} of {config.BASIS_SIZE} basis-library terms, ranked by Bayesian posterior; "
             "probabilities sum to 1 across the full library.",
             "#"]
    lines += [f"#   {i}. {g.display:<{width}} (p={g.weight:.2f})" for i, g in enumerate(guidance, 1)]
    lines += ["#", "# Use as inspiration; combine, modify, or ignore. Focus on physical meaning.",
              "# " + "=" * (len(REFERENCE_HEADER) - 2)]
    return "\n".join(lines)


def with_guidance(code: str, guidance: list[GuidanceTerm] | None) -> str:
    """The island prompt with the block before the first example program (``equation_v0``)."""
    if guidance is None:
        return code
    i = code.index(f"\ndef {EVOLVE}_v0(") + 1
    return code[:i] + reference_terms(guidance) + "\n\n\n" + code[i:]


class _NumpyCode(NumPyPrinter):
    """numpy source as the programs write it; Max and Min elementwise."""

    def _fold(self, name, e):
        out = self._print(e.args[0])
        for a in e.args[1:]:
            out = f"numpy.{name}({out}, {self._print(a)})"
        return out

    def _print_Max(self, e):
        return self._fold("maximum", e)

    def _print_Min(self, e):
        return self._fold("minimum", e)


def numpy_code(e: sp.Expr) -> str:
    return _NumpyCode({"fully_qualified_modules": True}).doprint(e).replace("numpy.", "np.")


def _parsed(gs: list[GuidanceTerm], inputs) -> tuple[list, list[float]]:
    terms, weights = [], []
    for g in gs:
        e = parse(g.term, inputs)
        t = canonical_term(e, inputs) if e is not None else None
        if t is not None:
            terms.append(t)
            weights.append(g.weight)
    return terms, weights


def _loss(terms: list, X: np.ndarray, y: np.ndarray):
    f = fit(terms, X, y)
    with np.errstate(all="ignore"):
        loss = nmse(f.predict(X), y)
    return (loss if math.isfinite(loss) else math.inf), f


def stepwise(terms: list, X: np.ndarray, y: np.ndarray, k_max: int) -> list[tuple[float, list, object]]:
    """Forward selection: each step adds the term that lowers the training NMSE most;
    every prefix of the path (1 .. k_max terms) with its fit."""
    chosen, path = [], []
    for _ in range(min(k_max, len(terms))):
        best = (math.inf, None, None)
        for i in range(len(terms)):
            if i not in chosen:
                loss, f = _loss([terms[j] for j in chosen + [i]], X, y)
                if loss < best[0]:
                    best = (loss, i, f)
        if best[1] is None:
            break
        chosen.append(best[1])
        path.append((best[0], [terms[j] for j in chosen], best[2]))
    return path


def build_programs(guidance: list[GuidanceTerm], inputs, X: np.ndarray, y: np.ndarray,
                   basis: list[GuidanceTerm] | None = None) -> list[str]:
    """Bodies of the evolved function for DASS's best fitted combinations.

    Two sources of combinations: the guidance terms, taken in the order PySR's guesses
    are (each term alone, then larger sets by decreasing total omega, at most
    MAX_TERMS_PER_SAMPLE terms; the first ``inject_from`` are fitted), and, given the
    whole basis B, the forward-selection path over B up to ``stepwise_max`` terms.  All
    are fitted on (X, y) with an intercept and the ``max_inject`` with the lowest NMSE
    kept.  The constants inside the terms are written in as fitted; each term and the
    intercept get a params weight scaled so that all-ones, where LLM-SR's BFGS starts,
    is the fitted combination.
    """
    terms, weights = _parsed(guidance, inputs)
    combos = [c for r in range(1, min(config.MAX_TERMS_PER_SAMPLE, len(terms)) + 1)
              for c in itertools.combinations(range(len(terms)), r)]
    combos = combos[:len(terms)] + sorted(combos[len(terms):], key=lambda c: (-sum(weights[i] for i in c), len(c)))
    fitted = {}
    for c in combos[: config.LLMSR["inject_from"]]:
        loss, f = _loss([terms[i] for i in c], X, y)
        fitted[tuple(sorted(terms[i].key for i in c))] = (loss, [terms[i] for i in c], f)
    if basis and config.LLMSR["stepwise_max"] > 0:
        for loss, ts, f in stepwise(_parsed(basis, inputs)[0], X, y, config.LLMSR["stepwise_max"]):
            fitted.setdefault(tuple(sorted(t.key for t in ts)), (loss, ts, f))
    bodies = []
    for _, ts, f in sorted((v for v in fitted.values() if math.isfinite(v[0])), key=lambda r: r[0])[: config.LLMSR["max_inject"]]:
        parts = [f"params[{j}] * {float(b)!r} * ({numpy_code(t.at(th))})"
                 for j, (t, b, th) in enumerate(zip(ts, f.coefs[1:], f.thetas))]
        b0 = float(f.coefs[0])
        parts.append(f"params[{len(ts)}] * {b0!r}" if b0 != 0 else f"params[{len(ts)}]")
        bodies.append(f"    output = {' + '.join(parts)}\n    return output\n")
    return bodies


def stage_prompts(stage: int, n_stages: int, total: int, per_prompt: int) -> list[int]:
    """Sizes of the prompts of one stage: its contiguous share of ``total`` samples."""
    share = total * (stage + 1) // n_stages - total * stage // n_stages
    return [per_prompt] * (share // per_prompt) + ([share % per_prompt] if share % per_prompt else [])


# --- program -> sympy -----------------------------------------------------------
def _same(a, *_, **__):
    return a


def _filled(value):
    def make(shape, *_, **__):
        if isinstance(shape, (int, np.integer)):
            return np.array([value] * int(shape), dtype=object)
        return value  # a *_like of an input: the constant itself

    return make


def _clip(a, lo, hi, *_, **__):
    a = a if lo is None else sp.Max(a, lo)
    return a if hi is None else sp.Min(a, hi)


# numpy (and math) as the equation programs use them, over sympy expressions
_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "absolute": sp.Abs, "fabs": sp.Abs, "tanh": sp.tanh, "sinh": sp.sinh,
    "cosh": sp.cosh, "arctan": sp.atan, "atan": sp.atan, "arcsin": sp.asin, "asin": sp.asin,
    "arccos": sp.acos, "acos": sp.acos, "sign": sp.sign,
    "square": lambda a: a ** 2, "power": lambda a, b: a ** b, "pow": lambda a, b: a ** b,
    "log10": lambda a: sp.log(a) / sp.log(10), "log2": lambda a: sp.log(a) / sp.log(2),
    "log1p": lambda a: sp.log(1 + a), "expm1": lambda a: sp.exp(a) - 1,
    "reciprocal": lambda a: 1 / a, "cbrt": lambda a: sp.sign(a) * sp.Abs(a) ** sp.Rational(1, 3),
    "maximum": sp.Max, "minimum": sp.Min, "fmax": sp.Max, "fmin": sp.Min, "clip": _clip,
    "divide": lambda a, b, *_, **__: a / b, "true_divide": lambda a, b, *_, **__: a / b,
    "multiply": lambda a, b, *_, **__: a * b, "add": lambda a, b, *_, **__: a + b,
    "subtract": lambda a, b, *_, **__: a - b, "negative": lambda a, *_, **__: -a,
    "asarray": _same, "array": _same, "asanyarray": _same, "atleast_1d": _same, "squeeze": _same,
    "ravel": _same, "copy": _same, "float64": _same, "float32": _same, "float_": _same,
    "zeros_like": lambda a, *_, **__: sp.Integer(0), "ones_like": lambda a, *_, **__: sp.Integer(1),
    "full_like": lambda a, c, *_, **__: c, "zeros": _filled(sp.Integer(0)), "ones": _filled(sp.Integer(1)),
    "concatenate": lambda seq, *_, **__: np.concatenate([np.asarray(x, dtype=object).ravel() for x in seq]),
    # guards on the data: a symbolic input is neither nan nor infinite
    "isnan": lambda a: False, "isinf": lambda a: False, "isfinite": lambda a: True,
    "any": lambda a, *_, **__: bool(np.any(a)), "all": lambda a, *_, **__: bool(np.all(a)),
    "pi": sp.pi, "e": sp.E, "inf": sp.oo, "ndarray": object, "float": float,
}
_NP = type("SymbolicNumpy", (), {k: staticmethod(v) if callable(v) and not isinstance(v, type) else v
                                 for k, v in _FUNCS.items()})


def _import(name, *_, **__):
    if name in ("numpy", "math"):
        return _NP
    raise ImportError(name)


_BUILTINS = {**{k: getattr(__builtins__, k) if not isinstance(__builtins__, dict) else __builtins__[k]
                for k in ("len", "range", "int", "float", "abs", "min", "max", "sum", "enumerate", "zip",
                          "isinstance", "tuple", "list", "ValueError", "TypeError", "Exception", "print")},
             "__import__": _import}


@contextmanager
def _time_limit(seconds: float):
    """A program that loops on its params could run long; stop it (main thread only)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def expire(*_):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def program_to_sympy(source: str, inputs, params) -> sp.Expr | None:
    """``equation`` with fitted ``params`` inlined, as a sympy expression over ``inputs``.

    The program is run once with sympy symbols for the inputs, the fitted params (a
    numpy object array of sympy floats) and a sympy stand-in for numpy/math, so length
    checks on params, helper functions, branches and loops on params, and ``np.clip``
    (as ``Max``/``Min``) come out as they execute.  Branches or loops on the data,
    ``np.where``, attributes of the inputs and anything outside the stand-in fail, and
    the result is None.
    ``transcribe`` still checks the expression against the program's own predictions.
    """
    try:
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    except (SyntaxError, StopIteration):
        return None
    if len(fn.args.args) != len(inputs) + 1:
        return None
    fn.decorator_list = []
    ns = {"np": _NP, "numpy": _NP, "math": _NP, "__builtins__": _BUILTINS}
    args = [*input_symbols(inputs).values(), np.array([sp.Float(float(p)) for p in params], dtype=object)]
    try:
        with _time_limit(5.0), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(ast.Module(body=[fn], type_ignores=[]), "<equation>", "exec"), ns)
            out = ns[fn.name](*args)
            if isinstance(out, np.ndarray):
                if out.size != 1:
                    return None
                out = out.reshape(-1)[0]
            e = sp.sympify(out)
            if not isinstance(e, sp.Expr) or e.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.I):
                return None
            # sympy's floats have unbounded exponents: extreme fitted params can give numbers
            # like 1e+217147, whose string sympy then reads back by building the exact integer
            # (hours of CPU); outside float64 the expression cannot match the program anyway
            if any(a != 0 and not 0 < abs(float(a)) < math.inf for a in e.atoms(sp.Float)):
                return None
            return e.xreplace({sp.E: sp.Float(math.e)})  # "E" would read back as a coefficient; pi stays pi
    except Exception:  # anything the stand-in does not cover
        return None


def transcribe(source: str, inputs, params, predict, X: np.ndarray) -> str:
    """``str`` of ``program_to_sympy``, or "" unless that string, read back by
    ``dass.terms.parse``, reproduces ``predict`` on ``X`` (where the program is finite)."""
    e = program_to_sympy(source, inputs, params)
    if e is None:
        return ""
    text = str(e)
    back = parse(text, inputs)
    if back is None:
        return ""
    try:
        f = sp.lambdify(list(input_symbols(inputs).values()), back, modules="numpy")
        with np.errstate(all="ignore"):
            want = predict(X)
            got = np.broadcast_to(np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), dtype=float), want.shape)
    except Exception:
        return ""
    ok = np.isfinite(want)
    if not ok.any() or not np.all(np.isfinite(got[ok])):
        return ""
    scale = float(np.max(np.abs(want[ok])))
    return text if np.allclose(got[ok], want[ok], rtol=1e-6, atol=1e-9 * scale) else ""


# --- adapter --------------------------------------------------------------------
class _Trajectory:
    """Takes the place of upstream's Profiler: every evaluated program, in sample order
    (the initial template program has sample order None)."""

    def __init__(self):
        self.programs: list[code_manipulation.Function] = []
        self._seen: set = set()
        self.best_score = -math.inf

    def register_function(self, fn: code_manipulation.Function) -> None:
        if fn.global_sample_nums in self._seen:
            return
        self._seen.add(fn.global_sample_nums)
        self.programs.append(fn)
        if fn.score is not None and fn.score > self.best_score:
            self.best_score = fn.score


class LLMSRAdapter:
    name = "llmsr"

    def __init__(self, endpoints: list[str], model: str, prompts_in_flight: int = 1,
                 api_key: str | None = None):
        self.client = LLMClient(endpoints, model, sampling=config.LLM_SAMPLING, api_key=api_key)
        self.prompts_in_flight = max(1, int(prompts_in_flight))
        self.max_samples = config.LLMSR["max_samples"]
        self.per_prompt = config.LLMSR["samples_per_prompt"]

    def start(self, task: Task, seed: int, workdir: Path) -> None:
        """Upstream ``pipeline.main`` up to the sampling loop: template, buffer,
        evaluator, and the template's own program scored and put on every island."""
        np.random.seed(seed)  # upstream's buffer draws from the global numpy RNG
        self.task = task
        spec, equation = specification(task)
        self.template = code_manipulation.text_to_program(spec)
        self.opt_template = code_manipulation.text_to_program(specs.optimization_spec + equation)
        self.cfg = llmsr_config.Config(
            experience_buffer=llmsr_config.ExperienceBufferConfig(num_islands=config.LLMSR["num_islands"]),
            samples_per_prompt=self.per_prompt, use_api=True)
        self.database = buffer.ExperienceBuffer(self.cfg.experience_buffer, self.template, EVOLVE)
        k = task.X_train.shape[1]
        self.data = {"inputs": [task.X_train[:, i] for i in range(k)], "outputs": task.y_train}
        self.evaluators = [evaluator.Evaluator(self.database, self.template, EVOLVE, RUN, {"data": self.data},
                                               timeout_seconds=self.cfg.evaluate_timeout_seconds,
                                               sandbox_class=evaluator.LocalSandbox)
                           for _ in range(self.cfg.num_evaluators)]
        self.trajectory = _Trajectory()
        self.var_y = float(np.var(task.y_train))
        self.n_samples = self.n_prompts = 0
        self.finish: Counter = Counter()
        self._built: dict = {}
        self.n_injected = 0
        self.inject_s = 0.0  # fitting, scoring, and registering injected DASS programs
        self._injected: set[str] = set()
        self._basis: list[GuidanceTerm] | None = None
        rows = np.random.default_rng(seed).permutation(len(task.y_train))[: config.FIT_ROWS]
        self._fit_X, self._fit_y = task.X_train[rows], task.y_train[rows]  # as DASS fits: FIT_ROWS rows
        initial = self.template.get_function(EVOLVE).body
        self.evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=self.trajectory)
        workdir.mkdir(parents=True, exist_ok=True)
        self.trace = (workdir / "trace.jsonl").open("w")  # every reply, what was taken from it, its score

    def run_stage(self, stage: int, n_stages: int, guidance: list[GuidanceTerm] | None) -> list[Candidate]:
        if guidance:
            self._inject(stage, guidance)
        todo = deque(stage_prompts(stage, n_stages, self.max_samples, self.per_prompt))
        pending: deque = deque()
        with ThreadPoolExecutor(self.prompts_in_flight * self.per_prompt) as pool:

            def issue():
                if self.trajectory.best_score == 0:  # upstream's sampler stops on a perfect score
                    todo.clear()
                    return
                prompt = self.database.get_prompt()
                text = "\n".join([INSTRUCTION, with_guidance(prompt.code, guidance)])
                messages = [{"role": "user", "content": text}]
                pending.append((prompt, [pool.submit(self.client.chat, messages) for _ in range(todo.popleft())]))
                self.n_prompts += 1

            while todo and len(pending) < self.prompts_in_flight:
                issue()
            while pending:
                prompt, futures = pending.popleft()
                for fut in futures:
                    reply = fut.result()
                    self.finish[str(reply.finish_reason)] += 1
                    self.n_samples += 1
                    chosen = np.random.choice(self.evaluators)
                    body = extract_body(reply.text, self.cfg)
                    chosen.analyse(body, prompt.island_id, prompt.version_generated, profiler=self.trajectory,
                                   global_sample_nums=self.n_samples)
                    last = self.trajectory.programs[-1]
                    score = last.score if last.global_sample_nums == self.n_samples else None
                    self.trace.write(json.dumps({"sample": self.n_samples, "stage": stage, "island": prompt.island_id,
                                                 "finish": reply.finish_reason, "score": score,
                                                 "reply": reply.text, "body": body}) + "\n")
                    self.trace.flush()
                if todo:
                    issue()
        return self._best()

    def set_basis(self, basis: list[GuidanceTerm]) -> None:
        """DASS's whole basis B by omega, handed over before a +DASS stage."""
        self._basis = basis

    def _inject(self, stage: int, guidance: list[GuidanceTerm]) -> None:
        """DASS's fitted combinations onto every island, scored by upstream's evaluator;
        numbered -1, -2, ... apart from the samples."""
        t0 = time.time()
        try:
            bodies = build_programs(guidance, self.task.inputs, self._fit_X, self._fit_y, self._basis)
            for body in bodies:
                if body in self._injected:
                    continue
                self._injected.add(body)
                self.n_injected += 1
                gid = -self.n_injected
                self.evaluators[0].analyse(body, None, None, profiler=self.trajectory, global_sample_nums=gid)
                last = self.trajectory.programs[-1]
                score = last.score if last.global_sample_nums == gid else None
                self.trace.write(json.dumps({"sample": gid, "stage": stage, "injected": True, "score": score,
                                             "body": body}) + "\n")
            self.trace.flush()
        finally:
            self.inject_s += time.time() - t0

    def final(self) -> Candidate:
        """Upstream's answer: the highest score over the run (earliest on ties)."""
        ranked = self._ranked()
        if not ranked:
            return Candidate("", math.inf)
        c = self._candidate(ranked[0])
        return c if c is not None else Candidate("", -ranked[0].score / self.var_y, source=str(ranked[0]))

    def cost(self) -> dict:
        drawn = [f for f in self.trajectory.programs if (f.global_sample_nums or 0) > 0]
        injected = [f for f in self.trajectory.programs if (f.global_sample_nums or 0) < 0]
        return {**self.client.usage(), "samples": self.n_samples, "prompts": self.n_prompts,
                "prompts_in_flight": self.prompts_in_flight,
                "programs_scored": sum(f.score is not None for f in drawn),
                "programs_failed": sum(f.score is None for f in drawn),
                "injected": len(injected), "injected_scored": sum(f.score is not None for f in injected),
                "inject_s": round(self.inject_s, 3),
                "finish_reasons": dict(self.finish)}

    # -- candidates ----------------------------------------------------------------
    def _ranked(self) -> list[code_manipulation.Function]:
        scored = [f for f in self.trajectory.programs if f.score is not None]
        return sorted(scored, key=lambda f: (-f.score, f.global_sample_nums or 0))

    def _best(self) -> list[Candidate]:
        out, seen = [], set()
        for f in self._ranked():
            source = str(f)
            if source in seen:
                continue
            seen.add(source)
            c = self._candidate(f)
            if c is not None:
                out.append(c)
            if len(out) == MAX_CANDIDATES:
                break
        return out

    def _candidate(self, f: code_manipulation.Function) -> Candidate | None:
        key = f.global_sample_nums or 0
        if key not in self._built:
            self._built[key] = self._build(f)
        return self._built[key]

    def _exec(self, template: code_manipulation.Program, body: str) -> dict:
        _, program = evaluator._sample_to_program(body, None, template, EVOLVE)
        namespace: dict = {}
        exec(program, namespace)
        return namespace

    def _build(self, f: code_manipulation.Function) -> Candidate | None:
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = np.asarray(self._exec(self.opt_template, f.body)["optimize"](self.data), dtype=float)
            equation = self._exec(self.template, f.body)[EVOLVE]
        except Exception:
            return None
        k = len(self.task.inputs)

        def predict(X: np.ndarray) -> np.ndarray:
            with np.errstate(all="ignore"):
                out = equation(*[X[:, i] for i in range(k)], params)
            return np.array(np.broadcast_to(np.asarray(out, dtype=float), (X.shape[0],)))

        source = str(f)
        expr = transcribe(source, self.task.inputs, params, predict, self.task.X_train[:CHECK_ROWS])
        return Candidate(expr=expr, train_nmse=-f.score / self.var_y, predict=predict, source=source)
