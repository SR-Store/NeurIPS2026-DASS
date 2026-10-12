import numpy as np
import pytest
import sympy as sp

from dass import config
from dass.interfaces import GuidanceTerm
from dass.solvers.pysr_solver import MAX_GUESSES, build_guesses, complexity, operators, render, share
from dass.terms import parse, term_from_text

PHYS = ("x", "t", "v")


def guidance(texts, inputs=PHYS):
    terms = [term_from_text(s, inputs) for s in texts]
    n = len(terms)
    return [GuidanceTerm(term=str(t.expr), weight=(n - i) / n, display=t.display) for i, t in enumerate(terms)]


def test_share_splits_the_budget():
    assert [share(65, s, 3) for s in range(3)] == [22, 22, 21]
    assert sum(share(10, s, 3) for s in range(3)) == 10


@pytest.mark.parametrize("text,rendered,size", [
    ("1.0*x + 1.0*x**3", "1.0*x^3 + 1.0*x", 9),
    ("1.0*exp(-1.0*t)", "1.0*exp(-1.0*t)", 6),
    ("1.0*x*exp(-Abs(x))", "1.0*x*exp(-1.0*abs(x))", 9),
    ("1.0*sqrt(x)", "1.0*sqrt(x)", 4),
    ("1.0*x/(1.0 + x)", "1.0*x/(x + 1.0)", 7),
])
def test_render_and_complexity(text, rendered, size):
    e = parse(text, PHYS)
    assert render(e) == rendered
    assert complexity(render(e)) == size


def test_operator_scan_ignores_literal_signs():
    assert operators("-1.0*x + exp(-2.5*t)") == ({"*", "+"}, {"exp"})
    assert operators("x - abs(v)^2.0") == ({"-", "^"}, {"abs"})


def data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(0, 10, n), rng.uniform(-1, 1, n)])
    return X, -0.8 * X[:, 0] - 0.5 * X[:, 0] ** 3 - 0.3 * X[:, 2]


def nmse_of(guess, X, y):
    e = parse(guess.replace("^", "**"), PHYS)
    f = sp.lambdify([sp.Symbol(n, real=True) for n in PHYS], e, "numpy")
    return float(np.mean((np.broadcast_to(f(*X.T), y.shape) - y) ** 2) / np.var(y))


def n_terms(guess):
    return len(sp.Add.make_args(parse(guess.replace("^", "**"), PHYS)))


def test_guesses_singles_then_fitted_combinations_within_maxsize():
    X, y = data()
    g = guidance(["x", "x**3", "v", "x*exp(-Abs(x))", "sin(omega*t)", "tanh(x)"])
    guesses, skipped = build_guesses(g, PHYS, X, y)
    assert skipped == [{"term": "tanh(x)", "reason": "operator tanh"}]
    assert [n_terms(s) for s in guesses[:5]] == [1] * 5
    assert len(guesses) == len(set(guesses)) <= MAX_GUESSES
    binary_ok, unary_ok = set(config.PYSR["binary_operators"]), set(config.PYSR["unary_operators"])
    for s in guesses:
        b, u = operators(s)
        assert complexity(s) <= config.PYSR["maxsize"] and b <= binary_ok and u <= unary_ok
    # the leading constants are fitted: the combination of the true terms is exact
    assert min(nmse_of(s, X, y) for s in guesses) < 1e-12
    # the first combination holds all five usable terms (complexity 30 at most)
    assert n_terms(guesses[5]) == 5


def test_guess_placeholders_are_fitted():
    X, _ = data()
    y = 1.5 * np.exp(-0.3 * X[:, 1])
    guesses, _ = build_guesses(guidance(["exp(alpha*t)"]), PHYS, X, y)
    assert nmse_of(guesses[0], X, y) < 1e-10


def test_terms_that_cannot_be_guesses_are_logged():
    X, y = data()
    guesses, skipped = build_guesses(guidance(["sqrt(x)", "sign(x)", "v"]), PHYS, X, y)
    assert [s["reason"] for s in skipped] == ["operator sign", "not finite on the data"]
    assert len(guesses) == 1


def test_guesses_are_capped():
    X, y = data()
    guesses, _ = build_guesses(guidance(["x", "x**2", "x**3", "v", "v**2", "x*v", "t", "sin(t)"]), PHYS, X, y)
    assert len(guesses) == MAX_GUESSES


@pytest.mark.integration
def test_pysr_stages_accept_guesses(tmp_path):
    pytest.importorskip("pysr")
    from dass.data import Task
    from dass.solvers.pysr_solver import PySRAdapter

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-2, 2, 300), rng.uniform(0, 10, 300), rng.uniform(-1, 1, 300)])
    y = -0.8 * X[:, 0] - 0.5 * X[:, 0] ** 3
    task = Task("SYN", "phys_osc", "dv_dt", PHYS, "", PHYS, X, y, X, y, X, y)
    config.PYSR.update(niterations=2, populations=2)
    try:
        a = PySRAdapter(threads=2)
        a.start(task, 0, tmp_path)
        cands = a.run_stage(0, 2, guidance(["x", "x**3"]))
        assert a.cost()["stages"][0]["n_guesses"] == 3 and cands[0].train_nmse < 1e-8
        a.run_stage(1, 2, None)
        assert a.final().expr
    finally:
        config.PYSR.update(niterations=65, populations=15)


@pytest.mark.parametrize("configured", [0, 2, 3])
def test_start_uses_explicit_julia_environment_only_when_complete(tmp_path, monkeypatch, configured):
    """A regular PySR install does not need the managed Julia depot variables."""
    import sys
    from types import ModuleType, SimpleNamespace
    from dass.data import Task
    from dass.solvers import pysr_solver

    class Regressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    pysr = ModuleType("pysr")
    pysr.PySRRegressor = Regressor
    julia_import = ModuleType("pysr.julia_import")
    julia_import.jl = SimpleNamespace(Threads=SimpleNamespace(nthreads=lambda: 2))
    monkeypatch.setitem(sys.modules, "pysr", pysr)
    monkeypatch.setitem(sys.modules, "pysr.julia_import", julia_import)
    monkeypatch.setattr(pysr_solver, "_warm", True)
    monkeypatch.delenv("DASS_SCRATCH", raising=False)
    monkeypatch.delenv("PYTHON_JULIAPKG_OFFLINE", raising=False)
    keys = ("PYTHON_JULIAPKG_EXE", "PYTHON_JULIAPKG_PROJECT", "JULIA_DEPOT_PATH")
    values = ("/opt/julia", "/tmp/julia-project", "/tmp/julia-depot")
    for i, (key, value) in enumerate(zip(keys, values)):
        if i < configured:
            monkeypatch.setenv(key, value)
        else:
            monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr(pysr_solver, "precompile_once", lambda *args: calls.append(args) or 1.5)
    X, y = data(n=12)
    task = Task("SYN", "phys_osc", "dv_dt", PHYS, "", PHYS, X, y, X, y, X, y)

    adapter = pysr_solver.PySRAdapter(threads=2)
    adapter.start(task, 0, tmp_path)

    assert adapter.julia_threads == 2
    assert adapter.model.kwargs["warm_start"] is True
    if configured == 3:
        assert calls == [("pysr", *values, tmp_path)]
        assert adapter.precompile_s == 1.5
        assert pysr_solver.os.environ["PYTHON_JULIAPKG_OFFLINE"] == "yes"
    else:
        assert calls == []
        assert adapter.precompile_s is None
        assert "PYTHON_JULIAPKG_OFFLINE" not in pysr_solver.os.environ


def test_guess_fitting_time_is_attributed_to_dass(monkeypatch):
    from types import SimpleNamespace
    from dass.solvers import pysr_solver

    clock = SimpleNamespace(now=0.0)

    def build(*args):
        clock.now += 3.0  # DASS fits the guesses before PySR's search starts.
        return ["x"], []

    def fit(*args, **kwargs):
        clock.now += 7.0  # The solver's own search time stays outside inject_s.

    monkeypatch.setattr(pysr_solver.time, "time", lambda: clock.now)
    monkeypatch.setattr(pysr_solver, "build_guesses", build)
    adapter = object.__new__(pysr_solver.PySRAdapter)
    adapter.task = SimpleNamespace(inputs=PHYS, X_train=np.zeros((2, 3)), y_train=np.zeros(2))
    adapter.X_fit, adapter.y_fit = adapter.task.X_train, adapter.task.y_train
    adapter.fit_kwargs = {}
    adapter.model = SimpleNamespace(set_params=lambda **kwargs: None, fit=fit)
    adapter._hall_of_fame = lambda: []
    adapter.stages = []
    adapter.inject_s = 0.0
    adapter.julia_threads = 1
    adapter.precompile_s = None

    adapter.run_stage(0, 2, guidance(["x"]))
    adapter.run_stage(1, 2, None)

    assert adapter.cost()["inject_s"] == 3.0
    assert [stage["fit_s"] for stage in adapter.cost()["stages"]] == [7.0, 7.0]
    assert adapter.cost()["stages"][1]["guess_s"] == 0.0
