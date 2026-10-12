import json

import numpy as np

from dass import config, runner
from dass.core import DASS
from dass.data import Task
from dass.interfaces import Candidate
from dass.terms import term_from_text

PHYS = ("x", "t", "v")


def synthetic_task(seed=0) -> Task:
    rng = np.random.default_rng(seed)

    def draw(n, hi):
        X = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(0, hi, n), rng.uniform(-1, 1, n)])
        return X, -0.8 * X[:, 0] - 0.5 * X[:, 0] ** 3 - 0.3 * X[:, 2]

    Xtr, ytr = draw(4000, 10)
    Xid, yid = draw(500, 10)
    Xood, yood = draw(500, 15)
    return Task("SYN1", "phys_osc", "dv_dt", PHYS, "acceleration", ("x", "t", "v"),
                Xtr, ytr, Xid, yid, Xood, yood, truth="-0.8*x - 0.5*x**3 - 0.3*v")


def basis(texts):
    terms = [term_from_text(s, PHYS) for s in texts]
    return [(t, 1.0) for t in terms]


def test_calibration_moves_mass_to_true_terms(monkeypatch):
    monkeypatch.setattr(config, "M_SAMPLES", 60)
    task = synthetic_task()
    d = DASS(task, basis(["x", "x**3", "v", "sin(omega*t)", "exp(alpha*t)", "t**2", "cos(t)", "t"]), seed=0)
    for _ in range(3):
        g = d.step()
    top = {x.display for x in g}
    assert {"x", "x**3", "v"} & top
    w = dict(zip([t.key for t in d.terms], d.omega))
    assert w["x**3"] > w["t**2"] and abs(sum(d.omega) - 1) < 1e-9


def test_reshape_inserts_a_useful_new_term(monkeypatch):
    monkeypatch.setattr(config, "M_SAMPLES", 60)
    task = synthetic_task()
    d = DASS(task, basis(["x", "sin(omega*t)", "exp(alpha*t)", "t**2", "cos(t)", "t"]), seed=0)
    d.step()
    d.reshape([Candidate("-0.8*x - 0.5*x**3 - 0.3*v", 0.0)])
    keys = [t.key for t in d.terms]
    assert "x**3" in keys and "v" in keys and len(keys) == 6
    assert d.log[-1].replaced and abs(sum(d.omega) - 1) < 1e-9



def test_oversized_solver_terms_cannot_enter_the_basis():
    from dass.core import admissible_new_term
    from dass.terms import canonical_term, parse

    big = canonical_term(parse("exp(-sin(0.3*x + 0.1 + v*sin(0.2*t)))*sin(0.4*t + 0.5*x)", PHYS), PHYS)
    assert big.n_params > config.MAX_NEW_TERM_PARAMS and not admissible_new_term(big)
    assert admissible_new_term(term_from_text("x*exp(alpha*t)*sin(beta*t + gamma)", PHYS))


def test_environment_fits_each_term_set_once(monkeypatch):
    from dass import fitting

    task = synthetic_task()
    env = fitting.make_environment("sparse", task.X_train, task.y_train, 1, np.random.default_rng(0))
    terms = [term_from_text(s, PHYS) for s in ("x", "x**3")]
    calls = []
    real = fitting.fit
    monkeypatch.setattr(fitting, "fit", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert env.fitness(terms) == env.fitness(list(terms)) and len(calls) == 1

class FakeAdapter:
    name = "fake"

    def __init__(self, **_):
        self.seen, self.bases = [], []
        self.inject_s = 0.0

    def start(self, task, seed, workdir):
        self.task = task

    def run_stage(self, stage, n_stages, guidance):
        self.seen.append(guidance)
        return [Candidate("-0.8*x - 0.5*x**3 - 0.3*v", 0.0)]

    def set_basis(self, b):
        self.bases.append(b)

    def final(self):
        return Candidate("-0.8*x - 0.5*x**3 - 0.3*v", 0.0)

    def cost(self):
        return {"calls": 0, "inject_s": self.inject_s}


def test_runner_base_and_dass(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "M_SAMPLES", 30)
    fake = FakeAdapter()
    monkeypatch.setattr(runner, "make", lambda name, **kw: fake)
    monkeypatch.setitem(config.ROUNDS, "pysr", 2)
    bdir = tmp_path / "basis"
    bdir.mkdir()
    rows = [{"key": t.key, "term": str(t.expr), "omega": 0.25} for t, _ in basis(["x", "v", "t", "t**2"])]
    (bdir / "phys_osc.json").write_text(json.dumps({"basis": rows}))
    monkeypatch.setattr(runner, "BASIS_DIR", bdir)
    task = synthetic_task()
    base = runner.run_one(task, "pysr", "base", 0, tmp_path)
    assert fake.seen == [None, None] and base["id"]["acc"] == 1.0 and base["ood"]["acc"] == 1.0
    assert fake.bases == []  # the base variant gets no basis either
    fake.seen.clear()
    fake.inject_s = 0.5
    rec = runner.run_one(task, "pysr", "dass", 0, tmp_path)
    assert all(g for g in fake.seen) and rec["dass"]["round"] == 2
    assert rec["time_dass_s"] >= 0.5  # adapter work inside a stage counts as DASS time
    b = fake.bases[-1]  # the whole basis by omega, the guidance its head
    assert len(fake.bases) == 2 and len(b) == 4 and [g.weight for g in b] == sorted((g.weight for g in b), reverse=True)
    assert b[: len(fake.seen[-1])] == fake.seen[-1]
    runner.write_atomic(tmp_path / "r.json", rec)
    assert json.loads((tmp_path / "r.json").read_text())["variant"] == "dass"
    fake.seen.clear()
    fake.inject_s = 0.0
    monkeypatch.setitem(config.PRE_ROUNDS, "pysr", 3)  # three rounds before the first stage, then one a stage
    rec = runner.run_one(task, "pysr", "dass", 0, tmp_path)
    assert len(fake.seen) == 2 and rec["dass"]["round"] == 5
    assert [r["env"] for r in rec["dass"]["log"][:3]] == list(config.ENVIRONMENTS)


def test_parallel_fits_give_the_serial_result(monkeypatch):
    monkeypatch.setattr(config, "M_SAMPLES", 40)
    task = synthetic_task()
    out = []
    for workers in (1, 2):
        d = DASS(task, basis(["x", "x**3", "v", "sin(omega*t)", "exp(alpha*t)", "t"]), seed=3, workers=workers)
        d.step()
        d.reshape([Candidate("-0.8*x - 0.5*x**3 - 0.3*v", 0.0), Candidate("x + t**2", 0.0)])
        d.step()
        out.append((d.omega.copy(), [t.key for t in d.terms]))
    assert out[0][1] == out[1][1] and np.array_equal(out[0][0], out[1][0])
