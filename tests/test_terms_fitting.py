import numpy as np
import pytest
import sympy as sp

from dass.basis_init import aggregate, parse_records, render
from dass.fitting import fit, make_environment
from dass.terms import canonical_term, parse, split_terms, term_from_text

PHYS = ("x", "t", "v")


@pytest.mark.parametrize("a,b", [
    ("alpha*x", "3.1*x"),
    ("alpha*sin(x)", "beta*sin(x)"),
    ("sigma*exp(-tau*x)", "2.5*exp(0.7*x)"),
    ("phi*x(t)*sin(gamma*t)", "x*sin(omega*t)"),
    ("beta*v**2", "-0.4*v**2.0"),
])
def test_same_term(a, b):
    assert term_from_text(a, PHYS).key == term_from_text(b, PHYS).key


@pytest.mark.parametrize("a,b", [("x", "x**3"), ("sin(omega*t)", "cos(omega*t)"), ("x*v", "x**2*v")])
def test_different_terms(a, b):
    assert term_from_text(a, PHYS).key != term_from_text(b, PHYS).key


def test_top_level_sum_is_not_one_term_and_constants_drop():
    assert term_from_text("alpha*x + beta", PHYS) is not None  # the constant is dropped
    assert term_from_text("alpha*x + beta*v", PHYS) is None
    assert term_from_text("gamma", PHYS) is None
    assert term_from_text("lambda*x", PHYS).key == term_from_text("x", PHYS).key


def test_product_with_inner_sum_is_one_term():
    bio = ("t", "P")
    logistic = term_from_text("r*P*(1 - P/K)", bio)
    assert logistic is not None and logistic.n_params == 1
    assert logistic.key == term_from_text("alpha*P*(1 - beta*P)", bio).key


def test_split_distributes_top_level():
    keys = {t.key for t in split_terms(parse("0.17*P*(1 - P/48.5)", ("t", "P")), ("t", "P"))}
    assert keys == {"P", "P**2"}
    keys = {t.key for t in split_terms(parse("-0.5*x - 1.3*x**3 + 0.2*sin(1.1*t)", PHYS), PHYS)}
    assert keys == {"x", "x**3", "sin(c*t)"}


def test_fit_recovers_nonlinear_parameter():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(400, 3))
    y = 1.5 * X[:, 0] - 0.7 * np.sin(1.3 * X[:, 1]) + 0.2
    terms = [term_from_text("x", PHYS), term_from_text("sin(omega*t)", PHYS)]
    f = fit(terms, X, y)
    assert np.mean((f.predict(X) - y) ** 2) < 1e-8


def test_environments_split_sizes():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(4000, 3))
    y = X[:, 0]
    ex = make_environment("extrap", X, y, 1, rng)
    assert ex.X_fit[:, 1].max() <= ex.X_eval[:, 1].min()
    sp_ = make_environment("sparse", X, y, 1, rng)
    assert len(sp_.y_fit) == 100 and len(sp_.y_eval) == 900
    assert term_from_text("x", PHYS) is not None and ex.fitness([term_from_text("x", PHYS)]) > 0.99


def test_basis_records_and_aggregation():
    text = ("alpha*x | Linear Spring Force | Restoring force\n"
            "beta*x**3 | Duffing | cubic\n"
            "gamma*x | Hooke | again linear\n"
            "garbage line without separator\n")
    recs = parse_records(text, PHYS)
    assert len(recs) == 3
    basis = aggregate([text, text], PHYS, n=25)
    assert basis[0]["key"] == "x" and basis[0]["count"] == 4
    assert abs(sum(b["omega"] for b in basis) - 1) < 1e-12
    assert "physics" in render("phys_osc") and "{count}" not in render("phys_osc")


def test_canonical_roundtrip():
    t = term_from_text("sigma*exp(-tau*x)*sin(omega*t)", PHYS)
    again = canonical_term(parse(str(t.expr), PHYS), PHYS)
    assert again.key == t.key and again.n_params == t.n_params == 2


def test_basis_for_a_task_missing_a_domain_variable():
    from dass.basis_init import load_basis
    from dass.runner import BASIS_DIR

    full = load_basis(BASIS_DIR / "phys_osc.json", ("x", "t", "v"))
    no_v = load_basis(BASIS_DIR / "phys_osc.json", ("x", "t"))
    assert len(no_v) < len(full) and all("v" not in t.key for t, _ in no_v)
    assert all(set(t.inputs) == {"x", "t"} for t, _ in no_v)


def test_an_exponent_beyond_double_range_is_a_placeholder():
    x = sp.Symbol("x", real=True)
    t = canonical_term(x ** sp.Float("1e400"), ["x"])
    assert t is not None and t.n_params == 1
