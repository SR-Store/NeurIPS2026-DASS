import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import sympy as sp

from dass import config, runner
from dass.data import Task
from dass.interfaces import GuidanceTerm
from dass.solvers import llmsr
from dass.terms import parse, term_from_text

# Replies for a synthetic oscillator task with inputs x, t, v.
TRUE = '''The restoring force saturates for large displacements, so:

```python
def equation_v1(x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Improved version of `equation_v0`."""
    damping = params[0] * v
    restoring = params[1] * x * np.exp(-np.abs(x))
    return damping + restoring
```
This keeps the physics simple.'''
CUBIC = '''```python
def equation_v1(x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    a, b, c = params[:3]
    output = a * x + b * x ** 3 + c * v
    return output
```'''
BRANCH = '''def equation_v1(x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    out = params[0] * x
    if params[1] > 0:
        out = out + params[2] * v
    return out
'''
NO_CODE = "I am not sure how to improve this function."


class StubServer:
    """OpenAI-compatible chat endpoint returning canned replies in arrival order."""

    def __init__(self, replies):
        self.replies, self.requests, self.lock = replies, [], threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with stub.lock:
                    stub.requests.append(body)
                    text = stub.replies[(len(stub.requests) - 1) % len(stub.replies)]
                payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": text},
                                                   "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 400, "completion_tokens": 60}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def prompts(self):
        return [r["messages"][0]["content"] for r in self.requests]


@pytest.fixture
def stub():
    s = StubServer([TRUE, CUBIC, BRANCH, NO_CODE])
    yield s
    s.server.shutdown()
    s.server.server_close()


@pytest.fixture(scope="module")
def po19():
    rng = np.random.default_rng(19)

    def sample(n, span):
        X = np.column_stack((rng.uniform(-span, span, n), rng.uniform(0, 10, n),
                             rng.uniform(-1, 1, n)))
        x, _, v = X.T
        return X, -0.8 * v - 1.2 * x * np.exp(-np.abs(x))

    X_train, y_train = sample(400, 2.0)
    X_id, y_id = sample(100, 2.0)
    X_ood, y_ood = sample(100, 2.2)
    return Task("PO19", "phys_osc", "dv_dt", ("x", "t", "v"),
                "Acceleration in Nonl-linear Harmonic Oscillator",
                ("Position at time t", "Time", "Velocity at time t"),
                X_train, y_train, X_id, y_id, X_ood, y_ood)


# LLM-SRBench's first prompt for PO19: LocalLLM's instruction, then the island prompt
# (spec docstring, eval_spec preface, the template program as v0, the v1 header).
BASE_FIRST_PROMPT = (
    "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
    "Complete the `equation` function below, considering the physical meaning and relationships of inputs. "
    "Explain your reasoning briefly before completing the function. Let's think step by step.\n\n"
    + '\n"""\nFind the mathematical function skeleton that represents Acceleration in Nonl-linear Harmonic Oscillator, '
    "given data on Position at time t, Time, and Velocity at time t.\n\"\"\"\n\n\n"
    "import numpy as np\n\n#Initialize parameters\nMAX_NPARAMS = 10\nparams = [1.0]*MAX_NPARAMS\n\n\n"
    "def equation_v0(x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
    '    """ Mathematical function for Acceleration in Nonl-linear Harmonic Oscillator\n\n'
    "    Args:\n"
    "        x: A numpy array representing observations of Position at time t.\n"
    "        t: A numpy array representing observations of Time.\n"
    "        v: A numpy array representing observations of Velocity at time t.\n"
    "        params: Array of numeric constants or parameters to be optimized\n\n"
    "    Return:\n"
    "        A numpy array representing Acceleration in Nonl-linear Harmonic Oscillator as the result of "
    "applying the mathematical function to the inputs.\n"
    '    """\n'
    "    output = params[0] * x + params[1] * t + params[2] * v + params[3]\n"
    "    return output\n\n\n"
    "def equation_v1(x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
    '    """Improved version of `equation_v0`."""\n\n'
)
BLOCK = re.compile(r"# ===== Reference Terms \(DASS\) =====\n(?:#.*\n)*?# =+\n\n\n")


def guidance():
    return [GuidanceTerm("x", 0.26, "x"), GuidanceTerm("x**3", 0.2, "x**3"),
            GuidanceTerm("sin(c0*t)", 0.11, "sin(alpha*t)")]


def test_stage_split():
    for T in (1, 2, 3, 4, 5, 9):
        sizes = [llmsr.stage_prompts(t, T, 100, 4) for t in range(T)]
        assert sum(map(sum, sizes)) == 100 and all(0 < n <= 4 for s in sizes for n in s)
    nine = [sum(llmsr.stage_prompts(t, 9, 100, 4)) for t in range(9)]
    assert set(nine) == {11, 12} and llmsr.stage_prompts(0, 9, 100, 4) == [4, 4, 3]


def test_reference_terms_block():
    code = "params = [1.0]*MAX_NPARAMS\n\n\ndef equation_v0(x):\n    return x\n"
    assert llmsr.with_guidance(code, None) is code
    guided = llmsr.with_guidance(code, guidance())
    lines = guided.splitlines()
    assert lines[3:8] == [
        "# ===== Reference Terms (DASS) =====",
        "# Top 3 of 25 basis-library terms, ranked by Bayesian posterior; "
        "probabilities sum to 1 across the full library.",
        "#",
        "#   1. x                (p=0.26)",
        "#   2. x**3             (p=0.20)",
    ]
    assert lines[8:12] == ["#   3. sin(alpha*t)     (p=0.11)", "#",
                           "# Use as inspiration; combine, modify, or ignore. Focus on physical meaning.",
                           "# " + "=" * 34]
    assert guided.index("Reference Terms") < guided.index("def equation_v0")
    assert BLOCK.sub("", guided) == code


@pytest.mark.parametrize("body, params, want", [
    ("    a = params[0] * x\n    b = params[1] * np.sin(params[2] * t)\n    return a + b\n",
     [2.0, -0.5, 1.5], "2.0*x - 0.5*sin(1.5*t)"),
    ('    """doc"""\n    p0, p1 = params[0:2]\n    out = np.zeros_like(x)\n    out += p0 * v\n'
     "    out -= p1 * x**3\n    return out\n", [0.3, 1.2], "0.3*v - 1.2*x**3"),
    ("    import math\n    k, *rest = params\n    return math.pi * x / params[-1]\n", [1.0, 4.0], None),
    ("    return params[0] * np.exp(-params[1] * t) * np.cos(np.pi * x) + np.sqrt(np.abs(v))\n",
     [1.5, 0.25], "sqrt(Abs(v)) + 1.5*exp(-0.25*t)*cos(pi*x)"),
    ("    return np.power(x, 2) / np.log10(t) + params[0] + np.e * np.cbrt(v)\n", [0.5], None),
    ("    return np.where(x > 0, params[0] * x, params[1] * v)\n", [1.0, 2.0], None),
    ("    for i in range(2):\n        x = x * params[i]\n    return x\n", [1.0, 2.0], None),
    # Defensive code in a model reply, including nested helpers and array checks.
    ("    params = np.asarray(params, dtype=float)\n    if len(params) < 4:\n"
     "        params = np.concatenate([params, np.zeros(4 - len(params))])\n"
     "    def get_param(i):\n        return params[i] if i < len(params) else 0.0\n"
     "    x = np.asarray(x)\n    out = np.zeros_like(x, dtype=float)\n"
     "    out += get_param(0) * x + get_param(1) * np.clip(v, -1.0, None)\n"
     "    if np.any(np.isnan(out)):\n        raise ValueError('nan')\n    return out + get_param(3)\n",
     [2.0, -0.5], None),
    ("    if x > 0:\n        return x\n    return v\n", [1.0], None),
])
def test_program_to_sympy(body, params, want):
    source = "def equation(x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:\n" + body
    e = llmsr.program_to_sympy(source, ("x", "t", "v"), params)
    if want is not None:
        assert str(e) == want
    elif "if x > 0" in body or "where" in body:  # decided by the data
        assert e is None
    else:  # supported, compare numerically with the program itself
        ns = {"np": np}
        exec(source, ns)
        X = np.random.default_rng(0).uniform(0.5, 2.0, (20, 3))
        f = sp.lambdify(sp.symbols("x t v", real=True), parse(str(e), ("x", "t", "v")), "numpy")
        assert np.allclose(f(*X.T), ns["equation"](*X.T, np.asarray(params)))


def test_program_with_numbers_beyond_float64_gives_none():
    """Extreme fitted params can make sympy compute numbers beyond float64 range."""
    source = ("def equation(t: np.ndarray, P: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
              "    return params[0] * P * (1.0 - (P * params[1])**params[2]) + params[3] * np.exp(params[4] * t)\n")
    t0 = time.time()
    assert llmsr.program_to_sympy(source, ("t", "P"), [0.1, 1e10, 5e4, 1.0, 0.5]) is None
    assert llmsr.program_to_sympy(source, ("t", "P"), [0.1, 1e10, -5e4, 1.0, 0.5]) is None  # 1e-500000
    assert llmsr.program_to_sympy(source, ("t", "P"), [0.1, 2.0, 1.5, 1.0, 0.5]) is not None
    assert time.time() - t0 < 5


def run(task, variant, stub, monkeypatch, tmp_path):
    monkeypatch.setitem(config.LLMSR, "max_samples", 6)
    monkeypatch.setitem(config.ROUNDS, "llmsr", 2)
    stub.requests.clear()
    return runner.run_one(task, "llmsr", variant, 0, tmp_path, "qwen", [stub.url])


def test_injection_time_includes_scoring_and_registration(monkeypatch):
    """DASS time includes both fitted-program construction and evaluator work."""
    import io
    from types import SimpleNamespace

    adapter = llmsr.LLMSRAdapter(["http://unused.invalid"], "stub")
    adapter.task = SimpleNamespace(inputs=("x",))
    adapter._fit_X = np.zeros((1, 1))
    adapter._fit_y = np.zeros(1)
    adapter._basis = None
    adapter._injected = set()
    adapter.n_injected = 0
    adapter.inject_s = 0.0
    adapter.trajectory = SimpleNamespace(programs=[])
    adapter.trace = io.StringIO()
    clock = iter((10.0, 11.0, 14.0, 15.0))

    def build(*args):
        assert next(clock) == 11.0
        return ["    return params[0] * x\n"]

    def analyse(body, island_id, version, **kwargs):
        assert next(clock) == 14.0
        adapter.trajectory.programs.append(SimpleNamespace(global_sample_nums=kwargs["global_sample_nums"],
                                                            score=-0.1))

    monkeypatch.setattr(llmsr.time, "time", lambda: next(clock))
    monkeypatch.setattr(llmsr, "build_programs", build)
    adapter.evaluators = [SimpleNamespace(analyse=analyse)]
    adapter._inject(0, [])

    assert adapter.inject_s == 5.0
    assert adapter.n_injected == 1
    assert json.loads(adapter.trace.getvalue())["score"] == -0.1


def test_end_to_end_base_and_dass(po19, stub, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "M_SAMPLES", 30)
    bdir = tmp_path / "basis"
    bdir.mkdir()
    terms = [term_from_text(s, po19.inputs) for s in ("x", "v", "x**3", "t", "sin(omega*t)", "exp(-alpha*t)")]
    rows = [{"key": t.key, "term": str(t.expr), "omega": 1 / len(terms)} for t in terms]
    (bdir / "phys_osc.json").write_text(json.dumps({"basis": rows}))
    monkeypatch.setattr(runner, "BASIS_DIR", bdir)

    base = run(po19, "base", stub, monkeypatch, tmp_path)
    prompts = stub.prompts()
    assert len(prompts) == 6 and prompts[0] == BASE_FIRST_PROMPT
    assert not any("Reference Terms" in p for p in prompts)
    body = stub.requests[0]
    assert body["model"] == config.BACKBONES["qwen"] and body["temperature"] == 0.8
    assert body["max_tokens"] == 8192 and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert base["cost"]["samples"] == 6 and base["cost"]["prompts"] == 2 and base["cost"]["calls"] == 6
    assert base["cost"]["programs_scored"] == 5 and base["cost"]["programs_failed"] == 1
    assert [s["n_candidates"] for s in base["stages"]] == [4, 4]  # template program + distinct scored ones
    assert "np.exp(-np.abs(x))" in base["final"]["source"] and base["id"]["nmse"] < 1e-6
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [r["sample"] for r in trace] == [1, 2, 3, 4, 5, 6] and [r["stage"] for r in trace] == [0] * 3 + [1] * 3
    assert sum(r["score"] is None for r in trace) == 1 and all(r["reply"] and r["finish"] == "stop" for r in trace)

    dass = run(po19, "dass", stub, monkeypatch, tmp_path)
    prompts = stub.prompts()
    assert len(prompts) == 6 and all(p.count("# ===== Reference Terms (DASS) =====") == 1 for p in prompts)
    head = BASE_FIRST_PROMPT[: BASE_FIRST_PROMPT.index("def equation_v0")]
    assert BLOCK.sub("", prompts[0]).startswith(head)  # the block, then the island's programs
    assert re.search(r"# Top \d of 25 basis-library terms", prompts[0])
    assert prompts[0].index("Reference Terms") < prompts[0].index("def equation_v0")
    assert dass["dass"]["round"] == 2 and dass["cost"]["samples"] == 6 and dass["cost"]["calls"] == 6
    # DASS's fitted combinations were put on the islands before the stage's prompts, and the
    # prompts show them (every island holds them and they beat the template)
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    injected = [r for r in trace if r.get("injected")]
    assert 1 <= dass["cost"]["injected"] == len(injected) <= 2 * config.LLMSR["max_inject"]
    assert [r["sample"] for r in injected] == list(range(-1, -len(injected) - 1, -1))
    assert dass["cost"]["injected_scored"] == len(injected) and all(r["score"] is not None for r in injected)
    lines = {r["body"].splitlines()[0].strip() for r in injected}
    assert any(line in prompts[0] for line in lines)
    assert [r["sample"] for r in trace if not r.get("injected")] == [1, 2, 3, 4, 5, 6]
    assert base["cost"]["injected"] == 0


def test_candidates(po19, stub, monkeypatch, tmp_path):
    monkeypatch.setitem(config.LLMSR, "max_samples", 8)
    adapter = llmsr.LLMSRAdapter([stub.url], "stub", prompts_in_flight=2)
    adapter.start(po19, 0, tmp_path)
    cands = adapter.run_stage(0, 1, None)
    assert adapter.cost()["prompts"] == 2 and len(stub.requests) == 8 and len(cands) == 4
    assert [c.train_nmse for c in cands] == sorted(c.train_nmse for c in cands)
    by_source = {("exp" in c.source, "if " in c.source, "** 3" in c.source): c for c in cands}
    true, branch, cubic = by_source[True, False, False], by_source[False, True, False], by_source[False, False, True]
    assert cands[0] is true and adapter.final().source == true.source
    assert branch.expr and branch.predict is not None  # the branch is on params, so it runs through
    X = po19.X_id[:50]
    for c in (true, cubic):
        e = parse(c.expr, po19.inputs)
        f = sp.lambdify(sp.symbols("x t v", real=True), e, "numpy")
        assert np.allclose(f(*X.T), c.predict(X), rtol=1e-8)
    assert "x*exp(-Abs(x))" in true.expr and "v" in true.expr
    assert abs(true.train_nmse - float(np.mean((true.predict(po19.X_train) - po19.y_train) ** 2)
                                       / np.var(po19.y_train))) < 1e-9


def test_body_comes_from_the_last_version_when_the_reply_echoes_the_prompt():
    from dass.solvers.llmsr import extract_body, llmsr_config

    cfg = llmsr_config.Config(use_api=True)
    echo = ("```python\nimport numpy as np\n\n"
            "def equation_v0(t: np.ndarray, P: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
            "    output = params[0] * t + params[1] * P + params[2]\n    return output\n\n\n"
            "def equation_v1(t: np.ndarray, P: np.ndarray, params: np.ndarray) -> np.ndarray:\n"
            "    r, K = params[0], params[1]\n    return r * P * (1 - P / K)\n```\n")
    body = extract_body(echo, cfg)
    assert "r * P * (1 - P / K)" in body and "params[2]" not in body
    single = "Here it is:\ndef equation_v1(t, P, params):\n    return params[0] * P\n"
    from dass.solvers.llmsr import sampler
    assert extract_body(single, cfg) == sampler._extract_body(single, cfg)


def test_programs_are_the_fitted_combinations():
    """A body from ``build_programs`` run at params all ones is DASS's fitted combination."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 2.0, size=(600, 3))
    x, t, v = X.T
    y = 2.0 * x ** 3 - 0.5 * v + 0.3 * np.exp(-1.7 * t) + 0.1
    guidance = [GuidanceTerm(term=str(term_from_text(s, ("x", "t", "v")).expr), weight=w, display=s)
                for s, w in (("x**3", 0.4), ("v", 0.3), ("exp(-alpha*t)", 0.2), ("Max(0, x)", 0.1))]
    bodies = llmsr.build_programs(guidance, ("x", "t", "v"), X, y)
    assert len(bodies) == config.LLMSR["max_inject"] and len(set(bodies)) == len(bodies)
    code = "import numpy as np\ndef equation(x, t, v, params):\n" + bodies[0]
    ns = {}
    exec(code, ns)
    pred = ns["equation"](x, t, v, np.ones(10))
    assert np.mean((pred - y) ** 2) / np.var(y) < 1e-8
    assert "np.maximum(0, x)" in llmsr.numpy_code(sp.Max(0, sp.Symbol("x", real=True)))


def test_programs_take_the_stepwise_path_over_the_basis():
    """A target that needs a basis term outside the guidance is fitted only with the basis."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.5, 2.0, size=(600, 3))
    x, t, v = X.T
    y = 2.0 * x ** 3 - 0.5 * v + 0.7 * np.sin(v) * x + 0.1

    def terms(pairs):
        return [GuidanceTerm(term=str(term_from_text(s, ("x", "t", "v")).expr), weight=w, display=s) for s, w in pairs]

    guidance = terms((("x**3", 0.4), ("v", 0.3), ("t", 0.2)))
    basis = guidance + terms((("t**2", 0.05), ("x*sin(v)", 0.03), ("exp(-alpha*t)", 0.02)))

    def best(bodies):
        ns = {}
        exec("import numpy as np\ndef equation(x, t, v, params):\n" + bodies[0], ns)
        return np.mean((ns["equation"](x, t, v, np.ones(10)) - y) ** 2) / np.var(y)

    assert best(llmsr.build_programs(guidance, ("x", "t", "v"), X, y)) > 1e-4
    with_basis = llmsr.build_programs(guidance, ("x", "t", "v"), X, y, basis)
    assert best(with_basis) < 1e-8 and "np.sin(v)" in with_basis[0]
    path = llmsr.stepwise(llmsr._parsed(basis, ("x", "t", "v"))[0], X, y, 7)
    assert [len(ts) for _, ts, _ in path] == [1, 2, 3, 4, 5, 6] and path[2][0] < 1e-8  # prefixes; three terms suffice
