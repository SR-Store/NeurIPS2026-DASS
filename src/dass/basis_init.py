"""Prior-driven initialization of the basis term set B(0).

The same prompt is sent n times per domain (memory-free sampling, temperature drawn
from U(0.7, 1.1) per call, {count}=10).  Records are parsed, coefficient names are
normalized (``alpha*sin(x)`` = ``beta*sin(x)``), and the top-n terms by frequency form
B(0) with omega(0) = c(s) / sum c.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import config
from .terms import Term, term_from_text

PROMPT = """You are a research scientist with broad knowledge of classical literature
in {field}.
Task
----
We are searching for mathematical function skeletons that may represent
{output_desc}, given data on {input_desc_list}.
That is, we seek expressions of the form
    {output_symbol} = F({input_symbols})
Drawing on classical and widely studied scientific literature in the relevant
field, propose {count} single-term candidate building blocks that could
plausibly appear on the right-hand side. For each term, identify the standard
field-specific family or named mechanism it belongs to; use a descriptive
category only when no standard name exists.
Each term must be:
- grounded in a canonical textbook- or review-level mechanism;
- a single building block: the top-level operator must not be + or -.
  Additions and subtractions inside denominators, exponents, or canonical
  mechanism forms are allowed;
- written in Python-compatible syntax, using only the input variables listed
  above and ASCII transliterations of Greek-letter coefficient names
  (alpha, beta, gamma, delta, ...; ASCII subscripts allowed, e.g. alpha_1);
- expressed with no other identifiers and no equals sign;
- free of the output symbol {output_symbol}.
Output format (strict)
----------------------
Output exactly {count} records, one per line, with three fields separated by "|":
    <term> | <family_or_mechanism_name> | <one-line mechanism description>
Constraints
-----------
- Do not use numeric constants as fitted coefficients; use Greek-letter names.
- Write exponentiation as **.
- Each line must be no longer than 200 characters.
- Output no header, no numbering, no blank lines, no commentary."""

# Per-domain prompt values; ``inputs`` are the dataset's variable names.
DOMAINS = {
    "phys_osc": dict(field="physics", output_desc="Acceleration in Nonlinear Harmonic Oscillator",
                     input_desc_list="Position at time t (x(t)), Velocity at time t (v(t)), and Time (t)",
                     output_symbol="dv/dt", input_symbols="x(t), v(t), t", inputs=("x", "t", "v")),
    "matsci": dict(field="materials science", output_desc="Stress",
                   input_desc_list="Strain (epsilon) and Temperature (T)",
                   output_symbol="sigma", input_symbols="epsilon, T", inputs=("epsilon", "T")),
    "chem_react": dict(field="chemistry", output_desc="Rate of change of concentration in chemistry reaction kinetics",
                       input_desc_list="Concentration at time t (A(t)) and Time (t)",
                       output_symbol="dA/dt", input_symbols="A(t), t", inputs=("t", "A")),
    "bio_pop_growth": dict(field="mathematical biology", output_desc="Population growth rate",
                           input_desc_list="Population at time t (P(t)) and Time (t)",
                           output_symbol="dP/dt", input_symbols="P(t), t", inputs=("t", "P")),
}
COUNT = 10
N_CALLS = 20
TEMPERATURE_RANGE = (0.7, 1.1)


def render(domain: str) -> str:
    d = DOMAINS[domain]
    return PROMPT.format(count=COUNT, **{k: v for k, v in d.items() if k != "inputs"})


def parse_records(text: str, inputs) -> list[tuple[Term, str]]:
    """``(term, family)`` for every well-formed record of one response."""
    out = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split("|")]
        if len(fields) < 2 or not fields[0]:
            continue
        term = term_from_text(fields[0].strip("`"), inputs)
        if term is not None:
            out.append((term, fields[1]))
    return out


def aggregate(responses: list[str], inputs, n: int = config.BASIS_SIZE) -> list[dict]:
    """Top-n terms by generation frequency with omega(0) = c / sum c (over the top n)."""
    counts: Counter[str] = Counter()
    first: dict[str, Term] = {}
    families: dict[str, Counter] = {}
    for text in responses:
        for term, family in parse_records(text, inputs):
            counts[term.key] += 1
            first.setdefault(term.key, term)
            families.setdefault(term.key, Counter())[family] += 1
    top = sorted(counts, key=lambda k: (-counts[k], k))[:n]
    total = sum(counts[k] for k in top)
    return [{"key": k, "term": str(first[k].expr), "display": first[k].display, "count": counts[k],
             "omega": counts[k] / total, "family": families[k].most_common(1)[0][0]} for k in top]


def load_basis(path: Path | str, inputs) -> list[tuple[Term, float]]:
    """The committed B(0) of a domain as (Term, omega(0)) pairs, for a task with ``inputs``.

    B(0) is generated once per domain over the domain's variables. Terms that use
    variables absent from the current task are left out.
    """
    from .terms import parse, canonical_term

    data = json.loads(Path(path).read_text())
    domain_inputs = DOMAINS[data.get("domain") or Path(path).stem]["inputs"]  # files are <domain>.json
    missing = set(domain_inputs) - set(inputs)
    out = []
    for r in data["basis"]:
        e = parse(r["term"], domain_inputs)
        if e is not None and missing & {s.name for s in e.free_symbols}:
            continue
        t = canonical_term(parse(r["term"], inputs), inputs)
        if t is None or t.key != r["key"]:
            raise ValueError(f"basis term {r['term']!r} does not round-trip to {r['key']!r}")
        out.append((t, float(r["omega"])))
    return out
