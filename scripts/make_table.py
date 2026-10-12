"""Summarize ID and OOD result records.

    python scripts/make_table.py DIR[:SOLVER,...] [DIR[:SOLVER,...] ...] [--out table.md]

Each DIR is a results directory; ``:pysr`` or ``:llmsr`` filters solvers in that
directory. Per (solver, backbone, variant, domain), report mean search time, mean NMSE
over finite runs, median NMSE, and Acc_0.01 in percent. Also report search time,
task-and-seed paired comparisons, and the source of LLM-SR +DASS final programs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

DOMAINS = ("phys_osc", "matsci", "chem_react", "bio_pop_growth")
SHORT = {"phys_osc": "Phys", "matsci": "MatSci", "chem_react": "Chem", "bio_pop_growth": "Bio"}
ORDER = [("llmsr", "qwen"), ("llmsr", "gemma"), ("pysr", None)]


def load(sources: list[str]) -> dict:
    cells = defaultdict(list)
    for src in sources:
        path, _, only = src.partition(":")
        keep = set(only.split(",")) if only else None
        for p in Path(path).rglob("*.json"):
            r = json.loads(p.read_text())
            if keep is None or r["solver"] in keep:
                r["_path"] = p
                cells[(r["solver"], r["backbone"], r["variant"], r["domain"])].append(r)
    return cells


def finite_mean(xs) -> tuple[float, int]:
    xs = list(xs)
    fin = [x for x in xs if math.isfinite(x)]
    return (sum(fin) / len(fin) if fin else math.inf), len(xs) - len(fin)


def summary(runs: list[dict], split: str) -> dict:
    mean, n_inf = finite_mean(r[split]["nmse"] for r in runs)
    return {"n": len(runs), "time": sum(r["time_s"] for r in runs) / len(runs),
            "dass_time": sum(r.get("time_dass_s", 0.0) for r in runs) / len(runs),
            "mean": mean, "n_inf": n_inf, "median": median(r[split]["nmse"] for r in runs),
            "acc": 100.0 * sum(r[split]["acc"] for r in runs) / len(runs)}


def rows(cells: dict):
    for solver, backbone in ORDER:
        if any(cells.get((solver, backbone, v, d)) for v in ("base", "dass") for d in DOMAINS):
            yield solver, backbone


def table(cells: dict, split: str) -> list[str]:
    lines = ["| Solver | Backbone | Variant | " + " | ".join(
        f"{SHORT[d]} Time (s) | {SHORT[d]} NMSE mean | {SHORT[d]} NMSE median | {SHORT[d]} Acc" for d in DOMAINS) + " |",
        "|---|---|---|" + "---:|" * (4 * len(DOMAINS))]
    for solver, backbone in rows(cells):
        for variant in ("base", "dass"):
            row = [solver, backbone or "-", "base" if variant == "base" else "+DASS"]
            for d in DOMAINS:
                runs = cells.get((solver, backbone, variant, d))
                if not runs:
                    row += [""] * 4
                    continue
                s = summary(runs, split)
                row += [f"{s['time']:.1f}", f"{s['mean']:.2E}" + (f" ({s['n_inf']} inf)" if s["n_inf"] else ""),
                        f"{s['median']:.2E}", f"{s['acc']:.1f}"]
            lines.append("| " + " | ".join(row) + " |")
    return lines


def times(cells: dict) -> list[str]:
    lines = ["| Solver | Backbone | Domain | Runs | base (s) | +DASS (s) | of which DASS (s) | +DASS / base |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for solver, backbone in rows(cells):
        for d in (*DOMAINS, "all"):
            b = [r for dd in DOMAINS if d in ("all", dd) for r in cells.get((solver, backbone, "base", dd), [])]
            x = [r for dd in DOMAINS if d in ("all", dd) for r in cells.get((solver, backbone, "dass", dd), [])]
            if not b or not x:
                continue
            sb, sx = summary(b, "id"), summary(x, "id")
            lines.append(f"| {solver} | {backbone or '-'} | {SHORT.get(d, 'All')} | {len(b)}/{len(x)} | {sb['time']:.1f} | "
                         f"{sx['time']:.1f} | {sx['dass_time']:.1f} | {sx['time'] / sb['time']:.2f} |")
    return lines


def log_ratio(base: float, dass: float) -> float:
    lb, ld = (math.log10(max(v, 1e-300)) for v in (base, dass))
    return 0.0 if lb == ld else ld - lb


def paired(cells: dict) -> list[str]:
    names = [SHORT[d] for d in DOMAINS] + ["All"]
    lines = ["| Solver | Backbone | Split | " + " | ".join(f"{n} lower | {n} log10 ratio" for n in names) + " |",
             "|---|---|---|" + "---:|" * (2 * len(names))]
    for solver, backbone in rows(cells):
        by_key = {}
        for d in DOMAINS:
            for variant in ("base", "dass"):
                for r in cells.get((solver, backbone, variant, d), []):
                    by_key.setdefault((d, r["task_id"], r["seed"]), {})[variant] = r
        pairs = [(d, v["base"], v["dass"]) for (d, _, _), v in by_key.items() if len(v) == 2]
        if not pairs:
            continue
        for split in ("id", "ood"):
            row = [solver, backbone or "-", split.upper()]
            for d in (*DOMAINS, "all"):
                ps = [(b, x) for dd, b, x in pairs if d in ("all", dd)]
                if not ps:
                    row += ["", ""]
                    continue
                lower = 100 * sum(x[split]["nmse"] < b[split]["nmse"] for b, x in ps) / len(ps)
                ratio = median(log_ratio(b[split]["nmse"], x[split]["nmse"]) for b, x in ps)
                row += [f"{lower:.0f}%", f"{ratio:+.2f}"]
            lines.append("| " + " | ".join(row) + " |")
    return lines


def final_is_injected(r: dict) -> bool | None:
    """Whether an LLM-SR +DASS final program is one DASS injected, as it was; None without a trace."""
    trace = r["_path"].with_suffix(".trace.jsonl")
    if not trace.exists():
        return None
    injected = set()
    with trace.open() as records:
        for line in records:
            t = json.loads(line)
            if t.get("injected"):
                injected.add(t["body"].strip())
    return r["final"]["source"].split('"""', 2)[-1].strip() in injected


def sources(cells: dict) -> list[str]:
    lines = ["| Backbone | Domain | Runs | Final = injected program | Final written by the LLM |", "|---|---|---:|---:|---:|"]
    for backbone in ("qwen", "gemma"):
        for d in (*DOMAINS, "all"):
            runs = [r for dd in DOMAINS if d in ("all", dd) for r in cells.get(("llmsr", backbone, "dass", dd), [])]
            flags = [f for f in map(final_is_injected, runs) if f is not None]
            if flags:
                k = sum(flags)
                lines.append(f"| {backbone} | {SHORT.get(d, 'All')} | {len(flags)} | {k} ({100 * k / len(flags):.0f}%) | "
                             f"{len(flags) - k} ({100 * (len(flags) - k) / len(flags):.0f}%) |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="results directories, each optionally :solver,solver")
    ap.add_argument("--out")
    args = ap.parse_args()
    cells = load(args.sources)
    text = []
    for split, name in (("id", "ID results"), ("ood", "OOD results")):
        text += [f"## {name}", "", *table(cells, split), ""]
    text += ["## Search time per run", "", *times(cells), ""]
    text += ["## Paired +DASS vs base (same task and seed)", "", *paired(cells), ""]
    src = sources(cells)
    if len(src) > 2:
        text += ["## Where the LLM-SR +DASS final program came from", "", *src, ""]
    out = "\n".join(text)
    if args.out:
        Path(args.out).write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
