"""One run = (task, solver, variant, backbone, seed) -> one result record (JSON).

    dass-run --data DATA.parquet --solver pysr --variant dass --out RESULTS_DIR

The record's ``time_s`` is the wall time of the T stages plus DASS itself (the
adapter's one-time ``start`` is excluded).  Base and DASS use the same T stages; the
base variant passes ``guidance=None``.  A record already on disk is skipped, so a
resubmitted job resumes where the previous one stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import sympy as sp

from . import config, metrics
from .basis_init import load_basis
from .core import DASS
from .data import Task, load_tasks
from .interfaces import Candidate
from .solvers import LLM_SOLVERS, make
from .terms import input_symbols, parse

SCHEMA = 1
BASIS_DIR = Path(__file__).resolve().parent / "basis"
RECORD_TZ = timezone(timedelta(hours=8))


def record_path(out: Path, solver: str, backbone: str | None, variant: str, task_id: str, seed: int) -> Path:
    return out / solver / (backbone or "na") / variant / f"{task_id}_s{seed}.json"


def predictor(c: Candidate, task: Task):
    if c.predict is not None:
        return c.predict
    e = parse(c.expr, task.inputs)
    if e is None:
        return None
    syms = list(input_symbols(task.inputs).values())
    f = sp.lambdify(syms, e, modules="numpy")
    return lambda X: np.broadcast_to(np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), dtype=float),
                                     (X.shape[0],))


def score(c: Candidate, task: Task) -> dict:
    f = predictor(c, task)
    out = {}
    for split, X, y in (("id", task.X_id, task.y_id), ("ood", task.X_ood, task.y_ood)):
        try:
            with np.errstate(all="ignore"):
                pred = None if f is None else np.asarray(f(X), dtype=float)
        except Exception:
            pred = None
        out[split] = {"nmse": metrics.nmse(pred, y), "acc": metrics.acc_tau(pred, y, config.TAU)}
    return out


def run_one(task: Task, solver: str, variant: str, seed: int, workdir: Path,
            backbone: str | None = None, endpoints: list[str] | None = None,
            model: str | None = None, api_key: str | None = None) -> dict:
    kwargs = {}
    if solver in LLM_SOLVERS:
        kwargs = {"endpoints": endpoints, "model": model or config.BACKBONES[backbone],
                  "api_key": api_key}
    adapter = make(solver, **kwargs)
    adapter.start(task, seed, workdir)
    T = config.ROUNDS[solver]
    started = datetime.now(RECORD_TZ).isoformat()
    t0 = time.time()
    dass_s = 0.0
    dass = None
    if variant == "dass":
        d0 = time.time()
        dass = DASS(task, load_basis(BASIS_DIR / f"{task.domain}.json", task.inputs), seed)
        for _ in range(config.PRE_ROUNDS.get(solver, 0)):
            dass.step()
        dass_s += time.time() - d0
    stages = []
    for t in range(T):
        guidance = None
        if dass is not None:
            d0 = time.time()
            guidance = dass.step()
            if hasattr(adapter, "set_basis"):  # a solver that also fits over the whole basis
                adapter.set_basis(dass.ranked_basis())
            dass_s += time.time() - d0
        s0 = time.time()
        cands = adapter.run_stage(t, T, guidance)
        stages.append({"stage": t, "wall_s": round(time.time() - s0, 3), "n_candidates": len(cands),
                       "best_train_nmse": min((c.train_nmse for c in cands), default=None)})
        if dass is not None:
            d0 = time.time()
            dass.reshape(cands)
            dass_s += time.time() - d0
    final = adapter.final()
    wall = time.time() - t0
    cost = adapter.cost()
    dass_s += cost.get("inject_s", 0.0)  # fitting and evaluating injected programs inside solver stages
    return {
        "schema": SCHEMA,
        "task_id": task.task_id, "domain": task.domain, "solver": solver, "variant": variant,
        "backbone": backbone, "model": kwargs.get("model"), "seed": seed, "rounds": T,
        "time_s": round(wall, 3), "time_dass_s": round(dass_s, 3),
        "final": {"expr": final.expr, "train_nmse": final.train_nmse, "source": final.source},
        **score(final, task),
        "stages": stages,
        "cost": cost,
        "dass": dass.state() if dass is not None else None,
        "commit": os.environ.get("DASS_COMMIT", ""),
        "overrides": config.OVERRIDES,
        "host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
        "started_at": started, "finished_at": datetime.now(RECORD_TZ).isoformat(),
    }


def write_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=float))
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DASS_DATA"),
                    help="task Parquet file (or set DASS_DATA)")
    ap.add_argument("--official", action="store_true", help="verify the 129-task benchmark domain counts")
    ap.add_argument("--solver", required=True, choices=sorted(config.ROUNDS))
    ap.add_argument("--variant", required=True, choices=("base", "dass"))
    ap.add_argument("--backbone", choices=sorted(config.BACKBONES))
    ap.add_argument("--model", help="override the served model name for --backbone")
    ap.add_argument("--endpoints", default="", help="comma-separated base URLs of the backbone")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tasks", default="all", help="comma-separated task ids, or all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=os.environ.get("TMPDIR", "/tmp"))
    args = ap.parse_args(argv)
    if not args.data:
        ap.error("--data is required unless DASS_DATA is set")
    if args.solver in LLM_SOLVERS and not (args.backbone and args.endpoints):
        ap.error("LLM solvers need --backbone and --endpoints")
    tasks = load_tasks(args.data, official=args.official)
    if args.tasks != "all":
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.task_id in wanted]
    out = Path(args.out)
    for seed in [int(s) for s in args.seeds.split(",")]:
        for task in tasks:
            path = record_path(out, args.solver, args.backbone, args.variant, task.task_id, seed)
            if path.exists():
                continue
            run_workdir = (Path(args.workdir) / args.solver / (args.backbone or "na") / args.variant
                           / f"{task.task_id}_s{seed}")
            rec = run_one(task, args.solver, args.variant, seed, run_workdir,
                          args.backbone, [u for u in args.endpoints.split(",") if u],
                          args.model, os.environ.get("DASS_API_KEY"))
            trace = run_workdir / "trace.jsonl"
            if args.solver in LLM_SOLVERS and trace.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(trace, path.with_suffix(".trace.jsonl"))
            write_atomic(path, rec)
            print(f"{path.name} time={rec['time_s']}s id_acc={rec['id']['acc']} ood_acc={rec['ood']['acc']}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
