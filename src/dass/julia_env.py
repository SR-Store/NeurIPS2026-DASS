"""Precompile an explicitly configured Julia environment before embedding it in Python.

When the caller supplies a Julia executable, project, and depot, ``precompile_once``
runs ``Pkg.precompile()`` in a standalone Julia process.  Repeated starts sharing a
scratch directory use a lock and reuse the completed package images.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path


def scratch_dir(fallback: Path) -> Path:
    """Use ``DASS_SCRATCH`` when supplied, otherwise the caller's work directory."""
    return Path(os.environ.get("DASS_SCRATCH") or fallback)


def precompile_once(name: str, julia: str, project: str, depot_path: str, scratch: Path) -> float | None:
    """Precompile ``project`` once per ``scratch``; seconds spent, or None if already done."""
    scratch.mkdir(parents=True, exist_ok=True)
    done = scratch / f"{name}.precompiled"
    with open(scratch / f"{name}.precompile.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if done.exists():
            return None
        t0 = time.time()
        env = dict(os.environ, JULIA_DEPOT_PATH=depot_path, JULIA_PKG_OFFLINE="true")
        with open(scratch / f"{name}.precompile.log", "w") as log:
            subprocess.run([julia, "--startup-file=no", f"--project={project}", "-e",
                            "using Pkg; Pkg.precompile()"], env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        seconds = round(time.time() - t0, 1)
        done.write_text(f"{seconds}\n")
        return seconds
