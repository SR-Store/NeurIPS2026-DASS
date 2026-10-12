import os
import stat
import threading

from dass.julia_env import precompile_once


def test_precompile_runs_once_per_scratch(tmp_path):
    calls = tmp_path / "calls"
    julia = tmp_path / "julia"
    julia.write_text(f"#!/bin/sh\necho \"$JULIA_DEPOT_PATH $JULIA_PKG_OFFLINE $*\" >> {calls}\nsleep 0.2\n")
    julia.chmod(julia.stat().st_mode | stat.S_IEXEC)
    scratch = tmp_path / "scratch"
    out = []
    threads = [threading.Thread(target=lambda: out.append(
        precompile_once("x", str(julia), "/proj", "/rw:/shared", scratch))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = calls.read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("/rw:/shared true --startup-file=no --project=/proj -e")
    assert sum(s is not None for s in out) == 1 and out.count(None) == 3
    assert precompile_once("x", str(julia), "/proj", "/rw:/shared", scratch) is None
    assert os.path.exists(scratch / "x.precompile.log")
