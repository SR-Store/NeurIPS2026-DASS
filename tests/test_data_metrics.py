import numpy as np

from dass.data import load_tasks
from dass.metrics import acc_tau, nmse, relative_errors


def test_load_tiny_sr_scientist_parquet(tiny_parquet, tiny_task):
    (t,) = load_tasks(tiny_parquet)
    assert t.task_id == tiny_task.task_id == "SYN1"
    assert t.output == "dv_dt" and t.inputs == ("x", "t", "v")
    assert t.X_train.shape == (64, 3) and t.X_id.shape == (24, 3) and t.X_ood.shape == (24, 3)
    assert np.allclose(t.y_train, -0.8 * t.X_train[:, 0] - 0.5 * t.X_train[:, 0] ** 3
                       - 0.3 * t.X_train[:, 2])


def test_data_path_is_explicit_and_official_counts_are_opt_in(tiny_parquet, monkeypatch):
    import pytest

    monkeypatch.delenv("DASS_DATA", raising=False)
    with pytest.raises(ValueError, match="dataset path required"):
        load_tasks()
    monkeypatch.setenv("DASS_DATA", str(tiny_parquet))
    assert [t.task_id for t in load_tasks()] == ["SYN1"]
    with pytest.raises(AssertionError, match="domain counts"):
        load_tasks(official=True)


def test_acc_tau_discards_worst_five_percent():
    y = np.linspace(1, 2, 100)
    pred = y.copy()
    pred[:5] *= 2.0  # 5% bad points are discarded
    assert acc_tau(pred, y) == 1.0
    pred[5] *= 1.02  # a sixth one is not
    assert acc_tau(pred, y) == 0.0


def test_near_zero_target_rule_is_preserved():
    y = np.array([1e-9, 1.0])
    pred = np.array([0.0, 1.0])
    assert np.array_equal(relative_errors(pred, y), [0.0, 0.0])


def test_nmse():
    y = np.array([1.0, 2.0, 3.0])
    assert nmse(y, y) == 0.0
    assert np.isinf(nmse(np.array([np.nan, 1, 1]), y))


def test_nmse_tiny_variance_is_not_zero():
    y = 1e-6 * np.linspace(0, 1, 50)  # variance ~ 1e-13
    assert 0 < nmse(y + 1e-7, y) < np.inf


def test_config_overrides_from_the_environment():
    import os
    import subprocess
    import sys

    code = "from dass import config; print(config.BETA, config.OVERRIDES)"
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, DASS_OVERRIDES='{"BETA": 1.0}', PYTHONPATH=src)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split()[0] == "1.0"
    bad = subprocess.run([sys.executable, "-c", code], env=dict(env, DASS_OVERRIDES='{"NOPE": 1}'),
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "NOPE" in bad.stderr
