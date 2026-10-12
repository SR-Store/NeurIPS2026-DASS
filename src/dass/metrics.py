"""Normalized mean squared error (NMSE) and threshold accuracy (Acc_tau).

Acc_tau: after discarding the worst 5% of test points, the maximum relative error of
the remaining points must not exceed tau.  A zero target counts as correct when the
prediction is within 1e-9 of zero.
"""

from __future__ import annotations

import math

import numpy as np

KEEP_FRACTION = 0.95


def nmse(y_pred: np.ndarray | None, y: np.ndarray) -> float:
    if y_pred is None:
        return math.inf
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.shape != y.shape:
        y_pred = np.broadcast_to(y_pred, y.shape)
    if not np.all(np.isfinite(y_pred)):
        return math.inf
    mse = float(np.mean((y_pred - y) ** 2))
    var = float(np.var(y))
    # Treat only an exactly zero target variance as constant; small positive
    # variances still need the usual NMSE denominator.
    if var == 0:
        return 0.0 if mse == 0 else math.inf
    return mse / var


def relative_errors(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    err = np.empty_like(y, dtype=float)
    zero = np.isclose(y, 0)
    err[~zero] = np.abs((y_pred[~zero] - y[~zero]) / y[~zero])
    err[zero] = np.where(np.isclose(y_pred[zero], 0, atol=1e-9), 0.0, np.inf)
    return err


def acc_tau(y_pred: np.ndarray | None, y: np.ndarray, tau: float = 0.01) -> float:
    if y_pred is None:
        return 0.0
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.shape != y.shape:
        y_pred = np.broadcast_to(y_pred, y.shape)
    if not np.all(np.isfinite(y_pred)):
        return 0.0
    err = np.sort(relative_errors(y_pred, y))
    keep = int(math.floor(KEEP_FRACTION * len(err)))
    return 1.0 if keep > 0 and err[keep - 1] <= tau else 0.0
