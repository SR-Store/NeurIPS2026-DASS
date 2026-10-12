"""Regenerate the deterministic one-task SR-Scientist-format test fixture."""

from pathlib import Path

import numpy as np
import pandas as pd


def rows(rng: np.random.Generator, count: int, time_limit: float) -> list[list[float]]:
    x = rng.uniform(-1.5, 1.5, count)
    t = rng.uniform(0, time_limit, count)
    v = rng.uniform(-1, 1, count)
    y = -0.8 * x - 0.5 * x**3 - 0.3 * v
    return np.column_stack((y, x, t, v)).tolist()


def main() -> None:
    rng = np.random.default_rng(2026)
    frame = pd.DataFrame([{
        "dataset_identifier": "LSR-Synth/phys_osc",
        "equation_idx": "SYN1",
        "symbols": ["dv_dt", "x", "t", "v"],
        "symbol_descs": ["Acceleration", "Position", "Time", "Velocity"],
        "expression": "-0.8*x - 0.5*x**3 - 0.3*v",
        "samples": {
            "train": rows(rng, 64, 4),
            "test": rows(rng, 24, 4),
            "ood_test": rows(rng, 24, 7),
        },
    }])
    frame.to_parquet(Path(__file__).with_name("tiny_lsr_synth.parquet"), index=False)


if __name__ == "__main__":
    main()
