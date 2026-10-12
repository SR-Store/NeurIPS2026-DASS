from pathlib import Path

import pytest

from dass.data import load_tasks


@pytest.fixture(scope="session")
def tiny_parquet() -> Path:
    return Path(__file__).parent / "fixtures" / "tiny_lsr_synth.parquet"


@pytest.fixture(scope="session")
def tiny_task(tiny_parquet):
    (task,) = load_tasks(tiny_parquet)
    return task
