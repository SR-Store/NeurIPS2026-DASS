"""One tiny DASS calibration step and a fixed-candidate score, with no solver service."""

import argparse
import json
from pathlib import Path

from dass import config
from dass.core import DASS
from dass.data import load_tasks
from dass.interfaces import Candidate
from dass.runner import score
from dass.terms import term_from_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    (task,) = load_tasks(args.data)
    terms = [term_from_text(text, task.inputs) for text in
             ("x", "x**3", "v", "t", "t**2", "sin(t)")]
    config.M_SAMPLES = 8  # only this one-process demonstration uses a tiny calibration budget
    dass = DASS(task, [(term, 1 / len(terms)) for term in terms], seed=0)
    guidance = dass.step()
    candidate = Candidate("-0.8*x - 0.5*x**3 - 0.3*v", 0.0)
    print(json.dumps({
        "task": task.task_id,
        "train_rows": len(task.y_train),
        "top_terms": [term.display for term in guidance],
        "score": score(candidate, task),
    }, indent=2))


if __name__ == "__main__":
    main()
