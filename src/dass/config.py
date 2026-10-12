"""Default DASS and solver settings used by the packaged implementation."""

import json
import os

# --- DASS -------------------------------------------------------------------
M_SAMPLES = 200  # expressions sampled from the subspace per DASS round
ELITE_RATIO = 0.30  # top-k% by fitness form the elite set
BASIS_SIZE = 25  # n = |B|
OMEGA0 = 0.5  # initial preference for a newly inserted term
GUIDE_RATIO = 0.30  # top-k% of terms by omega handed to the solver
ENVIRONMENTS = ("extrap", "sparse", "noisy")

BETA = 3.0  # Gibbs posterior strength
ETA = 0.3  # EMA update rate
EPS = 1e-8
MAX_TERMS_PER_SAMPLE = 5  # maximum terms in each sampled expression
SPARSE_FRACTION = 0.10  # fraction of rows used to fit in the sparse environment
NOISE_REL_STD = 0.05  # noise standard deviation relative to std(y)
MAX_REPLACE_PER_ROUND = 5
MAX_NEW_TERM_PARAMS = 3
MAX_NEW_TERM_OPS = 10
FIT_ROWS = 1000  # maximum rows used by DASS's own fits

# DASS rounds T per supported solver.
ROUNDS = {"llmsr": 9, "pysr": 3}
PRE_ROUNDS = {"llmsr": 0, "pysr": 0}

# --- Solver budgets ----------------------------------------------------------
LLMSR = {"max_samples": 100, "samples_per_prompt": 4, "num_islands": 10,
         # Fit guidance combinations and a forward-selection path over the basis.
         "max_inject": 3, "inject_from": 30, "stepwise_max": 7}
PYSR = {"niterations": 65, "populations": 15, "population_size": 33, "ncycles_per_iteration": 550,
        "maxsize": 30, "binary_operators": ["+", "-", "*", "/", "^"],
        "unary_operators": ["exp", "log", "sqrt", "sin", "cos", "abs"],
        "max_guesses": 30}  # guesses supplied per stage

# --- LLM backbones -----------------------------------------------------------
LLM_SAMPLING = {"temperature": 0.8, "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": False}}
BACKBONES = {"qwen": "qwen35-9b", "gemma": "gemma4-31b"}  # vLLM served names

SEEDS = (0, 1, 2)
TAU = 0.01

# Optional experiment settings are recorded in each result for reproducibility.
OVERRIDES = json.loads(os.environ.get("DASS_OVERRIDES") or "{}")
for _name, _value in OVERRIDES.items():
    if _name not in globals() or not _name.isupper():
        raise KeyError(f"DASS_OVERRIDES names no constant of dass.config: {_name}")
    globals()[_name] = _value
