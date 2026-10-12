"""Strings extracted verbatim from upstream LLM-SRBench `methods/llmsr/`.

`eval_spec` and `optimization_spec` are lines 10-82 of `searcher.py` (the rest of
that file is `LLMSRSearcher` glue for the upstream `bench` package and is not
vendored; it carries no per-file header and is covered by the repository's MIT
license).  `instruction_prompt` is lines 273-274 of `sampler.py`: the instruction
`LocalLLM` puts in front of every prompt, copied as it is, so the whitespace the
backslash continuation leaves inside the string is upstream's.  See
`../PROVENANCE.md`.
"""

eval_spec = '''

import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS


@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(*X, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss

        
'''

optimization_spec = '''

import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS


@evaluate.run
def optimize(data: dict):
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(*X, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    return optimized_params.tolist()


'''

instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")
