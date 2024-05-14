import itertools
import sys
import os
sys.path.append("../") # go to parent dir

import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import numpy as np
from scipy.stats import rankdata
import scipy.stats as ss
import seaborn as sns
from sklearn.model_selection import KFold

# from data.create_sim_data import *
import data.template_causl_simulations as causl_py
from data.run_all_simulations import plot_simulation_results
from frugal_flows.causal_flows import independent_continuous_marginal_flow, get_independent_quantiles, train_frugal_flow
from frugal_flows.bijections import UnivariateNormalCDF

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

# Activate automatic conversion of rpy2 objects to pandas objects
pandas2ri.activate()

# Import the R library causl
try:
    causl = importr('causl')
except Exception as e:
    package_names = ('causl')
    utils.install_packages(StrVector(package_names))

jax.config.update("jax_enable_x64", True)

hyperparams_dict = {
    'learning_rate': 5e-3,
    'RQS_knots': 8,
    'flow_layers': 5,
    'nn_width': 50,
    'nn_depth': 4,    
    'max_patience': 50,
    'max_epochs': 10000
}

def generate_param_combinations(param_grid):
    """
    Generate all possible combinations of parameters from a given parameter grid.

    Parameters:
    - param_grid (dict): A dictionary containing parameter names as keys and lists of possible values as values.

    Returns:
    - param_combinations (list): A list of dictionaries, where each dictionary represents a combination of parameters.

    Example:
    >>> param_grid = {'param1': [1, 2], 'param2': ['a', 'b']}
    >>> generate_param_combinations(param_grid)
    [{'param1': 1, 'param2': 'a'}, {'param1': 1, 'param2': 'b'}, {'param1': 2, 'param2': 'a'}, {'param1': 2, 'param2': 'b'}]
    """
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return param_combinations


def gaussian_outcome_hyperparameter_search(X, Y, Z_disc=None, Z_cont=None, param_combinations=None, seed=0):
    """
    Performs hyperparameter search for a frugal flow model with Gaussian outcome.

    Args:
        X (array-like): The treatment features.
        Y (array-like): The outcome variable.
        param_combinations (list, optional): List of dictionaries representing different combinations of hyperparameters. 
            Each dictionary should contain the hyperparameters to be tested. Defaults to None.
        Z_disc (array-like, optional): The discrete confounders. Defaults to None.
        Z_cont (array-like, optional): The continuous confounders. Defaults to None.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.

    Returns:
        pandas.DataFrame: A DataFrame containing the results of the hyperparameter search. Each row represents a 
        combination of hyperparameters and includes the following columns:
            - ate: The average treatment effect.
            - const: The constant term in the causal margin model.
            - scale: The scale parameter in the causal margin model.
            - min_loss: The minimum loss achieved during training.

    """
    results_list = []
    for i, param_set in enumerate(param_combinations):
        results = param_set.copy()
        fitted_flow = causl_py.frugal_fitting(X, Y, Z_disc=Z_disc, Z_cont=Z_cont, seed=i, frugal_flow_hyperparams=param_set)
        causal_margin = fitted_flow['causal_margin'] 
        results['ate'] = causal_margin.ate
        results['const'] = causal_margin.const
        results['scale'] = causal_margin.scale
        results['min_loss'] = min(fitted_flow['losses']['val'])
        results_list.append(
            pd.DataFrame.from_dict(results, orient='index').T
        )
    return pd.concat(results_list, axis=0)