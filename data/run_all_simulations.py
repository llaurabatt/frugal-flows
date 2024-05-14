import sys
import os
sys.path.append("../") # go to parent dir

import jax
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from frugal_flows.causal_flows import independent_continuous_marginal_flow, get_independent_quantiles, train_frugal_flow
from frugal_flows.bijections import UnivariateNormalCDF
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects import pandas2ri

sys.path.append("../")  # go to parent dir

import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.stats as ss

import data.template_causl_simulations as causl_py

import rpy2.robjects as ro

# Activate automatic conversion of rpy2 objects to pandas objects
pandas2ri.activate()

# Import the R library causl
try:
    causl = importr('causl')
except Exception as e:
    package_names = ('causl')
    utils.install_packages(StrVector(package_names))

jax.config.update("jax_enable_x64", True)

HYPERPARAMS_DICT = {
    'learning_rate': 5e-3,
    'RQS_knots': 8,
    'flow_layers': 5,
    'nn_width': 50,
    'nn_depth': 4,
    'max_patience': 50,
    'max_epochs': 10000
}
SEED = 0
NUM_ITER = 20

TRUE_PARAMS = {'ate': 1, 'const': 1, 'scale': 1}
CAUSAL_PARAMS = list(TRUE_PARAMS.values())[:-1]


def plot_simulation_results(results_df, true_params, save_fig_path=None):
    plt.figure(figsize=(12, 6))

    # Boxplot
    box = results_df.boxplot(column=["ate", "const", "scale"], grid=False)

    # Adding lines for the true parameters
    plt.axhline(y=true_params['ate'], color='r', linestyle='--', label='True ate')
    plt.axhline(y=true_params['const'], color='g', linestyle='--', label='True const')
    plt.axhline(y=true_params['scale'], color='b', linestyle='--', label='True scale')

    # Adding title and labels
    plt.title('Box and Whisker Plot for ATE, Const, and Scale')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(save_fig_path, format='pdf')
    return None


def main():
    gaussian_covariates_results = causl_py.run_simulations(
        causl_py.generate_gaussian_samples,
        seed=SEED,
        num_samples=10000,
        num_iter=NUM_ITER,
        causal_params=CAUSAL_PARAMS,
        hyperparams_dict=HYPERPARAMS_DICT
    )
    plot_simulation_results(gaussian_covariates_results, TRUE_PARAMS, 'gaussian_simulation_results.pdf')
    continous_covariates_results = causl_py.run_simulations(
        causl_py.generate_mixed_samples,
        seed=SEED,
        num_samples=10000,
        num_iter=NUM_ITER,
        causal_params=CAUSAL_PARAMS,
        hyperparams_dict=HYPERPARAMS_DICT
    )
    plot_simulation_results(continous_covariates_results, TRUE_PARAMS, 'mixed_simulation_results.pdf')

    discrete_covariates_results = causl_py.run_simulations(
        causl_py.generate_discrete_samples,
        seed=SEED,
        num_samples=10000,
        num_iter=NUM_ITER,
        causal_params=CAUSAL_PARAMS,
        hyperparams_dict=HYPERPARAMS_DICT
    )
    plot_simulation_results(discrete_covariates_results, TRUE_PARAMS, 'mixed_simulation_results.pdf')

if __name__ == "__main__":
    main()
