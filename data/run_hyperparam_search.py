import sys

import jax
import sys
import jax
import os
import matplotlib.pyplot as plt

import hyperparam_and_bootstrapping as hb
import template_causl_simulations as causl_py

sys.path.append("../")  # go to parent dir

jax.config.update("jax_enable_x64", True)

param_grid = {
    'RQS_knots': [4, 5, 6, 7, 8, 9, 10],
    'flow_layers': [2, 3, 4, 5, 6],
    'nn_width': [25, 30, 35, 40, 45, 50, 55, 60],
    'nn_depth': [2, 3, 4, 5, 6],
    'learning_rate': [5e-3],
    'batch_size': [1000],
    'max_patience': [50],
    'max_epochs': [10000]
}
param_grid = {
    'RQS_knots': [4, 5],
    'flow_layers': [4],
    'nn_width': [50],
    'nn_depth': [4],
    'learning_rate': [5e-3],
    'batch_size': [1000],
    'max_patience': [50],
    'max_epochs': [10000]
}

def main():
    output_dir = "hyperparam_results"
    param_combinations = hb.generate_param_combinations(param_grid)

#    # Gaussian hyperparam fits
#    gaussian_fp = os.path.join(output_dir, 'gaussian_sim.py')
#    if not os.path.exists(output_dir):
#        Z, X, Y = causl_py.generate_gaussian_samples(N=1000, causal_params=[1,1], seed=0).values()
#        gaussian_hyperparam_fits = hb.gaussian_outcome_hyperparameter_search(
#            X, Y, Z_disc=None, Z_cont=Z, param_combinations=param_combinations, seed=0
#        )
#        gaussian_hyperparam_fits.to_csv(gaussian_fp, index=False)
#    else:
#        print("File 'gaussian_hyperparam_fits' already exists. Fit will not be performed.")
#
#    # Mixed hyperparam fits
#    mixed_fp = os.path.join(output_dir, 'mixed_sim.py')
#    if not os.path.exists(output_dir):
#        Z, X, Y = causl_py.generate_mixed_samples(N=1000, causal_params=[1,1], seed=0).values()
#        Z_disc = Z.select_dtypes(include=['int'])
#        Z_cont = Z.select_dtypes(include=['float'])
#        mixed_hyperparam_fits = hb.gaussian_outcome_hyperparameter_search(
#            X, Y, Z_disc=Z_disc, Z_cont=Z_cont, param_combinations=param_combinations, seed=0
#        )
#        mixed_hyperparam_fits.to_csv(mixed_fp, index=False)
#    else:
#        print(f"File '{mixed_fp}' already exists. Fit will not be performed.")

    # Discrete hyperparam fits
    discrete_fp = os.path.join(output_dir, 'discrete_sim.py')
    if not os.path.exists(output_dir):
        Z, X, Y = causl_py.generate_discrete_samples(N=1000, causal_params=[1,1], seed=0).values()
        # Split the data
        Z_disc = Z[:, [0,-1]].astype(int)
        Z_cont = Z[:, [1,2]]
        print(Z[:10])
        print(Z_disc[:10])
        print(Z_cont[:10])
        discrete_hyperparam_fits = hb.gaussian_outcome_hyperparameter_search(
            X, Y, Z_disc=Z_disc, Z_cont=Z_cont, param_combinations=param_combinations, seed=0
        )
        discrete_hyperparam_fits.to_csv(discrete_fp, index=False)
    else:
        print(f"File '{discrete_fp}' already exists. Fit will not be performed.")

if __name__ == "__main__":
    main()