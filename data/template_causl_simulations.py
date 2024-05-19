import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

from frugal_flows.causal_flows import independent_continuous_marginal_flow, get_independent_quantiles, train_frugal_flow
from frugal_flows.bijections import UnivariateNormalCDF

import data.template_causl_simulations as causl_py
import wandb

# Activate automatic conversion of rpy2 objects to pandas objects
pandas2ri.activate()
base = importr('base')
utils = importr('utils')

# Import the R library causl
try:
    causl = importr('causl')
except Exception as e:
    package_names = ('causl')
    utils.install_packages(StrVector(package_names))

CAUSAL_PARAMS = [1, 1]

def calculate_ecdf(Z):
    """
    Calculate the empirical cumulative distribution function (ECDF) for each column of the input array.

    Parameters:
    Z (ndarray): The input array of shape (n_samples, n_features).

    Returns:
    ndarray: The ECDF values for each column of the input array, with shape (n_samples, n_features).

    """
    ecdf_transforms = []
    for d in range(Z.shape[1]):
        data = np.concatenate([Z[:,d], [np.inf, -np.inf]])
        ecdf_values = rankdata(data, method='average') / len(data)
        ecdf_transforms.append(ecdf_values[:-2])
    return jnp.vstack(ecdf_transforms).T


def generate_uz_samples(Z_disc=None, Z_cont=None, use_marginal_flow=False, seed=0, frugal_flow_hyperparams={}):
    """
    Generates the uz_samples for frugal fitting.

    Args:
        Z_disc (array-like, optional): The discrete pretreatment covariates. Defaults to None.
        Z_cont (array-like, optional): The continuous pretreatment covariates. Defaults to None.
        use_marginal_flow (boolean): If true, the quantiles will be learned by fitting a set
            of independent univariate flows to the Z_cont variables. Otherwise, an empirical
            CDF will be calculated.
        seed (int, optional): The random seed. Defaults to 0.
        frugal_flow_hyperparams (dict, optional): Hyperparameters for training the frugal flow model. Defaults to {}.

    Returns:
        dict: A dictionary containing the uz_samples.
    """
    key = jr.PRNGKey(seed)
    key, *subkeys = jax.random.split(key, num=3)

    uz_disc_samples = None
    uz_cont_samples = None
    if Z_disc is None and Z_cont is None:
        print("Warning: No pretreatment covariates provided")
    elif Z_disc is None:
        if use_marginal_flow:
            cont_marg_flow = get_independent_quantiles(
                key = subkeys[0],
                z_cont=Z_cont,
                **frugal_flow_hyperparams
            )
            uz_cont_samples = cont_marg_flow['u_z_cont']
        else:
            uz_cont_samples = calculate_ecdf(Z_cont)
    elif Z_cont is None:
        marg_flow = get_independent_quantiles(
           key = subkeys[0],
           z_discr=Z_disc
        )
        uz_disc_samples = marg_flow['u_z_discr']
    else:
        if use_marginal_flow:
            cont_marg_flow = get_independent_quantiles(
                key = subkeys[0],
                z_cont=Z_cont,
                **frugal_flow_hyperparams
            )
            uz_cont_samples = cont_marg_flow['u_z_cont']
        else:
            uz_cont_samples = calculate_ecdf(Z_cont)
        disc_marg_flow = get_independent_quantiles(
           key = subkeys[0],
           z_discr=Z_disc
        )
        uz_disc_samples = disc_marg_flow['u_z_discr']
    if uz_disc_samples == None:
        uz_samples = uz_cont_samples
    elif uz_cont_samples == None:
        uz_samples = uz_disc_samples
    else:
        uz_samples = jnp.hstack([uz_disc_samples, uz_cont_samples])
    return {
        'uz_disc': uz_disc_samples,
        'uz_cont': uz_cont_samples,
        'uz_samples': uz_samples
    }

def frugal_fitting(X, Y, use_marginal_flow=False, Z_disc=None, Z_cont=None, seed=0, frugal_flow_hyperparams={}, causal_model=None, causal_model_args={}):
    """
    Fits a frugal flow model to the given data.

    Args:
        X (array-like): The treatment variable.
        Y (array-like): The outcome variable.
        uz_samples (dict): A dictionary containing the uz_samples generated by generate_uz_samples.
        seed (int, optional): The random seed. Defaults to 0.
        frugal_flow_hyperparams (dict, optional): Hyperparameters for training the frugal flow model. Defaults to {}.

    Returns:
        dict: A dictionary containing the frugal flow model, the causal margin, and the pretreatment covariate samples.
    """
    key = jr.PRNGKey(seed)
    key, *subkeys = jax.random.split(key, num=3)
    uz_samples = generate_uz_samples(Z_disc, Z_cont, use_marginal_flow, seed, frugal_flow_hyperparams)

    uz_disc_samples = uz_samples['uz_disc']
    uz_cont_samples = uz_samples['uz_cont']

    # Learn Frugal Flow
    frugal_flow, losses = train_frugal_flow(
        key=subkeys[1],
        y=Y,
        u_z=uz_samples['uz_samples'],
        condition=X,
        **frugal_flow_hyperparams,
        causal_model=causal_model,
        causal_model_args=causal_model_args
    )
    causal_margin = frugal_flow.bijection.bijections[-1].bijection.bijections[0]
    min_loss = jnp.min(jnp.array(losses['val']))
    output = {
        'frugal_flow': frugal_flow,
        'losses': losses,
        'causal_margin': causal_margin,
        'uz_disc': uz_disc_samples,
        'uz_cont': uz_cont_samples,
        'val_loss': min_loss,
    }
    # wandb.log(data={'val_loss': min_loss})
    return output, min_loss

def run_simulations(data_generating_function: callable, seed: int, num_samples: int, num_iter: int, causal_params: list, hyperparams_dict: dict, causal_model_args: dict) -> pd.DataFrame:
    """
    Run simulations using the provided data generating function.

    Parameters:
    data_generating_function (callable): A function that generates data samples.
    seed (int): The seed for random number generation.
    num_samples (int): The number of samples to generate.
    num_iter (int): The number of iterations to run.
    hyperparams_dict (dict): A dictionary of hyperparameters for frugal_fitting.

    Returns:
    pd.DataFrame: A DataFrame containing the results of the simulations.
    """
    causal_param_names = ['ate', 'const', 'scale']
    results = []
    for i in range(num_iter):
        Z_disc, Z_cont, X, Y = data_generating_function(num_samples, seed=seed+i, causal_params=causal_params).values()
        frugal_fit, losses = causl_py.frugal_fitting(
            X, Y, Z_cont=Z_cont, Z_disc=Z_disc, seed=seed+i,
            frugal_flow_hyperparams=hyperparams_dict,
            causal_model='gaussian',
            causal_model_args=causal_model_args,
        )
        causal_margin = frugal_fit['causal_margin']
        results.append(np.array([
            causal_margin.ate,
            causal_margin.const,
            causal_margin.scale
        ]))
    return pd.DataFrame(np.array(results), columns=causal_param_names)


def generate_data_samples(rscript):
    """
    Generate data samples using an R script.

    Args:
        rscript (str): The R script to be executed. We recommend using the R `causl`
            package for generating these data.

    Returns:
        dict: A dictionary containing JAX arrays of the generated data samples.
            The dictionary has the following keys:
            - 'Z': JAX array of the pretreatment covariates.
            - 'X': JAX array of the treatment variable.
            - 'Y': JAX array of the outcome variable.

    Raises:
        ValueError: If the R script does not return a dataframe object named 'data_samples'
            with a single outcome labelled 'Y', a single treatment labelled 'X', and
            multiple pretreatment covariates labelled 'Z'.
    """
    rcode_compiled = SignatureTranslatedAnonymousPackage(rscript, "powerpack")
    data_xdyc = rcode_compiled.data_samples

    # Extract JAX arrays from the dataframe
    Z_disc = jnp.array(data_xdyc[[col for col in data_xdyc.columns if col.startswith('Zd')]].values).astype(int)
    Z_cont = jnp.array(data_xdyc[[col for col in data_xdyc.columns if col.startswith('Zc')]].values)
    if Z_cont.size == 0:
       Z_cont = None
    if Z_disc.size == 0:
       Z_disc = None
    X = jnp.array(data_xdyc['X'].values)[:, None]
    Y = jnp.array(data_xdyc['Y'].values)[:, None]

    return {'Z_disc': Z_disc, 'Z_cont': Z_cont, 'X': X, 'Y': Y}


def generate_gaussian_samples(N, causal_params, seed=0):
    gaussian_rscript = f"""
    library(causl)
    pars <- list(Zc1 = list(beta = 0, phi=1),
                 Zc2 = list(beta = c(1,1), phi=1),
                 Zc3 = list(beta = c(1,1), phi=1),
                 Zc4 = list(beta = c(0,1,1,1), phi=0.5),
                 X = list(beta = c(0,1,1,1)),
                 Y = list(beta = c({causal_params[0]}, {causal_params[1]}), phi=1),
                 cop = list(beta=matrix(c(2,1,0.5,1,1,1,1,1,1,1), nrow=1)))
    
    set.seed({seed})  # for consistency
    fams <- list(c(1,1,1,1),5,1,1)
    data_samples <- causalSamp({N}, formulas=list(list(Zc1~1, Zc2~Zc1, Zc3~Zc1, Zc4~Zc3+Zc2+Zc1), X~Zc1+Zc2+Zc3, Y~X, ~1), family=fams, pars=pars)
    """
    data = generate_data_samples(gaussian_rscript)
    return data


def generate_mixed_samples(N, causal_params, seed=0):
    mixed_cont_rscript = f"""
    library(causl)
    pars <- list(Zc1 = list(beta = c(1), phi=1),
                 Zc2 = list(beta = c(1), phi=1),
                 Zc3 = list(beta = c(1), phi=1),
                 Zc4 = list(beta = c(1), phi=1),
                 X = list(beta = c(-2,1,1,1,1)),
                 Y = list(beta = c({causal_params[0]}, {causal_params[1]}), phi=1),
                 cop = list(beta=matrix(c(0.5,0.3,0.1,0.1,
                                              0.4,0.1,0.1,
                                                  0.1,0.1,
                                                      0.1), nrow=1)))
    
    set.seed({seed})  # for consistency
    fams <- list(c(3,3,3,3),5,1,1)
    data_samples <- causalSamp({N}, formulas=list(list(Zc1~1, Zc2~1, Zc3~1, Zc4~1), X~Zc1+Zc2+Zc3+Zc4, Y~X, ~1), family=fams, pars=pars)
    """
    data = generate_data_samples(mixed_cont_rscript)
    return data


def generate_discrete_samples(N, causal_params, seed=0):
    disc_rscript = f"""
    library(causl)
    forms <- list(list(Zc1 ~ 1, Zc2 ~ 1, Zd3 ~ 1, Zd4 ~ 1), X ~ Zc1*Zc2+Zd3+Zd4, Y ~ X, ~ 1)
    fams <- list(c(1,1,5,5), 5, 1, 1)
    pars <- list(Zc1 = list(beta=0, phi=1),
                Zc2 = list(beta=0, phi=2),
                Zd3 = list(beta=0),
                Zd4 = list(beta=0),
                X = list(beta=c(-0.3,0.1,0.2,0.5,-0.2,1)),
                Y = list(beta=c({causal_params[0]}, {causal_params[1]}), phi=1),
                cop = list(beta=matrix(c(0.5,0.3,0.1,0.1,
                                             0.4,0.1,0.1,
                                                 0.1,0.1,
                                                     0.1), nrow=1)))
    set.seed({seed})
    data_samples <- rfrugalParam({N}, formulas = forms, family = fams, pars = pars)
    """
    data = generate_data_samples(disc_rscript)
    return data


def generate_many_discrete_samples(N, causal_params, seed=0):
    disc_rscript = f"""
    library(causl)
    forms <- list(list(Zc1 ~ 1, Zc2 ~ 1, Zc3 ~ 1, Zc4 ~ 1, Zc5 ~ 1, Zd1 ~ 1, Zd2 ~ 1, Zd3 ~ 1, Zd4 ~ 1, Zd5 ~ 1), X ~ Zc1+Zc2+Zc3+Zc4+Zc5+Zd1+Zd2+Zd3+Zd4+Zd5, Y ~ X, ~ 1)
    fams <- list(c(1,1,1,1,1,5,5,5,5,5), 5, 1, 1)
    pars <- list(Zc1 = list(beta=0, phi=1),
                Zc2 = list(beta=0, phi=1),
                Zc3 = list(beta=0, phi=1),
                Zc4 = list(beta=0, phi=1),
                Zc5 = list(beta=0, phi=1),
                Zd1 = list(beta=0),
                Zd2 = list(beta=0),
                Zd3 = list(beta=0),
                Zd4 = list(beta=0),
                Zd5 = list(beta=0),
                X = list(beta=c(-0.9,0.3,0.6,1.5,-0.6,3,0.9,-1.2,2.1,-0.3,2.7)),
                Y = list(beta=c({causal_params[0]}, {causal_params[1]}), phi=1),
                cop = list(beta=matrix(c(0.3,0.4,0.5,0.1,-0.2,-0.7,0.5,-0.4, 0.5,
                                            -0.3,0.6,-0.3,0.4,-0.4,0.6,0.3,  0.2,
                                                -0.5,0.2,-0.1,-0.1,0.0,-0.4,-0.4,
                                                    -0.2,-0.2,-0.5,0.5,0.3,  0.4,
                                                        -0.1,-0.1,-0.5,-0.6,-0.2,
                                                             -0.0,0.4,0.2,   0.5,
                                                                 -0.5,0.4,  -0.4,
                                                                      0.4,   0.4,
                                                                             0.4), nrow=1)))
    set.seed({seed})
    data_samples <- rfrugalParam({N}, formulas = forms, family = fams, pars = pars)
    """
    data = generate_data_samples(disc_rscript)
    return data


def generate_many_discrete_samples_sparse(N, causal_params, seed=0):
    disc_rscript = f"""
    library(causl)
    forms <- list(list(Zc1 ~ 1, Zc2 ~ 1, Zc3 ~ 1, Zc4 ~ 1, Zc5 ~ 1, Zd1 ~ 1, Zd2 ~ 1, Zd3 ~ 1, Zd4 ~ 1, Zd5 ~ 1), X ~ Zc1+Zc2+Zc3+Zc4+Zc5+Zd1+Zd2+Zd3+Zd4+Zd5, Y ~ X, ~ 1)
    fams <- list(c(1,1,1,1,1,5,5,5,5,5), 5, 1, 1)
    pars <- list(Zc1 = list(beta=0, phi=1),
                Zc2 = list(beta=0, phi=1),
                Zc3 = list(beta=0, phi=1),
                Zc4 = list(beta=0, phi=1),
                Zc5 = list(beta=0, phi=1),
                Zd1 = list(beta=0),
                Zd2 = list(beta=0),
                Zd3 = list(beta=0),
                Zd4 = list(beta=0),
                Zd5 = list(beta=0),
                X = list(beta=c(-0.9,0.3,0.6,1.5,-0.6,3,0.9,-1.2,2.1,-0.3,2.7)),
                Y = list(beta=c({causal_params[0]}, {causal_params[1]}), phi=1),
                cop = list(beta=matrix(c(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                                             0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                                                 0.0,0.0,0.0,0.0,0.0,0.0,0.5,
                                                     0.0,0.0,0.0,0.0,0.0,0.0,
                                                         0.0,0.0,0.0,0.0,0.0,
                                                             0.0,0.0,0.0,0.0,
                                                                 0.0,0.0,0.8,
                                                                     0.0,0.0,
                                                                         0.7), nrow=1)))
    set.seed({seed})
    data_samples <- rfrugalParam({N}, formulas = forms, family = fams, pars = pars)
    """
    data = generate_data_samples(disc_rscript)
    return data