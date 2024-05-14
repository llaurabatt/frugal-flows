import numpy as np
import jax 
import jax.numpy as jnp
import pandas as pd
import scipy.stats as ss

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

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

# Set random seed
np.random.seed(42)

# CORRELATION_MATRIX = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.4], [0.6, 0.4, 1.0]])


# # Select marginal distributions of Z
# MARGINAL_Z = {
#     #    "Z1": ss.bernoulli(p=0.5),
#     #    "Z2": ss.poisson(mu=5.0),
#     "Z1": ss.norm(loc=0, scale=1),
#     "Z2": ss.norm(loc=0, scale=1),
# }

# N = 2000
# TREATMENT_TYPE = "C"
# OUTCOME_TYPE = "C"
# PROP_SCORE_WEIGHTS = [2, 2]  # Check propscore weights are of same dim as Z
# # PROP_SCORE_WEIGHTS = [0, -0]  # Check propscore weights are of same dim as Z
# OUTCOME_WEIGHTS = [1, 1]

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
    Z = jnp.array(data_xdyc[[col for col in data_xdyc.columns if col not in ['X', 'Y']]].values)
    X = jnp.array(data_xdyc['X'].values)[:, None]
    Y = jnp.array(data_xdyc['Y'].values)[:, None]

    return {'Z': Z, 'X': X, 'Y': Y}
