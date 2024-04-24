import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import scipy.stats as ss

# Set random seed
np.random.seed(42)

# Upper triangle of correlation values for copula
CORRELATION_VALUES = [0.9, 0.6, 0, 0.2,
                           0.5, 0.5, 0,
                                0.2, 0.8,
                                     0.6]

CORRELATION_MATRIX = np.array([
    [1.0, CORRELATION_VALUES[0], CORRELATION_VALUES[1], CORRELATION_VALUES[2], CORRELATION_VALUES[3]],
    [CORRELATION_VALUES[0], 1.0, CORRELATION_VALUES[4], CORRELATION_VALUES[5], CORRELATION_VALUES[6]],
    [CORRELATION_VALUES[1], CORRELATION_VALUES[4], 1.0, CORRELATION_VALUES[7], CORRELATION_VALUES[8]],
    [CORRELATION_VALUES[2], CORRELATION_VALUES[5], CORRELATION_VALUES[7], 1.0, CORRELATION_VALUES[9]],
    [CORRELATION_VALUES[3], CORRELATION_VALUES[6], CORRELATION_VALUES[8], CORRELATION_VALUES[9], 1.0]
])

# Select marginal distributions of Z
MARGINAL_Z = {
    'Z1': ss.bernoulli(p=0.5),
    'Z2': ss.poisson(mu=5.),
    'Z3': ss.gamma(a=3.),
    'Z4': ss.norm(loc=0, scale=3)
}

N = 1000
TREATMENT_TYPE = 'C'
OUTCOME_TYPE = 'C'
PROP_SCORE_WEIGHTS = [2, 0.5, 0, -3] # Check propscore weights are of same dim as Z
OUTCOME_WEIGHTS = [1, -1]


def generate_copula_samples(N: int, correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Generate samples from a multivariate Gaussian distribution.

    Parameters:
    - N (int): Number of samples to generate.
    - correlation_matrix (np.ndarray): Symmetric matrix representing the correlation between variables.

    Returns:
    - np.ndarray: Transformed samples.
    """
    # Generate samples from multivariate Gaussian distribution
    mean = np.zeros(correlation_matrix.shape[0])
    samples = np.random.multivariate_normal(mean, correlation_matrix, N)

    # Transform samples with standard Gaussian univariate CDF
    transformed_samples = ss.norm.cdf(samples)

    return transformed_samples


def transform_z_quantiles_to_samples(z_quantiles: np.ndarray, marginal_z: dict) -> np.ndarray:
    """
    Transform quantiles to samples using marginal distributions.

    Parameters:
    - quantiles (np.ndarray): Matrix of quantiles.
    - marginal_z (dict): Dictionary of marginal distributions.

    Returns:
    - np.ndarray: Matrix of transformed samples.
    """
    marginal_samples = np.zeros_like(z_quantiles)
    for d in range(z_quantiles.shape[1]):
        marginal_samples[:, d] = marginal_z[f'Z{d+1}'].ppf(z_quantiles[:, d])
    return marginal_samples


def sample_treatment(N: int, treatment_type: str, weights: list, Z: np.ndarray) -> np.ndarray:
    """
    Sample treatment based on treatment type and weights.

    Parameters:
    - N (int): Number of data samples to draw.
    - treatment_type (str): Flag indicating treatment type ('C' for continuous, 'D' for discrete).
    - weights (list): List of floats representing weights.
    - Z (np.ndarray): Samples of variables.

    Returns:
    - np.ndarray: Sampled treatment.
    """
    weights = np.array([weights] * N)
    params = np.einsum('ij,ij->i', Z, weights)
    if treatment_type == 'D':
        p = 1 / (1 + np.exp(-params))
        treatment = np.random.binomial(1, p)
    if treatment_type == 'C':
        treatment = np.random.normal(loc=params, scale=1)
    return treatment


def sample_outcome(N: int, outcome_type: str, treatment: np.ndarray, outcome_quantiles: np.ndarray, outcome_weights: list[float]) -> np.ndarray:
    """
    Sample outcome based on outcome type and treatment.

    Parameters:
    - N (int): Number of data samples to draw.
    - outcome_type (str): Flag indicating outcome type ('C' for continuous, 'D' for discrete).
    - X (np.ndarray): Treatment values.
    - outcome_quantiles (np.ndarray): Matrix of outcome quantiles.

    Returns:
    - np.ndarray: Sampled outcome.
    """
    treatment = treatment.reshape(-1, 1)
    treatment = np.concatenate((np.ones((N, 1)), treatment), axis=1)
    print(treatment[:5])
    if outcome_type == 'D':
        p = 1 / (1 + np.exp(-np.dot(treatment, outcome_weights)))
        outcome = ss.binom(1, p).ppf(outcome_quantiles)
    elif outcome_type == 'C':
        mean = np.dot(treatment, outcome_weights)
        outcome = ss.norm(loc=mean, scale=1).ppf(outcome_quantiles)
    else:
        raise ValueError("Invalid outcome type. Please choose 'C' for continuous or 'D' for discrete.")
    return outcome
    

def simulate_data(N: int, correlation_matrix: np.ndarray, marginal_Z: dict, treatment_weights: list[float], treatment_type: str, outcome_weights: list[float], outcome_type: str) -> pd.DataFrame:
    """
    Simulate data based on specified parameters.

    Parameters:
    - N (int): Number of data samples to generate.
    - correlation_matrix (np.ndarray): Symmetric matrix representing the correlation between variables.
    - marginal_Z (dict): Dictionary of marginal distributions for Z variables.
    - treatment_weights (list): List of floats representing weights for treatment.
    - treatment_type (str): Flag indicating treatment type ('C' for continuous, 'D' for discrete).
    - outcome_weights (list): List of floats representing weights for outcome.
    - outcome_type (str): Flag indicating outcome type ('C' for continuous, 'D' for discrete).

    Returns:
    - pd.DataFrame: Simulated data with labeled columns.
    """
    copula_samples = generate_copula_samples(N, correlation_matrix)
    Z_quantile_samples = copula_samples[:,:-1]
    Z_marginal_samples = transform_z_quantiles_to_samples(Z_quantile_samples, marginal_Z)
    causal_quantile_samples = copula_samples[:,-1]
    treatment = sample_treatment(N, treatment_type, treatment_weights, Z_marginal_samples)
    outcome = sample_outcome(N, outcome_type, treatment, causal_quantile_samples, outcome_weights)
    data = np.concatenate([outcome.reshape(-1, 1), treatment.reshape(-1, 1), Z_marginal_samples], axis=1)
    data = pd.DataFrame(data, columns=['Y', 'X', 'Z1', 'Z2', 'Z3', 'Z4'])
    return data


def main():
    data_xcyc = simulate_data(N, CORRELATION_MATRIX, MARGINAL_Z, PROP_SCORE_WEIGHTS, 'C', OUTCOME_WEIGHTS, 'C')
    data_xcyd = simulate_data(N, CORRELATION_MATRIX, MARGINAL_Z, PROP_SCORE_WEIGHTS, 'C', OUTCOME_WEIGHTS, 'D')
    data_xdyc = simulate_data(N, CORRELATION_MATRIX, MARGINAL_Z, PROP_SCORE_WEIGHTS, 'D', OUTCOME_WEIGHTS, 'C')
    data_xdyd = simulate_data(N, CORRELATION_MATRIX, MARGINAL_Z, PROP_SCORE_WEIGHTS, 'D', OUTCOME_WEIGHTS, 'D')

    # Save each file locally
    data_xcyc.to_csv('data_xcyc.csv', index=False)
    data_xcyd.to_csv('data_xcyd.csv', index=False)
    data_xdyc.to_csv('data_xdyc.csv', index=False)
    data_xdyd.to_csv('data_xdyd.csv', index=False)


if __name__ == '__main__':
    main()