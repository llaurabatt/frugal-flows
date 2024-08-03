import sys
import jax
import jax.random as jr
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import time
import torch
from torch_two_sample import MMDStatistic, EnergyStatistic, FRStatistic, KNNStatistic
import wandb

# sys.path.append("../")  # go to parent dir
from frugal_flows.causal_flows import get_independent_quantiles, train_frugal_flow
from frugal_flows.sample_outcome import sample_outcome
from frugal_flows.sample_marginals import from_quantiles_to_marginal_cont, from_quantiles_to_marginal_discr
from frugal_flows.train_quantile_propensity_score import train_quantile_propensity_score
from frugal_flows.bijections import MaskedAutoregressiveHeterogeneous

# sys.path.append("../data/analysis/")  # go to parent dir
import data.template_causl_simulations as causl_py



hyperparam_dict = {
    "RQS_knots": 8,
    "nn_depth": 4,
    "nn_width": 50,
    "flow_layers": 4,
    "learning_rate": 5e-3,
    "max_epochs": 1000,
    "max_patience": 100,
}

class FrugalFlowModel:
    def __init__(self, Y, X, Z_disc=None, Z_cont=None, confounding_copula=None):
        self.Y = Y
        self.X = X
        self.Z_disc = Z_disc
        self.Z_cont = Z_cont
        self.subkeys =         self.res = None
        self.frugal_flow = None
        self.min_val_loss = None
        self.vmap_frugal_flow = None
        self.prop_flow = None
        self.vmap_prop_flow = None
        self.confounding_copula = confounding_copula
        if confounding_copula is None:
            self.confounding_copula = self._bivariate_gaussian_copula

    def _bivariate_gaussian_copula(self, key, N, rho):
        corr_matrix = jnp.array([
            [1., rho],
            [rho, 1.]
        ])
        mean = jnp.array([0,0])
        quantiles = jax.scipy.special.ndtr(
            jr.multivariate_normal(key=key, mean=mean, cov=corr_matrix, shape=(N,))
        )
        return quantiles[:, 0], quantiles[:, 1]

    def train_benchmark_model(self, 
                              training_seed,
                              marginal_hyperparam_dict, 
                              frugal_hyperparam_dict, 
                              causal_model, 
                              causal_model_args, 
                              prop_flow_hyperparam_dict):
        training_seeds = jr.split(training_seed, 20)
        self.train_marginal_cdfs(training_seeds[0], marginal_hyperparam_dict)
        self.train_frugal_flow(training_seeds[1], frugal_hyperparam_dict, causal_model, causal_model_args)
        self.train_propensity_flow(training_seeds[2], prop_flow_hyperparam_dict)

    def train_marginal_cdfs(self, key, marginal_hyperparam_dict):
        self.res = get_independent_quantiles(
            key=key,
            z_cont=self.Z_cont,
            z_discr=self.Z_disc,
            # max_epochs=hyperparam_dict["max_epochs"],
            # max_patience=hyperparam_dict["max_patience"],
            return_z_cont_flow=True,
            **marginal_hyperparam_dict
        )

    def train_frugal_flow(self, key, hyperparam_dict, causal_model, causal_model_args):
        uz_full_samples = jnp.hstack([self.res['u_z_cont'], self.res['u_z_discr']])
        self.frugal_flow, losses = train_frugal_flow(
            key=key,
            y=self.Y,
            u_z=uz_full_samples,
            condition=self.X,
            causal_model=causal_model,
            causal_model_args=causal_model_args,
            **hyperparam_dict
        )
        self.min_val_loss = jnp.min(jnp.array(losses['val']))
        self.vmap_frugal_flow = jax.vmap(fun=self.frugal_flow.bijection.transform, in_axes=(0))

    def train_propensity_flow(self, key, hyperparam_dict):
        self.prop_flow, _ = train_quantile_propensity_score(
            key=key,
            x=self.X.astype(int),
            condition=jnp.hstack([self.Z_disc, self.Z_cont]),
            **hyperparam_dict
        )
        prop_flow_cdf = self.prop_flow.bijection.transform
        self.vmap_prop_flow = jax.vmap(prop_flow_cdf, in_axes=(0,))

    def generate_samples(self, key, sampling_size, copula_param, outcome_causal_model, outcome_causal_args):
        subkeys = jr.split(key, 4)

        # Generate U*_y|x and U_x|z quantiles
        u_yx, u_xz = self.confounding_copula(subkeys[0], sampling_size, copula_param)
        u_yx = u_yx[:, None]
        u_xz = u_xz[:, None]
        
        # Sample U_z quantiles from frugal flow
        baseline_uz = jr.uniform(key=subkeys[1], shape=(sampling_size, 4))
        frugal_baselines = jnp.hstack([u_yx, baseline_uz])
        uz_samples = self.vmap_frugal_flow(x=frugal_baselines, condition=jnp.zeros(u_yx.shape))[:, 1:]

        # Inverse probability integral transform
        Z_cont_samples = from_quantiles_to_marginal_cont(
            key=subkeys[2],
            flow=self.res['z_cont_flows'],
            n_samples=sampling_size,
            u_z=uz_samples[:, :self.Z_cont.shape[1]]
        )
        Z_disc_samples = from_quantiles_to_marginal_discr(
            key=subkeys[3],
            mappings=self.res['z_discr_rank_mapping'],
            empirical_cdfs=self.res['z_discr_empirical_cdf_long'],
            nvars=self.res['u_z_discr'].shape[1],
            n_samples=sampling_size,
            u_z=uz_samples[:, self.Z_cont.shape[1]:]
        )
        full_Z_samples = jnp.hstack([Z_cont_samples, Z_disc_samples])

        # Calculate X quantiles
        u_x = self.vmap_prop_flow(u_xz, condition=full_Z_samples)
        ## Assumes X is binary treatment
        X_samples = (u_x > (1 - jnp.mean(self.X))).astype(int)
        
        # Sample outcomes
        if outcome_causal_model == 'location_translation':
            Y_samples = sample_outcome(
                frugal_flow=self.frugal_flow,
                key=subkeys[4],
                n_samples=sampling_size,
                causal_model=outcome_causal_model,
                causal_condition=X_samples[:, None],
                u_yx=u_yx.flatten(),
                **outcome_causal_args
            )
        else:
            Y_samples = sample_outcome(
                key=subkeys[4],
                n_samples=sampling_size,
                causal_model=outcome_causal_model,
                causal_condition=X_samples[:, None],
                u_yx=u_yx.flatten(),
                **outcome_causal_args
            )
        sim_data = np.hstack([Y_samples[:, None], X_samples, full_Z_samples])
        sim_data_df = pd.DataFrame(sim_data, columns=['Y', 'X', *[f"Z_{i+1}" for i in range(full_Z_samples.shape[1])]])
        # model_fits = valMethods.run_model_fits('Y', 'X', sim_data_df, sample_frac=0.8, repeats=1, replace=True)
        return sim_data_df


def compare_datasets(real_data, synth_data, alphas, k, n_permutations=1000, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_real = real_data.shape[0]
    n_synth = synth_data.shape[0]

    # Convert to torch tensors
    real_samples_var = torch.tensor(real_data, dtype=torch.float32)
    synth_samples_var = torch.tensor(synth_data, dtype=torch.float32)

    # Initialize test statistics
    mmd = MMDStatistic(n_real, n_synth)
    energy = EnergyStatistic(n_real, n_synth)
    fr = FRStatistic(n_real, n_synth)
    knn = KNNStatistic(n_real, n_synth, k)

    # Calculate p-values for each test
    start_time = time.time()
    print('Calculating Metrics...')
    
    start = time.time()
    mmd_matrix = mmd(real_samples_var, synth_samples_var, alphas, ret_matrix=True)[1]
    print(f'MMD Calculated in {time.time() - start:.4f} seconds...')
    
    start = time.time()
    energy_matrix = energy(real_samples_var, synth_samples_var, ret_matrix=True)[1]
    print(f'Energy Calculated in {time.time() - start:.4f} seconds...')

    # start = time.time()
    # fr_matrix = fr(real_samples_var, synth_samples_var, norm=2, ret_matrix=True)[1]
    # print(f'FR Calculated in {time.time() - start:.4f} seconds...')
    
    # start = time.time()
    # knn_matrix = knn(real_samples_var, synth_samples_var, norm=2, ret_matrix=True)[1]
    # print(f'KNN Calculated in {time.time() - start:.4f} seconds...')
    # print(f'Total Metrics Calculation Time: {time.time() - start_time:.4f} seconds')
    results = {
        "MMD pval": mmd.pval(mmd_matrix, n_permutations=n_permutations),
        "Energy pval": energy.pval(energy_matrix, n_permutations=n_permutations),
        # "Friedman-Rafsky pval": fr.pval(fr_matrix, n_permutations=n_permutations),
        # "kNN pval": knn.pval(knn_matrix, n_permutations=n_permutations)
    }

    return results

# real_data = np.random.randn(100, 5)  # Real dataset with 5 features
# synth_data = np.random.randn(100, 5)  # Synthetic dataset with 5 features
# alphas = [0.5, 1.0, 2.0]  # Define kernel parameters for MMD
# k = 3
# 
# results = compare_datasets(real_data, synth_data, alphas, k)
# print(results)


# seed = 0
# N = 500
# sampling_size = 100
# causal_params = [0, 1]
# keys, *subkeys = jr.split(jr.PRNGKey(seed), 20)
# hyperparam_dict = {
#     "RQS_knots": 8,
#     "nn_depth": 4,
#     "nn_width": 50,
#     "flow_layers": 4,
#     "learning_rate": 5e-3,
#     "max_epochs": 1000,
#     "max_patience": 100,
# }
# 
# Z_disc, Z_cont, X, Y = causl_py.generate_discrete_samples(N=N, seed=0, causal_params=causal_params).values()
# benchmark_flow = FrugalFlowModel(Y=Y, X=X, Z_disc=Z_disc, Z_cont=Z_cont, confounding_copula=None)
# benchmark_flow.train_benchmark_model(
#     training_seed=jr.PRNGKey(1),
#     marginal_hyperparam_dict=hyperparam_dict,
#     frugal_hyperparam_dict=hyperparam_dict,
#     causal_model='gaussian',
#     causal_model_args={'ate': 0, 'const': 0, 'scale': 1},
#     prop_flow_hyperparam_dict=hyperparam_dict
# )
# sim_data = benchmark_flow.generate_samples(
#     key=jr.PRNGKey(2),
#     sampling_size=1000,
#     copula_param=0.8,
#     outcome_causal_model='causal_cdf',
#     outcome_causal_args={'ate': 5, 'const': 3, 'scale': 0.5}
# )
# print(sim_data)