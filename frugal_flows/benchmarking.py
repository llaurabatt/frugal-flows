import sys
import jax
import numpy as np
import pandas as pd
from frugal_flows.causal_flows import get_independent_quantiles, train_frugal_flow
from frugal_flows.sample_outcome import sample_outcome
from frugal_flows.sample_marginals import from_quantiles_to_marginal_cont, from_quantiles_to_marginal_discr
from frugal_flows.train_quantile_propensity_score import train_quantile_propensity_score
import wandb

sys.path.append("../")  # go to parent dir
# sys.path.append("../data/analysis/")  # go to parent dir
# import data.template_causl_simulations as causl_py


import jax.random as jr
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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
        self.conf_shape = 0
        if Z_disc != None:
            self.conf_shape += self.Z_disc.shape[1]
        if Z_cont != None:
            self.conf_shape += self.Z_cont.shape[1]
        self.res = None
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

    def train_marginal_cdfs(self, key, hyperparam_dict):
        self.res = get_independent_quantiles(
            key=key,
            z_cont=self.Z_cont,
            z_discr=self.Z_disc,
            max_epochs=hyperparam_dict["max_epochs"],
            max_patience=hyperparam_dict["max_patience"],
            return_z_cont_flow=True
        )

    def train_frugal_flow(self, key, hyperparam_dict, causal_model, causal_model_args):
        if self.res['u_z_cont'] == None:
            uz_full_samples = self.res['u_z_discr']
        elif self.res['u_z_discr'] == None:
            uz_full_samples = self.res['u_z_cont']
        else:
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
        if self.Z_disc == None:
            condition = self.Z_cont
        elif self.Z_cont == None:
            condition = self.Z_disc
        else:
            condition = jnp.hstack([self.Z_disc, self.Z_cont])
        self.prop_flow, _ = train_quantile_propensity_score(
            key=key,
            x=self.X.astype(int),
            condition=condition,
            **hyperparam_dict
        )
        prop_flow_cdf = self.prop_flow.bijection.transform
        self.vmap_prop_flow = jax.vmap(prop_flow_cdf, in_axes=(0,))

    def generate_samples(self, key, sampling_size, copula_param, outcome_causal_model, outcome_causal_args, with_confounding=True):
        subkeys = jr.split(key, 4)

        # Generate U*_y|x and U_x|z quantiles
        u_yx, u_xz = self.confounding_copula(subkeys[0], sampling_size, copula_param)
        u_yx = u_yx[:, None]
        u_xz = u_xz[:, None]
        
        # Sample U_z quantiles from frugal flow
        baseline_uz = jr.uniform(key=subkeys[1], shape=(sampling_size, self.conf_shape))
        frugal_baselines = jnp.hstack([u_yx, baseline_uz])
        uz_samples = self.vmap_frugal_flow(x=frugal_baselines, condition=jnp.zeros(u_yx.shape))[:, 1:]

        # Inverse probability integral transform
        if self.Z_cont != None:
            Z_cont_samples = from_quantiles_to_marginal_cont(
                key=subkeys[2],
                flow=self.res['z_cont_flows'],
                n_samples=sampling_size,
                u_z=uz_samples[:, :self.Z_cont.shape[1]]
            )
        if self.Z_disc != None:
            if self.Z_cont == None:
                print(uz_samples.shape)
                Z_disc_samples = from_quantiles_to_marginal_discr(
                    key=subkeys[3],
                    mappings=self.res['z_discr_rank_mapping'],
                    empirical_cdfs=self.res['z_discr_empirical_cdf_long'],
                    nvars=self.res['u_z_discr'].shape[1],
                    n_samples=sampling_size,
                    u_z=uz_samples
                )
            else:
                Z_disc_samples = from_quantiles_to_marginal_discr(
                    key=subkeys[3],
                    mappings=self.res['z_discr_rank_mapping'],
                    empirical_cdfs=self.res['z_discr_empirical_cdf_long'],
                    nvars=self.res['u_z_discr'].shape[1],
                    n_samples=sampling_size,
                    u_z=uz_samples[:, self.Z_cont.shape[1]:]
                )
        if self.Z_disc == None:
            full_Z_samples = Z_cont_samples
        elif self.Z_cont == None:
            full_Z_samples = Z_disc_samples
        else:
            full_Z_samples = jnp.hstack([Z_cont_samples, Z_disc_samples])

        # Calculate X quantiles
        if with_confounding:
            u_x = self.vmap_prop_flow(u_xz, condition=full_Z_samples)
        elif not with_confounding:
            u_x = u_xz.copy()
        else:
            print("ERROR: Must specify propensity function.")
        ## Assumes X is binary treatment
        X_samples = (u_x > (1 - jnp.mean(self.X))).astype(int)
        
        # Sample outcomes
        if outcome_causal_model == 'location_translation':
            Y_samples = sample_outcome(
                frugal_flow=self.frugal_flow,
                key=subkeys[4],
                n_samples=sampling_size,
                causal_model=outcome_causal_model,
                causal_condition=X_samples,
                u_yx=u_yx.flatten(),
                **outcome_causal_args
            )[:, None]
        else:
            Y_samples = sample_outcome(
                key=subkeys[4],
                n_samples=sampling_size,
                causal_model=outcome_causal_model,
                causal_condition=X_samples,
                u_yx=u_yx.flatten(),
                **outcome_causal_args
            )[:, None]
        sim_data = np.hstack([Y_samples, X_samples, full_Z_samples])
        sim_data_df = pd.DataFrame(sim_data, columns=['Y', 'X', *[f"Z_{i+1}" for i in range(full_Z_samples.shape[1])]])
        # model_fits = valMethods.run_model_fits('Y', 'X', sim_data_df, sample_frac=0.8, repeats=1, replace=True)
        return sim_data_df