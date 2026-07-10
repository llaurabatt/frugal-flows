import sys
import jax
import numpy as np
import pandas as pd
from frugal_flows.causal_flows import get_independent_quantiles, train_frugal_flow
from frugal_flows.interventions import interventional_samples
from frugal_flows.outcome_transforms import as_outcome_transform
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
    """End-to-end frugal-flow benchmarking pipeline.

    Orchestrates the full workflow: fit stage-1 marginal CDFs
    (``train_marginal_cdfs``), train the frugal flow with an explicit causal
    parameter (``train_frugal_flow``), fit the quantile propensity score
    (``train_propensity_flow``), then draw a synthetic ``(Y, X, Z)`` dataset
    from a confounding copula (``generate_samples``). ``X`` is assumed binary.

    Args:
        Y: Outcome, shape (n, 1).
        X: Binary treatment, shape (n, 1).
        Z_disc: Discrete confounders, shape (n, n_disc), or None.
        Z_cont: Continuous confounders, shape (n, n_cont), or None.
        confounding_copula: Callable ``(key, N, rho) -> (u_yx, u_xz)``;
            defaults to a bivariate Gaussian copula.
        outcome_transform: ``None`` / kind-string / ``OutcomeTransform`` applied to
            ``Y`` before fitting the causal margin and inverted on sampled outcomes
            (see ``frugal_flows.outcome_transforms``). ``None`` -> identity (no-op,
            fully backward compatible). For a skewed / heavy-tailed outcome that would
            otherwise saturate the spline margin, pass an explicit
            ``OutcomeTransform("log", floor=b)`` or ``OutcomeTransform("asinh", floor=b)``
            (the bare ``"log"``/``"asinh"`` strings are rejected -- ``floor`` must be given).
    """

    def __init__(self, Y, X, Z_disc=None, Z_cont=None, confounding_copula=None,
                 outcome_transform=None):
        self.Y = Y
        self.X = X
        self.outcome_transform = as_outcome_transform(outcome_transform)
        self.Z_disc = Z_disc
        self.Z_cont = Z_cont
        self.conf_shape = 0
        if Z_disc is not None:
            self.conf_shape += self.Z_disc.shape[1]
        if Z_cont is not None:
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
        if self.res['u_z_cont'] is None:
            uz_full_samples = self.res['u_z_discr']
        elif self.res['u_z_discr'] is None:
            uz_full_samples = self.res['u_z_cont']
        else:
            uz_full_samples = jnp.hstack([self.res['u_z_cont'], self.res['u_z_discr']])
        # Fit the causal margin on the (optionally) transformed outcome. inverse is
        # applied to sampled outcomes in generate_samples so the estimand stays on
        # the original Y scale. identity (default) leaves y == self.Y unchanged.
        y_fit = self.outcome_transform.fit(self.Y).forward(self.Y)
        self.frugal_flow, losses = train_frugal_flow(
            key=key,
            y=y_fit,
            u_z=uz_full_samples,
            condition=self.X,
            causal_model=causal_model,
            causal_model_args=causal_model_args,
            **hyperparam_dict
        )
        self.min_val_loss = jnp.min(jnp.array(losses['val']))
        self.vmap_frugal_flow = jax.vmap(fun=self.frugal_flow.bijection.transform, in_axes=(0))

    def sample_do(self, key, t, n_mc):
        """Draw ``n_mc`` samples of ``Y | do(T = t)`` on the ORIGINAL outcome scale.

        Samples the fitted frugal flow's causal margin (output dim 0) at the fixed
        treatment level ``t`` (broadcast to all treatment columns) and inverts the
        outcome transform. ``key`` must be a typed key (``jax.random.key(...)``).
        """
        if self.frugal_flow is None:
            raise RuntimeError("sample_do requires a fitted flow; call train_frugal_flow first")
        cond = jnp.full((n_mc, self.X.shape[1]), float(t))
        y = self.frugal_flow.sample(key, condition=cond)[:, 0]
        return np.asarray(self.outcome_transform.inverse(y))

    def estimate_ate(self, key, n_mc=20000):
        """Model-agnostic ATE = ``E[Y|do(1)] - E[Y|do(0)]`` from the fitted causal margin.

        Paired common-random-number interventional read-out (see
        ``frugal_flows.interventions.interventional_samples``): samples the fitted
        flow at do(0) and do(1) with the SAME base key, inverts the outcome
        transform, and differences -- so the estimand is always on the original
        ``Y`` scale, whatever ``outcome_transform`` was used at fit time. ``key``
        must be a typed key (``jax.random.key(...)``). Requires ``train_frugal_flow``
        to have been run.

        Returns the full read-out dict (``ate``, ``tau_sd``, ``y0``/``y1``,
        means/vars, ...).
        """
        if self.frugal_flow is None:
            raise RuntimeError("estimate_ate requires a fitted flow; call train_frugal_flow first")
        return interventional_samples(
            key, self.frugal_flow, self.X.shape[1], n_mc,
            outcome_transform=self.outcome_transform,
        )

    def train_propensity_flow(self, key, hyperparam_dict):
        if self.Z_disc is None:
            condition = self.Z_cont
        elif self.Z_cont is None:
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
        if self.Z_cont is not None:
            Z_cont_samples = from_quantiles_to_marginal_cont(
                key=subkeys[2],
                flow=self.res['z_cont_flows'],
                n_samples=sampling_size,
                u_z=uz_samples[:, :self.Z_cont.shape[1]]
            )
        if self.Z_disc is not None:
            if self.Z_cont is None:
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
        if self.Z_disc is None:
            full_Z_samples = Z_cont_samples
        elif self.Z_cont is None:
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
        # Invert the outcome transform so sampled Y (and any downstream ATE) is on
        # the ORIGINAL scale; no-op for the identity default.
        Y_samples = np.asarray(self.outcome_transform.inverse(Y_samples))
        print(f"Y shape: {Y_samples.shape}")
        print(f"X shape: {X_samples.shape}")
        print(f"Z shape: {full_Z_samples.shape}")
        sim_data = np.hstack([Y_samples, X_samples, full_Z_samples])
        sim_data_df = pd.DataFrame(sim_data, columns=['Y', 'X', *[f"Z_{i+1}" for i in range(full_Z_samples.shape[1])]])
        # model_fits = valMethods.run_model_fits('Y', 'X', sim_data_df, sample_frac=0.8, repeats=1, replace=True)
        return sim_data_df