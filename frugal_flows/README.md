# Frugal Flows


frugal_flows/
├── __init__.py                          # exports masked_independent_flow only
├── benchmarking.py                      # FrugalFlowModel: end-to-end pipeline class
├── causal_flows.py                      # all flow training functions
├── basic_flows.py                       # flow architecture constructors
├── sample_outcome.py                    # outcome sampling from a trained flow
├── sample_marginals.py                  # quantile → original-scale inversion
├── train_quantile_propensity_score.py   # propensity score model
└── bijections/
    ├── loc_cond.py                      # LocCond bijection
    ├── univariate_normal_cdf.py         # UnivariateNormalCDF bijection
    ├── masked_independent.py            # MaskedIndependent bijection
    ├── masked_autoregressive_*.py       # several MAF variants
    └── rational_quadratic_spline_additive_cond.py


**The bijection stack, concretely.**

Scalar FF (current):

Uniform[-1,1]^{1+D}
  → MAF (first coord fixed, transforms the rest)   [learns copula c(V_Z | R)]
  → Invert(Affine)                                 [NonTrainable: [-1,1]→[0,1]]
  → UnivariateNormalCDF at index 0                 [NonTrainable: Y→R, scalar]


Architecture B:

Uniform[-1,1]^{K+D}
  → MAF (first K coords fixed, transforms the rest)  [learns copula c(V_Z | R_1..K)]
  → Invert(Affine)                                   [NonTrainable: [-1,1]→[0,1]]
  → AutoregressiveCausalMargin(T) at indices 0:K     [NonTrainable: Y→R, K-dim]
    with Identity at indices K:K+D                   [V_Z passes through unchanged]
