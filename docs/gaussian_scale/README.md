# The Gaussian-scale frugal flow

`frugal_flows.gaussian_scale` implements the frugal flow on a standard-normal scale. It is the recommended model for multivariate outcomes such as images. This note covers the model, how to use it, and what it cannot do.

## The model

A frugal flow parametrises the causal margin p(y | do(t)) directly. It fits everything else (the covariates and their dependence on the outcome) with a copula. This version writes the whole model on the Gaussian scale, linked to the uniform-base version by the probit map. A copula is invariant to monotone maps of its margins, so the model class is the same; only the geometry the network fits changes.

The joint flow, from base to data:

```
StandardNormal(K + d)
  -> GaussianCopulaBlock                 identity on the K outcome coordinates; the d covariate
                                         coordinates go through a conditional MAF given them
  -> [ margin(. ; T) , identity(d) ]     the causal margin on the K outcome coordinates
```

The pieces:

- **Causal margin:** a masked autoregressive spline flow (rational-quadratic splines on [−5, 5], identity tails) on a Normal base, with the treatment T as a conditioner of every layer.
  - It maps the outcome to scores that are N(0, I) at the optimum.
  - The outcome columns of `flow.sample(key, condition=t)` depend only on T and the first K base coordinates. So sampling at a fixed t draws from p(y | do(t)) without touching the copula. That is the frugal property.
- **Copula:** a conditional flow for the covariates' normal scores given the outcome scores, zero-initialised at independence.
  - It does not see T.
- **Covariates** enter as normal scores of their ranks.

The density factorises as the margin's density of y given T, times the copula's density of the covariate scores given the outcome scores.

Two margins are available:

| `causal_model` | margin | use |
|---|---|---|
| `flexible_continuous_gaussian` | spline flow conditioned on T | any effect: shifts, scale changes, quantile effects, heterogeneity across the outcome's distribution |
| `location_translation_gaussian` | spline flow blind to T, then a per-column shift `y = m(e) + ate·T` | effects that are pure location shifts |

## Using it

```python
import jax.random as jr
import frugal_flows as ff

flow, ot, info = ff.fit_gaussian_frugal_flow(jr.key(0), Y, Z, T)        # Y (n, K), Z (n, d), T (n,)
s = ff.interventional_samples(jr.key(1), flow, 1, 5000, outcome_transform=ot, dim_y=Y.shape[1])
s["ate"], s["y0"], s["y1"]                                              # ATE (K,), do(0)/do(1) draws
y_cf = ot.inverse(ff.counterfactual_gaussian(flow, ot.forward(Y), T, 1 - T))   # unit counterfactuals
ff.save_gaussian_flow("my_fit", flow, info["build_kwargs"], outcome_transform=ot)
flow, ot = ff.load_gaussian_flow("my_fit")
```

`fit_gaussian_frugal_flow` does three things:
1. Standardises each outcome column. Recommended: the Gaussian scale gains most when the outcome is standardised.
2. Ranks the covariates.
3. Fits with `PAPER_SETTINGS`:
   - margin: MAF width 48, depth 1, 4 layers, 8 knots;
   - copula: width 16, 4 layers, depth 1, 8 knots;
   - training: Adam lr 1e-3, batch 100, up to 1000 epochs, early stopping with patience 30 on a held-out split, best validation checkpoint returned.

The low-level entry point is `train_frugal_flow(..., causal_model="flexible_continuous_gaussian")`, which takes standardised Y and ranks yourself.

The counterfactual is abduction–action–prediction under rank preservation. Each unit keeps its rank under the causal margin when T is switched.

## Evidence (MorphoMNIST 8×8, n = 5000)

The comparison is against the uniform-base flexible flow (`flexible_continuous`) and frengression, over 6 experiments × 10 datasets:
- **Single fits:** the Gaussian spline beats the uniform-base flow on every experiment (0.49–0.70× its ATE MAE). The gain comes from the scale, not from standardisation alone.
- **The paper settings stand:** a 302-fit hyperparameter sweep, on separate tuning datasets, found nothing better.
- **Averaging:** averaging the ATE maps of 5 fits with different seeds improves the flow 1.3–1.5×.
- **Against frengression** (also averaged over 5 fits):

  | Experiment | Result |
  |---|---|
  | E1, E2, E3, E5 | better: 0.68–0.77× |
  | E6 | level: 0.97× |
  | E4 | behind: 1.13× |

- **Realism:** counterfactual images are closer to the true counterfactuals than the uniform flow's (pilot, E2: per-unit error 0.033 vs 0.066 on the logit scale). See `validation/morphomnist/scripts/exp_ate_recovery/realism.py`.

## Known limitation: the copula does not see T

The copula models p(g_Z | g_Y), the same in both treatment arms. The Z–Y dependence (how the outcome's ranks relate to the covariates) therefore cannot differ between treated and control.

When the effect varies with a covariate (a conditional average treatment effect, E4), that variation must be carried by the margin alone, which does not condition on Z. This is the leading explanation for the E4 gap, but it has not been tested yet. The candidate fix is a treatment-conditioned copula. It is not implemented here, and whether it keeps the parametrisation frugal is still open.

## Precision

The MorphoMNIST fits were run in float64. `load_gaussian_flow` loads a fit saved in either precision into the session's precision. Call `frugal_flows.set_x64(True)` before creating arrays to keep float64.
