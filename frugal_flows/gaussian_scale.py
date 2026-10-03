"""The frugal flow on the Gaussian scale, for multivariate outcomes (images, many columns).

Quick start (``Y`` (n, K) outcomes, ``Z`` (n, d) continuous covariates, ``T`` (n,) treatment)::

    import jax.random as jr
    from frugal_flows import fit_gaussian_frugal_flow, interventional_samples, counterfactual_gaussian

    flow, ot, info = fit_gaussian_frugal_flow(jr.key(0), Y, Z, T)
    s = interventional_samples(jr.key(1), flow, 1, 5000, outcome_transform=ot, dim_y=Y.shape[1])
    ate = s["ate"]                                          # (K,) on the original Y scale
    y_cf = ot.inverse(counterfactual_gaussian(flow, ot.forward(Y), T, 1 - T))   # unit counterfactuals

Selected through ``train_frugal_flow`` as ``causal_model="flexible_continuous_gaussian"`` (flexible
margin) or ``"location_translation_gaussian"`` (shift margin). Save and reload a fit with
``save_gaussian_flow`` / ``load_gaussian_flow``.

Every other arm in ``causal_flows`` works with uniform ranks: a Uniform base, a spline on a
bounded interval, ``atanh`` to reach the real line, and a copula over ranks in (0, 1). Here the
same model class is written on the standard-normal scale, tied to the uniform one by the probit
map ``g = Phi^{-1}(u)`` (a copula is invariant to monotone marginal maps, so the class is
unchanged; only the geometry the flow has to fit changes):

* the causal margin maps the outcome ``y`` (K columns) to scores ``g_Y`` that are N(0, I) at the
  optimum: a Normal base, an RQS spline on [-interval, interval] with identity tails, no ``tanh``;
* each covariate enters as a normal score ``g_Z = Phi^{-1}(u_Z)``;
* the copula is a conditional flow ``p(g_Z | g_Y)`` on a Normal base and does not see T.

The joint flow, base -> data, is

    StandardNormal(K + nvars)
      -> GaussianCopulaBlock   identity on the first K coords; the last nvars coords are pushed
                               through a conditional MAF given the first K (T is ignored)
      -> Concatenate([margin(T), Identity(nvars)])   the causal margin on the first K coords

so ``flow.sample(key, condition=T)`` returns ``(n, K + nvars)`` with the outcome in the first K
columns, and those columns depend only on the first K base coordinates and T (the frugal
property: the causal margin is sampled without the copula). The density factorises as
``p(y | T) * p(g_Z | g_Y)`` with ``g_Y = margin^{-1}(y; T)``. No uniform appears inside the model
(``Phi`` saturates in float32 beyond |g| ~ 5.4); uniforms are for reporting only.

Two margins:
``"flexible"``  each layer is ``MaskedAutoregressiveSpread`` with T as an unmasked conditioner
                (the Normal-base counterpart of ``flexible_continuous``);
``"shift"``     the same stack with T fed but MASKED from every output
                (``MaskedAutoregressiveMaskedCond``), then ``LocCond``: ``y = margin(e) + ate * T``
                per outcome column (the Normal-base counterpart of ``location_translation``).

Both margin stacks follow flowjax's ``masked_autoregressive_flow`` key splitting, so the same key
gives the same margin weights.

Known limitation: the copula ``p(g_Z | g_Y)`` does not see T, so the dependence between covariates
and the outcome's ranks is the same in both treatment arms. Effects that vary with a covariate are
carried by the margin alone. Standardising Y per column before fitting (``OutcomeTransform
("standardize")``, done by ``fit_gaussian_frugal_flow``) is the recommended default.
"""
from __future__ import annotations

import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import paramax
from flowjax.bijections import (
    AbstractBijection,
    Chain,
    Concatenate,
    Invert,
    MaskedAutoregressive,
    RationalQuadraticSpline,
    Scan,
)
from flowjax.bijections.utils import Identity
from flowjax.distributions import StandardNormal, Transformed
from flowjax.flows import _add_default_permute
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from frugal_flows.bijections.loc_cond import LocCond
from frugal_flows.bijections.masked_autoregressive_masked_cond import (
    MaskedAutoregressiveMaskedCond,
)
from frugal_flows.bijections.masked_autoregressive_spread import (
    MaskedAutoregressiveSpread,
)
from frugal_flows.training import fit_to_data

#: default clip for the probit of a uniform: Phi^{-1}(1e-6) = -4.75, finite in float32
NORMAL_SCORE_EPS = 1e-6
#: spline half-width on the Gaussian scale (identity outside)
DEFAULT_INTERVAL = 5.0
MARGINS = ("flexible", "shift")
#: the two ``train_frugal_flow`` names and the margin each selects
CAUSAL_MODELS = {"flexible_continuous_gaussian": "flexible", "location_translation_gaussian": "shift"}
#: margin size used when ``causal_model_args`` leaves a key out (the settings of the MorphoMNIST paper grid)
MARGIN_DEFAULTS = {"RQS_knots": 8, "nn_depth": 1, "nn_width": 48, "flow_layers": 4}


# ------------------------------------------------------------------ scales
def normal_scores_from_uniform(u: ArrayLike, eps: float = NORMAL_SCORE_EPS):
    """``Phi^{-1}(clip(u, eps, 1 - eps))``: finite for every u in [0, 1]."""
    u = jnp.asarray(u)
    return norm.ppf(jnp.clip(u, eps, 1.0 - eps))


def uniform_from_normal_scores(g: ArrayLike):
    """``Phi(g)``. For reporting only: never used inside the model."""
    return norm.cdf(jnp.asarray(g))


# ------------------------------------------------------------------ margins
def _normal_stack(key, dim, cond_dim, causal_model_args, mask_cond: bool):
    """Invert(Scan(L x [MAF layer with RQS(knots, interval), Permute])) with flowjax
    ``masked_autoregressive_flow``'s key split per layer. ``mask_cond``: T fed but masked from
    every output (``MaskedAutoregressiveMaskedCond``); else T unmasked (``MaskedAutoregressiveSpread``)."""
    a = {**MARGIN_DEFAULTS, **(causal_model_args or {})}
    transformer = RationalQuadraticSpline(knots=a["RQS_knots"], interval=a.get("interval", DEFAULT_INTERVAL))

    def make_layer(k):
        bk, pk = jr.split(k)
        if mask_cond:
            b = MaskedAutoregressiveMaskedCond(key=bk, transformer=transformer, dim=dim, cond_dim_mask=cond_dim,
                                               nn_width=a["nn_width"], nn_depth=a["nn_depth"])
        else:
            b = MaskedAutoregressiveSpread(key=bk, transformer=transformer, dim=dim, cond_dim=cond_dim,
                                           nn_width=a["nn_width"], nn_depth=a["nn_depth"])
        return _add_default_permute(b, dim, pk)

    return Invert(Scan(eqx.filter_vmap(make_layer)(jr.split(key, a["flow_layers"]))))


def gaussian_margin_flexible(key, dim: int, condition: ArrayLike, causal_model_args: dict) -> AbstractBijection:
    """Flexible causal margin on a Normal base: T is an unmasked conditioner of every layer."""
    return _normal_stack(key, dim, jnp.shape(condition)[1], causal_model_args, mask_cond=False)


def gaussian_margin_shift(key, dim: int, condition: ArrayLike, causal_model_args: dict) -> AbstractBijection:
    """Location-translation margin on a Normal base: the stack with T masked, then
    ``LocCond``: ``y = stack(e) + ate * T[0]``. ``causal_model_args["ate"]`` (scalar or length
    ``dim``; default 0) initialises the shift vector."""
    cond_dim = jnp.shape(condition)[1]
    ate = jnp.atleast_1d(jnp.asarray(causal_model_args.get("ate", 0.0), dtype=float))
    if ate.shape == (1,):
        ate = jnp.broadcast_to(ate, (dim,))
    if ate.shape != (dim,):
        raise ValueError(f"ate has length {ate.shape[0]} but the outcome has {dim} column(s)")
    stack = _normal_stack(key, dim, cond_dim, causal_model_args, mask_cond=True)
    return Chain([stack, LocCond(ate=ate, cond_dim=cond_dim)])


MARGIN_BUILDERS = {"flexible": gaussian_margin_flexible, "shift": gaussian_margin_shift}


# ------------------------------------------------------------------ copula
class GaussianCopulaBlock(AbstractBijection):
    """Bijection on ``(g_Y, e_Z) in R^{K + nvars}``: identity on the first K coordinates; the
    last nvars go through ``flow`` (a conditional bijection with ``cond_shape == (K,)``) given
    the first K. The block's own ``cond_shape`` is the treatment's, so it can sit in one Chain
    with the margin, but it never reads the condition: T does not enter the copula.

    Forward (base -> data): ``(g_Y, e_Z) -> (g_Y, flow(e_Z | g_Y))``.
    Inverse: ``(g_Y, g_Z) -> (g_Y, flow^{-1}(g_Z | g_Y))``. log|det| = the flow's.
    """

    flow: AbstractBijection
    dim_y: int = eqx.field(static=True)
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def __init__(self, flow: AbstractBijection, dim_y: int, cond_dim: int | None):
        if flow.cond_shape != (dim_y,) or len(flow.shape) != 1:
            raise ValueError(f"copula flow must have cond_shape ({dim_y},); got {flow.cond_shape}")
        self.flow = flow
        self.dim_y = int(dim_y)
        self.shape = (dim_y + flow.shape[0],)
        self.cond_shape = None if cond_dim is None else (int(cond_dim),)

    def transform_and_log_det(self, x, condition=None):
        g_y, e_z = x[: self.dim_y], x[self.dim_y:]
        g_z, log_det = self.flow.transform_and_log_det(e_z, g_y)
        return jnp.concatenate([g_y, g_z]), log_det

    def inverse_and_log_det(self, y, condition=None):
        g_y, g_z = y[: self.dim_y], y[self.dim_y:]
        e_z, log_det = self.flow.inverse_and_log_det(g_z, g_y)
        return jnp.concatenate([g_y, e_z]), log_det


def _zero_final_layer(b):
    """Zero the conditioner's output layer: spline parameters 0 = the identity spline."""
    zero = lambda lin: jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else x, lin)
    return eqx.tree_at(lambda m: m.masked_autoregressive_mlp.layers[-1], b, replace_fn=zero)


def gaussian_copula_block(key, dim_y: int, nvars: int, cond_dim: int | None, RQS_knots: int = 8,
                          nn_width: int = 50, nn_depth: int = 1, flow_layers: int = 4,
                          interval: float = DEFAULT_INTERVAL, zero_init: bool = True) -> GaussianCopulaBlock:
    """``flow_layers`` flowjax ``MaskedAutoregressive`` layers over the nvars covariate scores with
    ``cond_dim = dim_y`` (+ the default permutation), ``Invert(Scan(...))`` as flowjax's
    ``masked_autoregressive_flow`` builds it. The g_Y inputs have rank -1, so every hidden unit
    sees every g_Y. ``zero_init`` zeroes each conditioner's output layer, so at initialisation
    every spline is the identity and the copula is independence (up to a permutation of the
    covariate coordinates, which preserves N(0, I))."""
    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=interval)

    def make_layer(k):
        bk, pk = jr.split(k)
        b = MaskedAutoregressive(key=bk, transformer=transformer, dim=nvars, cond_dim=dim_y,
                                 nn_width=nn_width, nn_depth=nn_depth)
        if zero_init:
            b = _zero_final_layer(b)
        return _add_default_permute(b, nvars, pk)

    flow = Invert(Scan(eqx.filter_vmap(make_layer)(jr.split(key, flow_layers))))
    return GaussianCopulaBlock(flow, dim_y, cond_dim)


# ------------------------------------------------------------------ joint flow
def build_gaussian_frugal_flow(key, dim_y: int, nvars: int, cond_dim: int, margin: str = "flexible",
                               RQS_knots: int = 8, nn_depth: int = 1, nn_width: int = 50,
                               flow_layers: int = 4, causal_model_args: dict | None = None) -> Transformed:
    """The unfitted joint flow (see the module docstring).

    Args:
        key: JAX PRNG key.
        dim_y: number of outcome columns K.
        nvars: number of covariates.
        cond_dim: treatment dimension (1 for a binary or scalar treatment).
        margin: ``"flexible"`` or ``"shift"``.
        RQS_knots, nn_depth, nn_width, flow_layers: size of the COPULA.
        causal_model_args: size of the MARGIN, keys ``RQS_knots``, ``nn_depth``, ``nn_width``,
            ``flow_layers`` (missing keys take ``MARGIN_DEFAULTS``); optional ``interval`` (spline
            half-width, default 5), ``ate`` (shift margin start, scalar or length K, default 0),
            ``copula_interval`` (default 5), ``copula_zero_init`` (default True: the copula starts
            at independence).

    Returns:
        A flowjax ``Transformed`` distribution over ``(n, K + nvars)``: outcome first, then the
        covariates' normal scores, conditioned on the treatment.
    """
    if margin not in MARGINS:
        raise ValueError(f"margin must be one of {MARGINS}, got {margin!r}")
    a = dict(causal_model_args or {})
    k_m, k_c = jr.split(key)
    margin_bij = MARGIN_BUILDERS[margin](k_m, dim_y, jnp.zeros((1, cond_dim)), a)
    copula = gaussian_copula_block(k_c, dim_y, nvars, cond_dim, RQS_knots=RQS_knots, nn_width=nn_width,
                                   nn_depth=nn_depth, flow_layers=flow_layers,
                                   interval=a.get("copula_interval", DEFAULT_INTERVAL),
                                   zero_init=a.get("copula_zero_init", True))
    margin_block = Concatenate([margin_bij, Identity((nvars,))])
    return Transformed(StandardNormal((dim_y + nvars,)), Chain([copula, margin_block]))


def copula_of(flow) -> GaussianCopulaBlock:
    """The copula block of a fitted flow (acts on the covariate scores given g_Y)."""
    return paramax.unwrap(flow).bijection.bijections[0]


def margin_of(flow) -> AbstractBijection:
    """The causal margin bijection (on the first K coordinates, conditioned on T)."""
    return paramax.unwrap(flow).bijection.bijections[1].bijections[0]


def shift_vector(flow):
    """The fitted ``ate`` vector of a shift-margin flow (preprocessed outcome scale)."""
    m = margin_of(flow)
    if not (isinstance(m, Chain) and isinstance(m.bijections[-1], LocCond)):
        raise ValueError("not a shift-margin flow")
    return m.bijections[-1].ate


def outcome_scores(flow, y: ArrayLike, condition: ArrayLike):
    """g_Y = margin^{-1}(y; T), row-wise: N(0, I) at the optimum."""
    return jax.vmap(margin_of(flow).inverse)(jnp.asarray(y), jnp.asarray(condition))


def counterfactual_gaussian(flow, y: ArrayLike, t: ArrayLike, t_new: ArrayLike):
    """Unit-level counterfactuals by abduction, action and prediction.

    Abduction ``g_Y = margin^{-1}(y; t)``; action and prediction ``y' = margin(g_Y; t_new)``. Each
    unit keeps its ranks under the causal margin (rank preservation); the copula does not see T,
    so it plays no part. The Gaussian-scale counterpart of ``interventions.counterfactual_flexible``.

    Args:
        flow: a fitted Gaussian-scale frugal flow.
        y: (n, K) observed outcomes on the FITTING scale (``ot.forward(Y)`` if Y was standardised).
        t, t_new: (n,) or (n, cond_dim) observed and counterfactual treatments.

    Returns:
        (n, K) numpy array of counterfactual outcomes on the fitting scale (apply ``ot.inverse``).
    """
    y = jnp.asarray(y)
    c_old = jnp.asarray(t, dtype=y.dtype).reshape(len(y), -1)
    c_new = jnp.asarray(t_new, dtype=y.dtype).reshape(len(y), -1)
    m = margin_of(flow)
    g = jax.vmap(m.inverse)(y, c_old)
    return np.asarray(jax.vmap(m.transform)(g, c_new))


def copula_residuals(flow, g_y: ArrayLike, g_z: ArrayLike):
    """The copula's base coordinates for (g_Y, g_Z): N(0, I) if the copula fits."""
    x = jnp.hstack([jnp.asarray(g_y), jnp.asarray(g_z)])
    cop = copula_of(flow)
    # the block never reads the condition, but flowjax requires one of its cond_shape
    dummy = jnp.zeros((x.shape[0],) + (cop.cond_shape or ()), x.dtype)
    return jax.vmap(cop.inverse)(x, dummy)[:, jnp.shape(g_y)[1]:]


# ------------------------------------------------------------------ training
def train_frugal_flow_gaussian(
    key,
    y: ArrayLike,
    u_z: ArrayLike,
    condition: ArrayLike,
    margin: str = "flexible",
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    causal_model_args: dict | None = None,
    show_progress: bool = True,
    fit_kwargs: dict | None = None,
    optimizer: optax.GradientTransformation | None = None,
):
    """Fit the Gaussian-scale frugal flow by maximum likelihood on ``(concat[y, g_Z], T)``.

    Low-level entry point: ``y`` is used as given (no standardisation) and ``u_z`` must already be
    ranks. Most users want ``fit_gaussian_frugal_flow``, which does both.

    Args:
        key: JAX PRNG key.
        y: (n, K) outcomes on the fitting scale.
        u_z: (n, d) covariate ranks in (0, 1), converted here to normal scores.
        condition: (n, cond_dim) treatment.
        margin: ``"flexible"`` or ``"shift"``.
        RQS_knots, nn_depth, nn_width, flow_layers: copula size.
        learning_rate, max_epochs, max_patience, batch_size: Adam training settings; training stops
            after ``max_patience`` epochs without a validation improvement and returns the best
            validation checkpoint (``frugal_flows.training.fit_to_data``).
        causal_model_args: margin size and options (see ``build_gaussian_frugal_flow``).
        show_progress: tqdm progress bar.
        fit_kwargs: extra arguments for ``fit_to_data``.
        optimizer: optional optax optimizer (overrides ``learning_rate``).

    Returns:
        ``(flow, losses)``: the fitted flow and a dict of train/validation loss curves.
    """
    y = jnp.asarray(y)
    if condition is None:
        raise ValueError("the Gaussian-scale frugal flow needs a treatment condition")
    condition = jnp.asarray(condition, dtype=y.dtype)
    g_z = normal_scores_from_uniform(jnp.asarray(u_z, dtype=y.dtype)).astype(y.dtype)
    key, sub = jr.split(key)
    flow = build_gaussian_frugal_flow(sub, y.shape[1], g_z.shape[1], condition.shape[1], margin=margin,
                                      RQS_knots=RQS_knots, nn_depth=nn_depth, nn_width=nn_width,
                                      flow_layers=flow_layers, causal_model_args=causal_model_args)
    key, sub = jr.split(key)
    return fit_to_data(key=sub, dist=flow, data=(jnp.hstack([y, g_z]), condition), optimizer=optimizer,
                       learning_rate=learning_rate, max_epochs=max_epochs, max_patience=max_patience,
                       batch_size=batch_size, show_progress=show_progress, **dict(fit_kwargs or {}))


# ------------------------------------------------------------------ user-facing fit, save, load
def ecdf_ranks(z: ArrayLike):
    """Column-wise empirical ranks of ``z`` (n, d) scaled into (0, 1): ``rank / (n + 1)``, ties
    get their average rank."""
    from scipy.stats import rankdata
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    return rankdata(z, axis=0) / (z.shape[0] + 1)


#: settings of the MorphoMNIST paper grid (8x8, n = 5000; a hyperparameter sweep found nothing better)
PAPER_SETTINGS = dict(copula=dict(RQS_knots=8, nn_depth=1, nn_width=16, flow_layers=4),
                      margin=dict(MARGIN_DEFAULTS), learning_rate=1e-3, batch_size=100,
                      max_epochs=1000, max_patience=30)


def fit_gaussian_frugal_flow(key, Y: ArrayLike, Z: ArrayLike, T: ArrayLike, margin: str = "flexible",
                             standardize: bool = True, copula_args: dict | None = None,
                             margin_args: dict | None = None, learning_rate: float = PAPER_SETTINGS["learning_rate"],
                             batch_size: int = PAPER_SETTINGS["batch_size"],
                             max_epochs: int = PAPER_SETTINGS["max_epochs"],
                             max_patience: int = PAPER_SETTINGS["max_patience"], show_progress: bool = False,
                             fit_kwargs: dict | None = None):
    """Fit a Gaussian-scale frugal flow from raw data in one call.

    Standardises each outcome column (``OutcomeTransform("standardize")``), turns the covariates
    into empirical ranks (``ecdf_ranks``) and fits with ``train_frugal_flow_gaussian``. Defaults
    are ``PAPER_SETTINGS``.

    Args:
        key: JAX PRNG key.
        Y: (n, K) outcomes.
        Z: (n, d) continuous covariates (or (n,)). Discrete covariates can be passed as numbers;
            they are ranked like the rest.
        T: (n,) or (n, cond_dim) treatment.
        margin: ``"flexible"`` (any effect) or ``"shift"`` (a per-column location shift).
        standardize: standardise Y per column (recommended).
        copula_args, margin_args: override ``PAPER_SETTINGS["copula"]`` / ``["margin"]`` keys.
        learning_rate, batch_size, max_epochs, max_patience, show_progress, fit_kwargs: training.

    Returns:
        ``(flow, outcome_transform, info)``. ``outcome_transform`` maps Y to the fitting scale
        (``.forward``) and back (``.inverse``); pass it to ``interventional_samples``. ``info``
        holds the loss curves, the build settings (``info["build_kwargs"]``, for
        ``save_gaussian_flow``) and the covariate ranks ``u_z``.
    """
    from frugal_flows.outcome_transforms import OutcomeTransform

    Y = jnp.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    T = jnp.asarray(T, dtype=Y.dtype).reshape(Y.shape[0], -1)
    ot = OutcomeTransform("standardize" if standardize else "identity").fit(Y)
    u_z = ecdf_ranks(Z)
    cop = {**PAPER_SETTINGS["copula"], **(copula_args or {})}
    mar = {**PAPER_SETTINGS["margin"], **(margin_args or {})}
    flow, losses = train_frugal_flow_gaussian(key, ot.forward(Y), u_z, T, margin=margin, causal_model_args=mar,
                                              learning_rate=learning_rate, max_epochs=max_epochs,
                                              max_patience=max_patience, batch_size=batch_size,
                                              show_progress=show_progress, fit_kwargs=fit_kwargs, **cop)
    build = dict(dim_y=int(Y.shape[1]), nvars=int(u_z.shape[1]), cond_dim=int(T.shape[1]), margin=margin,
                 causal_model_args=mar, **cop)
    return flow, ot, {"losses": losses, "build_kwargs": build, "u_z": u_z}


def save_gaussian_flow(path: str, flow, build_kwargs: dict, outcome_transform=None) -> None:
    """Save a fitted Gaussian-scale flow to the directory ``path``.

    Writes ``model.eqx`` (the weights) and ``model_spec.json`` (``build_kwargs``: the arguments of
    ``build_gaussian_frugal_flow`` minus the key, and, when given, the outcome transform's
    per-column location and scale), so ``load_gaussian_flow`` needs nothing else.
    """
    os.makedirs(path, exist_ok=True)
    dtypes = {str(x.dtype) for x in jax.tree_util.tree_leaves(eqx.filter(flow, eqx.is_inexact_array))}
    spec = {"build_kwargs": _jsonable(build_kwargs), "dtype": dtypes.pop() if len(dtypes) == 1 else sorted(dtypes)}
    if outcome_transform is not None:
        spec["outcome_transform"] = outcome_transform_state(outcome_transform)
    with open(os.path.join(path, "model_spec.json"), "w") as f:
        json.dump(spec, f, indent=1)
    eqx.tree_serialise_leaves(os.path.join(path, "model.eqx"), flow)


def load_gaussian_flow(path: str):
    """Load a flow saved by ``save_gaussian_flow``. Returns ``(flow, outcome_transform_or_None)``.

    A flow saved in float64 loads in a float32 session (cast down) and vice versa; call
    ``frugal_flows.set_x64(True)`` first to keep full precision."""
    with open(os.path.join(path, "model_spec.json")) as f:
        spec = json.load(f)
    skeleton = build_gaussian_frugal_flow(jr.key(0), **spec["build_kwargs"])
    flow = eqx.tree_deserialise_leaves(os.path.join(path, "model.eqx"), skeleton, filter_spec=_load_cast)
    ot = spec.get("outcome_transform")
    return flow, (None if ot is None else outcome_transform_from_state(ot))


_OT_FITTED = ("_mean", "_sd", "_post_mean", "_post_sd")


def _load_cast(f, x):
    """Deserialise one leaf, cast to the skeleton leaf's dtype (a flow saved in float64 loads in a
    float32 session and vice versa)."""
    if isinstance(x, (jax.Array, np.ndarray)):
        return jnp.asarray(np.load(f)).astype(x.dtype)
    return eqx.default_deserialise_filter_spec(f, x)


def outcome_transform_state(ot) -> dict:
    """A JSON-able record of a fitted ``OutcomeTransform`` (constructor arguments and fitted
    per-column statistics)."""
    state = {"kind": ot.kind, "post_standardize": ot.post_standardize, "fitted": ot.fitted}
    for attr in ("floor", "asinh_scale") + _OT_FITTED:
        v = getattr(ot, attr, None)
        state[attr] = None if v is None else np.asarray(v).tolist()
    return state


def outcome_transform_from_state(state: dict):
    """Rebuild an ``OutcomeTransform`` from ``outcome_transform_state`` output."""
    from frugal_flows.outcome_transforms import OutcomeTransform
    kind = state["kind"]
    ot = OutcomeTransform(kind, floor=state.get("floor") if kind in ("log", "asinh") else None,
                          asinh_scale=state.get("asinh_scale"),
                          post_standardize=state.get("post_standardize") if kind in ("log", "asinh") else None)
    for attr in ("asinh_scale",) + _OT_FITTED:
        if state.get(attr) is not None:
            setattr(ot, attr, jnp.asarray(state[attr], dtype=float))
    ot.fitted = bool(state.get("fitted", True))
    return ot


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if hasattr(d, "tolist"):
        return d.tolist()
    return d
