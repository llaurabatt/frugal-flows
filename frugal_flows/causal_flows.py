from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flowjax.bijections import (
    Affine,
    Concatenate,
    Invert,
    RationalQuadraticSpline,
    Stack,
    Tanh,
)
from flowjax.bijections.utils import Identity
from flowjax.distributions import Transformed, Uniform, _StandardUniform
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from jaxtyping import ArrayLike
from paramax import NonTrainable

from frugal_flows.basic_flows import (
    masked_autoregressive_bijection,
    masked_autoregressive_bijection_masked_condition,
    masked_autoregressive_flow_first_uniform,
    masked_autoregressive_flow_heterogeneous,
    masked_independent_flow,
    univariate_marginal_flow,
)
from frugal_flows.bijections import LocCond, UnivariateNormalCDF
from frugal_flows.bijections.transformer_autoregressive import (
    transformer_autoregressive_bijection,
)


def _freeze_arrays(subtree):
    return jax.tree.map(
        lambda leaf: NonTrainable(leaf) if eqx.is_inexact_array(leaf) else leaf,
        subtree,
    )


def _build_flexible_margin(key, dim, condition, causal_model_args):
    """Build the flexible causal margin with the conditioner named in
    ``causal_model_args["conditioner"]``: ``"mlp"`` (default, the MADE-masked
    MLP) or ``"transformer"`` (causal-transformer conditioner, TarFlow-inspired;
    see ``bijections.transformer_autoregressive``). Both emit RQS spline
    parameters on [-1, 1] and return the same ``Invert(Scan(...))`` bijection,
    so the chain around the margin is conditioner-agnostic. The transformer
    reads two extra optional keys, ``nn_heads`` (default 4; must divide
    ``nn_width``) and ``expansion`` (default 2).
    """
    conditioner = causal_model_args.get("conditioner", "mlp")
    if conditioner == "mlp":
        return masked_autoregressive_bijection(
            key=key,
            dim=dim,
            condition=condition,
            nn_depth=causal_model_args["nn_depth"],
            nn_width=causal_model_args["nn_width"],
            RQS_knots=causal_model_args["RQS_knots"],
            flow_layers=causal_model_args["flow_layers"],
        )
    if conditioner == "transformer":
        return transformer_autoregressive_bijection(
            key=key,
            dim=dim,
            condition=condition,
            RQS_knots=causal_model_args["RQS_knots"],
            nn_depth=causal_model_args["nn_depth"],
            nn_width=causal_model_args["nn_width"],
            flow_layers=causal_model_args["flow_layers"],
            nn_heads=causal_model_args.get("nn_heads", 4),
            expansion=causal_model_args.get("expansion", 2),
        )
    raise ValueError(
        f"unknown conditioner {conditioner!r}; choose 'mlp' or 'transformer'"
    )


def train_copula_flow(
    key: jr.PRNGKey,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
):
    nvars = u_z.shape[1]
    key, subkey = jr.split(key)

    base_dist = Uniform(-jnp.ones(nvars), jnp.ones(nvars))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    copula_flow = masked_autoregressive_flow(  # masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
    )  # Support on [-1, 1]

    copula_flow = Transformed(
        copula_flow, Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2))
    )  # Unbounded support

    copula_flow = copula_flow.merge_transforms()

    assert isinstance(copula_flow.base_dist, _StandardUniform)

    copula_flow = eqx.tree_at(
        where=lambda copula_flow: copula_flow.bijection.bijections[0],
        pytree=copula_flow,
        replace_fn=_freeze_arrays,
    )

    copula_flow = eqx.tree_at(
        where=lambda copula_flow: copula_flow.bijection.bijections[-1],
        pytree=copula_flow,
        replace_fn=_freeze_arrays,
    )

    key, subkey = jr.split(key)

    # Train
    copula_flow, losses = fit_to_data(
        key=subkey,
        dist=copula_flow,
        data=u_z,
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return copula_flow, losses


def train_frugal_flow_location_translation(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    causal_model_args: dict | None = None,
):
    nvars = u_z.shape[1]
    dim_y = y.shape[1]
    # ate may be a scalar / length-1 array (univariate Y, legacy callers) or a
    # length-K vector (multivariate Y). Normalise to one shift per outcome dim;
    # a single value is broadcast across all K dims.
    ate = jnp.atleast_1d(jnp.asarray(causal_model_args["ate"]))
    if ate.shape != (dim_y,):
        if ate.shape == (1,):
            ate = jnp.broadcast_to(ate, (dim_y,))
        else:
            raise ValueError(
                f"ate has length {ate.shape[0]} but y has {dim_y} column(s)."
            )

    if condition is None:
        cond_dim = None
    else:
        cond_dim = condition.shape[1]
    if mask_condition:
        cond_dim_mask = cond_dim
        cond_dim_nomask = None
    else:
        cond_dim_mask = None
        cond_dim_nomask = cond_dim

    list_bijections_affine = [Identity((dim_y,))] + [
        Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2))
    ]
    bijections_affine = Concatenate(list_bijections_affine)

    key, subkey = jr.split(key)
    ate_maf_bijection = masked_autoregressive_bijection_masked_condition(
        key=subkey,
        dim=dim_y,
        condition=condition,
        RQS_knots=causal_model_args["RQS_knots"],
        nn_depth=causal_model_args["nn_depth"],
        nn_width=causal_model_args["nn_width"],
        flow_layers=causal_model_args["flow_layers"],
    )

    list_bijections_ate_maf = [ate_maf_bijection] + [Identity((1,))] * nvars
    bijections_ate_maf = Concatenate(list_bijections_ate_maf)

    list_bijections_tanh = [Invert(Tanh(()))] * dim_y + [Identity(())] * nvars
    bijections_tanh = Stack(list_bijections_tanh)

    list_bijections_loccond = [LocCond(ate=ate[k]) for k in range(dim_y)] + [
        Identity(())
    ] * nvars
    bijections_loccond = Stack(list_bijections_loccond)

    base_dist = Uniform(-jnp.ones(nvars + dim_y), jnp.ones(nvars + dim_y))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    key, subkey = jr.split(key)
    frugal_flow = masked_autoregressive_flow_first_uniform(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        invert=True,
        cond_dim_mask=cond_dim_mask,
        cond_dim_nomask=cond_dim_nomask,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
        cond_u_y_dim=dim_y,
    )  # Support on [-1, 1]

    frugal_flow = Transformed(
        frugal_flow,
        bijections_affine,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_ate_maf,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_tanh,
    )
    frugal_flow = Transformed(
        frugal_flow,
        bijections_loccond,
    )
    frugal_flow = frugal_flow.merge_transforms()
    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-4],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[0],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    key, subkey = jr.split(key)

    # Train
    key, subkey = jr.split(key)
    frugal_flow, losses = fit_to_data(
        key=subkey,
        dist=frugal_flow,
        data=(jnp.hstack([y, u_z]), condition),
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return frugal_flow, losses


def train_frugal_flow_flexible_continuous(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    causal_model_args: dict | None = None,
    pretrained_margin=None,  # AbstractBijection | None: warm-start graft (see below)
):
    nvars = u_z.shape[1]
    dim_y = y.shape[1]

    if condition is None:
        cond_dim = None
    else:
        cond_dim = condition.shape[1]
    if mask_condition:
        cond_dim_mask = cond_dim
        cond_dim_nomask = None
    else:
        cond_dim_mask = None
        cond_dim_nomask = cond_dim

    list_bijections_affine = [Identity((dim_y,))] + [
        Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2))
    ]
    bijections_affine = Concatenate(list_bijections_affine)

    key, subkey = jr.split(key)
    # condition is unmasked here
    causal_maf_bijection = _build_flexible_margin(
        key=subkey, dim=dim_y, condition=condition, causal_model_args=causal_model_args
    )

    list_bijections_ate_maf = [causal_maf_bijection] + [Identity((1,))] * nvars
    bijections_ate_maf = Concatenate(list_bijections_ate_maf)

    list_bijections_tanh = [Invert(Tanh(()))] * dim_y + [Identity(())] * nvars
    bijections_tanh = Stack(list_bijections_tanh)

    base_dist = Uniform(-jnp.ones(nvars + dim_y), jnp.ones(nvars + dim_y))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    key, subkey = jr.split(key)
    frugal_flow = masked_autoregressive_flow_first_uniform(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        invert=True,
        cond_dim_mask=cond_dim_mask,
        cond_dim_nomask=cond_dim_nomask,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
        cond_u_y_dim=dim_y,
    )  # Support on [-1, 1]

    frugal_flow = Transformed(
        frugal_flow,
        bijections_affine,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_ate_maf,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_tanh,
    )

    frugal_flow = frugal_flow.merge_transforms()

    # Optional warm-start: graft a pre-fitted causal margin into the flow before the
    # joint fit. The causal margin is the treatment-conditioned autoregressive
    # bijection at bijections[-2].bijections[0] (Concatenate([causal_maf, Identity...]));
    # it is left trainable (the freezing below only touches [-3] and [0]), so the
    # grafted margin co-adapts with the copula during fit_to_data. `pretrained_margin`
    # must have been built by the same margin builder at IDENTICAL hyperparameters
    # AND conditioner, else its pytree structure will not match the graft site.
    if pretrained_margin is not None:
        # Validate BEFORE replacing: eqx.tree_at(replace=...) swaps subtrees with NO
        # structure check, so a wrong object (e.g. a base distribution's unconditional
        # Affine) would graft silently and only fail statistically (ate == 0, ~50%
        # non-finite samples). Two guards catch that whole bug class up front:
        # (1) the pretrained margin must be conditional on exactly the same cond_shape
        #     as the causal-margin slot; (2) its array pytree structure must match the
        #     slot's, so the graft is weight-for-weight.
        original = frugal_flow.bijection.bijections[-2].bijections[0]
        assert pretrained_margin.cond_shape == original.cond_shape, (
            f"pretrained_margin.cond_shape {pretrained_margin.cond_shape} != "
            f"causal-margin slot cond_shape {original.cond_shape}"
        )
        assert jax.tree_util.tree_structure(
            eqx.filter(pretrained_margin, eqx.is_array)
        ) == jax.tree_util.tree_structure(eqx.filter(original, eqx.is_array)), (
            "pretrained_margin pytree structure does not match the causal-margin slot "
            "(was it built by masked_autoregressive_bijection with identical "
            "hyperparameters?)"
        )
        frugal_flow = eqx.tree_at(
            where=lambda f: f.bijection.bijections[-2].bijections[0],
            pytree=frugal_flow,
            replace=pretrained_margin,
        )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-3],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[0],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    key, subkey = jr.split(key)

    # Train
    key, subkey = jr.split(key)
    frugal_flow, losses = fit_to_data(
        key=subkey,
        dist=frugal_flow,
        data=(jnp.hstack([y, u_z]), condition),
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return frugal_flow, losses


def pretrain_causal_margin(
    key,
    y: ArrayLike,
    condition: ArrayLike,
    causal_model_args: dict,
    learning_rate: float = 5e-3,
    max_epochs: int = 100,
    max_patience: int = 10,
    batch_size: int = 100,
    show_progress: bool = False,
):
    """Warm-start builder for the ``flexible_continuous`` causal margin.

    Fits the causal margin ALONE -- ``Uniform[-1, 1] -> causal_maf(RQS | T) ->
    atanh -> Y`` -- by maximum likelihood on ``(y, condition)``, and returns the
    fitted treatment-conditioned bijection ready to graft into a full frugal flow
    via ``train_frugal_flow(..., pretrained_margin=...)``. Because the margin term
    moment-matches the interventional outcome by construction, pretraining it lands
    the ATE *level* at the identified point before the copula is introduced, which
    empirically removes the small-``n`` level bias of a cold spline fit.

    The margin is built by the SAME margin builder (identical hyperparameters
    AND conditioner, read from ``causal_model_args`` — see
    ``_build_flexible_margin``) that ``train_frugal_flow_flexible_continuous``
    uses, so the returned bijection's pytree matches the graft site exactly (the
    graft re-validates ``cond_shape`` + structure before replacing; a pretrain
    built with a different conditioner fails that structure check). The outcome
    dimension is read from ``y``, so the same ``y`` must be passed here and to
    the full frugal-flow fit.

    Args:
        key: a JAX PRNG key.
        y: outcome to fit the margin on, shape ``(n, K)`` (already on the fitting
            scale, i.e. after any ``outcome_transform``).
        condition: treatment, shape ``(n, cond_dim)``.
        causal_model_args: dict with ``nn_depth``/``nn_width``/``RQS_knots``/
            ``flow_layers`` (as produced for the ``flexible_continuous`` arm).
        learning_rate, max_epochs, max_patience, batch_size, show_progress: passed
            through to ``fit_to_data``.

    Returns:
        The fitted causal-margin bijection (``cond_shape == (condition.shape[1],)``).
    """
    cond_dim = condition.shape[1]
    dim_y = jnp.asarray(y).shape[1]
    key, subkey = jr.split(key)
    causal_maf = _build_flexible_margin(
        key=subkey, dim=dim_y, condition=condition, causal_model_args=causal_model_args
    )
    # base Uniform[-1, 1] -> causal_maf ([-1, 1]) -> Invert(Tanh) = atanh -> Y scale
    margin_flow = Transformed(Uniform(-jnp.ones(dim_y), jnp.ones(dim_y)), causal_maf)
    margin_flow = Transformed(margin_flow, Invert(Tanh((dim_y,))))
    margin_flow = margin_flow.merge_transforms()
    key, subkey = jr.split(key)
    trained, _ = fit_to_data(
        key=subkey,
        dist=margin_flow,
        data=(jnp.asarray(y), jnp.asarray(condition)),
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    # Extract the causal_maf by its cond_shape, NOT by position: flowjax's
    # Uniform(-1, 1) is itself Transformed(_StandardUniform, NonTrainable(Affine)),
    # so after merge_transforms() the chain is [Affine, causal_maf, Invert(Tanh)]
    # and bijections[0] is the base's UNCONDITIONAL Affine -- grafting that would
    # give ate == 0 exactly and ~50% non-finite samples. Exactly one element is
    # conditional; select it by cond_shape.
    conditional = [b for b in trained.bijection.bijections if b.cond_shape is not None]
    assert len(conditional) == 1 and conditional[0].cond_shape == (cond_dim,), (
        f"expected exactly one conditional bijection with cond_shape ({cond_dim},); "
        f"got {[b.cond_shape for b in trained.bijection.bijections]}"
    )
    return conditional[0]


def train_frugal_flow_flexible_discrete(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    causal_model_args: dict | None = None,
):
    nvars = u_z.shape[1]

    key, subkey = jr.split(key)
    outcome, _ = univariate_discrete_cdf(
        key=subkey, z_discr=y.flatten(), max_unique_z_discr_size=len(jnp.unique(y))
    )
    outcome = jnp.expand_dims(outcome, axis=1)

    if condition is None:
        cond_dim = None
    else:
        cond_dim = condition.shape[1]
    if mask_condition:
        cond_dim_mask = cond_dim
        cond_dim_nomask = None
    else:
        cond_dim_mask = None
        cond_dim_nomask = cond_dim

    list_bijections_affine = [Identity((1,))] + [
        Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2))
    ]
    bijections_affine = Concatenate(list_bijections_affine)

    key, subkey = jr.split(key)
    ate_maf_bijection = masked_autoregressive_bijection_masked_condition(
        key=subkey,
        dim=1,
        condition=condition,
        RQS_knots=causal_model_args["RQS_knots"],
        nn_depth=causal_model_args["nn_depth"],
        nn_width=causal_model_args["nn_width"],
        flow_layers=causal_model_args["flow_layers"],
    )

    list_bijections_ate_maf = [ate_maf_bijection] + [Identity((1,))] * nvars
    bijections_ate_maf = Concatenate(list_bijections_ate_maf)

    list_bijections_affine_output = [
        Invert(Affine(loc=-jnp.ones(1), scale=jnp.ones(1) * 2))
    ] + [Identity((nvars,))]
    bijections_affine_output = Concatenate(list_bijections_affine_output)

    base_dist = Uniform(-jnp.ones(nvars + 1), jnp.ones(nvars + 1))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    key, subkey = jr.split(key)
    frugal_flow = masked_autoregressive_flow_first_uniform(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        invert=True,
        cond_dim_mask=cond_dim_mask,
        cond_dim_nomask=cond_dim_nomask,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
    )  # Support on [-1, 1]

    frugal_flow = Transformed(
        frugal_flow,
        bijections_affine,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_ate_maf,
    )

    frugal_flow = Transformed(
        frugal_flow,
        bijections_affine_output,
    )

    frugal_flow = frugal_flow.merge_transforms()
    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-3],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-1],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[0],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    key, subkey = jr.split(key)

    # Train
    key, subkey = jr.split(key)
    frugal_flow, losses = fit_to_data(
        key=subkey,
        dist=frugal_flow,
        data=(jnp.hstack([outcome, u_z]), condition),
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return frugal_flow, losses


def train_frugal_flow_gaussian(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    u_z_hetero: ArrayLike | None = None,
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    causal_model_args: dict | None = None,
    causal_effect_idx: int = 0,
):
    # nvars = u_z.shape[1]
    input_vars = []
    if u_z_hetero is not None:
        input_vars.append(u_z_hetero)
    input_vars += [y, u_z]
    input_vars = jnp.hstack(input_vars)
    nvars = input_vars.shape[1]

    if condition is None:
        cond_dim = None
    else:
        cond_dim = condition.shape[1]
    if mask_condition:
        cond_dim_mask = cond_dim
        cond_dim_nomask = None
    else:
        cond_dim_mask = None
        cond_dim_nomask = cond_dim

    # list_bijections = [
    #     UnivariateNormalCDF(
    #         ate=causal_model_args["ate"],
    #         scale=causal_model_args["scale"],
    #         const=causal_model_args["const"],
    #         cond_dim=cond_dim,
    #     )
    # ] + [Identity(())] * nvars

    list_bijections = [Identity(())] * (nvars)
    list_bijections[causal_effect_idx] = UnivariateNormalCDF(
        ate=causal_model_args["ate"],
        scale=causal_model_args["scale"],
        const=causal_model_args["const"],
        cond_dim=cond_dim,
    )

    marginal_transform = Stack(list_bijections)
    base_dist = Uniform(-jnp.ones(nvars), jnp.ones(nvars))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    key, subkey = jr.split(key)
    frugal_flow = masked_autoregressive_flow_heterogeneous(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        invert=True,
        cond_dim_mask=cond_dim_mask,
        cond_dim_nomask=cond_dim_nomask,
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
        causal_effect_idx=causal_effect_idx,
    )  # Support on [-1, 1]

    frugal_flow = Transformed(
        frugal_flow,
        Invert(Affine(loc=-jnp.ones(nvars), scale=jnp.ones(nvars) * 2)),
    )
    frugal_flow = Transformed(
        frugal_flow,
        Invert(marginal_transform),
    )
    frugal_flow = frugal_flow.merge_transforms()
    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[-2],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    frugal_flow = eqx.tree_at(
        where=lambda frugal_flow: frugal_flow.bijection.bijections[0],
        pytree=frugal_flow,
        replace_fn=_freeze_arrays,
    )

    key, subkey = jr.split(key)

    # Train
    key, subkey = jr.split(key)
    frugal_flow, losses = fit_to_data(
        key=subkey,
        dist=frugal_flow,
        data=(input_vars, condition),
        optimizer=optimizer,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )

    return frugal_flow, losses


def train_frugal_flow(
    key: jr.PRNGKey,
    y: ArrayLike,
    u_z: ArrayLike,  # impose discrete
    u_z_hetero: ArrayLike | None = None,
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    nn_depth: int = 1,
    nn_width: int = 50,
    flow_layers: int = 4,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    condition: ArrayLike | None = None,
    mask_condition: bool = True,
    causal_model="gaussian",
    causal_model_args: dict | None = None,
    pretrained_margin=None,  # AbstractBijection | None: warm-start graft (flexible_continuous only)
):
    valid_causal_models = [
        "gaussian",
        "flexible_continuous",
        "flexible_discrete_output",
        "location_translation",
    ]

    if (causal_model != "gaussian") & (u_z_hetero is not None):
        raise ValueError("Only gaussian causal model supports heterogeneous effects.")

    if u_z_hetero is not None:
        assert condition.shape[1] == (
            u_z_hetero.shape[1] + 1
        ), "Both z_hetero and treatment must be included in the condition."
        causal_effect_idx = u_z_hetero.shape[1]
    else:
        causal_effect_idx = 0

    if causal_model == "gaussian":
        frugal_flow, losses = train_frugal_flow_gaussian(
            key=key,
            y=y,
            u_z=u_z,
            u_z_hetero=u_z_hetero,
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
            condition=condition,
            mask_condition=mask_condition,
            causal_model_args=causal_model_args,
            causal_effect_idx=causal_effect_idx,
        )

    elif causal_model == "flexible_discrete_output":
        frugal_flow, losses = train_frugal_flow_flexible_discrete(
            key=key,
            y=y,
            u_z=u_z,  # impose discrete
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
            condition=condition,
            mask_condition=mask_condition,
            causal_model_args=causal_model_args,
        )

    elif causal_model == "flexible_continuous":
        frugal_flow, losses = train_frugal_flow_flexible_continuous(
            key=key,
            y=y,
            u_z=u_z,
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
            condition=condition,
            mask_condition=mask_condition,
            causal_model_args=causal_model_args,
            pretrained_margin=pretrained_margin,
        )

    elif causal_model == "location_translation":
        frugal_flow, losses = train_frugal_flow_location_translation(
            key=key,
            y=y,
            u_z=u_z,
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            nn_depth=nn_depth,
            nn_width=nn_width,
            flow_layers=flow_layers,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
            condition=condition,
            mask_condition=mask_condition,
            causal_model_args=causal_model_args,
        )

    else:
        raise ValueError(f"Invalid choice. Please choose from: {valid_causal_models}")

    return frugal_flow, losses


def independent_continuous_marginal_flow(
    key: jr.PRNGKey,
    z_cont: ArrayLike,
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
):
    nvars = z_cont.shape[1]
    key, subkey = jr.split(key)

    base_dist = Uniform(-jnp.ones(nvars), jnp.ones(nvars))

    transformer = RationalQuadraticSpline(knots=RQS_knots, interval=1)

    flow = masked_independent_flow(  # masked_autoregressive_flow(
        key=subkey,
        base_dist=base_dist,
        transformer=transformer,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )  # Support on [-1, 1]

    flow = Transformed(flow, Invert(Tanh(flow.shape)))  # Unbounded support

    flow = flow.merge_transforms()

    assert isinstance(flow.base_dist, _StandardUniform)

    flow = eqx.tree_at(
        where=lambda flow: flow.bijection.bijections[0],
        pytree=flow,
        replace_fn=_freeze_arrays,
    )

    key, subkey = jr.split(key)

    # Train
    flow, losses = fit_to_data(
        key=subkey,
        dist=flow,
        data=z_cont,
        learning_rate=learning_rate,
        max_patience=max_patience,
        max_epochs=max_epochs,
        batch_size=batch_size,
        show_progress=show_progress,
        optimizer=optimizer,
        val_prop=val_prop,
    )

    return flow, losses


def univariate_discrete_cdf(
    key: jr.PRNGKey,
    z_discr: ArrayLike,
    max_unique_z_discr_size: int,
):
    if z_discr.ndim >= 2:
        _, dim = z_discr.shape
        if dim > 1:
            raise ValueError(
                "input must be 1D with shape (n_samples,) or 2D with shape (n_samples,1)"
            )

    if (z_discr.dtype != "int64") & (z_discr.dtype != "int32"):
        raise ValueError("type of input must be integer")

    n_samples = z_discr.shape[0]
    pmf_keys, pmf_vals = jnp.unique(
        z_discr, return_counts=True, size=max_unique_z_discr_size
    )
    # assert pmf_keys.all() == jnp.arange(max(pmf_keys)+1).all() # check increasing order is respected
    z_discr_empirical_pmf = pmf_vals / n_samples
    z_discr_empirical_cdf_long = z_discr_empirical_pmf.cumsum()

    def uniform_shift(standard_uniform, upper_index):
        # Function to handle the case where upper_index != 0
        def not_zero():
            lower = z_discr_empirical_cdf_long[upper_index - 1]
            upper = z_discr_empirical_cdf_long[upper_index]
            return standard_uniform * (upper - lower) + lower

        # Function to handle the case where upper_index == 0
        def zero():
            upper = z_discr_empirical_cdf_long[upper_index]
            return standard_uniform * upper

        # Using lax.cond to select which function to use
        return jax.lax.cond(upper_index != 0, not_zero, zero)

    uniforms = jr.uniform(key, (n_samples,))
    vmapped_uniform_shift = jax.vmap(uniform_shift)
    u_z_discr = vmapped_uniform_shift(uniforms, z_discr.flatten())

    return u_z_discr, z_discr_empirical_cdf_long


def get_independent_quantiles(
    key: jr.PRNGKey,
    z_discr: ArrayLike | None = None,  # impose discrete
    z_cont: ArrayLike | None = None,
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    return_z_cont_flow=False,
):
    assert (z_discr is not None) | (z_cont is not None)
    res = {"u_z_cont": None, "u_z_discr": None}

    key, subkey = jr.split(key)

    if z_cont is not None:
        partial_univariate_marginal_cdf = partial(
            univariate_marginal_cdf,
            key=subkey,
            optimizer=optimizer,
            RQS_knots=RQS_knots,
            flow_layers=flow_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            show_progress=show_progress,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            max_patience=max_patience,
            batch_size=batch_size,
        )
        u_z_cont = []
        z_cont_flows = []
        for i in jnp.arange(z_cont.shape[1]):
            (
                u_z_cont_univariate,
                z_cont_flow_univariate,
            ) = partial_univariate_marginal_cdf(z_cont=z_cont[:, i])
            u_z_cont.append(u_z_cont_univariate)
            z_cont_flows.append(z_cont_flow_univariate)
        u_z_cont = jnp.hstack(u_z_cont)
        res["u_z_cont"] = u_z_cont

        if return_z_cont_flow:
            res["z_cont_flows"] = z_cont_flows

        # z_cont_flow, z_cont_losses = independent_continuous_marginal_flow(
        #     key=key,
        #     z_cont=z_cont,
        #     optimizer=optimizer,
        #     RQS_knots=RQS_knots,
        #     flow_layers=flow_layers,
        #     nn_width=nn_width,
        #     nn_depth=nn_depth,
        #     show_progress=show_progress,
        #     learning_rate=learning_rate,
        #     max_epochs=max_epochs,
        #     max_patience=max_patience,
        #     batch_size=batch_size,
        # )
        # z_cont_marginal_cdf = jax.vmap(z_cont_flow.bijection.inverse, in_axes=(0,))

        # u_z_cont = z_cont_marginal_cdf(z_cont)
        # res["u_z_cont"] = u_z_cont

        # if return_z_cont_flow:
        #     res["z_cont_flow"] = z_cont_flow

    def rankdata(z_disc):
        z_disc_ordered = []
        z_rank_mapping = {}
        for d in range(z_disc.shape[1]):
            z_disc_d = z_disc[:, d]
            unique_z_disc_d = jnp.unique(z_disc_d)
            rank_mapping = {
                k: v
                for k, v in zip(
                    np.array(unique_z_disc_d), np.arange(len(unique_z_disc_d))
                )
            }
            z_disc_new = jnp.array([rank_mapping[i] for i in np.array(z_disc_d)])
            z_disc_ordered.append(z_disc_new)
            z_rank_mapping[d] = rank_mapping
        return jnp.vstack(z_disc_ordered).T, z_rank_mapping

    if z_discr is not None:
        z_discr_ordered, z_discr_rank_mapping = rankdata(z_discr)
        n_discr_ordered = z_discr_ordered.shape[1]
        keys = jr.split(key, n_discr_ordered)
        vmapped_get_discrete_quantiles = jax.vmap(
            univariate_discrete_cdf, in_axes=(0, 1, None)
        )
        u_z_discr_T, z_discr_empirical_cdf_long = vmapped_get_discrete_quantiles(
            keys, z_discr_ordered, len(jnp.unique(z_discr))
        )

        res["z_discr_empirical_cdf_long"] = z_discr_empirical_cdf_long
        res["u_z_discr"] = u_z_discr_T.T
        res["z_discr_rank_mapping"] = z_discr_rank_mapping

    return res


def univariate_marginal_cdf(
    key: jr.PRNGKey,
    z_cont: ArrayLike,
    optimizer: optax.GradientTransformation | None = None,
    RQS_knots: int = 8,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    show_progress: bool = True,
    learning_rate: float = 5e-4,
    max_epochs: int = 100,
    max_patience: int = 5,
    batch_size: int = 100,
    val_prop: float = 0.1,
):
    # Defensive: x64 is enabled globally, so the flow builds float64 params. A
    # float32 z_cont (e.g. image features passed straight into
    # get_independent_quantiles) would raise a lax.scan carry-dtype mismatch.
    # No-op for float64 callers.
    z_cont = jnp.asarray(z_cont, dtype=float)
    if z_cont.ndim == 1:
        # Reshape one-dimensional array to two dimensions with second dim as 1
        z_cont = z_cont.reshape(-1, 1)
    elif z_cont.ndim == 2:
        if z_cont.shape[1] > 1:
            raise ValueError(
                "Univariate input with shape (n_samples,) or (n_samples,1) is required"
            )
    else:
        raise ValueError(
            "Univariate input with shape (n_samples,) or (n_samples,1) is required"
        )

    z_cont_flow, z_cont_losses = univariate_marginal_flow(
        key=key,
        z_cont=z_cont,
        optimizer=optimizer,
        RQS_knots=RQS_knots,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
        show_progress=show_progress,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )
    z_cont_marginal_cdf = jax.vmap(z_cont_flow.bijection.inverse, in_axes=(0,))
    u_z_cont = z_cont_marginal_cdf(z_cont)
    return u_z_cont, z_cont_flow
