# Why the spline arm reports effect heterogeneity at small n

**Branch:** `spline-bias-analysis`  ·  **Arm:** `flexible_continuous` (treatment-conditioned RQS margin)  ·  **Date:** 2026-07-01

## Question

On a **homogeneous** DGP — Gaussian outcome, constant location-shift effect, so the true
per-quantile effect `τ(u) = Q_{Y|do1}(u) − Q_{Y|do0}(u)` is **flat at the ATE** and the true
`tau_sd` (its spread across quantiles) is **0** — the spline arm recovers the ATE at every n
but reports `tau_sd > 0` that shrinks with n (≈0.21 @ n=200 → 0.13 @ n=1000). Why does it
invent effect heterogeneity, and is it bias or noise?

Readout: paired common-random-numbers, `τ[i] = y1[i] − y0[i]` from the same base draw pushed
through do(0)/do(1); `tau_curve()` bins τ by control-outcome rank to get `τ(u)` on a fixed
40-bin grid, so per-seed curves stack for a **seed-mean (bias) vs seed-SD (variance)** split.

## Answer — it is two distinct mechanisms plus a clean positive control

### 1. A finite-sample VARIANCE floor that is independent of spline capacity
When treatment is unconfounded (`X ⟂ Z`), the spurious heterogeneity is **variance-dominated**:
the seed-**mean** `τ(u)` is essentially flat (shape RMS 0.054 @ n=200, 0.031 @ n=1000), while the
per-seed spread is larger (0.159 → 0.084) and **averages away**. Each fit wiggles differently; the
average is unbiased in shape.

`capacity_sweep` (E1) shows this floor **does not scale with the spline's capacity** — `tau_sd`
is flat across `RQS_knots ∈ {4,8,12}`, `nn_depth ∈ {2,4,6}`, `flow_layers ∈ {2,4,6}` (all ≈0.22 @
n=200, ≈0.13 @ n=1000). The only lever is **n**.

> **Mechanism:** intrinsic finite-sample noise in estimating *two separate* treatment-conditioned
> quantile functions from a split sample. It is not "over-flexibility": shrinking the spline
> (fewer knots / shallower conditioner) will **not** remove it. More data will.

### 2. A CONFOUNDING-induced BIAS that survives seed-averaging
The confounded baseline carries, on top of the variance floor, a **systematic** seed-averaged
shape: at n=1000 it is bias-dominated (mean-curve shape 0.109 > per-seed SD 0.102), and the mean
`τ(u)` bows **above** truth through the bulk and dips in the upper tail.

`confounding_sweep` (E2) confirms the driver: `tau_sd` rises **monotonically** with the Z→X
confounding coefficient β (0.14 → 0.17 → 0.22 → 0.23 at n=200 for β = 0, 0.5, 1, 1.5), while the
observational contrast grows 1.0 → 2.1 (generator sanity). Confounding also mildly inflates the
**ATE** upward (1.1–1.25) at small n.

> **Mechanism:** imperfect small-n deconfounding by the copula leaves residual Z–Y structure that
> the flexible margin renders as a *real* (non-noise) apparent effect heterogeneity. This is the
> genuine-bias part of the story, and it decays more slowly with n than the variance floor.

### 3. Genuine heterogeneity IS recovered (positive control)
On the **Gamma** outcome (log link ⇒ multiplicative effect ⇒ `τ(u)` genuinely **rising**), the
spline's seed-mean `τ(u)` **tracks the rising analytic truth** (under-shooting only the extreme
upper tail at small n). So `tau_sd` is not an artefact that fires on everything — it responds to
real quantile-heterogeneity. `tau_sd` here is large and does **not** decay with n (0.68 → 0.81):
it is signal, not noise.

## Calibration is necessary but not sufficient (E6)
Rosenblatt ranks stay ~Uniform even at n=200 (outcome KS 0.040, well under the ≈0.096 threshold;
Gamma similar). So the spurious heterogeneity **does not coincide with rank miscalibration** — a
well-calibrated frugal flow can still mis-allocate finite-sample variance into spurious margin
heterogeneity. Calibration checks won't catch it.

## Contrast: the additive arm's opposite failure (E5, profile likelihood)
Freezing the `gaussian`-arm `ate` at a grid and re-optimising everything else, the **profile NLL
minimum sits below truth at small n** (min @ 0.75 for n=100/200) and moves toward truth by n=1000
(min @ 1.25); the free optimiser lands **even lower** (0.23 / 0.26 / 0.50). So the additive arm's
small-n bias is **finite-sample non-identifiability** (the likelihood genuinely prefers a smaller
`ate`) compounded by optimisation stopping short. The spline escapes this *scalar* trap — it fits
the data without collapsing the effect — but pays for it with the variance/confounding
heterogeneity above.

## Practical takeaways
- Small-n spline `tau_sd` is **mostly finite-sample noise + confounding residual**, not evidence
  of real CATE structure. Do not read it as effect heterogeneity without a positive control.
- To reduce it: **more data** (not fewer knots, not more epochs) for the variance floor; **better
  deconfounding** for the bias part.
- The two arms fail oppositely: additive **collapses** the effect (scalar non-identifiability),
  spline **spreads** it (finite-sample + confounding heterogeneity). Both are small-n
  identifiability symptoms of the frugal margin/copula split.

## Follow-up: is the confounding bias copula-underfitting? (hyperparameter probe)

`spline_copula_hp` fits the confounded baseline (n=200, 8 seeds) varying **only** the
copula's capacity (top-level `nn_depth/nn_width/flow_layers/RQS_knots`) and the `u_z`
quality (`use_marginal_flow`), holding the margin fixed. Measured the seed-mean τ(u)
**bias shape** (curvature that survives averaging) separately from the ATE offset:

| config | bias shape ↓ | total dev (rms) | ATE (true 1.0) |
|---|---|---|---|
| baseline | 0.154 | 0.234 | 1.18 |
| cop_deep (depth 4→8) | **0.080** | 0.597 | 1.59 |
| cop_wide (width 50→100) | 0.168 | **0.168** | **0.99** |
| cop_knots (8→12) | 0.111 | 0.298 | 1.28 |
| cop_big (all up) | 0.114 | 0.375 | 1.36 |
| marginal_flow (u_z via flows) | 0.109 | 0.277 | 1.26 |
| mflow_cop_big | 0.119 | 0.632 | 1.62 |

**Verdict — partly, but it re-partitions rather than removes.** More copula capacity
(depth/knots/size) or better `u_z` **does flatten the quantile-varying bias shape**
(confirming the copula's finite-sample fit is part of the leak) — but it **inflates the
ATE** (level bias 1.3–1.6), so the *total* deviation from truth does not improve and often
worsens. The extra masked-copula capacity lets the joint NLL re-attribute the treated/control
difference, moving the leak from a τ(u) *shape* into an ATE *offset*. The one clean win was
widening the copula conditioner (`cop_wide`): best ATE recovery (0.99) at modest total
deviation. Net: the margin/copula split is finite-sample under-identified — hyperparameters
relocate the bias but only **more data** removes it (consistent with the β→0-floor and
n-decay). See `outputs/copula_hp.png`.

## Reproduce
```
cd validation
# battery (sharded by seed); writes outputs/*_shard*.csv
python -m diagnostics.spline_tau_curve  --seeds 0,1 --ns 200,1000 --out outputs/taucurve_shard0.csv   # E3 (+gamma control)
python -m diagnostics.spline_capacity   --seeds 0   --ns 200,1000 --out outputs/capacity_shard0.csv    # E1
python -m diagnostics.spline_confounding --seeds 0  --betas 0,0.5,1.0,1.5 --out outputs/confound_shard0.csv  # E2
python -m diagnostics.bias_profile      --seeds 0   --ns 100,200,1000 --out outputs/profile_shard0.csv  # E5
python -m diagnostics.rank_histogram    --family gamma --n 200 --arm flexible_continuous               # E6
python -m diagnostics.spline_bias_plot                                                                  # figures
```
Figures: `outputs/spline_tau_decomp.png` (centrepiece), `spline_capacity.png`, `spline_confounding.png`,
`additive_profile.png`, `rank_hist_flexible_continuous_*_n200.png`.
```
```
