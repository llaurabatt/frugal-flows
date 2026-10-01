# Leftover confounding in the E2 estimates — evidence and working hypothesis

> Scripts for this investigation: `validation/morphomnist/scripts/leftover_confounding/` (toys) and
> `validation/morphomnist/scripts/exp_ate_recovery/` (image runs). Results and logs stay in
> `validation/morphomnist/runs/leftover_confounding/` and `runs/exp_ate_recovery/`.

**For the current state of knowledge read STATUS.md first.** This README is the dated log; some early
entries use stronger words ("refuted", "ruled out") than the evidence supports — STATUS.md states each
claim with its scope.

## RESULT 2026-09-30 (regime): full batch is worse; regime is not the difference

```
Training-regime test (toy_regime.py, current arm, n=5923, 4 seeds, no early stopping; 16 fits, 0 failed):
  batches of 100, 400 epochs: end slope K=8 0.060-0.085, K=16 0.019-0.068 (at the early-stop epoch 0.044-0.122 / 0.051-0.097)
  full batch, 5000 steps (frengression's regime): end slope K=8 0.106-0.158, K=16 0.070-0.152 -> WORSE, flat over training
=> The training regime is not what makes frengression work. Remaining untested difference: the objective
   (likelihood vs energy score). Recommended to the user: stop the search, report the leftover with the
   sample-size ablation, proceed to the paper grid (after the E4/E6 T-blind-copula check).
```

## RESULT 2026-09-30: spread check confirms an under-confident copula; the reversed arm is WORSE

```
CORRECTION (2026-09-30, after the user asked whether the spread is the reason): NO. The spread check
confirms the DIRECTION (copula too wide) but not the SIZE. Gaussian regression-dilution calculation with
the exact toy values (true Var(Z|Y) 0.18 / 0.14; spread ratio 1.17 / 1.21; mean shrink 0.965 / 0.975)
predicts about 0.9 % / 1.1 % of the dependence lost, against a measured leftover of 8.4 % / 6.1 %.
So regression dilution through the copula is NOT the explanation. The copula's picture of thickness given
the image is nearly right; by elimination the missing dependence sits in the MARGIN (which carries part of
the thickness-driven difference between arms, as the September notes described). Why the margin does
this at n=5923 but not at 50000, and why frengression's does not, is still unknown.
Also: batch 500 (with early stopping, images, noise_fixes) was worse than batch 100 (slope 0.038 vs 0.029,
MAE 0.028 vs 0.019); full batch WITHOUT early stopping (frengression's regime) not tested before -> regime run.
```

```
Spread check (toy_spread.py, 4 seeds): the current arm's copula is UNDER-CONFIDENT at n=5923, as predicted.
  model variance of z given the image ranks / true variance: K=8 1.17 (se 0.03), K=16 1.21 (0.05); K=8 n=50000 1.06 (0.01)
  slope of the model's conditional mean on the true one: 0.965 / 0.975 (n=5923), 0.996 (n=50000)
Reversed arm on the toy (toy_reversed.py, 32 fits, 0 failed): WORSE than the current arm at n=5923.
  leftover slope (g-formula read-out), current arm, frengression on the same data:
  K=8:  w0 0.118 (se 0.012), w10 0.159, w100 0.068   | current 0.084 | frengression -0.008
  K=16: w0 0.163, w10 0.223, w100 0.167               | current 0.061 | frengression -0.009
  MAE K=8: reversed 0.19-0.26 vs current 0.12; K=16: 0.29-0.37 vs 0.11. At n=50000 (w0) slope 0.005.
  Rank KS (pooled ranks vs uniform): w0 0.18/0.11, w10 0.07/0.09, w100 0.05/0.08 (noise 0.018).
=> Reversing the copula direction does NOT fix the leftover in this implementation. (The copula IS too wide,
   but by far too little to explain the leftover: see CORRECTION above.)
Next (the plan's fallback): training regime. RUNNING screen regime: toy_regime.py, current arm without early
stopping, full batch 5000 steps vs batches of 100 for 400 epochs, K=8/16, seeds 1-4 (16 fits).
```

## 2026-09-30 — Reasoning for the reversed copula (STATUS: HYPOTHESES, STILL BEING CHECKED)

Nothing below is established unless marked. Tests planned: spread check (toy), then the reversed arm on the toy.

### Convinced (argued, not yet tested in code)
1. The reversed arm models p(y | u, t) = p*(y | t) c(r | u), c a flow over the image ranks r given u, blind to t.
   This is the Evans-Didelez conditional p(y | z, t) = p*(y|t) c(u, F*(y|t)); it integrates to 1 over y
   for every (u, t), so it is a proper conditional likelihood and can represent the truth.
2. The margin/copula split is not identified: margin -> h o margin and the copula reshaped to match give the
   same likelihood for any fixed map h of rank space (same for both arms and every u). The ATE read out by
   simulating u ~ U, r ~ c(r|u), y = F*^-1(r|t) is the g-formula of p(y|u,t), so it does not depend on h.
   Counterfactuals y' = F*^-1(F*(y|t)|t') do not depend on h either (h cancels).
3. Rank penalty: under the model, the rank of an observed image given (u, t) has density c(r|u). Pooled over
   the data (u uniform by construction) the ranks follow int c(r|u) du. So "pooled observed ranks uniform"
   <=> "copula's r-marginal uniform" <=> "p* is the causal margin". Computed in the fast direction.
4. This penalty only selects among solutions with the same likelihood (some h always makes the pooled
   ranks uniform without changing p(y|u,t)), so it does not compete with the fit. The u-marginal penalty in
   the current arm did compete. Caveats: the penalty uses observed ranks (equal to the model's only if the
   fit is good), and the margin's layers may not represent the needed h exactly.

### Corrected
- My earlier argument ("the current copula predicts a 1-D target from redundant pixels, so there is less
  signal") is WRONG as stated: the likelihood gained by learning the dependence is the mutual information
  I(U; Y | T), which is symmetric, so it is the same in both directions.

### Hypothesis for why the direction matters: regression dilution (TESTED 09-30: direction right, size ~1 % of 6-8 % -> NOT the explanation)
The effect needs E[Y | u, t]. The reversed arm estimates it directly (a fitted mean relationship; an
over-wide noise estimate does not bias it). The current arm learns q(u | r) and gets E[Y | u] only by
inverting with Bayes; Gaussian picture: u | y ~ N(beta y, s^2) gives an implied slope of y on u of
beta Var(Y) / (beta^2 Var(Y) + s^2), attenuated when s^2 is too large (copula under-confident, e.g. a fit
stopped early from a near-independent start). Fits: 5-10 % attenuation in the toy; shrinks with n;
frengression (models Y given Z) unaffected; the u-marginal penalty could not help.
Test (spread check, toy): the fitted copula's spread of u given the image ranks should be WIDER than the
true spread, by an amount consistent with the measured attenuation. Not a gate: we build the reversed arm
whatever it shows (frengression's result is the main evidence for the direction); only a copula that is
too NARROW would make us rethink.

### New issue found: E4 and E6 (UNTESTED)
A T-blind copula assumes the covariate-Y(t) dependence is the same for both t. True in E1/E2/E3/E5 (the
treated image is a per-pixel function of the untreated one that does not depend on the covariates, so ranks
are preserved). False in E4/E6, where Y(1) = Y(0) + tau(z): the copula is misspecified there by construction,
in both arms. Frugal theory allows a t-dependent copula; our code masks t because unmasking leaked in the
current direction. Possible explanation of the unexplained September E4 ring (+0.15). Needs its own look
before the paper grid.

### Next (agreed 2026-09-30)
- Spread check on the toy (screen, minutes) and build the reversed arm in parallel; the arm's toy result decides.


## RESULT 2026-09-28 (images): the leftover shrinks with sample size

```
Images, single fits (seeds k and 1001; plain setting), E2 leftover slope by sample size (digit 0):
  n=1500 0.109 (se 0.020) | n=3000 0.084 (0.013) | n=5923 0.058 (0.006)   [subsample_8x8.sh, 40 fits, 0 failed]
All ten digits (n=60000), seed k, 5 datasets [all_digits_8x8.sh, 10 fits, 0 failed]:
  E2 slope 0.009 (se 0.007), disc +0.005, MAE 0.011 (digit 0 same seeds: slope 0.059, disc +0.012, MAE 0.029)
  E1 MAE 0.0069 (digit 0: 0.019)
=> CONFIRMED on the images: the halo / leftover confounding is a small-sample effect. With 10x the data a
single fit keeps <1 % of the confounding.
```

## RESULT 2026-09-28 (mechanism): the copula misses 5-10 % of the dependence

```
Mechanism check (toy_mechanism.py, K=8, n=5923, 4 seeds). bhat_k = slope on Z of the fitted model's
E[Y_k | Z, T=0] (importance weights q(u|r) over 20000 uniform image-rank draws; ESS >= 1090).
Prediction err_k = (b_k - bhat_k) * (mean Z|T=1 - mean Z|T=0):
  correlation(actual err, predicted) per seed: 0.96, 0.83, 0.98, 0.99
  slope of actual on predicted:                1.02, 1.20, 1.51, 0.96
  share of the true dependence the copula misses: 6.0 %, 5.2 %, 8.2 %, 10.4 %
=> CONFIRMED in the toy: the leftover is the part of the covariate-image dependence the
validation-selected copula has not learned, times the covariate imbalance between arms.
```

## RESULT 2026-09-28 (later): K-pixel toy — the leftover is a small-sample effect

```
K-pixel toy (toy_multi.py; Y_k = b_k Z + tau_k T + AR(0.7) noise; true ranks Phi(Z)), 4 seeds each,
slope = share of the confounding left (single fits; OLS ~0 everywhere):
  K=8 : n=5923 0.084 (se 0.013) -> n=50000 0.002 (0.005)
  K=16: n=5923 0.061 (0.008)    -> n=50000 0.011 (0.012)
  K=8, n=5923 with patience 300: identical to patience 30 (kept epochs 27-39; the validation loss never
  improves again in 300 more epochs).
FF MAE at n=5923 0.08-0.16 vs OLS 0.026 (OLS is the correct model here).
Reading: the leftover reproduces in a known-truth toy at the image sample size and disappears with ~10x
the data. It belongs to the validation-selected fit at n~6000, not to stopping too early and not to the
objective's optimum. Image check: E2 with all ten digits (n ~ 60000).
```

## RESULT 2026-09-28: the penalty does not fix the leftover (revised wording — NOT a refutation of the mechanism)

```
E2, single fits at fit seed 2001, 5 datasets (slope = share of the confounding left; OLS ~0.005):
  plain   slope 0.067 (se 0.006)  disc -0.048  MAE 0.049  KS(copula u vs observed u) 0.042  val NLL -51.89
  ecdf    slope 0.074 (0.010)     disc -0.030  MAE 0.044  KS 0.038                          val NLL -52.35
  umw10   slope 0.082 (0.009)     disc -0.035  MAE 0.049  KS 0.024                          val NLL -51.93
  umw100  slope 0.085 (0.007)     disc -0.046  MAE 0.053  KS 0.023                          val NLL -52.15
  umw1000 slope 0.381 (0.069)     disc +0.192  MAE 0.085  KS 0.020                          val NLL -52.70
E1: plain MAE 0.021, ecdf 0.020, umw10 0.021, umw100 0.022, umw1000 0.014 (val NLL -51.74 -> -52.91).
Reading: the penalty brings the copula's u-marginal close to the data (KS 0.042 -> 0.020) and improves
the held-out likelihood, but the E2 leftover gets WORSE as the weight grows (0.067 -> 0.38 at 1000).
Revised (user challenge): this rules out "enforcing the copula u-marginal fixes it", not that the
normalisation is involved. "Zero at the truth, so it does not move the optimum" only holds if the fit can
reach the truth; it does not (slope 0.067 without the penalty), so constraining q moves a compromise in an
unknown direction. Firm only at weight 1000; weights 10/100 are ~2 se worse on one unlucky fit seed.
Checked: the true copula is the same in both arms in E2 (generator: Y1 = Y0 + tau in logit space, a
constant shift per pixel, ranks preserved), so a T-blind copula is correctly specified. The penalised
copula fits held-out data better (copula term -0.51 -> -0.72 at weight 100), so it was not broken.
ecdf ranks: no change (as expected once the stage-one ranks were found uniform).
Seed 2001 is an unlucky fit seed, not a regression: an old seed-1001 fit rerun with cf2a766 is identical.
```
Runs: ecdf_8x8.sh, umpen_8x8.sh, umarg_check_8x8.sh; analyse_leftover_fixes.py.

Written 2026-09-27. Status: **hypothesis, not yet tested.**

## The problem

On E2 (confounded, 8x8, digit 0, n = 5923) the flexible-continuous frugal flow leaves part of
the confounding in its effect estimate. Measure: regress, pixel by pixel, the estimate's error
map on the dataset's confounding map (observed treated-minus-untreated mean image minus the
true effect). The slope is the share of the confounding still in the estimate.

- 5-fit average (5 datasets x 5 fit seeds, copula width 16, lr 1e-3): slope 0.036 (se 0.003),
  disc error +0.022. OLS: slope about 0.005, disc +0.000.
- Single fits (20 datasets): slope 0.044 (se 0.005).

## What has been ruled out

| Candidate | Test | Result |
|---|---|---|
| Copula too small / too slow | step A: copula width 8; copula learning rate x3, x10 (150 fits) | slope 0.037 / 0.046 / 0.074; no improvement, faster copula worse |
| Too many parameters for n | existing fits with smaller margins (20 datasets each) | margin width 16: slope 0.062-0.064, i.e. worse than width 48 (0.044) |
| Stopping too early | 600 epochs without early stopping, effect read out every 20 epochs (10 fits, `training_length_8x8.sh`) | slope falls from ~0.19 (epoch 20) to ~0.05 by epoch 100-200, then flat to epoch 600 |

In the Gaussian toy (`toy_gaussian.py`, `toy_trajectory.py`; Z ~ N(0,1), T ~ Bern(sigmoid(2Z)),
Y = Z + T + e, true effect 1) the leftover WAS about stopping early: +0.085 at n = 5000 (7 % of
the confounding), shrinking to about +0.015 by epoch 400, and -0.001 (se ~0.014) at n = 50000.
So in the toy any bias from the cause below is at most ~2-3 % of the confounding. The images
behave differently: the leftover does not shrink with training.

## Hypothesis: the copula is only half normalised, so the propensity does not drop out

Notation: u = F_Z(z) covariate ranks (uniform over the whole data); p*(y | t) the causal margin;
r = F*(y | t) the image ranks under it.

Evans & Didelez: p(z, t, y) = p(z) p(t | z) p(y | z, t) with p(y | z, t) = p*(y | t) c(u, r).
p(z) and p(t | z) carry no margin/copula parameters, so they drop out and what is left to
maximise is  sum_i [ log p*(y_i | t_i) + log c(u_i, r_i) ]  — the same two terms we train.

That step needs p*(y | t) c(u, r) to be a proper density in y for every (z, t):
  int p*(y | t) c(u, r) dy = int c(u, r) dr = 1   for every u,
which holds because c is a copula (uniform in both arguments).

Our copula block (`masked_autoregressive_flow_first_uniform`, fitted on
`hstack([y, u_z])` given T, causal_flows.py ~L436) is a flow that produces u given r, blind to T.
It guarantees  int q(u | r) du = 1  for every r, but NOT  int q(u | r) dr = 1  for every u.
Without that, p* q is not a proper density of y given (z, t); what is maximised is the density
of (y, u) given t, which contains p(u | t) — the covariate distribution in each arm, i.e. the
propensity information. The only T-aware part of the model is the margin, so under
confounding it absorbs part of it. In E1, p(u | t) is uniform in both arms, the two objectives
coincide and nothing leaks.

Consistent with: no improvement from copula capacity, learning rate or training length; worse
with a smaller margin; E1 unaffected. Not yet directly tested.

## Update (same evening): the reversed copula alone is not enough — prefer a uniformity penalty on u

A copula needs BOTH marginals uniform. A flow gives uniformity for free only to the variable
that sits untouched in the base:
- current direction: r is untouched, so r is exactly uniform and p* IS the causal margin
  (the frugal property); the u-marginal  int q(u | r) dr  is not enforced -> propensity leaks.
- reversed direction: the u side is enforced, the r side is not. Then margin and copula are
  identified only up to a common reshaping h of rank space (margin -> h o margin, copula
  composed with h, same for both arms): "infinitely many ways to map to the base" (user's
  point). The ATE by simulation and the rank counterfactuals do not depend on h, but p* alone
  is no longer the causal margin, which breaks the explicit-margin / simulation claim.

Preferred fix: keep the current direction (p* exact) and enforce the missing half on u, which
is only d = 2 dimensional: a penalty making the copula's u-marginal uniform, estimated from
samples (r ~ U^K, v ~ U^d -> u = Q(v | r); distance of those u to U(0,1)^d, e.g. energy
distance or MMD). Exact alternative: the normaliser N(u) = int q(u | r) dr, loss
-log p* - log q + log N(u) (a proper conditional likelihood for any q), but N needs an
integral over 64-dim r where q is sharply peaked, so its Monte Carlo estimate would be noisy.
Direct check of the hypothesis on current fits: E2 fits should show a non-uniform model
u-marginal, E1 fits a uniform one.


Answers to the two follow-up questions (user, same evening):
- Does the penalty distort the fit? The true copula satisfies it (penalty zero at the truth), so it
  does not move the optimum; it removes only non-copula solutions. Its gradient reaches only the
  copula (r is drawn from the base, u = Q(v | r) goes through the copula blocks only); the margin
  moves only indirectly. Practical risks: the weight (too large hampers the copula, too small does
  nothing), sampling noise, and stage-one ranks being only approximately uniform (fallback: compare
  to the pooled observed u). E1 is the control: it should barely change.
- Counterfactuals: unchanged. They use only the margin, r_i = F*(y_i | t_i), y_i' = F*^{-1}(r_i | t');
  the copula is blind to T. Interventional sampling and the ATE read-out are also unchanged.

Next (agreed 2026-09-27): first the direct check — at the end of each fit, draw r ~ U^K, v ~ U^d,
push through the copula, and measure how far the resulting u is from uniform. Prediction: clearly
non-uniform on E2, roughly uniform on E1. Only if that holds, build the penalty.

## Earlier proposal (superseded above): reversed copula

Condition on u and let the copula produce r given u, still blind to T:
  p(y | u, t) = p*(y | t) c(r | u).
- The copula is a flow over r conditioned on u, so int c(r | u) dr = 1 by construction; u is a
  condition like t, p(u | t) is never modelled, and the propensity drops out as in E&D.
- Not guaranteed: int c(r | u) du = 1 (what makes p* exactly the interventional margin).
  For the ATE, read out by simulation: u ~ U(0,1), r ~ c(r | u), y = F*^{-1}(r | t) for
  t = 0, 1. For simulation with p* as the causal margin, check the gap with a diagnostic
  (uniformity of r when u ~ U(0,1)); add a penalty only if needed.
- Alternatives rejected at the time: an extra p(u | t) model in the current direction (cannot
  be normalised independently of the copula). (I also wrote "a penalty on the current copula's
  r-marginal" — wrong: in the current direction r is already uniform; the missing half is the
  u-marginal, which is what the update above proposes to penalise.)
- Build: conditioner that masks PART of the condition (copula reads u not t; margin reads t
  not u), new arm in train_frugal_flow (current arm untouched), read-out sampler, name tag,
  config / index / tables.

## Tests planned

1. Toy: still recovers the effect 1 (correctness only; the toy never showed this bias).
2. E2 + E1, same 5 datasets x 5 fit seeds as step A: slope and 5-fit-average error vs the
   current arm, OLS, frengression. Success: E2 slope towards ~0.005 without hurting E1.
3. Diagnostic: uniformity of r when u ~ U(0,1).
If (2) fails: test the stage-one covariate ranks (fitted marginal flows) against exact
empirical-CDF ranks.

## Files

- `toy_gaussian.py`, `run_toy.sh`, `results/` — the 32-fit toy grid.
- `toy_trajectory.py`, `run_trajectory.sh`, `trajectory/` — toy without early stopping.
- `../../scripts/exp_ate_recovery/training_length_8x8.sh`, `analyse_training_length.py` — images.
- `../../scripts/exp_ate_recovery/confounding_fixes_8x8.sh`, `analyse_confounding_fixes.py` — step A.
