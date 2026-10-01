# Halo diagnostic ladder — preregistration v1.1 (2026-10-01, after Codex round C0-1 and the smoke budget)

Status: v1, FROZEN at the first S1 launch (sha256 recorded in every stage's `_stage.json`).
Changes after freezing are appended as dated amendments, never edited in.
v0 → v1 changes come from Codex round 1 of session `halo-ladder-c0` (8 findings, all accepted):
sampling unit relabelled and a second corpus added (R1-F1); every gate made quantitative and no
attribution from a null (R1-F2); the "cancels in tau_hat" claim replaced by a measured endpoint (R1-F3);
LEAK redefined as excess leak on exact-floor pixels (R1-F4); n raised to 10 and primary families cut to
three contrasts (R1-F5); S4's u_z specified (R1-F6); the spread-rank stack made the reference arm with its
own P1 cell (R1-F7); template coefficients demoted to descriptive (R1-F8).

## Question

In the MorphoMNIST 8×8 (K=64) frugal-flow experiments a residual spatial pattern ("the halo") appears
in per-pixel model outputs. Is it a property of (a) the copula / propensity stages, (b) the causal
margin p(Y|do(T)) itself, (c) the Uniform+RQS+atanh margin construction the package uses (the only
FF-specific part of the margin; the masked-autoregressive body is flowjax's own), or (d) normalising
flows in general?

Two objects are called "halo" in the record and are NOT the same map:

- **Sense 1** — `E_tau = tau_hat − ATE` on the flexible margin under confounded E2: a positive ring on
  the active off-support stroke pixels (~+0.06 of δ=1), absent on randomised E1; ~13% of the
  unadjusted confounding bias survives (slope 0.14 vs frengression 0.014). Confounding-driven.
- **Sense 2** — with no copula and no treatment, a pattern "where there's variation inside the image
  (digit transitions)", "the same shape regardless of the simulated ATE" (15 Sep minutes). Algebraically
  `tau_hat − ATE = E_mu[1] − E_mu[0]`, so a per-arm mean error cancels in `E_tau` only if it is equal
  across arms. Whether it is equal is an S2 endpoint (`E_mu_diff`), not an assumption.
- Confounding-independent evidence already archived: per-pixel sd ratio model/oracle under do(0) is
  1.3–4.6× on the quiet background pixels on E1 and E2 alike.

Mechanism under test (H_sliver): the runner feeds raw logit pixels (±3.66) with no standardisation
into Uniform[−1,1] → RQS(8, interval 1, min_width 1e-3) → atanh. A pure-background pixel is exactly
`Y ~ logit(0.025 + 0.95·U/256)`: support [−3.6636, −3.5213] (width 0.142, sd 0.041); in tanh space a
sliver 4.3e-4 wide, 1.3e-3 from the boundary, narrower than the spline's minimum bin. Edge pixels are
mixtures of that sliver and a continuous part.

## Corpora and sampling unit

- **Corpus A** (primary for the mechanism; the setting in which the halo was observed): the fixed
  digit-0 training pool, n = 5923, all images in every cell. `seed_data` randomises dequantisation
  noise, treatment assignment and the validation split; `seed_fit` randomises initialisation and
  mini-batching. The unit of inference is therefore a **replication on the fixed digit-0 corpus**,
  not an image draw, and every null is worded that way.
- **Corpus B** (genuine image draws): all ten digits; one master permutation of the 60,000 training
  images (`ExpConfig(digit=None, n=None, seed=1000)`), sliced into 10 **disjoint** blocks of 5923.
  Pixel classes are recomputed on Corpus B's own data. Run only for the three primary contrasts
  (S1: A1s/P0 vs A1s/P1, A1s/P0 vs A2/P1; S2: FF/P0 vs FF/P1 at τ=1), so the mechanism claim is
  also tested across independent image samples.

Seeds: `seed_data ∈ {31,…,40}` (10), `seed_fit ∈ {41, 42}`, crossed (20 fits per config);
`seed_mc = 7` (+ 8 on seed_data 31 for the MC floor). Disjoint from all earlier campaign seeds
(1–5, 11–15, 101–105, 201–220) and from the seed-1 calibration. seed_fit is averaged within seed_data
before any contrast and reported as a variance component. Driver asserts seed_fit ≠ seed_data and one
XLA flag string for every cell.

Fit: lr 1e-2, batch 100, patience 30, max 300 epochs, 10% validation, best-validation checkpoint
(`frugal_flows.training.fit_to_data`, no EMA, no wall cap). float32.

**v1.1 budget decisions (made before freezing, after the smoke timing showed 15–29 h at n_mc 20000):**
- `n_mc = 5000` per arm (the production runner's own value). MC floor on an active pixel: SE of a mean
  ≈ 1.1/√5000 = 0.016, of an sd ≈ 0.011; every primary endpoint is a class mean over ≥ 12 pixels, so the
  MC contribution to a class mean is ≤ 0.005. The second `seed_mc` on seed_data 31 measures it directly.
- **Primary configs** (the arms in the three-contrast primary families, plus Corpus B) keep the full
  10 × 2 crossing. **Exploratory configs** run at `seed_fit = 41` only (10 fits): S1 A1s/P2, A1s/P3,
  A1w, A2/P0, A3, A4, C-smooth, C-zinf; S2 all τ = 0 arms and SEP; S3 all zuko arms; S4 the E1 full-FF
  configs and the two E2 anchors. The variance split (seed_data / seed_fit / residual) is therefore
  reported for primary configs only.
- Stage order S1 → S2 → S4 → S3. S3 may be partial by morning; it resumes on the next night and the
  analysis labels any S3 table built on an incomplete stage as PARTIAL with its cell count.
- Divergence guard: a cell whose |`E_mu`| exceeds 10 on any pixel (A3 produced ~4e4 at 5 epochs in the
  smoke) is marked `diverged`, reported, and excluded from means — the same treatment as non-finite cells.

## Maps (per pixel, logit space, shape (64,), shown 8×8)

gen = model draws (Uniform-base flows via `sample_clamped`; `n_clamped` and non-finite counts
recorded); ref = exact `Y0`, `Y1` of all n units (S1: the fitted `Y`).

| map | definition | reads |
|---|---|---|
| `E_mu[t]` | mean(gen_t) − mean(ref_t) | Sense-2 candidate |
| `E_mu_diff` | `E_mu[1] − E_mu[0]` (= `E_tau` algebraically; reported per class with CI) | equal-across-arms test |
| `E_sd[t]`, `R_sd[t]` | sd(gen_t) − sd(ref_t); log2(sd ratio) | quiet over-dispersion |
| `E_tau` | tau_hat − ATE (S2+) | Sense-1 object |
| `KS[t]`, `W1[t]` | per-pixel two-sample KS statistic and exact 1-D W1 | fidelity |
| `LEAK_X` | P_model(Y_k ∉ [−3.6636, −3.5213]) − P_ref(same) | **sliver test, S1 primary** |
| `FLOORMASS` | P_model(Y_k < −3.52) − P_ref(Y_k < −3.52) | floor-component mass |
| `NBCORR` | \|corr_gen − corr_ref\| over edge-adjacent pixel pairs | dependence / rank-mask |

Pixel sets, computed on the REAL data of each (corpus, seed_data) and applied unchanged to the
synthetic controls of that seed: `exact_floor` = P(raw pooled pixel = 0) = 1 (Corpus A seed 1: 15 px);
`pure_floor` ≥ 0.95 (31 px); `mixture` ∈ (0.05, 0.95) (22 px); `ink` ≤ 0.05 (11 px); `quiet` =
sd(Y) < 0.3 (32 px); `active_off` = not quiet and ATE = 0 (20 px); `disc` = ATE ≠ 0 (12 px; the
geometric disc at τ = 0); regions `disc / ring / far` from `region_masks(8, 2)`. Thresholds are
fixed; a Corpus-A cell whose quiet count falls outside [30, 34] is flagged, not re-thresholded.
`LEAK_X` is undefined on C-smooth (no atoms) and is not computed there; C-smooth and C-zinf are
scored on `R_sd` and `KS` over the same pixel indices.

"Where the pattern lives" is read from class means with CIs on the cross-table floor class
{exact, pure, mixture, ink} × region {disc, ring, far}. The template OLS (templates `t_mix`, `t_sd`,
`t_imb`, `t_thick`, `t_ring`) is **descriptive only**: its R² and the 5×5 template correlation matrix are
reported; coefficients are not endpoints (corr(t_sd, t_thick) ≈ 0.93; corr(t_imb, t_thick) ≈ 0.98 under
E2). Thickness-leakage vs imbalance is decided by the E1-vs-E2 design contrast on the same arm and seeds.

Noise floors beside every map: 200-bootstrap per-pixel SE (mean, sd, naive diff); split-half of the
data; MC floor from the second `seed_mc`; oracle split-half floor for KS/W1. Seed-1 calibration:
split-half |Δmean| max quiet 0.004 / active 0.050; |Δsd| max 0.012 / 0.051; null naive T-difference on
Y0: active sd 0.017, max 0.051. An E1 `E_tau` of 0.02–0.06 is at the data floor.

## Gates (every gate is this conjunction; nothing else counts as "resolved")

A contrast is **resolved** iff (i) Holm-adjusted paired Wilcoxon p < 0.05 within its declared primary
family, (ii) the paired 95% bootstrap CI excludes 0, and (iii) the paired effect meets the minimum:
`LEAK_X` ≥ 0.02; median log2 `R_sd` on `exact_floor` ≥ 0.5; |`E_tau`| class mean ≥ 0.03 (≈ the E1 data
floor 0.029); S4 slope difference ≥ 0.05.
An arm is **elevated** on an endpoint iff its arm-vs-reference-data contrast is resolved; a fix
**removes** an elevation iff the fix-vs-status-quo contrast is resolved AND the fixed arm is not itself
elevated. A null is reported as "no resolved change at this n (10 replications on the fixed digit-0
corpus / 10 disjoint all-digit draws): paired 95% CI [a, b] includes 0" and **never produces an
attribution**.

## Design

**S0 data only.** Classes, templates (+ correlation matrix), floors, closed-form quiet density;
model-free maps: E1 imbalance and E2 unadjusted bias over the 10 seed_data; Corpus B index 0 check.

**S1 unconditional p(Y), no T, no Z.**

| arm | stack | preproc |
|---|---|---|
| **A1s** (reference = current package margin) | Uniform base → 4 × [`MaskedAutoregressiveSpread` RQS(8, interval 1), w48 d1, Permute] → atanh; pytree structure asserted equal to Laura's `_margin_dist` | P0 raw logit α=0.05; P1 per-column standardise; P2 α=0.3; P3 affine pixel→[−0.9,0.9], no logit |
| A1 (historical) | A1s with flowjax's modulo-rank `MaskedAutoregressive` (the 4 Sep archived stack) | P0 |
| A1w | A1s with width 72 (≥ dim) | P0 |
| A2 | Normal base, RQS(8, interval 5), MAF w48 L4, no tanh | P0, P1 |
| A3 | Normal base, affine MAF | P1 |
| A4 | Normal base, RQS coupling flow | P1 |
| C-smooth | A1s/P0 on synthetic Gaussian images (data mean + covariance, no atoms) | P0 |
| C-zinf | the same images → clip[0,1] → quantise 1/256 → dequantise → logit (atoms restored) | P0 |
| P4 demo | A1s, no dequantisation; 2 cells, demonstration only | — |
| Corpus B | A1s/P0, A1s/P1, A2/P1 | — |

P1 is estimand-preserving; P2/P3 are not and are never compared to P0 on `E_tau`.

Primary endpoints: `LEAK_X` mean on `exact_floor`; median log2 `R_sd` on `exact_floor`.
**Primary family (Holm, 3):** A1s/P0 vs A1s/P1; A1s/P0 vs A2/P1; A1s/P0 vs A1/P0. Each also run on
Corpus B except the third. Secondary (exploratory, CIs only): all other arms vs A1s/P0; `E_sd` and
`FLOORMASS` on `mixture`; `NBCORR` on active pixels; C-smooth vs C-zinf on `R_sd`/`KS`.

Attribution rules (positive findings only):
- A1s/P0 elevated AND P1 removes it → H_sliver (preprocessing) on the current stack.
- A1s/P0 elevated AND A2/P1 not elevated AND A1s/P0 vs A2/P1 resolved → the Uniform+atanh construction.
- A1s/P0 elevated AND A2/P1 elevated → not construction-specific; S3 decides library-generality.
- A1s/P0 not elevated → "no resolved Sense-2 artefact in the unconditional margin at this n".
- C-zinf elevated AND C-smooth not (exploratory) → supports the atom mechanism; stated as such.
- A1/P0 vs A1s/P0 resolved on `NBCORR` only → the rank defect is a dependence defect, not the halo.

**S2 conditional margin Y|T, randomised (E1), τ ∈ {0, 1}.** Arms: FF-cond (spread stack, T as
unmasked conditioner) at P0 and P1; A2/P1 conditional; LT (location-translation margin: A1s + `LocCond`,
exact parametric tau_hat); SEP (one unconditional A1s/P0 flow per arm). 5 × 2 × 20 = 200 cells; plus
Corpus B FF/P0 and FF/P1 at τ=1 (40).
Primary endpoints: `E_tau` class mean on `disc` (bias) and RMS of the seed-mean `E_tau` on `active_off`.
**Primary family (τ=1, Holm, 3):** FF/P0 vs FF/P1; FF/P1 vs A2/P1; FF/P1 vs LT.
Secondary: `E_mu_diff` per class (equal-across-arms test); variance split seed_data / seed_fit /
residual (primary configs only); slope of `E_tau` on `t_imb` (≈1 with residual at the MC floor is CORRECT for a margin-only
model under randomisation); `E_mu[t]`, `E_sd[t]`, `LEAK_X`; corr of seed-mean `E_mu^0` across τ.
Identification statement: "a pure-flowjax conditional flow identifies the effect" iff the `disc`
seed-mean tau_hat lies within ±0.05 of 1.0 AND the `active_off` RMS is ≤ the E1 floor (0.03) for at
least one of {FF/P1, A2/P1}; a Sense-2 structure in `E_mu[t]` whose `E_mu_diff` is not resolved is
reported as a fidelity problem, not an identification problem.

**S3 different library (PyTorch, zuko 1.6).** `NSF` (autoregressive RQS, Normal base), `MAF` (affine),
NSF coupling. S1 tasks at P0/P1 and S2 tasks at τ 0/1, P0/P1, same seeds; own loop mirroring
`fit_to_data`. 18 × 20 = 360 cells. Exploratory family: each zuko arm vs A1s/P0 (S1) and vs FF/P1 (S2),
CIs only; S3 "decides library-generality" only through resolved elevations of the zuko arms vs their
own reference data, reported with the same (i)–(iii) conjunction but labelled exploratory.

**S4 copula and Z re-added.** `train_frugal_flow(causal_model="flexible_continuous")` called directly:
`u_z` = ECDF midranks rank/(n+1) of thickness (no stage-1 flow; E1/E2 use thickness only); copula knots
8 / depth 1 / width 50 / 4 layers (package defaults, Laura's spread ranks — the fetched function has no
rank switch); margin 8/1/48/4 mlp; lr 1e-2, 300 epochs, patience 30, batch 100. Matrix: {E1, E2} ×
{P0, P1} full FF + anchors FF-cond/P0 and FF-cond/P1 on E2 (margin only) = 6 × 20 = 120 cells.
Primary: E2 off-support slope of `E_tau` on `t_imb`, P0 vs P1, paired by seed_data (family of 1).
Secondary: E2 ring decomposed as (S2 margin-only E1 residual) + (confounding residual); E1 full-FF vs
S2 margin-only; the anchors give the slope-scale endpoints (margin-only on E2 ≈ 1.0).
Attribution: resolved slope reduction under P1 → the Sense-1 ring has a margin component of that size.
A null → "no resolved margin component of the E2 ring at this n"; the ring's attribution stays open.

## Analysis and wording

Paired Wilcoxon signed-rank (n = 10, minimum two-sided p = 0.00195) with 95% paired bootstrap CI;
Holm within each declared primary family; everything else labelled exploratory. Cells with > 0.1%
clamped or non-finite draws are reported and excluded from means. Cells from differing XLA flag strings
are never pooled. All S0–S4 cells run in one overnight batch (Dan's decision), so gates are applied post
hoc and the analysis states, for S3 and S4, which S1/S2 rule they would have served.

---

## Amendment A1 (2026-10-01 09:30, after S0–S4 and Codex C1 round 1; appended, not edited in)

**Trigger.** The frengression comparator (`exp_frengression_recovery.py:314-322`, frozen tuning setting
`y_scaling=per_pixel, y_sd_floor=0.25`) fits a per-pixel standardised target with the sd FLOORED at a
quarter of the global sd, and inverts on samples. The frugal-flow runner fitted raw logits. Dan asked
whether the same scaling works for the frugal-flow architecture.

**New preprocessing P5 (frengression-style):** per column, `Z_k = (Y_k − mean_k) / s_k` with
`s_k = max(sd_k, 0.25 · sd(Y_all))`, fitted on the fitting data, inverted on samples. Affine, so
estimand-preserving.

**Cells (stage S5, seeds 31–40 × {41, 42} = 20 fits per config):** S1-type A1s/P5 (Corpus A and B);
S2-type ff_cond/P5 at τ = 1 (Corpus A); S4-type ff_full/P5 on E1 and E2. 100 cells.

**Primary endpoints and family (Holm, 3 contrasts, same gates as v1.1):** on `LEAK_X` exact-floor and
median log2 `R_sd` exact-floor: A1s/P5 vs A1s/P0 (does the floored scaling remove the elevation);
A1s/P5 vs A1s/P1 (does it differ from full standardisation); elevation of A1s/P5 vs reference data.
Exploratory: Corpus-B replication; S2 E_tau disc bias and floor-referenced active-off RMS, ff/P5 vs
ff/P1 and vs ff/P0; S4 E2 slope and E2 − E1 paired disc bias under P5. Attribution rules: a resolved
removal under P5 that is not resolvedly different from P1 → "floored scaling suffices"; P5 elevated
while P1 is not → "the floor leaves part of the artefact"; nulls are worded as in v1.1.

## Amendment A1.1 (2026-10-01 10:05, after Codex C1 round 2 finding R2-F4; appended BEFORE any S5
outcome has been opened, aggregated or inspected — attested)

The A1 rule "P5 not resolvedly different from P1 → floored scaling suffices" rested on a null and is
replaced. Non-inferiority margins equal to the v1.1 minimum effects are declared: `LEAK_X` exact-floor
0.02; median log2 `R_sd` exact-floor 0.5. "Floored scaling suffices" requires ALL of: (i) A1s/P5 vs
A1s/P0 resolved (removal); (ii) A1s/P5 not elevated vs reference data; (iii) the upper bound of the
paired A1s/P5 − A1s/P1 95% bootstrap CI below the margin on BOTH endpoints. If (i) and (ii) hold but
(iii) fails: "floored scaling removes the elevation but is not shown to match full standardisation
within the margin". If (i) fails: "floored scaling does not resolve the elevation at this n". The
exploratory S2/S4-type P5 comparisons are reported with CIs only.

## Amendment A2 (2026-10-01 10:20, after the S2/S3 no-covariate comparison; appended before any S6 cell exists)

**Trigger.** On randomised E1 with no covariates (S2/S3, τ=1, standardised), the frugal flexible margin's
ATE MAE (0.022) exceeds an otherwise identical flowjax Normal-base spline MAF (0.017), the PyTorch flows
(0.012–0.018) and the naive difference in means (0.012); the location-translation margin reaches 0.007.
The frugal margin differs from the flowjax arm by three coupled choices (Uniform(−1,1) base; spline on
[−1,1]; atanh after the spline) plus the MADE rank class. Hypothesis H_construct: the bounded
construction accounts for the frugal-vs-flowjax gap. H_shift: an explicit per-pixel shift accounts for
the remainder down to the location-translation level.

**Stage S6 (all P1 standardised, preset exp1_rct_homogeneous, Corpus A).** New arms, built by composing
existing pieces only: **N** = Normal base, RQS(8, interval 5), `MaskedAutoregressiveSpread` conditioner
with T as unmasked input, 4 layers with permutations, no tanh (the S2 A2 arm with spread ranks);
**LT-N** = the same stack unconditional (T masked) followed by the package `LocCond` per-pixel shift.
Comparators from S2: U = ff_cond/P1, A2 = a2_cond/P1, LT = lt/P0 (lt was run at P0; an **LT/P1** arm is
added so the LT comparison is also at P1). Cells: N, LT-N, LT/P1 at τ=1 with seeds 31–40 × {41,42}
(primary, 60) and at τ=0 with seed_fit 41 (exploratory, 30). 90 cells.

**Endpoints.** Primary: per-cell ATE MAE (mean |E_tau| over 64 px) and `E_tau` disc-class mean; secondary:
off-support MAE, `KS` on active pixels, log2 `R_sd` quiet. **Primary family (Holm over 3 contrasts × 2
endpoints):** U − N (H_construct); N − A2 (ranks); N − LT-N (H_shift). Gate = v1.1 conjunction with minimum
effect 0.003 on ATE MAE (≈ 2 SE at n=10) and 0.01 on disc bias. Exploratory: LT − LT/P1, LT/P1 − LT-N,
τ=0 arms, distributional endpoints. Predictions, stated before running: U − N > 0 and ≥ half of U − A2;
N − A2 not resolved; N − LT-N > 0. Nulls worded as v1.1; no attribution from a null.

## Amendment A3 (2026-10-01 16:15, appended before any S7 cell exists)

**Trigger.** Laura's fix of 2026-09-25/26 (commits 1b2510a, 8cb4086): flowjax's rule
`hidden_ranks = arange(nn_width) % dim` leaves the copula's covariate outputs unable to depend on
outcome ranks ≥ nn_width when nn_width < dim (14 of 64 pixels at 8×8 with width 50; 48 of 64 with the
paper-setting width 16; 206 of 256 at 16×16 with width 50). Her `autoregressive_hidden_ranks` spreads the
ranks so every input is reachable at any width. Every full-model cell in S4/S5 already ran WITH the fix
(the halo branch is based on 6eecd77, which contains it; the rank files are unchanged up to 5cdac8c).
Dan asked whether the fix has a positive impact. S1 and S6 only compared rank classes in the MARGIN.

**Stage S7.** Full frugal flow (`train_frugal_flow`, flexible_continuous, ECDF u_z of thickness), Corpus A
(digit 0, 8×8), harness fit settings (lr 1e-2, ≤300 epochs, patience 30), P0 raw logit (as the runner
feeds it). Factor R: copula hidden ranks **old** (the exact pre-1b2510a rule, restored for the copula
only by patching the name the copula module imports; margin untouched) vs **new** (the fix). Factor W:
copula width 50 (historical default) and 16 (paper setting). Cells: E2 with seeds 31–40 × {41, 42}; E1 with
seed_fit 41. The W=50/new cells are the existing S4 ff_full/P0 cells. New cells: W50/old (E2 20, E1 10),
W16/old (20, 10), W16/new (20, 10) = 90. Plus a **paper-setting check**: lr 1e-3, ≤1000 epochs, width 16,
E2, seeds 31–35 × 41, old vs new (10 cells), exploratory.
A deterministic **connectivity table** (number of outcome ranks that can reach the covariate outputs)
is computed from the constructed masks for old/new × width {16, 50} × K {64, 256}.

**Endpoints.** Primary on E2: `E_tau` disc-class mean (confounding bias), per-cell ATE MAE, and the
off-support slope of `E_tau` on the naive-bias map. **Primary family (Holm over 2 widths × 3 endpoints):**
old − new at W=50 and at W=16, paired by (seed_data, seed_fit). Gate = v1.1 conjunction with minimum
effects 0.01 (disc bias), 0.003 (ATE MAE), 0.05 (slope). Secondary/exploratory: E2 − E1 paired disc bias
at seed_fit 41; E1 ATE MAE (does the fix cost anything under randomisation); W16 vs W50 under the fix;
the paper-setting check. Predictions, stated before running: at W=16 old − new > 0 on all three
endpoints (the copula cannot adjust for thickness through 48 unseen pixels); at W=50 the contrast is
small (the 14 unseen pixels are mostly the quiet bottom rows). Nulls worded as v1.1; no attribution from
a null. Not tested here: 16×16, and all-digit data (Laura's paper grid).

## Amendment A4 (2026-10-01 17:20, appended before any S8/S9 cell exists) — Gaussian-scale reformulation

**Trigger (Dan).** The difficulties sit at the boundaries of the uniform-rank construction. Replace
"marginally uniform ranks" by "marginally standard-Gaussian scores" g = Φ⁻¹(u) throughout: the margin maps
Y to g_Y ~ N(0, I) (Normal base, unbounded spline, no atanh); covariates become normal scores
g_Z = Φ⁻¹(midrank/(n+1)); the copula is a conditional flow p(g_Z | g_Y) on a Normal base, blind to T. The
model class is unchanged (copulas are invariant to monotone marginal maps); the geometry is. Pixels stay
logit + per-pixel standardisation (P1), so the estimand is unchanged. No conversion to the uniform scale
inside the model.

**S8 — treatment and response only (calibration read-out).** The gated contrasts for this step are S6's
(U − N, N − LT-N). S8 refits arms U (ff_cond), N (n_cond), LT-N (lt_n) on E1, τ = 1, P1, Corpus A,
seeds 31–40 × seed_fit 41 (30 cells), storing per cell: KS of each latent coordinate (data pushed through
the fitted margin's inverse) against its base law, on the 10% validation rows and on all rows; mean
|off-diagonal correlation| of the latent coordinates; best epoch and epochs run. Descriptive; no gate.

**S9 — full model on the Gaussian scale.** Joint flow on (Y_std, g_Z) given T: base N(0, I_{K+nvars});
copula block = conditional flow of the nvars covariate coordinates given the K outcome coordinates
(flowjax masked autoregressive flow over nvars dims with cond_dim = K, RQS(8, interval 5), MLP width 50
depth 1, 4 layers; identity on the first K coordinates); margin block on the first K coordinates given T.
Arms: **GFF-flex** (margin N) and **GFF-shift** (margin LT-N: T masked from the residual, explicit
per-pixel `LocCond` shift, tau_hat read from the shift vector). Harness fit settings (lr 1e-2, ≤ 300
epochs, patience 30, batch 100), n_mc 5000, read-out by sampling g_Y ~ N(0, I) through the margin at
T = 0, 1. Corpus A, P1. Cells: E2 seeds 31–40 × {41, 42}; E1 × 41; per arm → 60. Paper-setting check
(lr 1e-3, ≤ 1000 epochs), E2, seeds 31–35 × 41, both arms → 10, exploratory, queued last. Total 70.
Comparator **FF-uniform**: the S4 cells ff_full/P1 (E2: 20; E1: 10).

**Endpoints (E2).** `E_tau` disc-class mean; per-cell ATE MAE; off-support slope of `E_tau` on the
naive-bias map. **Primary family (Holm over 3 contrasts × 3 endpoints, n = 10 seed_data):**
GFF-flex − FF-uniform; GFF-shift − FF-uniform; GFF-shift − GFF-flex. Gate = v1.1 conjunction with minimum
effects 0.01 (disc bias), 0.003 (ATE MAE), 0.05 (slope). Secondary/exploratory: E2 − E1 paired disc bias
at seed_fit 41; E1 ATE MAE and disc bias; latent calibration (g_Y KS; KS of the model-implied g_Z marginal
against N(0,1) from 20,000 draws; KS of the data g_Z pushed through the copula's inverse); best epoch;
non-finite draw counts (expected 0, no clamp); the paper-setting check against S7's paper-setting
new-rank cells. Predictions, stated before running: GFF-flex lowers E2 ATE MAE relative to FF-uniform;
GFF-shift has the lowest E1 ATE MAE; the effect of either on the E2 disc bias is not predicted. Nulls
worded as v1.1; no attribution from a null. Not tested: probit pixel transform (not estimand-preserving),
16×16, all-digit data, heterogeneous presets.
