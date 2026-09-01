# MorphoMNIST causal experiments

Synthetic image-outcome experiments for validating frugal flows, built so that
**the per-pixel ATE is exactly known** — imposed by construction, not measured
with Monte-Carlo error.

Everything here lives in `validation/morphomnist/` and is run from that
directory.

---

## Quick start

```bash
cd validation/morphomnist

# 1. does everything still work? (~2 min, writes nothing permanent)
python exp_ate_recovery.py --selftest

# 2. inspect a dataset without fitting anything
python prepare_morphomnist_exps.py --preset exp4_covariate_cate

# 3. fit one cell
python exp_ate_recovery.py --preset exp4_covariate_cate --size 8

# 4. a fast end-to-end cycle while developing (recovers nothing; proves plumbing)
python exp_ate_recovery.py --size 4 --n 400 --max-epochs 3 \
    --marginal-max-epochs 3 --n-mc 200

# 5. look at what you have
python exp_ate_recovery.py --collect
```

If you change anything in either script, run `--selftest` before trusting a
result. It asserts 106 invariants and exits non-zero on failure.

---

## The files

| file | what it is |
|---|---|
| `prepare_morphomnist_exps.py` | the data generator. All simulation settings live here. |
| `exp_ate_recovery.py` | fits a frugal flow to a generated dataset, scores it, archives the run. |
| `dataset.py` | MorphoMNIST loader (images + thickness/intensity morphometrics). |

Other scripts in this directory are earlier single-purpose versions, superseded by
these two. Don't extend them.

---

## The data-generating process

Everything happens in **logit space**, where the treatment effect is additive.
Images are downsampled to `size × size`, dequantised, and logit-transformed, so
the outcome is `K = size²` unbounded reals.

```
Y_i(0) = the (transformed) image
Y_i(1) = Y_i(0) + τ_i

τ_ik   = m_k · factor_ik
T_i    ~ Bernoulli( σ(β₀ + β₁ · standardised thickness_i) )
Y_i    = T_i · Y_i(1) + (1 − T_i) · Y_i(0)          ← the only Y the model sees
```

* `m_k` — the **spatial effect map** (a centred disc by default).
* `factor_ik` — the **per-unit modulation**, built from rank scores.

### Why the ATE is exact

Every modulator is built from ranks and then has its **empirical** mean
subtracted, so it has sample mean *exactly* zero:

```
h_i = φ(u_i) − mean_j φ(u_j),    u_i = (rank_i + 0.5)/n
```

Therefore

```
ATE_k = mean_i τ_ik = m_k · (1 + a·mean(h) + …) = m_k
```

**The centering does the work, not the linearity.** `φ` can be any function —
non-monotone, discontinuous, whatever — and the identity still holds. The same
argument is why spatial patterns are free (below). The generator asserts this at
build time; the realised residual is ~1e-15.

> **Consequence:** `ATE` is a number you *set*, not one you measure. Score
> against it directly.

---

## Why thickness and brightness are pre-treatment covariates

This comes up, so it's worth settling. The images are the **DeepSCM synthetic
MorphoMNIST dataset** ([Pawlowski et al. 2020](https://arxiv.org/abs/2006.06485)),
whose SCM is known in closed form — attributes are sampled first and the image is
*synthesised to realise them*:

```
A (thickness) = 0.5 + Gamma(10, 5)
B (intensity) = 191·σ( N((A − 2.5)·2, 0.5) ) + 64
x (image)     = clip( SetThickness(A)(mnist_digit) · B / measured_B, 0, 255 )
```

Verified against `data/`: thickness mean 2.503 / sd 0.632 vs theoretical 2.500 /
0.632, and the intensity residual on the logit scale has sd 0.5007 against the
generator's `scale=0.5` (recorded in `data/args.txt`). The CSV holds the
**sampled** values, not quantities re-measured from the image — so there isn't
even measurement error between the latent factors and what we condition on.

So `A` and `B` are causal **parents** of the image, not summaries of it:

```
A ──→ B ──→ x = Y(0) ──→ Y(1)
│           ↑
│    digit ─┘   (exogenous: style)
└──→ T = 1{U < σ(β₀ + β₁·Ã)},   U ⊥ everything
```

1. **Z is a non-descendant of T** — the definition of a pre-treatment covariate.
   Verifiable, not assumed: hold the seed fixed and move `--ps-slope` from 0 to 3
   (which swings corr(A, T) from −0.01 to +0.69) and `THICKNESS`, `BRIGHTNESS`
   and `Y0` come back **bitwise identical**.
2. **Z is a textbook confounder** — a common cause of `T` and `Y`. Not a mediator
   (`T` doesn't cause `A`), not a collider (`A` has no incoming edge from `T` or
   `Y`). The back-door paths `T ← A → x → Y` and `T ← A → B → x → Y` are both
   blocked by conditioning on `A`.
3. **Ignorability holds by construction**, not by assumption: `T` is built from
   `A` plus an independent uniform, so `T ⊥ (Y(0), Y(1)) | A` is a fact about the
   code. The oracle-IPW row in the table below is the empirical confirmation.

⚠️ **"You can predict thickness from the image" is not a counter-argument.** A
linear predictor on the 64 pixels gets R² = 0.94 — because `SetThickness` *wrote*
the thickness into the image. Inverting a mechanism doesn't reverse its arrow;
the discriminating test is interventional, and that's point 1. In the generative
direction the mechanism is nowhere near degenerate: the attributes explain only
R² = 0.14 of pixel variance, the rest being digit identity and style.

---

## Simulation settings

### 1. The six presets

| preset | confounded | effect depends on | ATE = ATT? |
|---|---|---|---|
| `exp1_rct_homogeneous` | no | nothing (identical for all units) | yes |
| `exp2_confounded_homogeneous` | yes | nothing | yes |
| `exp3_confounded_heterogeneous` | yes | thickness **and own Y(0) rank** | no |
| `exp4_covariate_cate` | yes | thickness, brightness, interaction | no |
| `exp5_quantile_effect` | yes | outcome quantile `u` only | no |
| `exp6_spatial_cate` | yes | as E4, plus a spatial gradient | no |

Designed as a ladder where one thing changes at a time:

* **E1 → E2** changes only assignment ⇒ isolates confounding.
* **E2 → E3/E4/E5/E6** changes only the effect ⇒ each isolates one kind of variation.
* **E4 → E6** adds only spatial structure.

Realised design diagnostics at `--size 8` (n = 5923, single digit class):

| | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|
| corr(thickness, T) | −0.01 | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |
| ATE exact to | 0 | 0 | 5e-15 | 3e-15 | 4e-15 | 3e-15 |
| \|ATT − ATE\| max | 0 | 0 | 0.223 | 0.188 | 0.131 | 0.215 |
| ITE sd across units | 0 | 0 | 0.086 | 0.076 | 0.065 | 0.054 |
| τ(u) paired-vs-marginal gap | 0 | 0 | 0.312 | 0.371 | **0** | 0.368 |
| naive bias (max abs) | 0.064 | 0.754 | 0.972 | 0.942 | 0.878 | 0.904 |
| **oracle-IPW bias (max abs)** | **0.064** | **0.068** | **0.068** | **0.068** | **0.068** | **0.068** |

The last row is the **sampling-noise floor** — what inverse-probability
weighting by the *true* propensity achieves. No estimator can beat it. Judge
recovery against ~0.065, not against zero.

### 2. `--effect-mode` — what τ is allowed to depend on

**`outcome_coupled`** (default; E1–E3)

```
factor = 1 + a_cov·h(thickness) + b_quant·g(own Y(0) rank)
```

⚠️ The `b_quant` term keys on the unit's **own outcome rank**, so it is a
*coupling between the potential outcomes*, not covariate heterogeneity. The
coupling itself is **not identified** from observational data — only its
consequence for the Y(1) margin is. Use this mode deliberately, and don't
describe the `b` term as heterogeneity in writing.

**`covariate_only`** (E4, E6)

```
factor = 1 + a_cov·h(thickness) + a_bright·h(brightness) + a_inter·h(thickness)·h(brightness)
```

No `Y(0)` dependence anywhere — unambiguously treatment-effect heterogeneity.
Brightness is added to `Z` automatically, because a covariate the effect uses
**must** be observed or the CATE is not identified. The interaction term keeps
the CATE surface non-additive.

**`quantile_primitive`** (E5)

```
factor = 1 + b_quant·ψ(u)      ⇒   δ_k(u) specified DIRECTLY
```

Here `Q1(u) − Q0(u) = δ_k(u)` **exactly**, returned as `TAU_ANALYTIC` — the
quantile effect in closed form rather than measured after the fact. The identity
needs the shift to be a function of `(pixel, u)` alone, so `a_cov` and
`a_spatial` are **refused** in this mode: any unit-level term would give two
units at the same `u` different shifts. The generator also verifies the treated
margin stayed monotone (`RANK_PRESERVED`) and raises if a steep `δ` reorders it.

### 3. `--h-shape` — the shape of the CATE

How the effect varies with the thickness rank. Multiplier by decile:

| shape | thin → thick |
|---|---|
| `linear` | 0.50 0.62 0.75 0.87 1.00 1.12 1.25 1.37 1.50 |
| `cubic` | 0.50 0.79 0.94 0.99 1.00 1.01 1.06 1.21 1.50 |
| `quadratic` | 1.50 1.17 0.94 0.80 **0.75** 0.80 0.94 1.17 1.50 |
| `sine` | 1.00 1.35 1.50 1.35 1.00 0.65 0.50 0.65 1.00 |
| `step` | 0.50 0.50 0.50 0.50 0.50 **1.50** 1.50 1.50 1.50 |

`quadratic` is U-shaped, `sine` fully non-monotone, `step` discontinuous — use
these when the point is recovering a *non-trivial* CATE. `--g-shape` does the
same for the quantile term (and for `δ(u)` in E5).

### 4. `--spatial-basis` / `--a-spatial` — spatially coherent heterogeneity

Makes the covariate's influence vary **across pixels**, so units differ in the
*shape* of their effect, not just its size. Same disc, three units, under
`gradient_x`:

```
THINNEST        MEDIAN          THICKEST
0.90 0.90       1.02 1.02       1.30 1.30
0.70 0.70       1.02 1.02       1.50 1.50
0.50 0.50       1.02 1.02       1.70 1.70
0.30 0.30       1.02 1.02       1.90 1.90
```

Opposite tilts, averaging to exactly 1.0.

Bases: `none` (default), `gradient_x`, `gradient_y`, `diagonal`, `radial`
(centre ↔ periphery). Works in `outcome_coupled` and `covariate_only`; refused
in `quantile_primitive`.

### 5. `--effect` / `--radius` / `--base-shift` — the effect map

`circle` (default, ~20% of pixels), `ring`, `const`, `gradient`.
**`gradient` is the hardest** — no flat regions and no exact zeros, so there is
no structure for an estimator to lock onto.

`--radius` defaults to `round(size/4)`, holding the map at ~20% of pixels as `K`
changes. A radius tuned at 8×8 covers only 4.7% at 16×16, which is a much
sparser target — hence the auto-scaling.

### 6. `--ps-slope` / `--ps-intercept` — confounding

`p = σ(β₀ + β₁ · standardised thickness)`. `β₁ = 0` is an RCT; `β₁ = 1.2` gives
corr(thickness, T) ≈ 0.45 with propensities spanning [0.06, 0.999]. `β₀` shifts
the treated fraction (0 ⇒ ~50/50).

### 7. `--size` / `--digit` / `--n` — dimensionality and sample

| flag | notes |
|---|---|
| `--size` | `K = size²`. 4 → 16 (debugging), 8 → 64 (default), 16 → 256 (full res). |
| `--digit` | default 0 ⇒ n = 5923, `Z` = thickness alone (discrete stage bypassed). |
| `--all-digits` | all ten classes ⇒ n = 60000, `Z` 11-dimensional, mixed continuous/discrete stage exercised. A different, harder experiment. |
| `--n` | cap the sample size. |

### ⚠️ The one rule to remember

**Keep the coefficients summing below 1:**

```
a_cov + a_bright + a_inter + a_spatial + b_quant  <  1
```

Above 1, some units' effects flip sign against the map. That's a legitimate DGP
but a *different* one — `frac_factor_negative` in `summarise()` reports it
(measured on the effect's support, since off support the multiplier multiplies
zero and is meaningless).

---

## Fitting: `exp_ate_recovery.py`

### Estimator arms

**`--arm location_translation`** (default) — treatment is masked from the margin
flow, so the whole effect rides on a per-pixel `LocCond` shift. `tau_hat` is an
exact model **parameter**. Correctly specified for E1/E2; misspecified wherever
the effect varies with `u`.

**`--arm flexible_continuous`** — treatment enters the K-dimensional spline
margin. `tau_hat` is **estimated** by paired common-random-number interventional
sampling, and `τ(u)` comes with it. Correctly specified throughout. Runs on
either conditioner:

* `--conditioner mlp` (default) — MADE-masked MLP.
* `--conditioner transformer` — causal-transformer conditioner (TarFlow-style).
  **Requires `--nn-width` divisible by `--nn-heads`**; the default width of 48
  satisfies the default 4 heads.

### ⚠️ The learning rate is not neutral between arms

`location_translation` carries an explicit additive parameter that must travel
from `--ate-init` to the true effect, and it **under-trains badly at small
rates** — its shift parameters stay bunched near their initial value. The
flexible arms are far less sensitive, and the transformer can **diverge** at
larger rates. Tune per arm and say so in writing; a single shared rate quietly
advantages whichever arm it happens to suit.

### Modes

```bash
python exp_ate_recovery.py --preset exp4_covariate_cate     # one cell
python exp_ate_recovery.py --sweep --size 8                 # 6 presets × 3 arms = 18 cells
python exp_ate_recovery.py --sweep --skip-done              # resume an interrupted sweep
python exp_ate_recovery.py --collect                        # table of completed runs
python exp_ate_recovery.py --replot runs/exp_ate_recovery/<run-id>
python exp_ate_recovery.py --selftest
```

⚠️ `--skip-done` keys on `metrics.json` existing, so a run that finished with a
non-finite score still counts as done and **will be skipped**. Delete such
folders before resuming.

### Run folders

Each run writes a self-contained folder under `runs/exp_ate_recovery/`:

```
<UTC-stamp>_<preset>_<arm>_s<seed>_k<K>_<suffix>/
    config.json    every knob + git commit/dirty + library versions
    log.txt        live training output (`tail -f` it)
    metrics.json   recovery scores + timings
    arrays.npz     tau_hat, truth, τ(u) curves, losses (replottable)
    plots/         every figure as PNG
```

`config.json` and `log.txt` appear at launch, so a folder holding only those two
is still training (or died). Only `arrays.npz` is gitignored — completed run
folders can be committed at negligible size.

---

## Reading the results

| key | meaning |
|---|---|
| `ate_mae` | headline: mean \|tau_hat − ATE\| over all K pixels |
| `ate_mae_on_support` | …restricted to pixels with a nonzero true effect |
| `ate_mae_off_support` | …restricted to pixels whose true effect is **exactly zero** |
| `ate_corr` | spatial correlation — is the map in the right *place*? |
| `att_mae` / `atc_mae` | same score against ATT and ATC |
| `design_oracle_ipw_bias_maxabs` | the design's sampling-noise floor |
| `mc_frac_dropped` | fraction of non-finite interventional draws discarded |

**Read the split, not just the total.** A radius-2 disc covers 12 of 64 pixels,
so ~80% of the plain MAE's weight sits on pixels whose true effect is exactly
zero. A model that recovers the magnitude perfectly but smears a little effect
everywhere scores worse than one that does neither well. "Recovered the
magnitude" and "kept the zeros at zero" are different failures.

**Judge against the floor, not against zero.** See the table above: ~0.065 at
n ≈ 5900, K = 64.

**On E1 and E2, `ate_mae` == `att_mae` == `atc_mae` by construction** — the
effect is homogeneous, so all three estimands coincide. They separate on E3–E6,
where the comparison tells you *which* estimand the fit landed on.

**`best_val_loss` is not a model-selection criterion here.** Better density fits
have been observed to recover the ATE *worse*, by spending capacity on
treatment-dependence in pixels that have none. Score against the truth.

**`mc_frac_dropped`** — a spline margin occasionally throws a draw into its tails
and overflows to ±inf. A plain mean lets one such draw among 5000 poison an
entire pixel (this happened). Non-finite draws are dropped and the fraction
reported: a handful is a numerical artefact, a large fraction is a real
pathology.

---

## Known caveats

1. **Single seed.** Nothing in the archive is replicated. Use `--seed-fit` for a
   proper seed study before trusting any ranking.
2. **The ground truth is sample-relative.** The ATE *value* is `m_k` for every
   sample, but because the modulators use within-sample ranks, the **CATE
   function** changes if you resample. Fine for ATE recovery; a problem for CATE
   recovery, which needs a population-level `τ(z)`. Fixable by freezing the rank
   transform on the full 60k pool — not yet implemented.
3. **E3's `b_quant` term is a coupling, not heterogeneity.** It keys on the
   unit's own `Y(0)` rank, so `τ` is not a function of pretreatment variables
   alone. This does **not** break identification — `T` is generated from
   thickness and independent noise whatever form `τ` takes, so ignorability
   still holds and E3 is a valid ATE experiment. But a CATE is not well defined
   for that component, and the coupling itself is unidentifiable from
   observational data. Use E4/E6 for clean heterogeneity and E5 for a stated
   quantile effect. Don't call E3's `b` term "heterogeneity" in writing.
4. **The transformer's read-out is expensive.** Training cost is comparable to
   the MLP (`log_prob` is one parallel pass), but sampling solves one coordinate
   per `lax.scan` step with a full attention pass at each, so it scales far worse
   in `K`. Cut `--n-mc` before cutting epochs.

---

## TO DO 

The built-in sweep
```bash
python exp_ate_recovery.py --sweep --size 8
```

Add --skip-done to resume after an interruption — but delete any folder whose ate_mae is inf/nan first, or it counts as done and gets skipped.

Axis sweeps

CATE shape, on the three presets where it bites:

```bash
for p in exp3_confounded_heterogeneous exp4_covariate_cate exp6_spatial_cate; do for s in linear cubic quadratic sine step; do python exp_ate_recovery.py --preset $p --h-shape $s --arm flexible_continuous --size 8; done; done
```

Spatial pattern:
```bash
for b in gradient_x gradient_y diagonal radial; do python exp_ate_recovery.py --preset exp6_spatial_cate --spatial-basis $b --arm flexible_continuous --size 8; done
```

Effect map (the gradient map is the hard one — no flat regions, no exact zeros):
```bash
for e in circle ring const gradient; do python exp_ate_recovery.py --preset exp4_covariate_cate --effect $e --arm flexible_continuous --size 8; done
```

Confounding strength:
```bash
for s in 0.0 0.6 1.2 1.8 2.4; do python exp_ate_recovery.py --preset exp4_covariate_cate --ps-slope $s --arm flexible_continuous --size 8; done
```
What I'd actually run first: Seeds, not breadth. Every one of your 18 archived runs is a single seed, and the transformer has already swung between diverging and best-in-matrix under settings differing only in learning rate. Adding more axes to an unreplicated matrix multiplies the number of rankings you can't trust.
```bash
for s in 1 2 3 4 5; do for c in mlp transformer; do python exp_ate_recovery.py --preset exp4_covariate_cate --arm flexible_continuous --conditioner $c --size 8 --seed-fit $s; done; done
```
