# MorphoMNIST causal experiments

Synthetic image-outcome experiments for validating frugal flows, built so that
**the per-pixel ATE is exactly known** — imposed by construction, not measured
with Monte-Carlo error.

Everything here lives in `validation/morphomnist/` and is run from that
directory.

---

## Current state (2026-10-01) — read this first

* **Benchmark: all ten MNIST digits** (`--all-digits`, n = 60000). Digit 0 (n = 5923, the
  default `--digit 0`) was the development setting only.
* **Method:** the frugal flow with the flexible-continuous margin (`--arm flexible_continuous
  --conditioner mlp`) in the paper setting, which is now the default for every other knob
  (learning rate 1e-3, copula width 16, margin width 48 / 8 knots, batch 100, patience 30, max 1000
  epochs). **The estimate is the average of 5 fits** with fit seeds `{k, 1001, 1002, 1003, 1004}`
  on dataset `k`: single fits are 1.5–2× worse.
* **Criterion:** "performs as frengression on the ATE" — on the same datasets, the flow's 5-fit
  average has an effect-map error not larger than frengression's (one fit, seed `k`) beyond noise.
* **Paper grid, 8×8, all digits** (E1–E6 × datasets 1–10): launcher
  `scripts/exp_ate_recovery/grid_8x8_alldigits_v2.sh`, analysis
  `scripts/exp_ate_recovery/analyse_grid_8x8_alldigits.py` → `runs/exp_ate_recovery/analysis/`.
  Early results: the flow matches or beats frengression on E1, E2, E3, E5, E6; **E4 is behind**.
* **Known problems** (evidence and scope in `docs/leftover_confounding/STATUS.md`):
  1. *Leftover confounding at higher resolution.* At 16×16 the flow keeps ~9 % of the E2
     confounding even on all digits (error 2.5–3× frengression's); larger networks do not help.
     8×8 on all digits is clean. Not yet fixed.
  2. *E4 / E6:* the copula is blind to the treatment, which is misspecified when the effect
     depends on the covariates (E4, E6). E4 is the preset where the flow lags.
  3. *32×32 cost:* the effect read-out samples one pixel at a time (~2.7 h per fit).
* **Layout:** code at the top level; launch and analysis scripts in `scripts/` (index and how to
  run them: [`scripts/README.md`](scripts/README.md)); write-ups in
  `docs/`; everything generated (run folders, indexes, logs, the dataset cache) in `runs/`,
  which is gitignored.

---

## Quick start

```bash
# From the repository root: one environment contains both JAX and PyTorch stacks.
micromamba create -f environment-frengression.yaml
micromamba activate frugal-flows-frengression
cd validation/morphomnist

# 1. does everything still work? (~2 min, writes nothing permanent)
python exp_ate_recovery.py --selftest

# 2. inspect a dataset without fitting anything
python prepare_morphomnist_exps.py --preset exp4_covariate_cate

# 3. fit one cell in the paper setting (all digits, flexible arm; the other knobs are the defaults)
python exp_ate_recovery.py --all-digits --preset exp2_confounded_homogeneous \
    --arm flexible_continuous --size 8 --seed-assign 1 --seed-fit 1
#    (the default --arm is still location_translation: pass --arm explicitly)

# 4. fit Frengression through the same Config/run_one experiment interface
python exp_ate_recovery.py --preset exp4_covariate_cate --size 8 \
    --arm frengression

# 5. a fast FF end-to-end cycle while developing (recovers nothing; proves plumbing)
python exp_ate_recovery.py --size 4 --n 400 --max-epochs 3 \
    --marginal-max-epochs 3 --n-mc 200

# 6. look at every FF and Frengression run in the shared archive
python exp_ate_recovery.py --collect
```

The two runners also retain their focused smoke tests:

```bash
python exp_ate_recovery.py --selftest
python exp_frengression_recovery.py --selftest
```

If you change a runner, run its `--selftest` before trusting a result. Each
exits non-zero on failure.

---

## The files

| file | what it is |
|---|---|
| `prepare_morphomnist_exps.py` | the data generator. All simulation settings live here. |
| `exp_ate_recovery.py` | public experiment interface for the existing FF arms plus Frengression. |
| `exp_frengression_recovery.py` | isolated official-package Frengression adapter used by the public interface. |
| `compare_frengression_ff.py` | fail-closed complete-grid comparison across estimators. |
| `dataset.py` | MorphoMNIST loader (images + thickness/intensity morphometrics). |
| `copula_diagnostics.py` | the copula checks `exp_ate_recovery.py` runs at the end of every fit with a copula (section below). |
| `baselines.py`, `run_index.py`, `run_tables.py`, `check_runs.py` | baselines as run folders, the two indexes, the per-run summary tables, the consistency check. |
| `sample_diagnostics.py` | generated-outcome quality checks run at the end of every flow fit (section below). |
| `dataset_store.py` | rebuilds a run's dataset from its `config.json` (hash-checked) and caches it in `runs/datasets/`; `run_arrays(run_dir)` returns a run's arrays with `Y` / `ITE` filled in. Runs no longer store the data. |
| `strip_dataset_arrays.py` | one-off: removed `Y` / `ITE` from old runs after a per-run exact match (dry run by default). |
| `scripts/` | launch and analysis scripts for every batch (`exp_ate_recovery/`, `frengression/`, `baselines/`, `leftover_confounding/` toys). |
| `docs/` | write-ups: `leftover_confounding/STATUS.md` (what is known, with evidence levels) and `README.md` (dated log). |

Other scripts in this directory are earlier single-purpose versions. Do not
extend them for the Frengression comparison.

## One experiment interface

The adapter calls the official `frengression.Frengression.train_y`; it does not
reimplement or extend the model. The frozen reporting profile is learning rate
`1e-3`, width 100, three layers, noise dimension 64, and per-pixel outcome
scaling with an SD floor.

The existing `Config` and `run_one` are now the only interface a caller needs.
The original FF paths and defaults are unchanged; `arm="frengression"` lazily
dispatches to the adapter, so ordinary FF use does not require importing Torch.

```python
from exp_ate_recovery import Config, run_one

metrics = run_one(Config(
    preset="exp4_covariate_cate",
    arm="frengression",
    seed_data=1,
    seed_fit=1,
))
```

The same applies at the command line. `--sweep` retains location translation,
flexible/MLP, and flexible/transformer and adds Frengression as one extra cell
for each E1-E6 preset.

### Fast complete-path check

```bash
# Tiny budgets prove the plumbing; these are not reporting results.
python exp_ate_recovery.py --sweep --size 4 --n 300 \
    --max-epochs 2 --marginal-max-epochs 2 --n-mc 200 \
    --frengression-num-iters 20 --frengression-n-mc 800 \
    --frengression-threads 1
```

### Paper grid (current workflow, 2026-10-01)

```bash
# all ten digits, 8x8: E1-E6 x datasets 1..10 x flow fit seeds {k,1001..1004} + frengression seed k.
# A pool of 48 slots x 5 cores (taskset); skips cells that are done (result + saved weights) or running.
bash scripts/exp_ate_recovery/grid_8x8_alldigits_v2.sh
bash scripts/baselines/baselines_grid_8x8_alldigits.sh         # OLS / IPW / AIPW on the same datasets
python scripts/exp_ate_recovery/analyse_grid_8x8_alldigits.py   # -> runs/exp_ate_recovery/analysis/grid_8x8_alldigits.md
```

The analysis reports, per preset and method (flow 5-fit average, flow single fits, frengression
seed-k fit, OLS): the error over all pixels, signed error on disc / ring / background, the leftover
slope (error map regressed on the dataset's confounding map), and the pass/fail against the
criterion. Timing on 5 cores per fit: flow ~2.1 h, frengression ~3.3 h (all digits, 8×8).

### Earlier reporting workflow (pre-2026-09-30, digit 0, location translation included)

This is the complete run-then-compare sequence. It uses E1-E6 at K=64 and five
reporting seeds. Each seed changes both the generated dataset and the learned
model's initialisation. The learned-model sweep is long and sequential;
`--skip-done` makes the loop resumable.

```bash
# 0. Check both implementations before starting the reporting grid.
python exp_ate_recovery.py --selftest
python exp_frengression_recovery.py --selftest

# 1. Classical estimators (naive, IPW, OLS, AIPW, oracle IPW), one dataset per call
#    or every dataset the flow runs used that has no baseline folder yet:
python baselines.py --preset exp1_rct_homogeneous --size 8 --seed-data 1
python baselines.py --from-index

# 2. E1-E6 x FF location translation, FF MLP, FF transformer and Frengression.
for seed in 1 2 3 4 5; do
    python exp_ate_recovery.py --sweep --size 8 \
        --seed-data "$seed" --seed-fit "$seed" --skip-done
done

# 3. Optional inspection of every completed learned-model run.
python exp_ate_recovery.py --collect

# 4. ATE comparison across all nine methods, with seed-averaged ATE maps.
python compare_frengression_ff.py \
    --size 8 --seeds 1 2 3 4 5 \
    --out runs/comparison-ate

# 5. Distributional comparison for the three models that estimate tau(u).
python compare_frengression_ff.py \
    --metric tau_u_rmse_vs_marginal \
    --methods ff_flexcont_mlp ff_flexcont_transformer frengression \
    --size 8 --seeds 1 2 3 4 5 \
    --out runs/comparison-tau --no-plots
```

The learned models write below `runs/exp_ate_recovery/`; the classical methods
write below `runs/baselines/` (see "Baselines" below). The final commands create:

- `runs/comparison-ate/{runs.csv,summary.csv,summary.md,ate_maps_*.png}`
- `runs/comparison-tau/{runs.csv,summary.csv,summary.md}`

The comparison refuses missing seeds, duplicate cells, non-finite shared
scores, and mismatched dataset designs. It accepts the existing baseline CSV
format without changing `baselines.py`; size is recovered from `n_pixels`, with
the existing default radius and digit used for the comparison identity.

W&B remains optional. For a direct run add `--wandb`; for the frozen reporting
grid use `sweeps/frengression_report.yaml` with
`frengression_sweep_agent.py`. Keep reporting seeds 1-5 separate from tuning
seeds 101+.

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

### 8. `--seed-data` / `--seed-assign` / `--seed-fit` — what each seed decides

| flag | decides |
|---|---|
| `--seed-data` | one generator, used in order for: the shuffle of the digit's images (and which are kept under `--n`); the dequantisation noise added to every pixel before the logit; and, unless `--seed-assign` is set, the treatment assignment `T = (uniform < propensity)`. Two data seeds differ in all three. |
| `--seed-assign` | default `None`: the assignment is drawn from the continuation of the `--seed-data` stream — every dataset before 2026-09-18 was built this way, and this keeps them reproducible. An integer: an independent generator for the assignment only, with the images and their noise held fixed by `--seed-data`. To bootstrap over assignments, fix `--seed-data` and vary `--seed-assign`. The run name gets an `sa<k>` tag and the dataset a different `dataset_id`. |
| `--seed-fit` | network initialisation and batch order. Never touches the data. |

The effect map `ATE` is deterministic given the preset and size; no seed changes it.

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

### What is fitted: `--model`

**`--model ff`** (default) — the frugal flow: the arm's margin plus the copula on the
covariate ranks, after the stage-1 quantile fit. Every run before 2026-09-18 is this.

**`--model margin`** — the treatment-conditioned image margin alone: no stage-1
quantiles, no copula. The same spline blocks `ff` puts after its copula, fitted
directly to `p(y | t)` with the same `fit_to_data` call. A baseline for what the
copula adds; `flexible_continuous` only. With `--base-shift 0` (no treatment
effect anywhere) the run carries the `effect0` tag, like any model on that data.

**`--model margin_sep`** — one unconditional margin per treatment arm, each fitted
on that arm's images only; the effect is the difference of their paired samples.
`metrics.json` carries arm 0 under the standard keys and arm 1 as
`best_val_loss_arm1` / `n_epochs_run_arm1`; `arrays.npz` gains `loss_train_arm1`
and `loss_val_arm1`. `flexible_continuous` only.

Margin-only runs save no `u_z` and no copula split (`val_copula_nll`). The
design-check figure's middle panel says so instead of showing the stage-1 quantiles.

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
python exp_ate_recovery.py --sweep --size 8                 # 6 presets × 4 configurations = 24 cells
python exp_ate_recovery.py --sweep --skip-done              # resume an interrupted sweep
python exp_ate_recovery.py --collect                        # table of completed runs
python exp_ate_recovery.py --replot runs/exp_ate_recovery/<run-id>
python exp_ate_recovery.py --selftest
python exp_ate_recovery.py --model margin --runs-root /some/scratch/dir   # trial run kept OUT of runs/
python run_tables.py runs/exp_ate_recovery/<run-id>        # the six summary tables -> tables.md
python check_runs.py [--no-wandb] [--root DIR]             # naming/config/wandb consistency of every run
```

`--runs-root DIR` writes the run folder under `DIR` instead of `runs/exp_ate_recovery/`;
use it for any fit that must not land among the real runs.

### The training loop: `frugal_flows.training.fit_to_data`

Since 2026-09-21 every fit trains through `frugal_flows.training.fit_to_data`, a drop-in
replacement for flowjax's routine. With the default options it reproduces the library bit
for bit (same key sequence, same step, same batching, same best-epoch and patience rule;
asserted by `tests/test_training_loop.py` and by refitting a grid cell), so runs made
before and after the switch are comparable. What it adds:

* **The validation split is recorded.** flowjax shuffles the rows and keeps the last 10 %
  for early stopping without saying which; the loop now stores them (`train_idx`,
  `val_idx` in `arrays.npz`, `n_train` / `n_val` in `metrics.json`). Everything computed
  "on the held-out rows" (the copula term, the copula and sample diagnostics) uses them.
  Older runs fall back to `_fit_val_indices`, which replays the key sequence and checks the
  result by the train/held-out loss gap; `val_split_source` says which (`recorded` /
  `reconstructed` / `none`).
* `--wall-cap-s S`: stop after the first epoch that ends past `S` seconds
  (`termination = wall_cap`; otherwise `patience` or `epoch_cap`, now recorded rather than
  inferred from the epoch count).
* `--select-on {joint,copula}`: what early stopping and the reported checkpoint follow.
  `joint` is the library rule (the batched validation loss). `copula` uses the validation
  copula NLL on the whole validation set after every epoch, the term that carries
  deconfounding (see "The copula is what buys identification"); the series is saved as
  `loss_select`, the joint loss at the chosen epoch as `val_loss_at_best`, and `best_select`
  is the criterion's minimum.
* `--track-every N` (with `--track-n-mc`): every `N` epochs, a small interventional
  read-out (`ate_mae`, signed disc / ring / far error against the truth, and the leftover
  `slope` on the dataset's confounding map) is stored under `metrics["track"]`. Each read-out
  compiles new code: keep to ~40 per fit (the process runs out of memory mappings beyond ~100).
* `--ema-epochs N`: keep a running average of the weights over about `N` epochs and use it for
  validation and the reported fit (tag `ema<N>`). Lowers fit-to-fit spread; small gain on top of
  the 5-fit average.

### Defaults and other options (2026-09-30 / 10-01)

The defaults are the paper setting: `--learning-rate 1e-3`, `--copula-nn-width 16`,
`--max-epochs 1000` (changed 2026-09-30), with `--nn-width 48`, `--rqs-knots 8`, `--batch-size 100`,
`--max-patience 30`. **Name tags still mark departures from the historical reference** (learning
rate 1e-2, copula width 50), so a paper-setting fit is named `..._lr0.001_copw16_...`.

Options that were tested and **not adopted** (kept for reproducibility; evidence in
`docs/leftover_confounding/STATUS.md`): `--copula-lr-mult` (tag `coplr`), `--copula-umarg-weight`
(penalty making the copula's covariate marginal uniform; tag `umw`), `--u-z-method ecdf`
(empirical-CDF covariate ranks; tag `ecdf`), and the library arm `flexible_reversed`
(`frugal_flows/reversed_copula.py`; toy only, not wired into this script).

**Under test, not the default** (2026-10-02): `--margin-order fixed` (tag `mfix`, wandb tag
`margin_order_fixed`) keeps one raster pixel order in every layer of the image margin instead of a
random permutation after each layer. The margin is then the Rosenblatt map in that order, which makes
conditioning on the first pixels exact and gives counterfactuals a plain reading. Test:
`scripts/exp_ate_recovery/margin_order_8x8.sh` (E1/E2 × datasets 1–3 × 5 seeds, paired with the grid).

`--save-model` (default on): the fitted weights go to `model.eqx`; `load_model(run_dir)` rebuilds
the flow (identical effect map). Frengression saves `model.pt` (`exp_frengression_recovery.load_model`).

### Using a saved fit: effect map, samples, counterfactuals

Fits made from 2026-10-01 save their weights (`model.eqx`). This reloads one, reproduces its effect
map, and transports observed images to the other treatment (checked on an all-digits E2 fit: the
effect map matches the saved one exactly; counterfactual error 0.037 per pixel against the truth,
versus 0.19 for leaving the image unchanged).

```python
import exp_ate_recovery as E, dataset_store as DS
import jax.random as jr
from frugal_flows.interventions import interventional_samples, counterfactual_flexible

run = "runs/exp_ate_recovery/<run-id>"        # a flexible_continuous fit with model.eqx
flow = E.load_model(run)                       # the fitted flow (rebuilt, weights filled in)
a = DS.run_arrays(run)                         # saved arrays + Y / ITE rebuilt from config.json

# interventional samples under do(T=0) and do(T=1), paired; their mean difference is the effect map
draws = interventional_samples(jr.key(0), flow, cond_dim=1, n_mc=5000, dim_y=a["Y"].shape[1])
tau = (draws["y1"] - draws["y0"]).mean(0)      # = a["tau_hat"] (same key and draws as the fit's read-out)

# counterfactual images of observed units under the other treatment (rank-preserving margin transport)
t = a["X"][:, 0]
y_cf = counterfactual_flexible(flow, a["Y"][:10], a["u_z"][:10], t[:10], 1 - t[:10])
```

Images are on the model's scale (dequantised logits of the pooled pixels). Frengression fits save
`model.pt`; `exp_frengression_recovery.load_model(run)` returns the model and its scaled inputs.

**Reproducibility.** A fit is determined by its config and seeds, but floating-point sums depend
on how many CPU threads the process gets: the same fit run unpinned and pinned to 5 cores gives
different results (effect maps differ by up to ~0.06, kept epoch 203 vs 168), while two runs pinned to
*different* sets of 5 cores are identical (max difference 0). So a fit reproduces exactly given the same
**number** of cores. The paper grid pins every fit to 5 cores (`taskset`); reproduce a grid fit with
`taskset -c <any 5 cores>`. Fits made before 2026-10-01 ran unpinned and reproduce only unpinned.

### The run index: `runs/exp_ate_recovery/index.csv`

One row per run, every column a query can need: identity (run id, uid, launch stamp,
wandb name/id/url, model, preset, arm, conditioner, `variant`), setup (digit, n,
split sizes, seeds, size, K, radius, region sizes, treatment slope, effect size,
draws), architecture and optimisation, training (stopping rule, epochs, best epoch,
loss values, wall time), performance (MAE/RMSE, signed error and MAE per region, both
arm errors per region, standard errors, non-finite counts), then every raw config key
as `cfg.<key>`. Missing values are empty cells, so numeric columns stay numeric.

The folders are the source of truth; the index is derived from them and can always
be rebuilt. Every finished run appends or replaces its own row automatically.

```bash
python run_index.py                                   # rebuild from every folder
python run_index.py --upsert runs/exp_ate_recovery/<run-id>
python run_index.py --query "preset == 'E1' and variant.isna() and termination == 'patience'"
```

or in pandas: `pd.read_csv("runs/exp_ate_recovery/index.csv")` and `groupby`. A plain
fit has an empty `variant`; `coplam4`, `copw200`, `effect0.5`, `rct`, … name what was
changed from it.

### The effect-map figure (`plots/ate_maps.png`) and the "against the images" metrics

Two rows of five panels, the disc outlined in black, region averages printed under
every error panel (signed error over all pixels, disc, ring, far; MAE and RMSE under
the two signed-error panels).

Row 1 is against the **truth**: the estimated effect, the true effect, their
difference, and — when the model samples both arms — each arm's sampled mean minus
that arm's true population mean.

Row 2 is against the **images**: the observed treated-minus-untreated difference over
all `n` images; the finite-sample imbalance (observed minus true — what a raw group
comparison gets wrong on this dataset); the estimate minus the observed difference;
and each arm's sampled mean minus the mean of that arm's images. The last two differ by
exactly the panel before them.

The same quantities are in `metrics.json` (`imb_*`, `vsobs_*`, `d0_*`, `d1_*`),
in Table 2 of `tables.md`, and as columns of both indexes, so "does the model reproduce
the imbalance in its data, and by how much does it depart from it" is a query, not a
new script. A panel whose input the folder does not hold (no sampled arms for
`loctrans` and the baselines other than `naive`) says so instead of drawing.

### Copula diagnostics (`copula_diagnostics.py`, `plots/copula_*.png`, `cop_*` metrics)

Every fit with a copula (`--model ff`) ends with a set of checks on the copula alone,
computed while the flow is still in memory (the fitted flow is not saved, so they
cannot be computed afterwards; the arrays behind the figures are, so `--replot`
redraws them). They run on flowjax's held-out rows when
`exp_ate_recovery._fit_val_indices` can reconstruct that split, otherwise on all rows;
`cop_rows` says which, and every caption repeats it. If the checks themselves fail,
the fit's normal outputs are still written and the error text is stored as
`cop_error`. They add about 30 s at 8×8 (`--copula-diag-n-mc`, default 20, sets the
number of conditional draws).

Notation. Before the main flow is trained, each covariate gets its own
one-dimensional flow; an observation's covariate value is replaced by its position in
that fitted distribution, a number in [0, 1] called the covariate rank `U_Z`. The main
flow's causal margin turns each pixel value into an outcome rank `R` given the
treatment, so `R` has one entry per pixel. The copula part learns how `U_Z` depends on
the whole `R`; the treatment does not enter it (its conditioning input is fully
masked). Running the copula backwards on an actual observation returns the noise it
would have needed to produce that observation's `U_Z` from its `R`; that noise is the
base coordinate `v`, again in [0, 1], and uniform if the copula has learned the
dependence.

Four figures, with their numbers printed on them:

* `copula_margins.png`, one row per covariate: histogram and uniform Q-Q plot of `U_Z`
  (does the covariate's own margin fit?) and of `v` (does the copula explain that
  covariate?), each with its KS distance from uniform.
* `copula_dependence.png`: scatter plots of one covariate rank against the other
  (E4/E6 only) and of one pixel's `R` against each covariate rank, for the pixel with
  the largest signed error in the disc, the ring and the far region. Three columns per
  covariate: the actual ranks; ranks the copula draws given that observation's whole
  `R` (one of the `cop_n_mc` draws); the base coordinate after the inverse. Spearman
  correlations in the titles, with the spread over draws for the predicted ones.
* `copula_dependence_maps.png`, one row per covariate: per pixel, the Spearman
  correlation between `R_k` and the covariate rank in the data; the same for ranks
  drawn from the copula given `R` (mean over draws); their difference, with disc, ring
  and far averages and the largest pixel under it; the correlation between `R_k` and
  `v` after the inverse (should be zero); and the signed error of the effect map for
  comparison. The difference map is the direct test of whether the copula misses
  dependence where the effect is wrong.
* `copula_calibration.png`, one row per covariate, six groups: untreated, treated, and
  the four quartiles of the mean logit intensity over the disc. Each panel plots the
  fraction of `v` at or below `q` against `q` for `q = 0.1 .. 0.9`, with a ±1.96
  binomial band for that group's size, the group's `n` and the largest gap.

In `metrics.json` and the wandb summary every scalar is under the `cop_` prefix. The
index keeps, per covariate, `cop_ks_u_*`, `cop_ks_v_*`, `cop_rho_ru_*_gap_{disc,ring,far,maxabs}`,
`cop_rho_rv_*_maxabs`, `cop_cal_*_{t0,t1,qmax}`, plus `cop_rows`, `cop_n`, `cop_error`
and the three thickness–brightness correlations; the per-quartile values, group sizes
and Monte Carlo spreads stay in `metrics.json`. Table 5 of `tables.md` lists them with
definitions. Runs before 2026-09-20 and margin-only runs have these columns empty.

### Generated-outcome quality (`sample_diagnostics.py`, `plots/samples_*.png`, `gen_*` metrics)

Every fit that samples its arms (all three models on the flexible arm) also checks the
samples themselves, per treatment arm, from the same draws the effect read-out used.
The reference is the true potential outcome `Y(t)` of the held-out units for both `t`,
exact from the generator, so the check does not depend on which arm a unit was observed
in. Reference and generated are two samples of one population distribution, not paired
individuals. Everything numeric is on the logit scale; only the gallery is shown in
pixel intensity. Same conventions as the copula diagnostics: held-out rows when the
split can be reconstructed (margin-only fits have no joint likelihood to reconstruct it
from and use all rows), `gen_rows` says which, a failure is stored as `gen_error`.

Three figures: `samples_gallery.png` (random reference and generated images per arm,
one intensity scale); `samples_moments.png` (reference and generated per-pixel mean and
SD maps and their differences, with MAE, RMSE and the Monte Carlo standard error of the
generated mean); `samples_distributions.png` (CDF overlays at the disc, ring and far
pixels the copula figure uses, with the two-sample KS distance; Pearson correlation of
every edge-adjacent pixel pair, generated against reference; the ROC of a logistic
regression telling reference units from an equal number of draws, five-fold
cross-validated, whose AUC is 0.5 when a linear rule cannot tell them apart).

The read-out now draws through `frugal_flows.interventions.sample_clamped`, which is
`flow.sample` with base coordinates on the support boundary moved inward (an exact 0
from the uniform sampler otherwise reaches arctanh and returns −inf); `mc_n_clamped`
counts how many were moved, `mc_frac_dropped` the draws still discarded as non-finite.

The index keeps, per arm, `gen_mean_mae_t*`, `gen_sd_mae_t*`, `gen_ks_max_t*`,
`gen_nbcorr_mad_t*`, `gen_auc_t*`, plus `gen_rows`, `gen_n_ref`, `gen_error`,
`mc_frac_dropped` and `mc_n_clamped`; Table 6 of `tables.md` has the full set with
definitions. Runs before 2026-09-21 have these columns empty.

### Baselines: `baselines.py`, `runs/baselines/`

The classical per-pixel estimators — naive difference in means, Hajek IPW with an
estimated propensity, per-pixel OLS on `[T, basis(Z)]`, 5-fold cross-fitted AIPW,
and oracle IPW with the true propensity — run on **one dataset per call**, built by
the very function and `Config` class the flow runs use (`exp_ate_recovery.build_data`),
with the same defaults (radius `round(size/4)`, digit 0, …) and the same generator
overrides (`--base-shift`, `--ps-slope`, …). One call writes one run folder:

```
runs/baselines/<UTC stamp>_baselines_<preset>_[<variant>_]k<K>_sd<seed>_d<digit>_<uid>/
    config.json    run_id, uid, dataset_id, data_hash, the generator config, basis
    arrays.npz     ATE, ATT, ATC, Y, X, ITE, PROPENSITY, tau_hat_<method> (x5)
    metrics.json   whole-image and regional scores per method
    plots/         ate_maps_<method>.png
```

`sd<seed>` is the *data* seed — a baseline has no fit seed. `--from-index` runs every
distinct dataset in the flow index that has no baseline folder yet.

**Joining baselines to flow runs.** Every dataset carries two fingerprints, computed
in `prepare_morphomnist_exps.dataset_identity` and recorded in every `config.json`,
`metrics.json` and both indexes: `dataset_id`, a hash of the preset and every
generator knob (so the same arguments give the same id, on either side), and
`data_hash`, an md5 of the built `Y`, `X` and `ATE` arrays (so equal ids can be
checked to have produced identical bytes). `data_hash` does not cover `Z`: with the
effect switched off, exp2/exp3/exp5 build the same `Y` and `X` with `Z` = thickness,
and exp4/exp6 the same `Y` and `X` with `Z` = thickness + brightness, so five presets
share one `data_hash` while giving two different fits. A third field, `z_hash` (md5 of
`Z`), records that difference; runs made before 2026-09-20 have no `z_hash` and the
index leaves the column empty for them. `runs/baselines/index.csv` has one row per
(dataset, method) with the same column names as the flow index wherever the meaning is
the same, and

```bash
python run_index.py --compare "preset == 'E1' and K == 64"   # every method on every dataset
python run_index.py --baselines                               # rebuild runs/baselines/index.csv
```

joins the two on `dataset_id`, warning if any dataset's `data_hash` disagrees between
rows. A flow row is labelled `<model>_<arm>[-trf]_[<variant>_]s<fit seed>`, a baseline
row by its method.

⚠️ `--skip-done` keys on `metrics.json` existing, so a run that finished with a
non-finite score still counts as done and **will be skipped**. Delete such
folders before resuming.

### Run folders

Each run writes a self-contained folder under `runs/exp_ate_recovery/`, named after
its wandb run with the launch time in front:

```
<UTC-stamp>_<wandb name>/        e.g. 2026-09-10T14-29-26Z_ff_e1_flexcont_k64_s101_d0_85c851
    config.json    run_id, wandb_name, uid, every knob, git commit/dirty, library versions,
                   dataset_id / data_hash / z_hash (the dataset's fingerprint)
    wandb.json     id, name and url of the wandb run (written once wandb.init returns)
    log.txt        live training output (`tail -f` it)
    metrics.json   recovery scores + timings
    arrays.npz     tau_hat, ATE and other truth summaries, τ(u) curves, losses, diagnostics.
                   NOT the dataset: Y and ITE are rebuilt by dataset_store (from 2026-10-01;
                   older runs were stripped after a per-run exact check)
    model.eqx      the fitted weights (from 2026-10-01; load_model)
    plots/         every figure as PNG
```

The wandb name is `<model>_<preset>_<arm>[-trf]_[<variant>_]k<K>_s<seed>_d<digit>_<uid>`:
`ff` because this script fits margin **and** copula (`margin`, `margin_sep`
are reserved for copula-free fits from other scripts); `e1`…`e6` the preset; `flexcont` or
`loctrans` the arm, `-trf` for a transformer conditioner; a variant tag only when a setting
differs from the reference (`lr<value>`, `copw<W>`, `mw<W>`, `mkn<K>`, `batch<B>`, `ema<N>`, `coplr<m>`,
`ep<N>`, `pat<N>`, `cap<s>`, `n<N>`, `ecdf`, `umw<w>`, `mfix`, `copsel`, `coplam<λ>`, `effect<size>`, `sa<k>`, `rct`, …;
`check_runs.py` checks every tag against `config.json`); `s<seed>`
the fit seed; `d0` for the single digit class, `d0-9` for all ten; `uid` six hex characters.
The stamp is UTC (`2026-09-10T14-29-26Z` = 10 Sep 2026, 14:29:26). Built by `run_id_for` /
`wandb_name_for`; what a run *is* should always be read from `config.json`, not from its name.

`config.json` and `log.txt` appear at launch, so a folder holding only those two
is still training (or died). The whole `runs/` directory is gitignored; scripts and
write-ups belong in `scripts/` and `docs/`, which are tracked. To share
a result, copy the selected `summary.md`/`summary.csv` to an explicitly tracked
results location rather than accidentally committing model archives.

---

## Reading the results

| key | meaning |
|---|---|
| `ate_mae` | headline: mean \|tau_hat − ATE\| over all K pixels |
| `ate_mae_on_support` | …restricted to pixels with a nonzero true effect |
| `ate_mae_off_support` | …restricted to pixels whose true effect is **exactly zero** |
| `ate_corr` | spatial correlation — is the map in the right *place*? |
| `att_mae` / `atc_mae` | same score against ATT and ATC |
| `tau_u_rmse_vs_marginal` | existing 40-bin `tau_curve` RMSE against `TAU_MARGINAL` |
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

1. **Single fits are noisy.** The effect is a tiny part of the likelihood, so where training
   stops moves the estimate. The method is the average of 5 fit seeds; compare methods on
   that, over several datasets (`--seed-assign`).
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

5. **Leftover confounding grows with resolution at fixed data** (see "Current state" and
   `docs/leftover_confounding/STATUS.md`): fine at 8×8 on all digits, ~9 % at 16×16.
6. **Run many fits through a slot pool.** ~60 unpinned fits on 240 cores halved every fit's
   speed; the grid launcher pins each fit to 5 cores with `taskset`.

---

## TO DO (2026-10-01)

1. Finish the 8×8 all-digits paper grid and read `runs/exp_ate_recovery/analysis/grid_8x8_alldigits.md`.
2. 16×16: choose a fix for the leftover confounding (candidates: an energy-score term in the flow's
   training; an AIPW-style correction of the estimate) or report it as a limitation; then the 16×16 grid.
3. E4: investigate the treatment-blind copula's misspecification when the effect depends on covariates.
4. 32×32: make the read-out cheaper before running it.
5. Remove the 10 weightless all-digit 8×8 flow fits from 2026-09-28 once the grid's refits are confirmed identical.

The older to-do list (axis sweeps on digit 0) is superseded; see git history for it.
