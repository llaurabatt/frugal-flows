# Identification diagnostics

Scripts that reproduce the **identification** findings of
`validation/Continous_Frugal_Flows.ipynb` — *does the frugal flow recover the
causal-margin parameters it was trained to identify?* — as runnable, sense-checkable
scripts instead of notebook cells.

**This folder is purely additive.** Every script only *imports and calls* existing
code in `frugal_flows/` and `validation/`. Nothing in the core package or the
existing `validation/*.py` modules is modified — `git diff` outside this folder is
empty.

## Environment

Everything runs in the one working env (see project memory `project-working-env`):

```bash
micromamba run -n frugal-flows-flowjax python -m diagnostics.<script> [flags]
```

Run from the `validation/` directory (so the `diagnostics` package and the existing
`data_processing_and_simulations` package are both importable). Ground-truth data
comes from the R `causl` package via rpy2 — already installed in that env.

## The three diagnostics

| Script | Question | Output (`outputs/`) |
|---|---|---|
| `d1_ate_recovery.py` | **Does it identify `(ate, const, scale)`?** Fit over N seeds, extract the learned causal margin, compare to truth. | `d1_<gen>_recovery.csv`, bias table (printed), `d1_<gen>_recovery.pdf` (boxplot vs truth) |
| `d2_margin_shape.py` | **What does the causal margin look like?** Interventional outcome quantile functions / densities for `do(X=0)` vs `do(X=1)`; the gap is the ATE. | `d2_<gen>_margin.pdf`, sense-checks (printed) |
| `d3_moment_match.py` | **Are the treatment/covariate moments what causl intended?** Per-variable mean/var vs a causl ground-truth reference. | `d3_<gen>_moments.csv`, table (printed), `d3_<gen>_moments.pdf` (histograms) |

Run them individually, or all three with one config:

```bash
# fast feedback (small n, few seeds, short training) — sanity check the pipeline
micromamba run -n frugal-flows-flowjax python -m diagnostics.run_diagnostics --smoke

# notebook-scale identification run on the mixed (Gaussian+Gamma) generator
micromamba run -n frugal-flows-flowjax python -m diagnostics.run_diagnostics \
    --generator mixed --const 1 --ate 1 --n-samples 20000 --n-iter 25
```

Key flags (shared): `--generator {gaussian,mixed,discrete,many_discrete}`,
`--const` / `--ate` (the generator's `causal_params = [const, ate]`),
`--n-samples`, `--seed`, `--smoke`. `d1`/`run` also take `--n-iter`; `d3`/`run`
take `--ref-n` (causl reference size).

## How the truth is defined

The causl generators encode the ground truth as **`causal_params = [const, ate]`**:
the outcome formula is `Y ~ X` with linear predictor `const + ate * X` and a Gaussian
family at `phi = 1`, so the causal-margin truth is `{ate, const, scale=1}`.
`_harness.true_params_from_causal_params` derives this.

> The notebook sets `TRUE_PARAMS = {ate:1, const:0, scale:1}` while generating with
> `CAUSAL_PARAMS = [2, 5]` — an internal inconsistency. These scripts derive the truth
> from `causal_params` so the comparison is always correct.

## causl conventions used by d3

d3's ground truth comes from causl itself. Primary route: a large `causalSamp` draw
(Monte-Carlo population reference) — works for every variable including the treatment
`X`, which is binomial *conditional* on `Z` and has no closed-form marginal. For
intercept-only covariates the closed forms cross-check it (verified — see project
memory `reference-causl-conventions`):

| causl family | code | link | mean | variance |
|---|---|---|---|---|
| Gamma | 3 | log | `exp(beta)` | `phi * exp(beta)^2` |
| binomial | 0/5 | logit | `expit(beta)` | `p(1-p)` |
| gaussian | 1 | identity | `beta` | `phi` |

(e.g. the `mixed` generator's `Zc~1, beta=1, phi=1` Gammas have true mean `e=2.718`,
var `7.389` — which the causl reference reproduces.)

## Design notes

- **The shared step is `_harness.fit_once`**, a thin wrapper over the existing
  `causl_sim_data_generation.frugal_fitting`. It returns the recovered params (d1),
  the fitted margin object + flow (d2), and the input data (d3).
- We deliberately bypass the notebook's `run_simulations` (it hardcodes
  `causal_model='gaussian'`, prints intermediate state, and self-references
  `causl_py.`) and loop `frugal_fitting` directly — same finding, plus we keep the
  fitted-flow object d2/d3 need. This is a *choice not to call* that function, not a
  modification of it.
- **d3 mode 2 (deferred):** compare causl truth against the *model's reconstruction*
  of X/Z (does the flow distort the covariates?). That needs the `FrugalFlowModel`
  sampler and will be added as a second mode — still without touching core code.
- `outputs/` holds regenerated artefacts (CSVs + PDFs); safe to delete.
