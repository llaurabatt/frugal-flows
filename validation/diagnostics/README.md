# Frugal-flow diagnostics

A small, script-based suite (no notebooks) for the two questions that matter when
sanity-checking a frugal-flow causal-margin fit:

1. **Treatment-effect estimation** — does the flow recover the true ATE, and how
   does that recovery behave from small `n` to large `n`?
2. **Margin shape** — not just the scalar ATE, but *what the learned
   interventional outcome distribution looks like*, in the small-data and
   large-data regimes.

Everything here is a **consumer of the `frugal_flows` package** — it never
reimplements model logic. The interventional read-out itself
(`do(0)` / `do(1)` common-random-number sampling → invert the outcome transform →
difference) lives in the package as `frugal_flows.interventions` /
`FrugalFlowModel.estimate_ate`; these scripts just wrap it around causl-generated
ground-truth DGPs so the estimate can be scored against a known truth.

Run everything in the one working env:

```bash
cd validation
micromamba run -n frugal-flows-flowjax python -m diagnostics.<script> --help
```

## Treatment-effect estimation

| Script | Question | Output |
|---|---|---|
| `d1_ate_recovery.py` | **Does the flow identify the causal-margin parameters?** Generate data with a known `(ate, const, scale)`, fit over several seeds, recover the params, compare to truth. | `outputs/d1_<gen>_recovery.csv` + boxplot PDF + printed bias table |
| `ate_sweep.py` | **How does ATE recovery scale with sample size?** Grid of (family × arm × `n` × seed); one CSV row per fit; model-agnostic paired-CRN ATE. Shardable over disjoint `--seeds`. | `outputs/sweep_*.csv` (one shard per run) |
| `ate_sweep_plot.py` | Aggregate the `sweep_*.csv` shards. | `outputs/ate_sweep_recovery.png` (ATE ± SD vs `n`), `ate_sweep_relerr.png` |

`ate_extraction_suite.py` is the single-shot scorecard version: fit every
(family × arm) once, score ATE / interventional-mean recovery, render a margin
grid + ATE scorecard. It also re-exports `intervene` / `tau_curve` from
`frugal_flows.interventions` under their historical names.

### Small-`n` bias study (adjudicated findings)

`plot_te_bias_findings.py` is **pure plotting** — it reads the committed study
CSVs in `outputs/flexible_te/te_bias_study/` (no training) and regenerates two
figures that record what the multi-seed Gamma runs settled:

| Figure | Shows |
|---|---|
| `fig5_small_n_ate_bias.png` | The log-cold ATE is unbiased at `n≥2000` but biased **upward**, growing as `n` shrinks (Panel A). Standardizing the fitting scale (log / +center / +scale / +standardize) is a **no-op** — all four coincide within SEM (Panel B). |
| `fig6_confounding_location.png` | That small-`n` bias is **confounding**: it collapses on the unconfounded DGP (`gamma_b0`, Panel A), and it is a **location** effect — under confounding the fitted control mean `E[log Y \| do(0)]` is pulled below the truth while the treated mean is not (Panel B). It is **not** fixable by any outcome transform. |
| `fig7_mean_ate_forest.png` | Forest plot of the **mean ATE estimate** (dot = mean, thick = ±SEM, thin = ±SD, faint = per-seed) for *every* study condition against the true-ATE line — the whole study at a glance: n-sweep means descend onto truth by `n≥2000`, the four standardize arms cluster indistinguishably, and the confounding pair straddles truth at `n=500` then converges. |

Source CSVs (committed, one row per fit): `nsweep_logcold.csv` (log-cold ATE vs
`n`, 5 seeds), `std_decompose.csv` (log / center / scale / standardize × {500, 2000}
× 15 seeds), `confound_test.csv` (`gamma_b0` vs `gamma_b1` × {500, 2000} × 15 seeds,
with fitted interventional log-means). Regenerate the figures with:

```bash
cd validation && micromamba run -n frugal-flows-flowjax python -m diagnostics.plot_te_bias_findings
```

## Margin shape (small vs big data)

| Script | Question | Output |
|---|---|---|
| `d2_margin_shape.py` | **What does the learned margin look like?** Plot the interventional quantile functions `Q_0(u)` vs `Q_1(u)`; the treatment effect is the gap between the curves. | `outputs/d2_<gen>_margin.csv` + margin PDF |
| `plot_margin_densities.py` | **Fitted vs true interventional densities** on the Gamma DGP. Panel A (`n=20,000`): spline vs additive margin against the analytic truth. Panel B (`n=2,000`): three restarts — density *shape* is stable even when the restart-level ATE wobbles. | interventional-density figure |

## Shared library modules (imported, not run)

- `outcome_families.py` — causl ground-truth generators keyed by outcome family
  (Gaussian, Gamma, …), each exposing `true_ate` / `mean_do` / `sample_truth`.
- `quick_sense_check.py` — `fit_model`, `base_hyperparams`, `model_args`: fit one
  frugal flow for a given (family, arm).
- `_harness.py` — path handling (makes `data_processing_and_simulations`
  importable) + the fit-once plumbing used by `d1` / `d2`.
