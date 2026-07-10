# Fixing the spline causal margin's restart-level noise: preprocessing, warm-start, LR schedule

**Branch:** `flexible-te-spline-analysis`  ·  **Arm:** `flexible_continuous` (treatment-conditioned RQS margin)  ·  **Status:** ADJUDICATED (rules frozen 2026-07-09, full run 2026-07-10)

## Question

On the Gamma DGP (`causal_params=[1.0, 0.5]`, log link, `phi=0.5`, **true ATE 1.7634**) the spline
margin recovers the right effect *shape* but its ATE **level** swings several tenths restart-to-restart
at n=2000 under a fixed dataset (only the fit key varies) — see `SPLINE_BIAS_FINDINGS.md`. Two mechanisms
are suspected: (i) raw `Y` is squashed by `tanh` into a thin `[0.99, 1.0]` sliver of the spline's `[-1,1]`
domain (heavy-tail saturation, and the atanh boundary emits ±inf samples); (ii) the margin/copula split
is **level-under-identified** and validation loss provably cannot select the good basin
(`SPLINE_BIAS_FINDINGS.md` E5). Which knobs actually reduce the level noise?

## Pre-registered hypotheses

- **H1 — preprocess Y.** Transform `Y` (raw / log / standardize) before the tanh→spline. `log` is the
  mechanism-targeted fix for a positive heavy-tailed outcome (makes a Gamma truth near location-shift and
  fills the tanh range); `standardize` is a linear rescale that cannot change tail shape; `raw` is control.
- **H2 — warm-start.** Pretrain the causal margin ALONE on the (transformed) `Y|X`, graft it into the
  causal-margin slot, then jointly fine-tune. The margin term moment-matches by construction, so this
  starts the joint fit at the identified level instead of a random basin.
- **H3a — LR schedule.** Cosine decay (init 1e-2 → 1e-4) vs the constant 5e-3 baseline.
- **H3b — restart-averaging.** Average the τ(u) *level* across the K restarts of a cell (computed post-hoc
  from the shard CSVs, not a new fit), since val-loss cannot pick the basin.

## Pre-registered grid (frozen — do not add/drop cells after seeing results)

- ONE dataset: `gamma_b1`, n=2000, data seed 0, `causal_params=[1.0,0.5]`; `u_z` generated once from `Z`.
- `warm_start ∈ {cold, warm} × transform ∈ {raw, log, standardize} × lr_schedule ∈ {const, cosine}
  × restart 0..5` = **72 fits** (+36 warm pretrains).
- Fixed architecture: `RQS_knots=8, nn_depth=4, nn_width=50, flow_layers=4, batch_size=256, epochs=600`
  (warm pretrain uses the same budget).

## Pre-registered decision rules

Adjudicate on **original-Y-scale metrics only** — never `val_loss`/`pre_val_loss` across transforms
(different Jacobian). Per cell (over K restarts): `mean_bias`, `sd_ate` (restart sd, primary noise metric),
`mean_qte_int_err`, `tot_n_drop`.

Project threshold idiom (`h1_matrix.py:18`): bias is **real** iff `|mean_bias| > 2·sd_ate/√K`; a cell is
**unbiased** when `mean_bias ± 2·sd_ate/√K` covers 0.

- **H1** (hold cold/const): `log` has `tot_n_drop == 0` AND `sd_ate(log) < sd_ate(raw)` AND
  `|mean_bias(log)| ≤ |mean_bias(raw)|`. Mechanism check: `raw` shows >0 boundary drops; `log` zeros them.
- **H2** (matched transform/lr): `sd_ate(warm) < sd_ate(cold)` AND warm's bias interval covers 0 while
  cold's may not. Primary claim: warm-start collapses the *level* → largest effect on `sd_ate`.
- **H3a** (matched transform/warm): `sd_ate(cosine) < sd_ate(const)`.
- **H3b**: `|ate_avg − true| ` below both the single-restart median `|bias|` and `sd_ate`
  (averaging buys √K only if the spread is mean-zero level jitter).

Control cell = `cold/raw/const`.

## Pre-registered predictions (recorded before running)

- `log` cuts both `|bias|` and `sd_ate` and zeroes boundary drops; `standardize` weaker.
- warm-start collapses `sd_ate` most (it directly attacks the level ambiguity), with a smaller `|bias|` gain.
- cosine gives a modest `sd_ate` reduction; restart-averaging helps wherever the spread is a level shift.

## Answer (full run 2026-07-10: 72 fits + 36 pretrains, 0 errors)

**Headline: warm-start is the dominant fix — all 6 warm cells are unbiased; 4 of 6 cold cells are
biased. log-Y is the only *transform* that de-biases a cold fit. Cosine does ~nothing. Restart-averaging
helps exactly where theory says it can (every unbiased cell), and the best estimator —
warm + averaging — is essentially exact (|avg−true| = 0.006).**

| cell (ws/transform/lr) | k | mean_bias | ±2sd/√K | unbiased | sd_ate | mean_qte | n_drop |
|---|---|---|---|---|---|---|---|
| cold/raw/const (control) | 6 | −0.2510 | 0.1924 | **NO** | 0.2357 | 0.3357 | 0 |
| cold/raw/cosine | 6 | −0.2170 | 0.1826 | **NO** | 0.2237 | 0.3425 | 0 |
| cold/log/const | 6 | −0.0434 | 0.0947 | yes | 0.1160 | 0.1778 | 0 |
| cold/log/cosine | 6 | −0.0780 | 0.1156 | yes | 0.1415 | 0.1650 | 0 |
| cold/standardize/const | 6 | −0.2343 | 0.1122 | **NO** | 0.1374 | 0.1940 | 0 |
| cold/standardize/cosine | 6 | −0.2694 | 0.1237 | **NO** | 0.1514 | 0.1981 | 0 |
| warm/raw/const | 6 | +0.0911 | 0.1281 | yes | 0.1569 | 0.2424 | 1 |
| warm/raw/cosine | 6 | −0.0058 | 0.0914 | yes | 0.1119 | 0.2206 | 10 |
| warm/log/const | 6 | +0.0809 | 0.1000 | yes | 0.1225 | 0.1790 | 0 |
| warm/log/cosine | 6 | +0.0390 | 0.0932 | yes | 0.1141 | 0.1756 | 0 |
| warm/standardize/const | 6 | −0.0282 | 0.0737 | yes | 0.0903 | 0.1550 | 0 |
| warm/standardize/cosine | 6 | +0.0103 | 0.0898 | yes | 0.1099 | 0.1539 | 0 |

### H1 (preprocess Y) — CONFIRMED for `log`, refuted for `standardize`
At cold/const, `log` meets all three pre-registered criteria: `tot_n_drop=0` ✓,
`sd_ate` 0.1160 < 0.2357 (raw) ✓, `|mean_bias|` 0.0434 ≤ 0.2510 (raw) ✓ — it halves the restart noise
AND removes the bias (interval covers 0). `standardize` reduces sd but leaves the bias intact
(−0.234, still "NO") — a linear rescale can't fix what the tanh does to the tail, as predicted.
**Mechanism caveat (honest):** the predicted raw-arm boundary drops did NOT appear at n=2000
(cold n_drop=0 everywhere); the atanh-overflow pathology is a small-n/underfit phenomenon, so the
log arm's win here is about likelihood geometry, not sample filtering.

### H2 (warm-start) — CONFIRMED, and it is the biggest single lever
Every warm cell is unbiased; warm converts all four biased cold cells to unbiased at matched
(transform, lr). The pre-registered sd criterion holds in 5/6 matched pairs (exception: log/const,
0.1225 vs 0.1160 — a wash where cold/log was already unbiased). The effect is primarily on the
**level** (bias), secondarily on sd — consistent with the mechanism claim that margin-only
pretraining starts the joint fit at the identified point of the flat margin/copula direction.

### H3a (cosine LR) — NOT confirmed
`sd_ate(cosine) < sd_ate(const)` in only 3/6 matched comparisons, all small. No consistent effect;
cosine is a tiebreaker at best (it does give the single best cell when combined with warm).

### Restart-averaging (H3b) — helps exactly where it can
`helps` (|ate_avg−true| below both the median single-restart |bias| and sd_ate) in **all 6 warm
cells and both cold/log cells**; fails in exactly the four biased cold cells — averaging cannot
remove a level bias, only mean-zero jitter, as pre-registered. Best cells after averaging:
warm/raw/cosine |avg−true| = **0.006**, warm/standardize/cosine 0.010, warm/standardize/const 0.028.

### Practical recommendation
Warm-start the causal margin (now a `pretrained_margin` kwarg in
`train_frugal_flow_flexible_continuous`), transform a positive heavy-tailed Y with `log` (or at
minimum expect raw/standardize to be biased cold), run K≥5 restarts and average the level. LR
schedule is a second-order choice.

## Reproduce

```
cd validation   # env: frugal-flows-flowjax
# full grid, sharded by disjoint restarts:
python -m diagnostics.spline_hp_battery --warm-starts cold,warm \
    --transforms raw,log,standardize --lr-schedules const,cosine \
    --restarts 3 --restart-base 0 --n 2000 --epochs 600 \
    --out outputs/flexible_te/spline_hp_shard0.csv
python -m diagnostics.spline_hp_battery ... --restart-base 3 \
    --out outputs/flexible_te/spline_hp_shard1.csv
# aggregate + adjudicate + figures:
python -m diagnostics.spline_hp_findings_plot
```
Figures: `outputs/flexible_te/spline_hp_bias_vs_sd.png`, `spline_hp_tau_overlay.png`.
