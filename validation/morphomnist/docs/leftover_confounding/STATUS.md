# Leftover confounding ("halo") in the frugal flow — status as of 2026-09-30

> Scripts for this investigation: `validation/morphomnist/scripts/leftover_confounding/` (toys) and
> `validation/morphomnist/scripts/exp_ate_recovery/` (image runs). Results and logs stay in
> `validation/morphomnist/runs/leftover_confounding/` and `runs/exp_ate_recovery/`.

This file states what we know, with the evidence and its scope. The dated log of how we got here is
`README.md` in this folder. Labels:

- **MEASURED** — directly measured, with the replication stated.
- **TESTED, NOT A FIX (in the tested setting)** — we changed one thing and the leftover did not go
  down; the scope is stated and the claim does not extend beyond it.
- **ARGUED** — follows from a derivation we checked, not from a run.
- **INFERRED** — deduced from other results; not measured directly.
- **UNKNOWN** — open.

## The measure

For one dataset, the *confounding map* is the naive treated-minus-untreated mean image minus the true
effect map. The *leftover slope* is the slope of the estimate's error map regressed, pixel by pixel, on the
confounding map: 0 means no confounding left, 1 means the naive difference. A *5-fit average* averages the
effect maps of 5 fits of one dataset that differ only in the fit seed. Unless stated otherwise: E2 preset,
8×8, digit 0 (n = 5923), joint frugal flow (flexible-continuous arm), lr 1e-3, copula width 16, margin
width 48 / 8 knots, batch 100, early stopping with patience 30, fixed rank rule (`spread_all`).

## MEASURED

1. **The leftover exists on E2 at n = 5923.** 5-fit average over 5 datasets: slope 0.036 (se 0.003),
   disc error +0.022. Single fits: 0.044 (se 0.005, 20 datasets); other seed sets give 0.029–0.067, and
   fit seed 2001 is unusually poor on all 5 datasets. OLS on the same datasets: slope about 0.005, disc
   error about 0. Frengression: slope −0.001 (se 0.0035, 10 single fits).
2. **It shrinks with sample size on the images.** Single fits, seeds k and 1001: slope 0.109 (se 0.020,
   n = 1500, 10 fits), 0.084 (0.013, n = 3000, 10 fits), 0.058 (0.006, n = 5923, 25 fits). All ten digits
   (n = 60000, 5 fits, seed k): 0.009 (0.007). Caveat: all-digits also changes the problem (ten digit
   shapes, an 11-dimensional covariate), so on its own it does not isolate sample size; the digit-0
   subsample series does.
3. **A Gaussian K-pixel toy reproduces it** (`toy_multi.py`: Y_k = b_k Z + τ_k T + correlated noise,
   true ranks Φ(Z), 4 seeds). n = 5923: K = 8 slope 0.084 (se 0.013), K = 16 0.061 (0.008). n = 50000:
   0.002 (0.005) and 0.011 (0.012). OLS about 0 throughout.
4. **Frengression has no leftover on the same toy data** (`toy_frengression.py`, 4 seeds): n = 5923,
   K = 8: −0.008 (0.009); K = 16: −0.009 (0.002). So at this sample size the difference is between the
   methods, not a limit of the data.
5. **Per pixel, the error equals the missing part of the image's dependence on Z times the imbalance in Z**
   (toy K = 8, 4 seeds; correlation 0.83–0.99). Note: this holds almost by construction for any model
   whose arm means match the data, so it says *where* the error is (in the fitted dependence of the image
   on Z), not *why* the fit misses it. The fitted model misses 5–10 % of the true dependence.
6. **Reproducibility.** Rerunning a 2026-09-27 fit with the code of 2026-09-28 gave an identical estimate
   (max difference 0, same kept epoch). The 2026-09-30 rerun is in
   `../../scripts/exp_ate_recovery/code_check_20260930.log`.

## TESTED, NOT A FIX (in the tested setting)

| Change | Setting and replication | Result |
|---|---|---|
| Copula width 8 | E2, 5 datasets × 5 seeds | 0.037 (se 0.005) vs 0.044 at width 16: slightly lower, not removed |
| Copula width 50 vs 16 | E2, 20 datasets, **before the margin rank fix** | 0.060 vs 0.046 |
| Copula learning rate ×3, ×10 | E2, 5-fit averages, 5 datasets | 0.046, 0.074 vs 0.036: worse (×10 stops earlier, margin undertrained) |
| Smaller margin (width 16) | E2, 20 datasets | 0.062–0.064 vs 0.044: worse |
| Training 600 epochs, no early stopping | images E2, 5 fits (seed k) | slope falls to ~0.05 by epoch 100–200, then flat to 600 |
| Training 400 epochs, batch 100, no early stopping | toy K = 8 / 16, 4 seeds | end 0.060–0.085 / 0.019–0.068: about flat (K = 16 drifts down somewhat) |
| Full batch, 5000 steps, no early stopping (frengression's regime) | toy K = 8 / 16, 4 seeds | end 0.106–0.158 / 0.070–0.152: worse |
| Batch 500 (with early stopping) | images E2 | 0.038 vs 0.029 at batch 100: worse |
| Copula u-marginal penalty, weight 10 / 100 / 1000 | images E2, 5 datasets, **one fit seed (2001)** | 0.082 / 0.085 / 0.381 vs 0.067: worse; firm only at 1000 (10 and 100 are ~2 se) |
| Exact empirical-CDF covariate ranks | images E2, 5 datasets, seed 2001 | 0.074 vs 0.067: no change (stage-one ranks were already uniform: KS 0.011 on all rows) |
| Reversed copula (image ranks given u), rank-penalty weight 0 / 10 / 100 | **toy only**, K = 8 / 16, 4 seeds | K = 8: 0.118 / 0.159 / 0.068; K = 16: 0.163 / 0.223 / 0.167, vs 0.084 / 0.061: worse or no better. Copula width 16 was not re-tuned for a K-dimensional copula, so this shows "this implementation does not fix it", not "the direction cannot matter" |

In the one-pixel toy (`toy_gaussian.py`, `toy_trajectory.py`) the leftover **did** shrink with longer
training (from ~7 % to ~1 % by epoch 400 at n = 5000). So "training longer does not help" holds for the
images and the K-pixel toy, not in general.

## ARGUED

- The copula must be blind to T for the frugal likelihood to identify the margin in the current arm
  (unmasking T lets it absorb p(u | t) and sends the margin to the naive answer).
- A T-blind copula is correctly specified when each treated image is a per-pixel function of the
  untreated image that does not depend on the covariates: true for E1, E2, E3, E5. **It is misspecified by
  construction when the effect depends on the covariates (E4, E6)**, in any arm. Not yet tested.
- For the reversed copula: the effect read out by simulating the intervention and the rank counterfactuals
  do not depend on how the work is split between margin and copula; the pooled-rank penalty is
  equivalent to "p\* is the causal margin" and does not compete with the likelihood.

## INFERRED (weaker)

- **The copula's spread is not the main cause.** Toy: the current copula's spread of Z given the image is
  17–21 % too wide at n = 5923 (6 % at n = 50000), and its mean relationship is shrunk by 2.5–3.5 %. A
  Gaussian calculation with the exact toy values says this loses about 1 % of the dependence, against a
  6–8 % leftover. The fitted model is not Gaussian, so this is an approximation.
- **So the missing dependence probably sits in the margin**, which carries part of the thickness-driven
  difference between arms. This is by elimination; it has not been measured directly.

## UNKNOWN

- **Why** the frugal flow's fit misses 5–10 % of the dependence at n ≈ 6000 while frengression, on the
  same data, does not. The one difference we have not tested is the objective: likelihood (flow) vs
  energy score (frengression). Testing it would mean training the flow with an energy-score term.
- Whether the image leftover has the same mechanism as the toy's (the toy reproduces its size, its
  dependence on n, and the gap to frengression; nothing more).
- E3–E6 have not been run with the fixed rank rule. The September E4 ring (+0.15, old code) is
  unexplained; the E4/E6 copula misspecification above may be why.

## 16×16 (2026-09-30, IN PROGRESS)

- MEASURED (updated 18:10): the ~20 % leftover at 16×16 on digit 0 is typical, and larger sizes do not fix it:
```
16x16, digit 0, E2, single fits (confirm_16x16 + size_variants_16x16, partial at 18:10 UTC):
  plain (copula 16, margin 48): slope 0.200 (1 fit; reproduces the timing fit exactly)
  + weight averaging (ema20): 0.163-0.215 (3 fits) | margin 128: 0.205-0.228 (3) | copula 64: 0.259 (1)
  copula 64 + margin 128: 0.269-0.361 (3). Naive MAE ~0.22-0.23; FF MAE 0.041-0.082.
MEASURED: ~20 % leftover at 16x16 on digit 0 is typical (7 fits, 3 datasets, 2 seeds), vs 4-6 % at 8x8 on the same images.
TESTED, NOT A FIX (16x16, digit 0, E2, seed k, 3 datasets): larger margin (no change), larger copula (worse).
Weight averaging: similar (within spread).
Not tested, deprioritised (user 09-30): a SMALLER margin at 16x16 (it was worse at 8x8: width 16 slope 0.062 vs 0.044).
Open: all ten digits at 16x16 (running, screens all16 / frall16).
```
- Settings were tuned at 8×8 only. Running: confirmation (is 0.20 typical?), copula/margin size variants at
  16×16, and frengression at 16×16 as the reference.

### Update 2026-09-30 23:30 UTC
```
Digit 0, 16x16, all finished (48 fits). E2 (3 datasets x 2 seeds): FF plain slope 0.170 (0.08-0.22), MAE 0.047;
FF ema20 0.166, MAE 0.045; frengression -0.052 (-0.07 to -0.02), MAE 0.021; FF copula 64 / 128: 0.43 / 0.64 (MAE 0.094 / 0.139).
E1: FF MAE 0.020 vs frengression 0.023. => at 16x16 on digit 0 FF FAILS the criterion on E2 (MAE ~2x frengression).
Weight averaging, pre-set rule: lower MAE in 8/12 pairs, mean change E1 -0.0016, E2 -0.0024, slope unchanged -> passes, small gain.
All digits 16x16 (partial, 23:22 UTC): E1 sa3 MAE 0.011 (digit 0: 0.020); E2 pending.
32x32 timing (digit 0, training capped 1 h, 121 epochs at 30 s): read-out 2.7 h, copula diagnostics 0.45 h, total 4.2 h.
  -> read-out is the 32x32 bottleneck; make it cheaper before 32x32 runs.
```

### Update 2026-10-01 01:40 UTC
```
ALL TEN DIGITS, 16x16, E2 (single fits, seed k; 2 of 3 datasets finished at 01:40 UTC):
  flow: slope 0.089 / 0.089, MAE 0.025 / 0.024, disc +0.049 / +0.047
  frengression: slope -0.043 / -0.020, MAE 0.010 / 0.008 (naive MAE ~0.185)
  E1 flow MAE 0.007-0.011 (3 datasets).
MEASURED: with 10x the images the 16x16 leftover halves (digit 0 ~0.17 -> 0.09) but FF's E2 error is still 2.5-3x frengression's.
=> FAILS the criterion at 16x16 on the paper data. More data is exhausted (60000 = all MNIST); averaging / EMA gave
   ~10-20 % at 8x8. A method change is needed. Lead: the flexible margin (Sept: location-translation had no halo with the
   same copula, old code); proposed next = re-test loctrans vs flexcont on the K-pixel toy with today's code.
```

## Decision (2026-09-30, agreed with the user)

Stop searching for a fix. Report the leftover with its evidence: the sample-size ablation (1500 → 60000),
the 5-fit average (3.6 % at n = 5923), and frengression as the more data-efficient model for population
effects at this sample size. Move on to the paper runs, after checking the code and the E4/E6 question.
