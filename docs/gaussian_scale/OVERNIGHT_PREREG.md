# Gaussian-scale frugal flow: overnight paper-grid run (pre-registration)

Written 2026-10-02 ~00:05 BST, before the first grid fit. Branch `multi-y-e2-weights` (local, unpushed).
Status: **exploratory**. The design was fixed after seeing digit-0 results (S9/S10/S11), and the comparators
(U-raw, FR) were fitted on Laura's machine; pipeline equivalence was checked (dataset-1 hashes match, her
stored dataset-1 seed-1 fit re-reads to ate_mae 0.011837681635029185, bitwise her number).

## Setting

8x8 MorphoMNIST, all ten digits, n = 60,000, Z = thickness + digit one-hot, `seed_data 101`, dataset k =
`seed_assign k`. Paper fit setting for every flow: lr 1e-3, <= 1000 epochs, patience 30, batch 100,
margin 48/1/4 with 8 knots, n_mc 5000, seed_mc 0, 1 thread per fit with pinned XLA flags.

| Code | `--arm` | Y | Copula width | Purpose |
|---|---|---|---|---|
| U-raw | `flexible_continuous` | raw logit | 16 | Laura's stored fits (comparator, not refitted except the Round C machine check) |
| FR | frengression | floored per-pixel | - | Laura's stored fits (comparator) |
| U-std | `flexible_continuous` | standardised | 16 | Standardisation alone |
| G-raw | `flexible_continuous_gaussian` | raw logit | 50 | Scale change alone |
| **G-std** | `flexible_continuous_gaussian` | standardised | 50 | **The proposal** |
| G-LT | `location_translation_gaussian` | standardised | 50 | Shift arm, `shift_init naive` |

## Queue (each round completes before the next starts; within a round, dataset-major)

| Round | Cells | Fits |
|---|---|---|
| A | E2, fit seed k: G-std datasets 1-6; G-LT, U-std, G-raw datasets 1-3 | 15 |
| B | E2, G-std fit seeds 1001-1004 on datasets 1-3 (Laura's 5-fit protocol) | 12 |
| C | Anchors, E2: G-std shuffled Z (dataset 1); G-LT shuffled Z (datasets 1-3); U-raw refit dataset 1 seed 1; G-std copula width 16 (datasets 1-3) | 8 |
| D | E1 (randomised), fit seed k: G-std datasets 1-6; G-LT datasets 1-3 | 9 |
| E | If time: U-std and G-raw seed 1001 (datasets 1-3); G-LT datasets 4-6; G-std seeds 1001-1004 datasets 4-6 | up to 21 |

Shuffled Z = rows of the stage-1 covariate ranks permuted jointly (`--z-shuffle-seed 7`) before the joint fit.

## Decision rules (fixed before any grid fit)

- **Primary endpoint**: `ate_mae` on E2, paired by dataset, seed-k fit vs seed-k fit, datasets 1-6.
  - *Beats the current flow*: G-std minus U-raw negative on every dataset AND the mean difference below
    -2 SE; Wilcoxon signed-rank reported (n = 6: two-sided p = 0.031 at best).
  - *Beats frengression*: the same rule for G-std minus FR. Also reported Laura's way (5-fit mean vs FR,
    datasets 1-3).
- **Not a standardisation artefact**: G-std < U-std and G-raw < U-raw on datasets 1-3; the standardisation
  effect is U-std minus U-raw and G-std minus G-raw. Limitation, stated: the Gaussian arms change the margin
  construction and the copula architecture together; S6 isolates the margin part only, without covariates.
- **Genuinely adjusting**: G-std with shuffled Z must return roughly the naive answer (`rho_retained` >= 0.8).
  If G-LT with shuffled Z still recovers the truth (`rho_retained` <= 0.1), G-LT is reported as a benchmark
  shortcut on this data too and excluded from any adjustment claim, whatever its ATE MAE.
- **Secondary** (reported, no decision rule): `signed_disc`, `mae_disc`, `signed_ring`, `mae_far`,
  `slope_imb`, `gen_sd_mae_t0/t1` (the background ring), best epoch, `gz_implied_ks`.
- **Known risk**: the digit one-hot enters as scores with a hard break at Phi^{-1}(0.9). Untested for the
  Gaussian arm (S9 had one continuous covariate). If `gz_implied_ks` is large or fits diverge, that is the
  first suspect.
- Every rule is printed as **met / not met / not resolved at this n** by
  `scripts/exp_ate_recovery/analyse_gaussian_scale.py`. A rule needing datasets that did not finish is
  "not resolved".

Definitions: `rho_retained` = <tau_hat - ATE, naive - ATE> / ||naive - ATE||^2 over pixels (0 = truth,
1 = naive treated-minus-untreated difference); `slope_imb` = slope of `np.polyfit(imbalance, tau_hat - ATE, 1)`
with imbalance = naive - ATE (as `analyse_grid_8x8_alldigits.py`).

## Pre-launch checks

Recorded in `~/work/halo-runs/gsw/checks.md` (existing arms bitwise unchanged; new runner tests; smoke with
W&B; shift-arm test at lr 1e-3; timing). Any amendment after launch (e.g. from the Codex round on this
plan) is appended below with a timestamp and applies to rounds not yet started.

## Amendments

### Launch decisions (2026-10-02 ~00:20, before the first grid fit; from the pre-launch checks)

1. **G-LT uses `--shift-lr-mult 10`** (Adam rate x10 on the LocCond shift leaves only). Pre-launch test, digit 0,
   E2, paper setting, standardised, naive start: without it the disc bias was +0.109 / +0.207 (2 seeds, fails the
   +-0.05 gate; ate_mae 0.029 / 0.057); with it +0.011 / +0.017 (ate_mae 0.0056 / 0.0064). Every G-LT cell, the
   shuffled-Z ones included, carries it (run names `..._shinaive_shlr10_...`).
2. **Timing** (all digits, E2, 3 epochs, 1 thread, machine under load): G-std 32 s/epoch, U-std 40 s/epoch, plus
   ~2-4 min of stage-1 / read-out / diagnostics per fit. Laura's U-raw paper fits ran 195-420 epochs, so one fit is
   ~1.5-3.5 h and A-D (44 fits) projects to ~12 h on 8 slots, far past 07:30. Per the plan: **Round B cut to fit
   seeds 1001, 1002** (6 fits) and **Round E skipped**.
3. **Round order A, C, B, D** (plan: A, B, C, D) because Rounds A and C must run: a round's fits start only after
   every fit of the earlier rounds has started (slots are not left idle at a round's tail), so C's 8 anchors take
   the slots freed by A before any B fit.
4. **No new fit starts after 05:30**; fits running then finish. Cells not started are listed in
   `~/work/halo-runs/gsw/launcher.log` and can be continued by re-running the launcher (skip-done).
