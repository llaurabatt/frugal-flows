# `scripts/` — launch and analysis scripts

Every batch of fits and every analysis behind the MorphoMNIST results, kept in git. Read the main
[`../README.md`](../README.md) first (section "Current state"); the evidence behind the design
choices is in [`../docs/leftover_confounding/STATUS.md`](../docs/leftover_confounding/STATUS.md).

## How these scripts work

* **Run them from anywhere**: each script `cd`s to `validation/morphomnist/` itself
  (`bash scripts/exp_ate_recovery/<script>.sh`). Environment: `frugal-flows-frengression`
  (has JAX, flowjax, torch and frengression; see the main README's Quick start).
* **Long batches run in a `screen`** and keep going unattended, e.g.
  `screen -S grid8 -dm bash -c "bash scripts/exp_ate_recovery/grid_8x8_alldigits_v2.sh > runs/exp_ate_recovery/_scripts/grid.log 2>&1"`.
* **Outputs never go into `scripts/`.** Run folders go to `runs/exp_ate_recovery/`,
  `runs/frengression/`, `runs/baselines/`; logs to `runs/<root>/_scripts/<name>_logs/`; analysis
  tables to `runs/exp_ate_recovery/analysis/`. `runs/` is gitignored.
* **Resumable.** Launchers skip a cell that is already done (has `metrics.json`, and from
  2026-10-01 also its saved weights), so a launcher can be rerun after an interruption.
* **Big batches use a slot pool** (`grid_8x8_alldigits*.sh`): 48 slots of 5 cores, each fit
  pinned with `taskset`. Without it ~60 fits fought over 240 cores and each ran at half speed.
  Note: a fit is bit-reproducible only at the same number of cores (see the main README).
* **Every fit logs to wandb** (`--wandb --wandb-group <group>`, project `proj-lb/Frugal Images`);
  the group names below find them.
* **Historical scripts record what was run, with the code of their date.** The defaults changed on
  2026-09-30 (learning rate, copula width, max epochs) and the layer fix landed on 2026-09-25/26,
  so rerunning an old script with today's code does not reproduce its old runs. To reproduce an old
  run, check out the git commit recorded in that run's `config.json` (`git.commit`). Runs made before
  2026-10-01 have `git.dirty: true` almost always, because the flag then also counted untracked files
  (e.g. a local `CLAUDE.md`); from 2026-10-01 it counts only changes to tracked files.

## Current: the paper grid (8×8, all ten digits)

| script | what it does | output / wandb group |
|---|---|---|
| `exp_ate_recovery/grid_8x8_alldigits_v2.sh` | **the paper grid**: E1–E6 × datasets (assignment seeds) 1–10; flow (flexible arm, paper defaults, weights saved) with fit seeds {k, 1001–1004} → the method is the 5-fit average; frengression one fit per dataset (seed k). Slot pool 48 × 5 cores. | `runs/exp_ate_recovery/`, `runs/frengression/`; `grid_8x8_alldigits`, `grid_8x8_alldigits_frengression` |
| `exp_ate_recovery/grid_8x8_alldigits.sh` | first version of the same grid (frengression with 5 seeds); replaced by `_v2` mid-run on 2026-10-01 | same |
| `baselines/baselines_grid_8x8_alldigits.sh` | naive / IPW / OLS / AIPW / oracle IPW on the grid's 60 datasets (seconds each) | `runs/baselines/` |
| `exp_ate_recovery/analyse_grid_8x8_alldigits.py` | per preset: flow 5-fit average, flow single fits, frengression, OLS — error over all pixels, disc / ring / background errors, leftover slope; **pass/fail vs frengression** (paired over datasets, diff < 2 se) | `runs/exp_ate_recovery/analysis/grid_8x8_alldigits.{md,csv}` |
| `exp_ate_recovery/grid_8x8_alldigits_watch.sh` | reruns the analysis every 2 h while the grid runs | `.../_scripts/grid_8x8_alldigits_watch.log` |
| `exp_ate_recovery/make_e2_weights_branch.sh` | builds the local results branch `multi-y-e2-weights` (E2 fits' weights, config, metrics) once the grid's E2 cells are done; does not push | git branch |
| `exp_ate_recovery/cleanup_paused_launcher.sh` | one-off: ended the paused first grid launcher once its fits finished. **Bug:** its last line, `screen -S grid8 -X quit`, matched the running screen `grid8b` by prefix and killed 48 fits; use `end_paused_launcher.sh` instead | — |
| `exp_ate_recovery/end_paused_launcher.sh <pid>` | ends a paused (SIGSTOPped) launcher once its fits finish, by pid only. To change the core count mid-grid: `kill -STOP` the launcher, kill its slot-waiting helper (the child running `sleep 10`), start `SLOTS=<n> bash grid_8x8_alldigits_v2.sh` in a new screen, and run this on the old pid | — |

## Paper outputs (2026-10-02)

| script | what it does | output |
|---|---|---|
| `paper/make_paper_outputs.py <topic>` | figures and LaTeX tables for the paper, one topic at a time; each topic also compiles a one-column preview at AISTATS text width. Topics so far: `setup` (table of the six presets with the size of their confounding at 8×8 and 16×16 over the 10 datasets; one truth figure per resolution) | `runs/paper/<topic>/` |

## Margin pixel order (test, 2026-10-02)

| script | what it does | wandb group |
|---|---|---|
| `exp_ate_recovery/margin_order_8x8.sh` | waits for the paper grid to end, then 30 fits with `--margin-order fixed` (tag `mfix`; one pixel order in every margin layer, no permutation): E1/E2 × datasets 1–3 × fit seeds {k, 1001–1004}, otherwise the grid's settings, so each pairs with a grid cell. Not the default | `margin_order_8x8` |

## Resolution checks (16×16, 32×32)

| script | what it does | wandb group |
|---|---|---|
| `exp_ate_recovery/timing_sizes.sh` | one E2 fit at 16×16 and at 32×32 (digit 0), training capped at 1 h: timing (32×32 read-out ~2.7 h) | `timing_sizes` |
| `exp_ate_recovery/confirm_16x16.sh` | 16×16 digit 0, E1/E2 × 3 datasets × 2 seeds, plain vs weight averaging | `confirm_16x16` |
| `exp_ate_recovery/size_variants_16x16.sh` | 16×16 digit 0, E2: larger copula / margin (worse or no help) | `size_variants_16x16` |
| `exp_ate_recovery/alldigits_16x16.sh` | 16×16 **all digits**, E1/E2 × datasets 1–3, one fit each (flow keeps ~9 % of E2 confounding; frengression ~0) | `alldigits_16x16` |
| `frengression/frengression_16x16.sh`, `frengression/frengression_alldigits_16x16.sh` | frengression on the same 16×16 datasets | `confirm_16x16_frengression`, `alldigits_16x16_frengression` |

## The leftover-confounding investigation (2026-09-27 → 30)

Write-up: [`../docs/leftover_confounding/STATUS.md`](../docs/leftover_confounding/STATUS.md).

Image runs (`exp_ate_recovery/`), each paired with earlier plain fits on the same data:

| script | tested | analysis |
|---|---|---|
| `confounding_fixes_8x8.sh` | copula width 8; copula learning rate ×3, ×10 | `analyse_confounding_fixes.py` |
| `training_length_8x8.sh` | 600 epochs without early stopping, effect tracked every 20 epochs | `analyse_training_length.py` |
| `umarg_check_8x8.sh` | the copula's own covariate marginal (diagnostic) | — |
| `umpen_8x8.sh` | penalty making that marginal uniform (weights 10/100/1000) | `analyse_leftover_fixes.py` |
| `ecdf_8x8.sh` | exact empirical-CDF covariate ranks | `analyse_leftover_fixes.py` |
| `subsample_8x8.sh` | digit 0 capped at n = 3000 / 1500 (leftover grows as n shrinks) | — |
| `all_digits_8x8.sh` | E1/E2 on all ten digits, one fit per dataset (leftover ~1 %) | — |

Toys (`leftover_confounding/`, small Gaussian problems with a known effect; each `run_*.sh`
launches its `toy_*.py`; results in `runs/leftover_confounding/results_*`):

| toy | question |
|---|---|
| `toy_gaussian.py` / `run_toy.sh` | 1 pixel: is the leftover there at n = 5000 and n = 50000? |
| `toy_trajectory.py` / `run_trajectory.sh` | 1 pixel: how does the leftover evolve without early stopping? |
| `toy_multi.py` / `run_multi.sh` | K = 8 / 16 pixels: reproduces the image-size leftover; vanishes at n = 50000 |
| `toy_mechanism.py` | is the error the missing image–covariate dependence times the covariate imbalance? (yes) |
| `toy_frengression.py` / `run_frengression_toy.sh` | frengression on the same toy data (no leftover) |
| `toy_spread.py` / `run_spread.sh` | is the copula under-confident? (yes, but too little to explain the leftover) |
| `toy_reversed.py` / `run_reversed_toy.sh` | the reversed-copula arm (`frugal_flows/reversed_copula.py`): worse |
| `toy_regime.py` / `run_regime.sh` | frengression's training regime (full batch, no early stopping) for the flow: worse |

## Historical: tuning on digit 0 (2026-09-19 → 27)

| script | what it did |
|---|---|
| `exp_ate_recovery/assignment_grid_8x8.sh` + `analyse_assignment_grid_8x8.py` | 8×8 digit-0 grid, E1–E6 × assignment seeds 1–20 (before the layer fix) |
| `frengression/frengression_grid_8x8.sh` | frengression on that grid's datasets |
| `exp_ate_recovery/ranks_fix_8x8.sh` | the copula hidden-rank fix on E1/E2, paired with the grid |
| `exp_ate_recovery/copsel_8x8.sh` | early stopping on the held-out copula loss |
| `exp_ate_recovery/lr1e-3_8x8.sh`, `copw_lr1e-3_8x8.sh` | learning rate 1e-3; copula width 50 vs 16 |
| `exp_ate_recovery/margin_size_8x8.sh` | margin width {48, 16} × knots {8, 4} |
| `exp_ate_recovery/margin_only_lr1e-3_8x8.sh`, `margin_only.py` | margin-only fits (no copula) as a diagnostic |
| `exp_ate_recovery/fitseed_control_8x8.sh` + `analyse_fitseed_control.py` | fit-seed control: single-fit error is mostly training noise |
| `exp_ate_recovery/noise_fixes_8x8.sh` + `analyse_noise_fixes.py` | weight averaging (ema20) and batch 500 |
| `exp_ate_recovery/compare_copula_versions.py` | compares the copula variants of the same cells |
| `exp_ate_recovery/queue.sh`, `overnight.py` | an early overnight batch runner (MLP and transformer setups) |
| `exp_ate_recovery/code_check_20260930.sh` | the pre-grid code check: self-test, index, checker with wandb, a reproduction |
