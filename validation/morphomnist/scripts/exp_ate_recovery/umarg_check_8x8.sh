#!/usr/bin/env bash
# Is the copula's u-marginal non-uniform on E2 and uniform on E1? (2026-09-27)
# Hypothesis (docs/leftover_confounding/README.md): the copula flow guarantees int q(u|r) du = 1
# but not int q(u|r) dr = 1; under confounding the fit uses that freedom and the margin absorbs
# part of the propensity. copula_diagnostics now draws r ~ U^K from the base and u from the
# copula and records cop_ks_umarg_* / cop_mean_umarg_*. Plain setting (joint ff, lr 0.001,
# copula width 16, margin 48/8, batch 100, no averaging, patience 30), E2 and E1, datasets
# 1..5, fit seed 2001 (a new seed, so these are extra plain fits, not duplicates). 10 fits.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=umarg_check_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/umarg_check_logs"; mkdir -p "$LOGDIR"
COMMON="--model ff --arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16
        --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== umarg check start $(date -u +%FT%TZ) ==="
for k in 1 2 3 4 5; do
  for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
    name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16_k64_s2001_d0"
    if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit 2001 $COMMON > "$LOGDIR/$name.log" 2>&1
      echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== umarg check end $(date -u +%FT%TZ) ==="
