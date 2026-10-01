#!/usr/bin/env bash
# E2 and E1 on all ten digits (--all-digits: n = 60000, Z = thickness + digit one-hot, 11 dims;
# name ends d0-9), 2026-09-28. Does the E2 leftover confounding shrink with ~10x the data, as in
# the K-pixel toy? Note this also changes the problem (ten digit shapes, a discrete covariate),
# so subsample_8x8.sh (digit 0 at n = 3000 / 1500) is the cleaner sample-size test.
# Plain setting otherwise (joint ff, lr 0.001, copula width 16, margin 48/8, batch 100, no
# averaging, patience 30), datasets 1..5, fit seed = dataset seed. 10 fits, ~1-1.5 h each.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=all_digits_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/all_digits_logs"; mkdir -p "$LOGDIR"
COMMON="--all-digits --model ff --arm flexible_continuous --conditioner mlp --size 8 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16
        --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== all digits start $(date -u +%FT%TZ) ==="
for k in 1 2 3 4 5; do
  for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
    name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16_k64_s${k}_d0-9"
    if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$k" $COMMON > "$LOGDIR/$name.log" 2>&1
      echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
    sleep 20   # stagger the start: micromamba's environment lock
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== all digits end $(date -u +%FT%TZ) ==="
