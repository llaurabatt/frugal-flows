#!/usr/bin/env bash
# All ten digits at 16x16 (2026-09-30), the paper benchmark. Does the 16x16 leftover seen on digit 0
# (one fit, slope 0.20) persist with 60000 images? E2 and E1, datasets 1..3, fit seed = dataset
# seed, paper setting (defaults: lr 1e-3, copula 16, margin 48/8, batch 100, patience 30, max 1000).
# Frengression on the same datasets: scripts/frengression/frengression_alldigits_16x16.sh.
# 6 fits, extrapolated ~4-6 h each.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
LOGDIR="runs/exp_ate_recovery/_scripts/alldigits_16x16_logs"; mkdir -p "$LOGDIR"
COMMON="--all-digits --model ff --arm flexible_continuous --conditioner mlp --size 16 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16
        --wandb --wandb-group alldigits_16x16"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== all digits 16x16 start $(date -u +%FT%TZ) ==="
for k in 1 2 3; do for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
  name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16_k256_s${k}_d0-9"
  if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
  echo "start $name $(date -u +%FT%TZ)"
  ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$k" $COMMON > "$LOGDIR/$name.log" 2>&1
    echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  sleep 20
done; done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== all digits 16x16 end $(date -u +%FT%TZ) ==="
