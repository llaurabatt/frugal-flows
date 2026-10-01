#!/usr/bin/env bash
# Timing at 16x16 and 32x32 (2026-09-30), confirmation step 0: one E2 fit per size in the agreed
# setting (joint ff, lr 0.001, copula width 16, margin 48/8, batch 100, patience 30, max 1000
# epochs), training capped at 3600 s (tag cap3600). Records s/epoch, read-out and diagnostic times.
# 32x32 has never been fitted before.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
LOGDIR="runs/exp_ate_recovery/_scripts/timing_logs"; mkdir -p "$LOGDIR"
echo "=== timing start $(date -u +%FT%TZ) ==="
for size in 16 32; do
  ( $PY exp_ate_recovery.py --preset exp2_confounded_homogeneous --seed-assign 1 --seed-fit 1 \
      --model ff --arm flexible_continuous --conditioner mlp --size $size --digit 0 --seed-data 101 \
      --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100 \
      --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16 --wall-cap-s 3600 \
      --wandb --wandb-group timing_sizes > "$LOGDIR/e2_${size}x${size}.log" 2>&1
    echo "end size $size rc=$? $(date -u +%FT%TZ)" ) &
  sleep 20
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== timing end $(date -u +%FT%TZ) ==="
