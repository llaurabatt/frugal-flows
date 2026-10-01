#!/usr/bin/env bash
# Is the E2 leftover confounding a small-sample effect on the images? (2026-09-28)
# The K-pixel toy (runs/leftover_confounding/toy_multi.py) leaves ~6-8 % at n=5923 and ~0 at
# n=50000. Here digit 0 is capped at n=3000 and n=1500 (--n; tag n<N>); prediction: the
# leftover slope grows as n shrinks. Plain setting otherwise (joint ff, lr 0.001, copula width
# 16, margin 48/8, batch 100, no averaging, patience 30), E2 and E1, datasets 1..5, fit seeds k
# and 1001; the full-n reference is the noise_fixes_8x8 plain fits at the same seeds. 40 fits.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=subsample_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/subsample_logs"; mkdir -p "$LOGDIR"
COMMON="--model ff --arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16
        --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== subsample start $(date -u +%FT%TZ) ==="
for n in 3000 1500; do
  for k in 1 2 3 4 5; do
    for fs in "$k" 1001; do
      for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
        name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16_n${n}_k64_s${fs}_d0"
        if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
        echo "start $name $(date -u +%FT%TZ)"
        ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$fs" --n "$n" $COMMON > "$LOGDIR/$name.log" 2>&1
          echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
      done
    done
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== subsample end $(date -u +%FT%TZ) ==="
