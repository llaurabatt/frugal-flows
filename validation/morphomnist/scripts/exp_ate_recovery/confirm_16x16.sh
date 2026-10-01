#!/usr/bin/env bash
# Confirmation step 2 (2026-09-30): 16x16, E1 and E2, datasets 1..3, fit seeds k and 1001,
# plain and with weight averaging over ~20 epochs (--ema-epochs 20, tag ema20), paired by dataset
# and seed. Paper setting (joint ff, lr 0.001, copula width 16, margin 48/8, batch 100, patience 30,
# max 1000 epochs). Question 1: is the E2 leftover at 16x16 around 0.20 (one timing fit) or was
# that one fit unusual? Question 2: does weight averaging lower the effect-map error pairwise
# without raising the leftover slope? 24 fits, ~35 min each alone.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=confirm_16x16
LOGDIR="runs/exp_ate_recovery/_scripts/confirm_16x16_logs"; mkdir -p "$LOGDIR"
COMMON="--model ff --arm flexible_continuous --conditioner mlp --size 16 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16
        --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== confirm 16x16 start $(date -u +%FT%TZ) ==="
for ema in 0 20; do
  for k in 1 2 3; do
    for fs in "$k" 1001; do
      for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
        tag=""; [ "$ema" != 0 ] && tag="_ema$ema"
        name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16${tag}_k256_s${fs}_d0"
        if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
        echo "start $name $(date -u +%FT%TZ)"
        ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$fs" --ema-epochs "$ema" $COMMON > "$LOGDIR/$name.log" 2>&1
          echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
        sleep 15
      done
    done
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== confirm 16x16 end $(date -u +%FT%TZ) ==="
