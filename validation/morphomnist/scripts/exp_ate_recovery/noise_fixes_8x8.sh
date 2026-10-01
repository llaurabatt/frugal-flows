#!/usr/bin/env bash
# Noise fixes (2026-09-27): does a single fit get less variable with a running average of the
# weights (--ema-epochs 20) and / or larger batches (--batch-size 500)? Joint fit (ff), lr
# 0.001, copula width 16, margin width 48 / 8 knots, fixed layers (hidden_ranks_rule
# spread_all), on E1 and E2 (effect 1.0), datasets = assignment seeds 1..5, fit seeds
# {k, 1001..1004}, four versions: plain, ema20, batch500, batch500 + ema20.
# The plain version with fit seed k exists (margin_size_8x8.sh, 48/8), so it is skipped.
# Measures (analyse_noise_fixes.py): spread of the error between fit seeds on one dataset,
# a typical single fit's error vs the error of the 5-fit average, and on E2 the slope of the
# error on the dataset's imbalance (leftover confounding).
# Names carry lr0.001, copw16, batch500 / ema20 when used, sa<k>, s<fit seed>; config.json and
# the wandb config record batch_size and ema_epochs; metrics record ema_decay; index columns
# batch_size, ema_epochs; Table 4 states the weight averaging. 190 fits (40 plain + 150).
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=noise_fixes_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/noise_fixes_logs"; mkdir -p "$LOGDIR"
PARALLEL=${PARALLEL:-12}
COMMON="--model ff --arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --copula-nn-width 16 --max-epochs 1000 --max-patience 30 --n-mc 5000
        --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== noise fixes start $(date -u +%FT%TZ) parallel=$PARALLEL ==="
for k in 1 2 3 4 5; do
  for fs in "$k" 1001 1002 1003 1004; do
    for preset in exp1_rct_homogeneous exp2_confounded_homogeneous; do
      for spec in "100 0" "100 20" "500 0" "500 20"; do
        set -- $spec; bs=$1; ema=$2
        tag=""; [ "$bs" != 100 ] && tag="${tag}_batch$bs"; [ "$ema" != 0 ] && tag="${tag}_ema$ema"
        name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16${tag}_k64_s${fs}_d0"
        # done only if a folder of this name finished under the current layer rule (spread_all)
        if grep -l '"hidden_ranks_rule": "spread_all"' runs/exp_ate_recovery/*_"${name}"_*/config.json 2>/dev/null \
             | while read -r c; do [ -f "$(dirname "$c")/metrics.json" ] && echo ok; done | grep -q ok; then
          echo "skip  $name (done)"; continue
        fi
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
        echo "start $name $(date -u +%FT%TZ)"
        ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$fs" \
            --batch-size "$bs" --ema-epochs "$ema" $COMMON > "$LOGDIR/$name.log" 2>&1
          echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
      done
    done
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== noise fixes end $(date -u +%FT%TZ) ==="
