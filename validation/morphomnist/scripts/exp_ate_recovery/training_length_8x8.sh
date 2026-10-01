#!/usr/bin/env bash
# Does the E2 leftover confounding shrink with longer training? (2026-09-27)
# The toy (runs/leftover_confounding) showed the leftover is not built into the objective
# (zero at n=50000) and shrinks slowly over training at n=5000, past the point where
# patience-30 early stopping stops. Here: the plain setting of the step-A comparison
# (joint ff, lr 0.001, copula width 16, margin 48/8, batch 100, no weight averaging,
# spread_all layers), trained 600 epochs with patience 600 (so no early stopping), with the
# effect read out every 20 epochs (5000 draws) into metrics["track"]: MAE, disc/ring/far
# signed error, and the slope of the error on the dataset's confounding map. E2 and E1
# (control), datasets = assignment seeds 1..5, fit seed = the dataset's seed (the first of
# the plain fits' five). 10 fits. Names carry ep600_pat600.
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=training_length_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/training_length_logs"; mkdir -p "$LOGDIR"
COMMON="--model ff --arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 600 --max-patience 600 --n-mc 5000 --copula-nn-width 16
        --track-every 20 --track-n-mc 5000 --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== training length start $(date -u +%FT%TZ) ==="
for k in 1 2 3 4 5; do
  for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
    name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16_ep600_pat600_k64_s${k}_d0"
    if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$k" $COMMON > "$LOGDIR/$name.log" 2>&1
      echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== training length end $(date -u +%FT%TZ) ==="
