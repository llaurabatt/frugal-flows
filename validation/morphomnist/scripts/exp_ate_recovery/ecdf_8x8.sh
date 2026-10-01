#!/usr/bin/env bash
# Leftover confounding, 2026-09-28 (docs/leftover_confounding/README.md): covariate ranks from the empirical CDF (--u-z-method ecdf). 10 fits.
# Plain setting otherwise (joint ff, lr 0.001, copula width 16, margin 48/8, batch 100, no
# averaging, patience 30, spread_all layers), E2 and E1, datasets 1..5, fit seed 2001, so each
# fit pairs with the plain fit of umarg_check_8x8.sh (same data, same keys).
# Analyse with analyse_leftover_fixes.py.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=ecdf_ranks_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/ecdf_logs"; mkdir -p "$LOGDIR"
COMMON="--model ff --arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16
        --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== ecdf start $(date -u +%FT%TZ) ==="
for spec in "ecdf"; do
  case $spec in ecdf) flag="--u-z-method ecdf";; umw*) flag="--copula-umarg-weight ${spec#umw}";; esac
  for k in 1 2 3 4 5; do
    for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
      name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16_${spec}_k64_s2001_d0"
      if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
      echo "start $name $(date -u +%FT%TZ)"
      ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit 2001 $flag $COMMON > "$LOGDIR/$name.log" 2>&1
        echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
    done
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== ecdf end $(date -u +%FT%TZ) ==="
