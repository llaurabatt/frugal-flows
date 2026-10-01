#!/usr/bin/env bash
# Margin-only fits (no copula) at learning rate 0.001 with the fixed margin layer
# (hidden_ranks_rule spread_all), margin width 48, 8 knots: E1 and E2, effect 1.0,
# assignment seeds 1..20 (2026-09-26). Same data and fit seeds as the joint fits of
# margin_size_8x8.sh at 48/8, which use the same margin class, so each pair differs only
# in whether a copula (width 16) is trained with the margin. Question: on E1, does the
# joint fit's margin miss each arm's mean image more than the margin fitted alone, i.e. does
# the copula term pull the margin away from its own data? Names: margin_..._lr0.001_...
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=margin_only_lr1e-3_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/margin_only_lr1e-3_logs"; mkdir -p "$LOGDIR"
PARALLEL=${PARALLEL:-12}
COMMON="--arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100
        --max-epochs 1000 --max-patience 30 --n-mc 5000 --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== margin only start $(date -u +%FT%TZ) parallel=$PARALLEL ==="
for k in $(seq 1 20); do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous; do
    name="margin_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_k64_s${k}_d0"
    if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then
      echo "skip  $name (done)"; continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --preset "$preset" --model margin --seed-assign "$k" --seed-fit "$k" $COMMON \
        > "$LOGDIR/$name.log" 2>&1; echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== margin only end $(date -u +%FT%TZ) ==="
