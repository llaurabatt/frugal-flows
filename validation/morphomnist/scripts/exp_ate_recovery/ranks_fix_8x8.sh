#!/usr/bin/env bash
# The copula hidden-rank fix (2026-09-25) on a small set: E1 and E2, effect 1.0, assignment
# seeds 1..5, joint fit (ff). Everything else is exactly the 2026-09-21 grid cell with the
# same seed (same data, same fit seed, same settings), so each new fit pairs with one grid
# fit whose only difference is the rank rule (hidden_ranks_rule spread vs legacy in the
# index). Every fit goes through exp_ate_recovery.py into runs/exp_ate_recovery/; the
# copula and sample diagnostics run as usual. Ten fits, five at a time.
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=copula_ranks_fix_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/ranks_fix_logs"; mkdir -p "$LOGDIR"
PARALLEL=5
COMMON="--arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.01 --batch-size 100
        --max-epochs 1000 --max-patience 30 --n-mc 5000 --wandb --wandb-group $GROUP"
echo "=== ranks fix start $(date -u +%FT%TZ) ==="
for k in 1 2 3 4 5; do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
    name="ff_${preset:0:4}_sa${k}"
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --preset "$preset" --model ff --seed-assign "$k" --seed-fit "$k" $COMMON \
        > "$LOGDIR/$name.log" 2>&1; echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== ranks fix end $(date -u +%FT%TZ) ==="
