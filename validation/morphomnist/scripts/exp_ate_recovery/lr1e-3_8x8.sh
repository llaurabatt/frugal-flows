#!/usr/bin/env bash
# Learning rate 0.001 instead of 0.01 (2026-09-25), fixed hidden-unit numbering, joint
# stopping: E1 and E2, effect 1.0, assignment seeds 1..5, joint fit (ff). Data, fit seeds
# and every other setting equal the 2026-09-21 grid cells and the ranks-fix and copsel
# refits of the same cells, so each fit is the fourth version of its cell. At lr 0.01 the
# copsel fits blew up after ~40 epochs; the question is whether a smaller step lets the
# margin and the copula both train to convergence. Names carry lr0.001; config.json and the
# wandb config record learning_rate; the index column is lr; Table 4 shows it.
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=lr_1e-3_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/lr1e-3_logs"; mkdir -p "$LOGDIR"
PARALLEL=5
COMMON="--arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --batch-size 100
        --max-epochs 1000 --max-patience 30 --n-mc 5000 --learning-rate 0.001 --wandb --wandb-group $GROUP"
echo "=== lr1e-3 start $(date -u +%FT%TZ) ==="
for k in 1 2 3 4 5; do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
    name="ff_${preset:0:4}_sa${k}_lr1e-3"
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --preset "$preset" --model ff --seed-assign "$k" --seed-fit "$k" $COMMON \
        > "$LOGDIR/$name.log" 2>&1; echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== lr1e-3 end $(date -u +%FT%TZ) ==="
