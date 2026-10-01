#!/usr/bin/env bash
# Early stopping on the held-out copula loss (2026-09-25), with the fixed hidden-unit
# numbering (commit 1b2510a): E1 and E2, effect 1.0, assignment seeds 1..5, joint fit (ff),
# --select-on copula. Data, fit seeds and every other setting equal the 2026-09-21 grid
# cells and the ranks-fix refits of the same cells, so each fit forms a triple with them:
# legacy numbering + joint stopping / spread + joint / spread + copula. Names carry the
# copsel tag; config.json and the wandb config record select_on; metrics record
# selected_on, best_select and the per-epoch copula series (loss_select in arrays.npz).
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=copula_select_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/copsel_logs"; mkdir -p "$LOGDIR"
PARALLEL=5
COMMON="--arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.01 --batch-size 100
        --max-epochs 1000 --max-patience 30 --n-mc 5000 --select-on copula --wandb --wandb-group $GROUP"
echo "=== copsel start $(date -u +%FT%TZ) ==="
for k in 1 2 3 4 5; do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
    name="ff_${preset:0:4}_sa${k}_copsel"
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --preset "$preset" --model ff --seed-assign "$k" --seed-fit "$k" $COMMON \
        > "$LOGDIR/$name.log" 2>&1; echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== copsel end $(date -u +%FT%TZ) ==="
