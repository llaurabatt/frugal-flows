#!/usr/bin/env bash
# Copula hidden width 50 vs 16 at learning rate 0.001 (2026-09-25), fixed hidden-unit
# numbering, joint stopping: E1 (control, no confounding) and E2, effect 1.0, assignment
# seeds 1..20, joint fit (ff). Data, fit seeds and every other setting equal the 2026-09-21
# grid, so every fit has the grid fit, OLS/AIPW/IPW/frengression on the same dataset.
# Width 50 at seeds 1..5 already exist (lr1e-3_8x8.sh, group lr_1e-3_8x8) and are skipped.
# Names carry lr0.001 and, for width 16, copw16; config.json and the wandb config record
# learning_rate and copula_nn_width; index columns lr and copula_nn_width; Table 4 shows both.
# A cell whose folder already holds metrics.json is skipped, so a relaunch resumes.
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=copula_width_lr1e-3_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/copw_lr1e-3_logs"; mkdir -p "$LOGDIR"
PARALLEL=${PARALLEL:-10}
COMMON="--arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100
        --max-epochs 1000 --max-patience 30 --n-mc 5000 --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== copw start $(date -u +%FT%TZ) parallel=$PARALLEL ==="
for k in $(seq 1 20); do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous; do
    for w in 50 16; do
      tag=""; [ "$w" != 50 ] && tag="_copw$w"
      name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001${tag}_k64_s${k}_d0"
      if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then
        echo "skip  $name (done)"; continue
      fi
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
      echo "start $name $(date -u +%FT%TZ)"
      ( $PY exp_ate_recovery.py --preset "$preset" --model ff --seed-assign "$k" --seed-fit "$k" \
          --copula-nn-width "$w" $COMMON > "$LOGDIR/$name.log" 2>&1
        echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
    done
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== copw end $(date -u +%FT%TZ) ==="
