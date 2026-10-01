#!/usr/bin/env bash
# Fit-seed control (2026-09-26): does the margin's own error come from the dataset or from
# training randomness? E1 (randomised), effect 1.0, datasets = assignment seeds 1..5; each
# refitted margin-only with fit seeds 1001..1004. Together with the margin-only fit of
# margin_only_lr1e-3_8x8.sh (fit seed = k) each dataset has 5 fits that differ only in the
# fit seed (network initialisation, validation split, mini-batch order). Settings equal those
# fits: lr 0.001, margin width 48, 8 knots, fixed layers (hidden_ranks_rule spread_all).
# Names: margin_e1_flexcont_sa<k>_lr0.001_k64_s<fitseed>_d0 (the s field is the fit seed,
# the sa tag the assignment seed); config.json and wandb config record seed_fit and
# seed_assign; index columns seed_fit, seed_assign. 20 new fits.
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=fitseed_control_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/fitseed_control_logs"; mkdir -p "$LOGDIR"
PARALLEL=${PARALLEL:-10}
COMMON="--preset exp1_rct_homogeneous --model margin --arm flexible_continuous --conditioner mlp --size 8
        --digit 0 --seed-data 101 --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8
        --learning-rate 0.001 --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000
        --wandb --wandb-group $GROUP"
echo "=== fitseed control start $(date -u +%FT%TZ) parallel=$PARALLEL ==="
for k in 1 2 3 4 5; do
  for fs in 1001 1002 1003 1004; do
    name="margin_e1_flexcont_sa${k}_lr0.001_k64_s${fs}_d0"
    if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then
      echo "skip  $name (done)"; continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --seed-assign "$k" --seed-fit "$fs" $COMMON \
        > "$LOGDIR/$name.log" 2>&1; echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== fitseed control end $(date -u +%FT%TZ) ==="
