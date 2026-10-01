#!/usr/bin/env bash
# Leftover-confounding fixes that keep the likelihood objective (2026-09-27), plan step A:
#   A1  copula width 8           (--copula-nn-width 8)
#   A2  copula learning rate x3  (--copula-nn-width 16 --copula-lr-mult 3)
#       copula learning rate x10 (--copula-nn-width 16 --copula-lr-mult 10)
# against the plain version (copula width 16, one learning rate; the batch-100, no-averaging
# fits of noise_fixes_8x8.sh, same data and fit seeds). Joint fit (ff), lr 0.001, margin 48/8,
# batch 100, no weight averaging, fixed layers (hidden_ranks_rule spread_all), on E2 and E1 as
# control, effect 1.0, datasets = assignment seeds 1..5, fit seeds {k, 1001..1004}: each
# dataset's 5 fits are averaged (the method agreed on 09-27). 150 fits.
# Names carry copw8 / coplr3 / coplr10; config.json and the wandb config record
# copula_nn_width and copula_lr_mult; index columns copula_nn_width, copula_lr_mult; Table 4
# states both. Analyse with analyse_confounding_fixes.py.
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=confounding_fixes_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/confounding_fixes_logs"; mkdir -p "$LOGDIR"
PARALLEL=${PARALLEL:-12}
COMMON="--model ff --arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000
        --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== confounding fixes start $(date -u +%FT%TZ) parallel=$PARALLEL ==="
for k in 1 2 3 4 5; do
  for fs in "$k" 1001 1002 1003 1004; do
    for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
      for spec in "8 1" "16 3" "16 10"; do
        set -- $spec; cw=$1; mult=$2
        tag="_copw$cw"; [ "$mult" != 1 ] && tag="${tag}_coplr$mult"
        name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001${tag}_k64_s${fs}_d0"
        if grep -l '"hidden_ranks_rule": "spread_all"' runs/exp_ate_recovery/*_"${name}"_*/config.json 2>/dev/null \
             | while read -r c; do [ -f "$(dirname "$c")/metrics.json" ] && echo ok; done | grep -q ok; then
          echo "skip  $name (done)"; continue
        fi
        while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
        echo "start $name $(date -u +%FT%TZ)"
        ( $PY exp_ate_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$fs" \
            --copula-nn-width "$cw" --copula-lr-mult "$mult" $COMMON > "$LOGDIR/$name.log" 2>&1
          echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
      done
    done
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== confounding fixes end $(date -u +%FT%TZ) ==="
