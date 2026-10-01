#!/usr/bin/env bash
# Outcome-margin size sweep (2026-09-26): margin hidden width {48, 16} x spline knots {8, 4},
# with learning rate 0.001, copula width 16, fixed hidden-unit numbering, joint stopping,
# on E1 (control, no confounding) and E2, effect 1.0, assignment seeds 1..20, joint fit.
# Data, fit seeds and all other settings equal the 2026-09-21 grid, so every fit has the
# grid fit, OLS/AIPW/IPW/frengression on the same dataset. Width 48 / knots 8 is the
# copw16 setting of copw_lr1e-3_8x8.sh; rerun here because the margin changed (commit of
# 2026-09-26, hidden_ranks_rule spread_all): 160 fits.
# Names carry lr0.001, copw16 and, when not default, mw<width> / mkn<knots>; config.json
# and the wandb config record nn_width and rqs_knots; index columns nn_width, rqs_knots;
# Table 4 shows both. A cell whose folder holds metrics.json is skipped; relaunch resumes.
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=margin_size_lr1e-3_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/margin_size_logs"; mkdir -p "$LOGDIR"
PARALLEL=${PARALLEL:-12}
COMMON="--arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-depth 1 --flow-layers 4 --learning-rate 0.001 --batch-size 100 --copula-nn-width 16
        --max-epochs 1000 --max-patience 30 --n-mc 5000 --wandb --wandb-group $GROUP"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== margin size start $(date -u +%FT%TZ) parallel=$PARALLEL ==="
for k in $(seq 1 20); do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous; do
    for spec in "48 8" "16 8" "48 4" "16 4"; do
      set -- $spec; w=$1; kn=$2
      tag=""; [ "$w" != 48 ] && tag="${tag}_mw$w"; [ "$kn" != 8 ] && tag="${tag}_mkn$kn"
      name="ff_${SHORT[$preset]}_flexcont_sa${k}_lr0.001_copw16${tag}_k64_s${k}_d0"
      # done only if a folder of this name finished under the current rule (spread_all):
      # the 48/8 copw16 fits of copw_lr1e-3_8x8.sh share the name but used flowjax's margin
      if grep -l '"hidden_ranks_rule": "spread_all"' runs/exp_ate_recovery/*_"${name}"_*/config.json 2>/dev/null \
           | while read -r c; do [ -f "$(dirname "$c")/metrics.json" ] && echo ok; done | grep -q ok; then
        echo "skip  $name (done)"; continue
      fi
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
      echo "start $name $(date -u +%FT%TZ)"
      ( $PY exp_ate_recovery.py --preset "$preset" --model ff --seed-assign "$k" --seed-fit "$k" \
          --nn-width "$w" --rqs-knots "$kn" $COMMON > "$LOGDIR/$name.log" 2>&1
        echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
    done
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== margin size end $(date -u +%FT%TZ) ==="
