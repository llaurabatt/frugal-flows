#!/usr/bin/env bash
# Test (2026-10-02): the image margin with ONE fixed pixel order (no permutation between its layers,
# --margin-order fixed, tag mfix) against the default shuffled margin. Paired with the 8x8 all-digits
# grid's cells: E1, E2 x datasets (assignment seeds) 1-3 x fit seeds {k, 1001-1004}, all other
# settings identical to grid_8x8_alldigits_v2.sh. 30 fits. NOT the default; a test only.
# Waits for the grid launcher to write its end line, then runs all 30 at once, 5 cores each.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
GLOG=runs/exp_ate_recovery/_scripts/grid_8x8_alldigits_driver.log
FLOG=runs/exp_ate_recovery/_scripts/margin_order_8x8_logs; mkdir -p "$FLOG"
n0=$(grep -c "grid 8x8 all digits end" "$GLOG")
echo "waiting for the grid to end ($(date -u +%FT%TZ))"
while [ "$(grep -c "grid 8x8 all digits end" "$GLOG")" -le "$n0" ]; do sleep 300; done
echo "=== margin order 8x8 start $(date -u +%FT%TZ) ==="
slot=0
for k in 1 2 3; do
  for fs in "$k" 1001 1002 1003 1004; do
    for preset in exp1_rct_homogeneous exp2_confounded_homogeneous; do
      p=e${preset:3:1}
      n="ff_${p}_flexcont_sa${k}_lr0.001_copw16_mfix_k64_s${fs}_d0-9"
      if ls runs/exp_ate_recovery/*_${n}_*/model.eqx >/dev/null 2>&1; then echo "skip  $n (done)"; continue; fi
      a=$((slot*5)); b=$((slot*5+4))
      echo "start $n cores $a-$b $(date -u +%FT%TZ)"
      ( taskset -c $a-$b micromamba run -n frugal-flows python exp_ate_recovery.py --all-digits --preset "$preset" \
          --seed-assign "$k" --seed-fit "$fs" --model ff --arm flexible_continuous --conditioner mlp --size 8 --seed-data 101 \
          --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100 \
          --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16 --save-model --margin-order fixed \
          --wandb --wandb-group margin_order_8x8 > "$FLOG/$n.log" 2>&1
        echo "end   $n rc=$? $(date -u +%FT%TZ)" ) &
      slot=$((slot+1)); sleep 3
    done
  done
done
wait
echo "=== margin order 8x8 end $(date -u +%FT%TZ) ==="
