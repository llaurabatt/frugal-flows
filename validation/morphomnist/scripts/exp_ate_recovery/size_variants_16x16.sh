#!/usr/bin/env bash
# 16x16 size variants (2026-09-30). One 16x16 E2 timing fit kept ~20 % of the confounding (8x8:
# ~4-6 %). Sizes were tuned at 8x8 only. E2, datasets 1..3, fit seed = dataset seed, so each pairs
# with the plain fit of confirm_16x16.sh. Variants: copula width 64; copula width 128; margin width
# 128 (tag mw128); copula 64 + margin 128. Pre-set rule: a candidate fix lowers the E2 slope on all
# 3 pairs; it is then checked on E1 and more seeds. 12 fits.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=size_variants_16x16
LOGDIR="runs/exp_ate_recovery/_scripts/size_variants_16x16_logs"; mkdir -p "$LOGDIR"
COMMON="--model ff --arm flexible_continuous --conditioner mlp --size 16 --digit 0 --seed-data 101
        --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
        --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000
        --wandb --wandb-group $GROUP"
echo "=== size variants 16x16 start $(date -u +%FT%TZ) ==="
for spec in "64 48" "128 48" "16 128" "64 128"; do
  set -- $spec; cw=$1; mw=$2
  for k in 1 2 3; do
    tag="_copw$cw"; [ "$mw" != 48 ] && tag="${tag}_mw$mw"
    name="ff_e2_flexcont_sa${k}_lr0.001${tag}_k256_s${k}_d0"
    if ls runs/exp_ate_recovery/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
    echo "start $name $(date -u +%FT%TZ)"
    ( $PY exp_ate_recovery.py --preset exp2_confounded_homogeneous --seed-assign "$k" --seed-fit "$k" \
        --copula-nn-width "$cw" --nn-width "$mw" $COMMON > "$LOGDIR/$name.log" 2>&1
      echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
    sleep 15
  done
done
wait
$PY run_index.py; $PY check_runs.py --no-wandb | tail -1
echo "=== size variants 16x16 end $(date -u +%FT%TZ) ==="
