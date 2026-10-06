#!/usr/bin/env bash
# Copula-width check at 16x16, all digits (2026-10-05, user: priority). The Gaussian 16x16 grid (copula width
# 16) puts a positive spurious effect on the ring around the disc in every confounded preset (E2-E6, not E1);
# a copula too narrow to model the covariates given 256 pixel scores is the leading hypothesis (bug #18 halo).
# Same settings as gaussian_16x16_complete.sh except --copula-nn-width 32 / 64; E2 and E4, dataset 1, fit
# seeds 1 and 1001, so each fit pairs with an existing width-16 fit on the same data and seed. Weights saved.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
CORES=5
LOCK=runs/exp_ate_recovery/_scripts/copula_width_16x16_slots; LOGD=runs/exp_ate_recovery/_scripts/copula_width_16x16_logs
mkdir -p "$LOCK" "$LOGD"
busy_slots() {   # slots that some running fit (any launcher) is pinned to
  for p in $(ps -u "$(id -u)" -o pid=,args= | awk '$2=="python" && $3 ~ /exp_(ate|frengression)_recovery/ {print $1}'); do
    a=$(taskset -c -p "$p" 2>/dev/null | awk -F': ' '{print $2}'); [ -n "$a" ] && echo $(( ${a%%[-,]*} / CORES ))
  done | sort -u
}
acquire() {
  while true; do
    busy=" $(busy_slots | tr '\n' ' ') "
    for i in $(seq 0 47); do
      case "$busy" in *" $i "*) continue;; esac
      mkdir "$LOCK/$i" 2>/dev/null && { echo $i; return; }
    done
    sleep 30
  done
}
launch() {
  local name=$1; shift
  local slot; slot=$(acquire); local a=$((slot*CORES)) b=$((slot*CORES+CORES-1))
  echo "start $name slot $slot cores $a-$b $(date -u +%FT%TZ)"
  ( taskset -c $a-$b "$@" > "$LOGD/$name.log" 2>&1; rc=$?; rmdir "$LOCK/$slot"; echo "end   $name rc=$rc $(date -u +%FT%TZ)" ) &
  sleep 20        # let the fit start and pin itself before the next slot is chosen
}
G="--model ff --conditioner mlp --size 16 --seed-data 101 --all-digits --arm flexible_continuous_gaussian
   --y-scaling standardize --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001
   --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --save-model
   --wandb --wandb-group copula_width_16x16"
echo "=== copula width 16x16 start $(date -u +%FT%TZ) ==="
for w in 64 32; do
  for preset in exp2_confounded_homogeneous exp4_covariate_cate; do
    for fs in 1 1001; do
      launch "G16_${preset%%_*}_copw${w}_s${fs}" micromamba run -n frugal-flows python exp_ate_recovery.py $G \
        --copula-nn-width $w --preset $preset --seed-assign 1 --seed-fit $fs
    done
  done
done
wait
echo "=== copula width 16x16 end $(date -u +%FT%TZ) ==="
