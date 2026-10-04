#!/usr/bin/env bash
# Completing 16x16, all digits (2026-10-04, user decision): Dan's Gaussian flexible flow on datasets 3-5,
# E1-E6, the grid's five fit seeds (datasets 1-2 done by gaussian_16x16_check/more.sh), so 5 datasets x 5
# seeds on every preset; frengression and baselines for E3, E5 on datasets 3-5 (the 16x16 grid has the
# other presets). Weights saved.
# Slots: any 5-core block (48 of them) that no running fit is pinned to; own locks for its own fits.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
CORES=5
LOCK=runs/exp_ate_recovery/_scripts/gaussian_16x16_complete_slots; LOGD=runs/exp_ate_recovery/_scripts/gaussian_16x16_complete_logs
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
   --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16 --save-model
   --wandb --wandb-group gaussian_16x16"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2 [exp3_confounded_heterogeneous]=e3
                  [exp4_covariate_cate]=e4 [exp5_quantile_effect]=e5 [exp6_spatial_cate]=e6)
echo "=== gaussian 16x16 complete start $(date -u +%FT%TZ) ==="
for k in 3 4 5; do
  for preset in exp3_confounded_heterogeneous exp5_quantile_effect; do
    nice -n 19 micromamba run -n frugal-flows python baselines.py --all-digits --preset $preset --size 16 \
      --seed-data 101 --seed-assign $k --no-plots > "$LOGD/baselines_${SHORT[$preset]}_sa${k}.log" 2>&1
  done
done
for k in 3 4 5; do
  for preset in exp3_confounded_heterogeneous exp5_quantile_effect; do
    launch "F16_${SHORT[$preset]}_sa${k}" micromamba run -n frugal-flows-frengression python exp_frengression_recovery.py \
      --all-digits --preset $preset --seed-assign $k --seed-fit $k --size 16 --seed-data 101 --num-iters 5000 \
      --threads 4 --runs-root runs/frengression --wandb --wandb-group gaussian_16x16_frengression
  done
done
for k in 3 4 5; do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous \
                exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate; do
    for fs in $k 1001 1002 1003 1004; do
      launch "G16_${SHORT[$preset]}_sa${k}_s${fs}" micromamba run -n frugal-flows python exp_ate_recovery.py $G \
        --preset $preset --seed-assign $k --seed-fit $fs
    done
  done
done
wait
echo "=== gaussian 16x16 complete end $(date -u +%FT%TZ) ==="
