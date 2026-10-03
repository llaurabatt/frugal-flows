#!/usr/bin/env bash
# Sanity checks for Dan's Gaussian-scale arms after merging gaussian-spline (2026-10-03).
# A) reproduce his five example fits (examples/morphomnist_8x8_n5000: E2, dataset 1, n = 5000,
#    fit seed 1) with their exact settings;
# B) his models on our 8x8 grid's data (all digits, n = 60000), E2 and E4, dataset 1:
#    Gaussian flexible with the grid's five fit seeds (5-fit average comparable with the grid's
#    uniform flow and frengression on the same data), and the two location-translation arms once.
# Slot pool on cores 150-239 (18 slots x 5 cores), its own locks. Throwaway: wandb group gaussian_sanity.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
SLOTS=18; CORES=5; OFFSET=150
LOCK=runs/exp_ate_recovery/_scripts/gaussian_sanity_slots; LOGD=runs/exp_ate_recovery/_scripts/gaussian_sanity_logs
mkdir -p "$LOCK" "$LOGD"
acquire() { while true; do for i in $(seq 0 $((SLOTS-1))); do mkdir "$LOCK/$i" 2>/dev/null && { echo $i; return; }; done; sleep 10; done; }
launch() {
  local name=$1; shift
  local slot; slot=$(acquire); local a=$((OFFSET+slot*CORES)) b=$((OFFSET+slot*CORES+CORES-1))
  echo "start $name slot $slot cores $a-$b $(date -u +%FT%TZ)"
  ( taskset -c $a-$b micromamba run -n frugal-flows python exp_ate_recovery.py "$@" > "$LOGD/$name.log" 2>&1; rc=$?
    rmdir "$LOCK/$slot"; echo "end   $name rc=$rc $(date -u +%FT%TZ)" ) &
  sleep 3
}
COMMON="--model ff --conditioner mlp --size 8 --seed-data 101 --all-digits --wandb --wandb-group gaussian_sanity"
U_FLEX="--arm flexible_continuous"
U_LT="--arm location_translation"
G_FLEX="--arm flexible_continuous_gaussian --y-scaling standardize"
G_LT_HEAD="--arm location_translation_gaussian --y-scaling standardize --shift-init naive --shift-lr-mult 10"
G_LT_STD="--arm location_translation_gaussian --y-scaling standardize --shift-init scalar"
echo "=== gaussian sanity start $(date -u +%FT%TZ) ==="
# A) reproduce the examples
for m in U-LT-raw:U_LT U-flex-raw:U_FLEX G-flex-std:G_FLEX G-LT-head:G_LT_HEAD G-LT-std:G_LT_STD; do
  name=${m%%:*}; var=${m##*:}
  launch "A_${name}" $COMMON ${!var} --preset exp2_confounded_homogeneous --n 5000 --seed-assign 1 --seed-fit 1
done
# B) our 8x8 data, dataset 1
for preset in exp2_confounded_homogeneous exp4_covariate_cate; do
  p=e${preset:3:1}
  for fs in 1 1001 1002 1003 1004; do
    launch "B_${p}_G-flex-std_s${fs}" $COMMON $G_FLEX --preset $preset --seed-assign 1 --seed-fit $fs \
      --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100 \
      --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16 --save-model
  done
  for m in G-LT-head:G_LT_HEAD U-LT-raw:U_LT; do
    name=${m%%:*}; var=${m##*:}
    launch "B_${p}_${name}_s1" $COMMON ${!var} --preset $preset --seed-assign 1 --seed-fit 1 --save-model
  done
done
wait
echo "=== gaussian sanity end $(date -u +%FT%TZ) ==="
