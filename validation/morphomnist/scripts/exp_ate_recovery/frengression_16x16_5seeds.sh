#!/usr/bin/env bash
# Frengression with five seeds at 16x16, all digits (2026-10-04, user decision): fit seeds 1001-1004 on
# E1-E6 x datasets 1-5 (seed k exists already), so frengression has 5 datasets x 5 seeds like the
# Gaussian flexible flow. 120 fits, weights saved, 5000-draw read-out (default).
# Slots: any 5-core block (48 of them) that no running fit is pinned to; own locks for its own fits.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
CORES=5
LOCK=runs/exp_ate_recovery/_scripts/frengression_16x16_5seeds_slots; LOGD=runs/exp_ate_recovery/_scripts/frengression_16x16_5seeds_logs
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
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2 [exp3_confounded_heterogeneous]=e3
                  [exp4_covariate_cate]=e4 [exp5_quantile_effect]=e5 [exp6_spatial_cate]=e6)
echo "=== frengression 16x16 5 seeds start $(date -u +%FT%TZ) ==="
for fs in 1001 1002 1003 1004; do
  for k in 1 2 3 4 5; do
    for preset in exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous \
                  exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate; do
      n="frengression_${SHORT[$preset]}_sa${k}_k256_s${fs}_d0-9"
      if ls runs/frengression/*_${n}_*/model.pt >/dev/null 2>&1; then echo "skip  $n (done)"; continue; fi
      launch "$n" micromamba run -n frugal-flows-frengression python exp_frengression_recovery.py \
        --all-digits --preset $preset --seed-assign $k --seed-fit $fs --size 16 --seed-data 101 --num-iters 5000 \
        --threads 4 --runs-root runs/frengression --wandb --wandb-group frengression_16x16_5seeds
    done
  done
done
wait
echo "=== frengression 16x16 5 seeds end $(date -u +%FT%TZ) ==="
