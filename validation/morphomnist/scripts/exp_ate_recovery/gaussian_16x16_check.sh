#!/usr/bin/env bash
# Dan's Gaussian flexible flow on our 16x16 grid's data (2026-10-03): E2 and E4, dataset 1, all digits
# (n = 60000), the grid's five fit seeds, grid settings -> 5-fit average comparable with the grid's
# uniform flow and frengression on the same data. Shares the gaussian_sanity slot pool (cores 150-239).
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
SLOTS=18; CORES=5; OFFSET=150
LOCK=runs/exp_ate_recovery/_scripts/gaussian_sanity_slots; LOGD=runs/exp_ate_recovery/_scripts/gaussian_sanity_logs
mkdir -p "$LOCK" "$LOGD"
acquire() { while true; do for i in $(seq 0 $((SLOTS-1))); do mkdir "$LOCK/$i" 2>/dev/null && { echo $i; return; }; done; sleep 10; done; }
echo "=== gaussian 16x16 start $(date -u +%FT%TZ) ==="
for preset in exp2_confounded_homogeneous exp4_covariate_cate; do
  p=e${preset:3:1}
  for fs in 1 1001 1002 1003 1004; do
    name="C16_${p}_G-flex-std_s${fs}"
    slot=$(acquire); a=$((OFFSET+slot*CORES)); b=$((a+CORES-1))
    echo "start $name slot $slot cores $a-$b $(date -u +%FT%TZ)"
    ( taskset -c $a-$b micromamba run -n frugal-flows python exp_ate_recovery.py --model ff --conditioner mlp \
        --size 16 --seed-data 101 --all-digits --arm flexible_continuous_gaussian --y-scaling standardize \
        --preset $preset --seed-assign 1 --seed-fit $fs --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 \
        --learning-rate 0.001 --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 \
        --copula-nn-width 16 --save-model --wandb --wandb-group gaussian_sanity > "$LOGD/$name.log" 2>&1
      rc=$?; rmdir "$LOCK/$slot"; echo "end   $name rc=$rc $(date -u +%FT%TZ)" ) &
    sleep 3
  done
done
wait
echo "=== gaussian 16x16 end $(date -u +%FT%TZ) ==="
