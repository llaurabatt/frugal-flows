#!/usr/bin/env bash
# Paper grid, 8x8, ALL TEN DIGITS (2026-10-01). E1-E6 x datasets (assignment seeds) 1..10 x fit seeds
# {k, 1001, 1002, 1003, 1004}: 300 flow fits (paper defaults: lr 1e-3, copula 16, margin 48/8, batch 100,
# patience 30, max 1000 epochs, weights saved) + 300 frengression fits (adapter defaults, weights saved).
# A cell is done only if it has a result AND saved weights, so the 10 weightless all-digit 8x8 fits from
# 2026-09-28 are refitted (identical by determinism; old folders to be removed after a per-cell check).
# CPU: a pool of 48 slots x 5 cores; each fit is pinned to its slot's cores with taskset.
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
SLOTS=${SLOTS:-48}; CORES=5
LOCK=runs/exp_ate_recovery/_scripts/grid_8x8_alldigits_slots; rm -rf "$LOCK"; mkdir -p "$LOCK"
FLOG=runs/exp_ate_recovery/_scripts/grid_8x8_alldigits_logs; RLOG=runs/frengression/_scripts/grid_8x8_alldigits_logs
mkdir -p "$FLOG" "$RLOG"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2 [exp3_confounded_heterogeneous]=e3
                  [exp4_covariate_cate]=e4 [exp5_quantile_effect]=e5 [exp6_spatial_cate]=e6)
PRESETS="exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate"

acquire() { while true; do for i in $(seq 0 $((SLOTS-1))); do mkdir "$LOCK/$i" 2>/dev/null && { echo $i; return; }; done; sleep 10; done; }
launch() {   # name logfile command...
  local name=$1 log=$2; shift 2
  local slot; slot=$(acquire); local a=$((slot*CORES)) b=$((slot*CORES+CORES-1))
  echo "start $name slot $slot cores $a-$b $(date -u +%FT%TZ)"
  ( taskset -c $a-$b "$@" > "$log" 2>&1; rc=$?; rmdir "$LOCK/$slot"; echo "end   $name rc=$rc $(date -u +%FT%TZ)" ) &
  sleep 3
}
ff_done() { for d in runs/exp_ate_recovery/*_"$1"_*/; do [ -f "$d/metrics.json" ] && [ -f "$d/model.eqx" ] && return 0; done; return 1; }
fr_done() { for d in runs/frengression/*_"$1"_*/; do grep -q '"status": "ok"' "$d/metrics.json" 2>/dev/null && [ -f "$d/model.pt" ] && return 0; done; return 1; }

echo "=== grid 8x8 all digits start $(date -u +%FT%TZ) slots=$SLOTS x $CORES cores ==="
for k in 1 2 3 4 5 6 7 8 9 10; do
  for fs in "$k" 1001 1002 1003 1004; do
    for preset in $PRESETS; do
      p=${SHORT[$preset]}
      n="ff_${p}_flexcont_sa${k}_lr0.001_copw16_k64_s${fs}_d0-9"
      if ff_done "$n"; then echo "skip  $n (done)"; else
        launch "$n" "$FLOG/$n.log" micromamba run -n frugal-flows python exp_ate_recovery.py --all-digits --preset "$preset" \
          --seed-assign "$k" --seed-fit "$fs" --model ff --arm flexible_continuous --conditioner mlp --size 8 --seed-data 101 \
          --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100 \
          --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16 --save-model \
          --wandb --wandb-group grid_8x8_alldigits
      fi
      r="frengression_${p}_sa${k}_k64_s${fs}_d0-9"
      if fr_done "$r"; then echo "skip  $r (done)"; else
        launch "$r" "$RLOG/$r.log" micromamba run -n frugal-flows-frengression python exp_frengression_recovery.py --all-digits \
          --preset "$preset" --seed-assign "$k" --seed-fit "$fs" --size 8 --seed-data 101 --num-iters 5000 --threads 4 \
          --runs-root runs/frengression --wandb --wandb-group grid_8x8_alldigits_frengression
      fi
    done
  done
done
wait
micromamba run -n frugal-flows python run_index.py
micromamba run -n frugal-flows python run_index.py --baselines
micromamba run -n frugal-flows python check_runs.py --no-wandb | tail -1
echo "=== grid 8x8 all digits end $(date -u +%FT%TZ) ==="
