#!/usr/bin/env bash
# 16x16 grid, all ten digits (2026-10-03, user decision): E1, E2, E4, E6 x datasets (assignment seeds) 1-5.
# Flow (flexible arm, paper defaults, weights saved) with fit seeds {k, 1001-1004} -> the method is the
# 5-fit average; frengression one fit per dataset (seed k). Same settings as grid_8x8_alldigits_v2.sh
# except --size 16. 100 flow fits + 20 frengression fits.
# Slot pool of SLOTS x 5 cores (default 48), each fit pinned with taskset; its own lock directory.
# Resumable: skips cells that are done (metrics.json + saved weights) or still running (folder without
# metrics.json). The all-digit 16x16 runs of 2026-09-30 have no weights, so their cells are run again.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
SLOTS=${SLOTS:-48}; CORES=5
LOCK=runs/exp_ate_recovery/_scripts/grid_16x16_alldigits_slots
FLOG=runs/exp_ate_recovery/_scripts/grid_16x16_alldigits_logs; RLOG=runs/frengression/_scripts/grid_16x16_alldigits_logs
mkdir -p "$LOCK" "$FLOG" "$RLOG"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2 [exp4_covariate_cate]=e4 [exp6_spatial_cate]=e6)
PRESETS="exp1_rct_homogeneous exp2_confounded_homogeneous exp4_covariate_cate exp6_spatial_cate"
acquire() { while true; do for i in $(seq 0 $((SLOTS-1))); do mkdir "$LOCK/$i" 2>/dev/null && { echo $i; return; }; done; sleep 10; done; }
launch() {
  local name=$1 log=$2; shift 2
  local slot; slot=$(acquire); local a=$((slot*CORES)) b=$((slot*CORES+CORES-1))
  echo "start $name slot $slot cores $a-$b $(date -u +%FT%TZ)"
  ( taskset -c $a-$b "$@" > "$log" 2>&1; rc=$?; rmdir "$LOCK/$slot"; echo "end   $name rc=$rc $(date -u +%FT%TZ)" ) &
  sleep 3
}
state() {   # root name modelfile -> done | running | todo
  local d
  for d in "$1"/*_"$2"_*/; do [ -d "$d" ] || continue
    [ -f "$d/metrics.json" ] && [ -f "$d/$3" ] && { echo done; return; }
  done
  for d in "$1"/*_"$2"_*/; do [ -d "$d" ] || continue
    [ -f "$d/config.json" ] && [ ! -f "$d/metrics.json" ] && { echo running; return; }
  done
  echo todo
}
echo "=== grid 16x16 all digits start $(date -u +%FT%TZ) slots=$SLOTS x $CORES cores ==="
for k in 1 2 3 4 5; do
  for fs in "$k" 1001 1002 1003 1004; do
    for preset in $PRESETS; do
      p=${SHORT[$preset]}
      n="ff_${p}_flexcont_sa${k}_lr0.001_copw16_k256_s${fs}_d0-9"
      st=$(state runs/exp_ate_recovery "$n" model.eqx)
      if [ "$st" != todo ]; then echo "skip  $n ($st)"; else
        launch "$n" "$FLOG/$n.log" micromamba run -n frugal-flows python exp_ate_recovery.py --all-digits --preset "$preset" \
          --seed-assign "$k" --seed-fit "$fs" --model ff --arm flexible_continuous --conditioner mlp --size 16 --seed-data 101 \
          --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100 \
          --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16 --save-model \
          --wandb --wandb-group grid_16x16_alldigits
      fi
      [ "$fs" = "$k" ] || continue          # frengression: the dataset's own seed only
      r="frengression_${p}_sa${k}_k256_s${fs}_d0-9"
      st=$(state runs/frengression "$r" model.pt)
      if [ "$st" != todo ]; then echo "skip  $r ($st)"; else
        launch "$r" "$RLOG/$r.log" micromamba run -n frugal-flows-frengression python exp_frengression_recovery.py --all-digits \
          --preset "$preset" --seed-assign "$k" --seed-fit "$fs" --size 16 --seed-data 101 --num-iters 5000 --threads 4 \
          --runs-root runs/frengression --wandb --wandb-group grid_16x16_alldigits_frengression
      fi
    done
  done
done
wait
while [ -n "$(ls -A $LOCK 2>/dev/null)" ]; do sleep 60; done
micromamba run -n frugal-flows python run_index.py
micromamba run -n frugal-flows python run_index.py --baselines
micromamba run -n frugal-flows python check_runs.py --no-wandb | tail -1
echo "=== grid 16x16 all digits end $(date -u +%FT%TZ) ==="
