#!/usr/bin/env bash
# Continuation of grid_8x8_alldigits.sh (2026-10-01 ~11:00 UTC, user decision): frengression gets ONE fit
# seed per dataset (seed = dataset seed k); its 5-fit average was only 2-7 % better than a single fit and
# its fits vary 2-4x less than the flow's. The flow keeps 5 seeds {k, 1001..1004}.
# Takes over from the first launcher (stopped; its running fits continue and free their slots): same
# slot pool, locks NOT reset; skips cells that are done (result + weights) or still running (folder
# without metrics.json). Appends to the same driver log, so the watcher sees the end line.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
SLOTS=48; CORES=5
LOCK=runs/exp_ate_recovery/_scripts/grid_8x8_alldigits_slots
FLOG=runs/exp_ate_recovery/_scripts/grid_8x8_alldigits_logs; RLOG=runs/frengression/_scripts/grid_8x8_alldigits_logs
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2 [exp3_confounded_heterogeneous]=e3
                  [exp4_covariate_cate]=e4 [exp5_quantile_effect]=e5 [exp6_spatial_cate]=e6)
PRESETS="exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate"
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
echo "=== grid 8x8 all digits v2 (frengression 1 seed) start $(date -u +%FT%TZ) ==="
for k in 1 2 3 4 5 6 7 8 9 10; do
  for fs in "$k" 1001 1002 1003 1004; do
    for preset in $PRESETS; do
      p=${SHORT[$preset]}
      n="ff_${p}_flexcont_sa${k}_lr0.001_copw16_k64_s${fs}_d0-9"
      st=$(state runs/exp_ate_recovery "$n" model.eqx)
      if [ "$st" != todo ]; then echo "skip  $n ($st)"; else
        launch "$n" "$FLOG/$n.log" micromamba run -n frugal-flows python exp_ate_recovery.py --all-digits --preset "$preset" \
          --seed-assign "$k" --seed-fit "$fs" --model ff --arm flexible_continuous --conditioner mlp --size 8 --seed-data 101 \
          --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100 \
          --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16 --save-model \
          --wandb --wandb-group grid_8x8_alldigits
      fi
      [ "$fs" = "$k" ] || continue          # frengression: the dataset's own seed only
      r="frengression_${p}_sa${k}_k64_s${fs}_d0-9"
      st=$(state runs/frengression "$r" model.pt)
      if [ "$st" != todo ]; then echo "skip  $r ($st)"; else
        launch "$r" "$RLOG/$r.log" micromamba run -n frugal-flows-frengression python exp_frengression_recovery.py --all-digits \
          --preset "$preset" --seed-assign "$k" --seed-fit "$fs" --size 8 --seed-data 101 --num-iters 5000 --threads 4 \
          --runs-root runs/frengression --wandb --wandb-group grid_8x8_alldigits_frengression
      fi
    done
  done
done
wait
# fits started by the first launcher are not children of this one: wait for them too
while pgrep -u "$(id -u)" -f "grid-8x8-placeholder-never-matches" >/dev/null || [ -n "$(ls -A $LOCK 2>/dev/null)" ]; do sleep 60; done
micromamba run -n frugal-flows python run_index.py
micromamba run -n frugal-flows python run_index.py --baselines
micromamba run -n frugal-flows python check_runs.py --no-wandb | tail -1
echo "=== grid 8x8 all digits end $(date -u +%FT%TZ) ==="
