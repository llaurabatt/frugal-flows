#!/usr/bin/env bash
# Assignment-bootstrap grid at 8x8. First run 2026-09-19 (5 seeds, 70 fits); deleted and
# relaunched 2026-09-21 with 20 seeds and the copula + sample diagnostics (commit 0442682).
#
#   presets E1..E6  x  effect default 1.0  x  assignment seeds 1..20   -> ff     (120 fits)
#   E1, E2, E4      x  effect 0            x  assignment seeds 1..20   -> ff     ( 60 fits)
#   E1              x  effect {1.0, 0}     x  assignment seeds 1..20   -> margin ( 40 fits)
#   then baselines.py --from-index on every dataset those fits used     (180 datasets)
#
# With the effect off, E2/E3/E5 build one dataset (Z = thickness) and E4/E6 another
# (Z = thickness + brightness), so E3/E5/E6 zero-effect fits would repeat E2's / E4's
# byte for byte (they did in the first run) and are not queued.
#
# Fixed for every fit: 8x8 (K=64), digit 0, data seed 101 (images and their noise never
# change), MLP conditioner, width 48, depth 1, 4 flow layers, 8 knots, lr 0.01, batch 100,
# copula 50/1/4/8 (the script defaults), epoch cap 1000 with patience 30, 5000 MC draws.
# Replicate k uses --seed-assign k --seed-fit k: a fresh assignment AND a fresh initialisation.
# Every fit goes through exp_ate_recovery.py, so folder, config, wandb run and index row are
# the standard ones. A cell whose folder already exists (with metrics.json) is skipped, so
# the script can be re-launched after an interruption. Six fits run at a time (240 cores;
# one fit uses ~48 threads).
#
#   usage:  bash assignment_grid_8x8.sh                       # the real grid
#           PROTO=1 bash assignment_grid_8x8.sh               # 2 cells, 3-epoch cap, scratch root
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
GROUP=assignment_bootstrap_8x8
LOGDIR="runs/exp_ate_recovery/_scripts/grid_logs"; mkdir -p "$LOGDIR"
SEEDS="$(seq -s ' ' 1 20)"; CAP=1000; ROOT_ARG=""; PARALLEL=6
if [ "${PROTO:-0}" = "1" ]; then
  SEEDS="1"; CAP=3; PARALLEL=1; GROUP="${GROUP}_proto"     # separate wandb group, deleted afterwards
  ROOT_ARG="--runs-root ${PROTO_ROOT:?set PROTO_ROOT to a scratch directory}"
  LOGDIR="${PROTO_ROOT}/grid_logs"; mkdir -p "$LOGDIR"
fi
COMMON="--arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101
        --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.01 --batch-size 100
        --max-epochs $CAP --max-patience 30 --n-mc 5000 --wandb --wandb-group $GROUP $ROOT_ARG"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2 [exp3_confounded_heterogeneous]=e3
                  [exp4_covariate_cate]=e4 [exp5_quantile_effect]=e5 [exp6_spatial_cate]=e6)

run_cell () {   # model preset effect(""|0) seed
  local model=$1 preset=$2 effect=$3 k=$4 tag="" eff_arg=""
  if [ -n "$effect" ]; then tag="_effect${effect}"; eff_arg="--base-shift $effect"; fi
  local name="${model}_${SHORT[$preset]}_flexcont${tag}_sa${k}_k64_s${k}_d0"
  local root="runs/exp_ate_recovery"; [ -n "$ROOT_ARG" ] && root="$PROTO_ROOT"
  if ls -d "$root"/*_"${name}"_*/metrics.json >/dev/null 2>&1; then
    echo "skip  $name (done)"; return
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
  echo "start $name  $(date -u +%FT%TZ)"
  ( $PY exp_ate_recovery.py --preset "$preset" --model "$model" $eff_arg --seed-assign "$k" --seed-fit "$k" $COMMON \
      > "$LOGDIR/$name.log" 2>&1; echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
}

echo "=== grid start $(date -u +%FT%TZ)  cap=$CAP seeds=[$SEEDS] parallel=$PARALLEL ==="
for k in $SEEDS; do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous \
                exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate; do
    for effect in "" 0; do
      # zero-effect E3/E5 repeat E2, E6 repeats E4 (same data, same fit seed): not queued
      case "$effect:$preset" in
        0:exp3_confounded_heterogeneous|0:exp5_quantile_effect|0:exp6_spatial_cate) continue ;;
      esac
      run_cell ff "$preset" "$effect" "$k"
      [ "${PROTO:-0}" = "1" ] && [ "$preset" != exp1_rct_homogeneous ] && continue
      [ "$preset" = exp1_rct_homogeneous ] && run_cell margin "$preset" "$effect" "$k"
    done
    [ "${PROTO:-0}" = "1" ] && break
  done
done
wait
echo "=== fits done $(date -u +%FT%TZ) ==="
if [ -z "$ROOT_ARG" ]; then
  $PY baselines.py --from-index > "$LOGDIR/baselines.log" 2>&1; echo "baselines rc=$?"
  $PY run_index.py; $PY run_index.py --baselines; $PY check_runs.py --no-wandb | tail -1
fi
echo "=== grid end $(date -u +%FT%TZ) ==="
