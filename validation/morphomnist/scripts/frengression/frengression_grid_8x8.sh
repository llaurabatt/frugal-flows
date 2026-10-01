#!/usr/bin/env bash
# Frengression on the 180 datasets of the 8x8 assignment-bootstrap grid (2026-09-21),
# as a sixth baseline next to naive / OLS / IPW / AIPW / oracle IPW. Launched 2026-09-25.
#
#   presets E1..E6  x  effect default 1.0  x  assignment seeds 1..20   (120 fits)
#   E1, E2, E4      x  effect 0            x  assignment seeds 1..20   ( 60 fits)
#
# Same datasets as the flow grid: 8x8 (K=64), digit 0, data seed 101, --seed-assign k; the
# fit seed is k too. Each run records dataset_id / data_hash / z_hash, and
# `run_index.py --baselines` puts it in runs/baselines/index.csv with method frengression,
# so it joins the flow runs and the other baselines on dataset_id.
# Model settings are the adapter's defaults (exp_frengression_recovery.Config: 5000
# iterations, lr 1e-3, hidden 100, 3 layers, noise_dim 64, per-pixel y scaling, 50000 MC
# draws, CPU, 4 torch threads). Runs in the frugal-flows-frengression environment
# (environment-frengression.yaml; frengression pinned to commit 8a09055).
# A cell whose folder already holds metrics.json with status ok is skipped, so the script
# can be relaunched after an interruption.
#
#   usage:  bash frengression_grid_8x8.sh                         # the real grid
#           PROTO=1 PROTO_ROOT=/scratch/dir bash frengression_grid_8x8.sh   # 2 cells, 50 iterations
set -u
cd "$(dirname "$0")/../.."          # validation/morphomnist
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
PY="micromamba run -n frugal-flows-frengression python"
GROUP=assignment_bootstrap_8x8
ROOT="runs/frengression"
LOGDIR="$ROOT/_scripts/grid_logs"; mkdir -p "$LOGDIR"
SEEDS="$(seq -s ' ' 1 20)"; ITERS=5000; PARALLEL=${PARALLEL:-12}
if [ "${PROTO:-0}" = "1" ]; then
  SEEDS="1"; ITERS=50; PARALLEL=2; GROUP="${GROUP}_proto"
  ROOT="${PROTO_ROOT:?set PROTO_ROOT to a scratch directory}"
  LOGDIR="$ROOT/grid_logs"; mkdir -p "$LOGDIR"
fi
COMMON="--size 8 --digit 0 --seed-data 101 --num-iters $ITERS --wandb --wandb-group $GROUP --runs-root $ROOT"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2 [exp3_confounded_heterogeneous]=e3
                  [exp4_covariate_cate]=e4 [exp5_quantile_effect]=e5 [exp6_spatial_cate]=e6)

run_cell () {   # preset effect(""|0) seed
  local preset=$1 effect=$2 k=$3 tag="" eff_arg=""
  if [ -n "$effect" ]; then tag="_effect${effect}"; eff_arg="--base-shift $effect"; fi
  local name="frengression_${SHORT[$preset]}${tag}_sa${k}_k64_s${k}_d0"
  if grep -l '"status": "ok"' "$ROOT"/*_"${name}"_*/metrics.json >/dev/null 2>&1; then
    echo "skip  $name (done)"; return
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 15; done
  echo "start $name  $(date -u +%FT%TZ)"
  ( $PY exp_frengression_recovery.py --preset "$preset" $eff_arg --seed-assign "$k" --seed-fit "$k" $COMMON \
      > "$LOGDIR/$name.log" 2>&1; echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
}

echo "=== frengression grid start $(date -u +%FT%TZ)  iters=$ITERS seeds=[$SEEDS] parallel=$PARALLEL ==="
for k in $SEEDS; do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous \
                exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate; do
    run_cell "$preset" "" "$k"
    case "$preset" in exp1_rct_homogeneous|exp2_confounded_homogeneous|exp4_covariate_cate)
      [ "${PROTO:-0}" = "1" ] || run_cell "$preset" 0 "$k" ;;
    esac
    [ "${PROTO:-0}" = "1" ] && [ "$preset" = exp2_confounded_homogeneous ] && break
  done
done
wait
echo "=== fits done $(date -u +%FT%TZ) ==="
if [ "${PROTO:-0}" != "1" ]; then
  micromamba run -n frugal-flows python run_index.py --baselines
fi
echo "=== frengression grid end $(date -u +%FT%TZ) ==="
