#!/usr/bin/env bash
# Frengression at 16x16 (2026-09-30): reference for the flow's 16x16 confirmation. E1 and E2,
# datasets 1..3, fit seeds k and 1001 (the flow's pairs), adapter defaults (5000 iterations etc.).
set -u
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
PY="micromamba run -n frugal-flows-frengression python"
ROOT="runs/frengression"; LOGDIR="$ROOT/_scripts/grid_16x16_logs"; mkdir -p "$LOGDIR"
COMMON="--size 16 --digit 0 --seed-data 101 --num-iters 5000 --wandb --wandb-group confirm_16x16_frengression --runs-root $ROOT"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== frengression 16x16 start $(date -u +%FT%TZ) ==="
for k in 1 2 3; do for fs in "$k" 1001; do for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
  name="frengression_${SHORT[$preset]}_sa${k}_k256_s${fs}_d0"
  if grep -l '"status": "ok"' "$ROOT"/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
  echo "start $name $(date -u +%FT%TZ)"
  ( $PY exp_frengression_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$fs" $COMMON > "$LOGDIR/$name.log" 2>&1
    echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  sleep 15
done; done; done
wait
micromamba run -n frugal-flows python run_index.py --baselines
echo "=== frengression 16x16 end $(date -u +%FT%TZ) ==="
