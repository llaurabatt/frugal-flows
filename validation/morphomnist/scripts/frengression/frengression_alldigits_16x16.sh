#!/usr/bin/env bash
# Frengression, all ten digits, 16x16 (2026-09-30): reference for alldigits_16x16.sh.
# E2 and E1, datasets 1..3, fit seed = dataset seed, adapter defaults (5000 iterations etc.).
set -u
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
PY="micromamba run -n frugal-flows-frengression python"
ROOT="runs/frengression"; LOGDIR="$ROOT/_scripts/alldigits_16x16_logs"; mkdir -p "$LOGDIR"
COMMON="--all-digits --size 16 --seed-data 101 --num-iters 5000 --wandb --wandb-group alldigits_16x16_frengression --runs-root $ROOT"
declare -A SHORT=([exp1_rct_homogeneous]=e1 [exp2_confounded_homogeneous]=e2)
echo "=== frengression all digits 16x16 start $(date -u +%FT%TZ) ==="
for k in 1 2 3; do for preset in exp2_confounded_homogeneous exp1_rct_homogeneous; do
  name="frengression_${SHORT[$preset]}_sa${k}_k256_s${k}_d0-9"
  if grep -l '"status": "ok"' "$ROOT"/*_"${name}"_*/metrics.json >/dev/null 2>&1; then echo "skip  $name (done)"; continue; fi
  echo "start $name $(date -u +%FT%TZ)"
  ( $PY exp_frengression_recovery.py --preset "$preset" --seed-assign "$k" --seed-fit "$k" $COMMON > "$LOGDIR/$name.log" 2>&1
    echo "end   $name rc=$? $(date -u +%FT%TZ)" ) &
  sleep 20
done; done
wait
micromamba run -n frugal-flows python run_index.py --baselines
echo "=== frengression all digits 16x16 end $(date -u +%FT%TZ) ==="
