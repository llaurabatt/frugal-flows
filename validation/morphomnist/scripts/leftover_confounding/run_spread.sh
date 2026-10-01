#!/usr/bin/env bash
# Spread check (2026-09-30): toy_spread.py, K=8 and 16 at n=5923 plus K=8 at n=50000, seeds 1-4.
cd "$(dirname "$0")/../../../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
D=validation/morphomnist/runs/leftover_confounding; SD=validation/morphomnist/scripts/leftover_confounding; R=$D/results_spread; mkdir -p $R
run() { out=$R/$1.json; [ -f "$out" ] && return; shift
  ( micromamba run -n frugal-flows python $SD/toy_spread.py "$@" --out $out > ${out%.json}.log 2>&1; echo "end $out rc=$? $(date -u +%FT%TZ)" ) &
  sleep 10; }
echo "=== spread start $(date -u +%FT%TZ) ==="
for s in 1 2 3 4; do run K8_n5923_seed$s --K 8 --n 5923 --seed $s; run K16_n5923_seed$s --K 16 --n 5923 --seed $s; run K8_n50000_seed$s --K 8 --n 50000 --seed $s; done
wait; echo "=== spread end $(date -u +%FT%TZ) ==="
