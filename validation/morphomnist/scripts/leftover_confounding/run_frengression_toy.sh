#!/usr/bin/env bash
# Plan step 1 (2026-09-28): frengression on the K-pixel toy, paired with toy_multi.py's
# frugal-flow fits (same data per seed). K=8 at n=5923 and 50000, K=16 at n=5923, seeds 1-4.
cd "$(dirname "$0")/../../../.."
D=validation/morphomnist/runs/leftover_confounding; SD=validation/morphomnist/scripts/leftover_confounding; R=$D/results_frengression; mkdir -p $R
run() { out=$R/$1.json; [ -f "$out" ] && return; shift
  ( micromamba run -n frugal-flows-frengression python $SD/toy_frengression.py "$@" --threads 8 --out $out > ${out%.json}.log 2>&1; echo "end $out rc=$? $(date -u +%FT%TZ)" ) &
  sleep 15; }
echo "=== frengression toy start $(date -u +%FT%TZ) ==="
for seed in 1 2 3 4; do
  run K8_n5923_a2_seed$seed --K 8 --n 5923 --seed $seed
  run K16_n5923_a2_seed$seed --K 16 --n 5923 --seed $seed
  run K8_n50000_a2_seed$seed --K 8 --n 50000 --seed $seed
done
wait; echo "=== frengression toy end $(date -u +%FT%TZ) ==="
