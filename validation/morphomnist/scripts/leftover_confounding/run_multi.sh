#!/usr/bin/env bash
# K-pixel toy grid (2026-09-28): see toy_multi.py.
cd "$(dirname "$0")/../../../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
D=validation/morphomnist/runs/leftover_confounding; SD=validation/morphomnist/scripts/leftover_confounding; R=$D/results_multi; mkdir -p $R
run() { out=$R/$1.json; [ -f "$out" ] && return; shift
  ( micromamba run -n frugal-flows python $SD/toy_multi.py "$@" --out $out > ${out%.json}.log 2>&1; echo "end $out rc=$?" ) & }
for seed in 1 2 3 4; do
  for K in 8 16; do for n in 5923 50000; do run K${K}_n${n}_a2_seed${seed} --K $K --n $n --seed $seed; done; done
  for n in 5923 50000; do run K8_n${n}_a0_seed${seed} --K 8 --n $n --seed $seed --a 0; done
done
wait; echo "=== multi done ==="
