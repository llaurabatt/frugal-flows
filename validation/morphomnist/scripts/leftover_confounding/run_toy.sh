#!/usr/bin/env bash
# Toy check of the leftover confounding (2026-09-27): see toy_gaussian.py.
cd "$(dirname "$0")/../../../.."   # repo root
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
D=validation/morphomnist/runs/leftover_confounding; SD=validation/morphomnist/scripts/leftover_confounding
for n in 50000 5000; do for a in 2 0; do for s in 1 0.25; do for seed in 1 2 3 4; do
  out=$D/results/a${a}_sigma${s}_n${n}_seed${seed}.json
  [ -f "$out" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 16 ]; do sleep 5; done
  ( micromamba run -n frugal-flows python $SD/toy_gaussian.py --a $a --sigma $s --n $n --seed $seed --out $out > $D/results/${out##*/}.log 2>&1; echo "end $out rc=$?" ) &
done; done; done; done
wait; echo "=== toy done ==="
