#!/usr/bin/env bash
# Trajectory follow-up (2026-09-27): see toy_trajectory.py.
cd "$(dirname "$0")/../../../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
D=validation/morphomnist/runs/leftover_confounding; SD=validation/morphomnist/scripts/leftover_confounding
for seed in 1 2 3 4; do
  ( micromamba run -n frugal-flows python $SD/toy_trajectory.py --n 5000 --seed $seed --epochs 800 --every 5 \
      --out $D/trajectory/n5000_seed$seed.json > $D/trajectory/n5000_seed$seed.log 2>&1; echo "end n5000 seed$seed rc=$?" ) &
done
for seed in 1 2; do
  ( micromamba run -n frugal-flows python $SD/toy_trajectory.py --n 50000 --seed $seed --epochs 150 --every 2 \
      --out $D/trajectory/n50000_seed$seed.json > $D/trajectory/n50000_seed$seed.log 2>&1; echo "end n50000 seed$seed rc=$?" ) &
done
wait; echo "=== trajectory done ==="
