#!/usr/bin/env bash
# Training-regime test (2026-09-30): toy_regime.py, current arm without early stopping.
# Full batch: 5000 steps, read-out every 250 (20 read-outs). Batches of 100: 400 epochs, every 20.
# K=8 and 16, n=5923, seeds 1-4. 16 fits.
cd "$(dirname "$0")/../../../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
D=validation/morphomnist/runs/leftover_confounding; SD=validation/morphomnist/scripts/leftover_confounding; R=$D/results_regime; mkdir -p $R
run() { out=$R/$1.json; [ -f "$out" ] && return; shift
  ( micromamba run -n frugal-flows python $SD/toy_regime.py "$@" --out $out > ${out%.json}.log 2>&1; echo "end $out rc=$? $(date -u +%FT%TZ)" ) &
  sleep 10; }
echo "=== regime start $(date -u +%FT%TZ) ==="
for s in 1 2 3 4; do for K in 8 16; do
  run K${K}_full_seed$s --K $K --seed $s --batch 0 --epochs 5000 --every 250
  run K${K}_b100_seed$s --K $K --seed $s --batch 100 --epochs 400 --every 20
done; done
wait; echo "=== regime end $(date -u +%FT%TZ) ==="
