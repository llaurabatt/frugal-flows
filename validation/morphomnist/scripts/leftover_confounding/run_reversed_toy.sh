#!/usr/bin/env bash
# Plan step 3 (2026-09-30): reversed-copula arm on the K-pixel toy, paired with the current arm
# (results_multi) and frengression (results_frengression). K=8 and 16 at n=5923, seeds 1-4,
# rank-penalty weights 0, 10, 100; plus K=8 at n=50000 with weight 0 and 10. 32 fits.
cd "$(dirname "$0")/../../../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
D=validation/morphomnist/runs/leftover_confounding; SD=validation/morphomnist/scripts/leftover_confounding; R=$D/results_reversed; mkdir -p $R
run() { out=$R/$1.json; [ -f "$out" ] && return; shift
  ( micromamba run -n frugal-flows python $SD/toy_reversed.py "$@" --out $out > ${out%.json}.log 2>&1; echo "end $out rc=$? $(date -u +%FT%TZ)" ) &
  sleep 10; }
echo "=== reversed toy start $(date -u +%FT%TZ) ==="
for s in 1 2 3 4; do
  for w in 0 10 100; do
    run K8_n5923_w${w}_seed$s --K 8 --n 5923 --seed $s --rank-weight $w
    run K16_n5923_w${w}_seed$s --K 16 --n 5923 --seed $s --rank-weight $w
  done
  for w in 0 10; do run K8_n50000_w${w}_seed$s --K 8 --n 50000 --seed $s --rank-weight $w; done
done
wait; echo "=== reversed toy end $(date -u +%FT%TZ) ==="
