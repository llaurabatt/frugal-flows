#!/usr/bin/env bash
# Classical baselines (naive, IPW, OLS, AIPW, oracle IPW) on the 60 all-digits 8x8 datasets of the paper
# grid (E1-E6 x assignment seeds 1..10), 2026-10-01. Seconds each; pinned to cores 235-239.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
for k in 1 2 3 4 5 6 7 8 9 10; do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate; do
    taskset -c 235-239 micromamba run -n frugal-flows python baselines.py --all-digits --preset "$preset" --size 8 --seed-data 101 --seed-assign "$k" --no-plots 2>&1 | grep -E "^  (ols|aipw)|Error|Traceback" | sed "s/^/$preset sa$k /"
  done
done
micromamba run -n frugal-flows python run_index.py --baselines
echo "=== baselines grid end $(date -u +%FT%TZ) ==="
