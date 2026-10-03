#!/usr/bin/env bash
# naive / IPW / OLS / AIPW / oracle IPW on the 16x16 all-digits grid's 20 datasets
# (E1, E2, E4, E6 x assignment seeds 1-5; grid_16x16_alldigits.sh). Run at low priority, unpinned,
# while the grid's fits hold all slots.
set -u
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
for k in 1 2 3 4 5; do
  for preset in exp1_rct_homogeneous exp2_confounded_homogeneous exp4_covariate_cate exp6_spatial_cate; do
    nice -n 19 micromamba run -n frugal-flows python baselines.py --all-digits --preset "$preset" --size 16 --seed-data 101 --seed-assign "$k" --no-plots 2>&1 | grep -E "^  (ols|aipw)|Error|Traceback" | sed "s/^/$preset sa$k /"
  done
done
micromamba run -n frugal-flows python run_index.py --baselines
echo "=== baselines 16x16 grid end $(date -u +%FT%TZ) ==="
