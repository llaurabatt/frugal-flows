#!/usr/bin/env bash
# Reruns analyse_grid_8x8_alldigits.py every 2 h while the grid runs, and once after it ends
# (results: _scripts/analysis/grid_8x8_alldigits.md; history in grid_8x8_alldigits_watch.log).
cd "$(dirname "$0")/../.."
DRV=runs/exp_ate_recovery/_scripts/grid_8x8_alldigits_driver.log
run() { echo "=== analysis $(date -u +%FT%TZ): ff done $(grep -c 'end   ff_.*rc=0' $DRV), frengression done $(grep -c 'end   frengression_.*rc=0' $DRV), failed $(grep -c 'rc=[1-9]' $DRV)"
        taskset -c 230-234 micromamba run -n frugal-flows python scripts/exp_ate_recovery/analyse_grid_8x8_alldigits.py 2>&1 | grep -v Warn | head -12; }
while ! grep -q "grid 8x8 all digits end" $DRV; do run; sleep 7200; done
run; echo "=== watcher done $(date -u +%FT%TZ)"
