#!/bin/bash
# Overnight Gaussian-scale paper grid (docs/gaussian_scale/OVERNIGHT_PREREG.md), 2026-10-02.
# 8 slots x 1 thread, round priority A, C, B (fit seeds 1001,1002), D; G-LT with --shift-lr-mult 10
# (pre-launch check: the naive-start shift did not move enough at lr 1e-3); no new fit after 05:30;
# analysis after each round; S11 resumed when the queue ends. Re-running continues (skip-done).
# Usage: nohup validation/morphomnist/scripts/exp_ate_recovery/grid_gaussian_scale.sh > ~/work/halo-runs/gsw/grid.out 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${GSW_LOG:-$HOME/work/halo-runs/gsw}"
mkdir -p "$LOG" "$HOME/work/halo-runs/S11/_logs"
ORIG_PATH="$PATH"
S11="cd /Users/danielmanela/work/frugal-flows-gauss/validation/diagnostics_univariate && PYTHONPATH=/Users/danielmanela/work/frugal-flows-gauss nohup /Users/danielmanela/micromamba/bin/python3 driver.py --blocks article1,misspec --resume --conc 8 --conc-shared 2 > $HOME/work/halo-runs/S11/_logs/_driver_phase1b.out 2>&1"
AFTER="env -i HOME=$HOME USER=$USER PATH=$ORIG_PATH /bin/bash -c '$S11'"
exec caffeinate -i "${MAMBA_EXE:-$HOME/.local/bin/micromamba}" run -n frugal-flows-e2w python "$HERE/grid_gaussian_scale.py" \
  --slots "${GSW_SLOTS:-8}" --logdir "$LOG" --order "${GSW_ORDER:-ACBD}" --b-seeds "${GSW_B_SEEDS:-1001,1002}" \
  --shift-lr-mult 10 --start-cutoff "${GSW_CUTOFF:-05:30}" --after "${GSW_AFTER:-$AFTER}"
