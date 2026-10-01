#!/usr/bin/env bash
# Overnight batch.
# Wave 0: the five 8x8 MLP fits, run together (light, fast).
# Wave 1: the three transformer setups at fit seed 101, with the machine to themselves.
# Wave 2: the same three at fit seed 102.
# Every run carries patience 30, a 1000-epoch cap and a 2-hour wall cap, and records
# which of the three ended it.
set -u
cd /home/llaurabat/ff-project/frugal-flows/validation/morphomnist
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
SP=/tmp/claude-2002/-home-llaurabat-ff-project/7ae089eb-8ecf-45ac-98e7-c90c357166d0/scratchpad
Q=$SP/queue.log
: > $Q
echo "=== QUEUE START $(date -u +%FT%TZ) ===" >> $Q

run () {
  local tag="$1_s$2"
  ( echo "START $tag $(date -u +%T)" >> $Q
    micromamba run -n frugal-flows python $SP/overnight.py "$1" "$2" > $SP/run_$tag.log 2>&1
    echo "END   $tag rc=$? $(date -u +%T)" >> $Q ) &
}

echo "--- wave 0: 8x8 MLP ---" >> $Q
run zero_mlp_8 101
run zero_mlp_8 102
run ref_mlp_8  102
run sep_mlp_8  101
run sep_mlp_8  102
wait
echo "--- wave 1: transformer, fit seed 101 ---" >> $Q
run std_trf_16   101
run ff_trf_16_e1 101
run ff_trf_16_e2 101
wait
echo "--- wave 2: transformer, fit seed 102 ---" >> $Q
run std_trf_16   102
run ff_trf_16_e1 102
run ff_trf_16_e2 102
wait
echo "=== QUEUE DONE $(date -u +%FT%TZ) ===" >> $Q
