#!/usr/bin/env bash
# Builds a LOCAL branch multi-y-e2-weights = multi-y (at build time) + one commit with the E2 run
# folders of the 8x8 all-digits paper grid: weights, config.json, metrics.json, wandb.json only
# (flow: datasets 1-10 x 5 fit seeds; frengression: the dataset's own seed). Waits until all 60
# E2 cells are finished. No checkout (temporary index), so multi-y and the working tree are untouched.
# Does NOT push: run `git push origin multi-y-e2-weights` yourself.
set -u
cd "$(dirname "$0")/../../../.."          # repo root
MM=validation/morphomnist
count() {   # finished E2 cells with weights: flow (preset e2, all seeds) + frengression own seed
  local f=0 r=0 k fs d
  for k in 1 2 3 4 5 6 7 8 9 10; do
    for fs in $k 1001 1002 1003 1004; do
      for d in $MM/runs/exp_ate_recovery/*_ff_e2_flexcont_sa${k}_lr0.001_copw16_k64_s${fs}_d0-9_*/; do
        [ -f "$d/model.eqx" ] && [ -f "$d/metrics.json" ] && { f=$((f+1)); break; }; done
    done
    for d in $MM/runs/frengression/*_frengression_e2_sa${k}_k64_s${k}_d0-9_*/; do
      [ -f "$d/model.pt" ] && [ -f "$d/metrics.json" ] && { r=$((r+1)); break; }; done
  done
  echo "$f $r"
}
while true; do
  read f r < <(count); echo "$(date -u +%FT%TZ) E2 finished: flow $f/50, frengression $r/10"
  [ "$f" -ge 50 ] && [ "$r" -ge 10 ] && break; sleep 600
done
files=()
for k in 1 2 3 4 5 6 7 8 9 10; do
  for fs in $k 1001 1002 1003 1004; do
    for d in $MM/runs/exp_ate_recovery/*_ff_e2_flexcont_sa${k}_lr0.001_copw16_k64_s${fs}_d0-9_*/; do
      [ -f "$d/model.eqx" ] || continue
      for x in model.eqx config.json metrics.json wandb.json; do [ -f "$d/$x" ] && files+=("${d%/}/$x"); done; break
    done
  done
  for d in $MM/runs/frengression/*_frengression_e2_sa${k}_k64_s${k}_d0-9_*/; do
    [ -f "$d/model.pt" ] || continue
    for x in model.pt config.json metrics.json wandb.json; do [ -f "$d/$x" ] && files+=("${d%/}/$x"); done; break
  done
done
echo "${#files[@]} files, $(du -ch "${files[@]}" | tail -1 | cut -f1)"
export GIT_INDEX_FILE=$(mktemp -u /tmp/e2idx.XXXX)
git read-tree multi-y
git add -f "${files[@]}"
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p multi-y -m "data: E2 fits of the 8x8 all-digits paper grid (weights, config, metrics, wandb link)

Flow: datasets 1-10 x fit seeds {k,1001..1004} (50 fits, model.eqx); frengression: one fit per
dataset, seed k (10 fits, model.pt). Reload with exp_ate_recovery.load_model / exp_frengression_recovery.load_model;
datasets are rebuilt from config.json by dataset_store. Results branch, not for development.")
git branch -f multi-y-e2-weights "$commit"
rm -f "$GIT_INDEX_FILE"
echo "$(date -u +%FT%TZ) built branch multi-y-e2-weights at $commit on top of $(git rev-parse --short multi-y); push with: git push origin multi-y-e2-weights"
