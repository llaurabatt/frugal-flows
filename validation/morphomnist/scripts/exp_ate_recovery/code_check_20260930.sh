#!/usr/bin/env bash
# Code check before the paper runs (2026-09-30): self-test, index rebuild, full run check with
# wandb, and a reproduction of an old plain E2 fit (sa1, seed 1001, 2026-09-27) with today's code.
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
PY="micromamba run -n frugal-flows python"
echo "=== selftest"; $PY exp_ate_recovery.py --selftest 2>&1 | grep -v Warn | tail -2
echo "=== index"; $PY run_index.py 2>&1 | tail -2
echo "=== check_runs (with wandb)"; $PY check_runs.py 2>&1 | grep -v -E "^wandb:|Warn" | tail -8
echo "=== reproduction"; rm -rf /tmp/claude-2002/-home-llaurabat-ff-project/7ae089eb-8ecf-45ac-98e7-c90c357166d0/scratchpad/repro0930
$PY exp_ate_recovery.py --preset exp2_confounded_homogeneous --seed-assign 1 --seed-fit 1001 --model ff --arm flexible_continuous --conditioner mlp --size 8 --digit 0 --seed-data 101 --nn-width 48 --nn-depth 1 --flow-layers 4 --rqs-knots 8 --learning-rate 0.001 --batch-size 100 --max-epochs 1000 --max-patience 30 --n-mc 5000 --copula-nn-width 16 --runs-root /tmp/claude-2002/-home-llaurabat-ff-project/7ae089eb-8ecf-45ac-98e7-c90c357166d0/scratchpad/repro0930 > /tmp/claude-2002/-home-llaurabat-ff-project/7ae089eb-8ecf-45ac-98e7-c90c357166d0/scratchpad/repro0930.log 2>&1
$PY -c "
import glob,numpy as np,json
old=glob.glob('runs/exp_ate_recovery/2026-09-27T01-06-45Z_ff_e2_flexcont_sa1_lr0.001_copw16_k64_s1001_d0_*/')[0]; new=glob.glob('/tmp/claude-2002/-home-llaurabat-ff-project/7ae089eb-8ecf-45ac-98e7-c90c357166d0/scratchpad/repro0930/*/')[0]
a,b=np.load(old+'arrays.npz'),np.load(new+'arrays.npz'); mo,mn=json.load(open(old+'metrics.json')),json.load(open(new+'metrics.json'))
print('max |tau diff| %.2e | best epoch %s %s | val %.4f %.4f'%(np.abs(a['tau_hat']-b['tau_hat']).max(),mo['best_epoch'],mn['best_epoch'],mo['best_val_loss'],mn['best_val_loss']))" 2>&1 | grep -v Warn
echo "=== done $(date -u +%FT%TZ)"
