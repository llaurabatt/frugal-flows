"""Re-read the effect map of every saved Frengression run at 5000 paired draws (2026-10-03).

Frengression runs before 2026-10-03 read the effect out with 50000 draws, the frugal flow with 5000.
This redraws each run's effect map from its saved weights with 5000 draws (same seed, same pairing)
and writes readout_mc5000.npz next to it; exp_frengression_recovery.effect_map then returns it.
Runs already at 5000, or already re-read, are skipped. Rerun after new 50000-draw runs finish.

    python scripts/frengression/reread_mc5000.py
"""
import glob
import json
import os
import sys

MM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, MM)
os.chdir(MM)
import exp_frengression_recovery as F  # noqa: E402

N = 5000
done = skipped = 0
for d in sorted(glob.glob(os.path.join(MM, "runs", "frengression", "*_frengression_*_d0-9_*/"))):
    if not (os.path.exists(d + "metrics.json") and os.path.exists(d + "model.pt")):
        continue
    used = json.load(open(d + "config.json"))["config"].get("n_mc")
    if used == N or os.path.exists(d + F.READOUT_FILE.format(n=N)):
        skipped += 1
        continue
    F.reread_effect_map(d, N)
    done += 1
    print(f"re-read {os.path.basename(d.rstrip('/'))}", flush=True)
print(f"=== re-read {done}, skipped {skipped} ===")
