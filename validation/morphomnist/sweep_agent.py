"""Entry point for wandb sweeps over ``exp_ate_recovery``.

A sweep agent cannot call ``exp_ate_recovery.py`` directly: wandb emits
hyperparameters as ``--learning_rate=0.01`` (underscores) while the CLI defines
``--learning-rate`` (hyphens), and argparse will not accept the underscore form.
Rather than add an alias for every flag, this wrapper reads ``wandb.config`` and
builds the ``Config`` object itself.

It also does two things the raw CLI cannot:

* ties ``seed_data`` and ``seed_fit`` together when the sweep sets a single
  ``seed``, so each trial is a fresh dataset AND a fresh initialisation rather
  than one held fixed;
* attaches to the run the agent already opened instead of starting a second one.

Usage:
    wandb sweep --entity YOUR_TEAM --project morphomnist-ate sweeps/loctrans_ate_init_lr.yaml
    wandb agent YOUR_TEAM/morphomnist-ate/<sweep-id>

⚠️ Sweeps here optimise ``ate_mae``, which is computed from the ground truth.
That is legitimate ONLY on tuning instances that are disjoint from the ones you
report. Keep tuning seeds (101+) and reporting seeds (1-5) separate, and keep
tuning sweeps in a separate project or group from reported results.
"""

from __future__ import annotations

import os
import sys
from dataclasses import fields

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import wandb

import exp_ate_recovery as E


def main():
    wandb.init()
    cfg_in = dict(wandb.config)

    # One `seed` in the sweep space drives both RNGs, so a trial varies the
    # dataset and the fit together and the spread covers total variance.
    if "seed" in cfg_in:
        seed = int(cfg_in.pop("seed"))
        cfg_in.setdefault("seed_data", seed)
        cfg_in.setdefault("seed_fit", seed)

    names = {f.name for f in fields(E.Config)}
    unknown = sorted(set(cfg_in) - names)
    if unknown:
        print(f"sweep_agent: ignoring keys not in Config: {unknown}")

    cfg = E.Config(**{k: v for k, v in cfg_in.items() if k in names},
                   **{"wandb": True})
    E.jax.config.update("jax_enable_x64", cfg.x64)
    os.makedirs(E.RUNS_ROOT, exist_ok=True)
    E.run_one(cfg)


if __name__ == "__main__":
    main()
