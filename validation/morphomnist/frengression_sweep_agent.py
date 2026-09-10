"""W&B sweep entry point for ``exp_frengression_recovery``."""

from __future__ import annotations

import os
import sys
from dataclasses import fields

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import exp_frengression_recovery as E


def config_from_mapping(mapping) -> E.Config:
    """Translate W&B's underscore keys and combined seed into a Config."""
    cfg_in = dict(mapping)
    if "seed" in cfg_in:
        seed = int(cfg_in.pop("seed"))
        cfg_in.setdefault("seed_data", seed)
        cfg_in.setdefault("seed_fit", seed)

    names = {field.name for field in fields(E.Config)}
    unknown = sorted(set(cfg_in) - names)
    if unknown:
        print(f"frengression_sweep_agent: ignoring keys not in Config: {unknown}")

    values = {key: value for key, value in cfg_in.items() if key in names}
    values["wandb"] = True
    return E.Config(**values)


def main():
    import wandb

    wandb.init()
    E.run_one(config_from_mapping(wandb.config))


if __name__ == "__main__":
    main()
