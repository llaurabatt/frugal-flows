Overnight batch of 2026-09-15, produced by overnight.py (queued by queue.sh), both copied here,
and the standalone margin fit of 2026-09-14, produced by margin_only.py (also here). Their run
folders live one level up, alongside the exp_ate_recovery.py runs, and are the 12 folders
named margin_*, margin_zero_*, margin_sep_* and ff_*-trf_* (wandb groups overnight_batch and
margin_only). Each overnight.py run folder holds:
  config.json   config copied from the wandb run on 2026-09-18 (model, preset, size, seeds, caps)
  wandb.json    id, name, url of the wandb run
  result.json   fit-time config (setup, seed_fit, lr, margin sizes) and all metrics
  result.npz    tau_hat, ATE, e0_map, e1_map, train/val indices, per-epoch history
  best.eqx / last.eqx   checkpoints (best_arm0/1, last_arm0/1 for the two-flow runs)
  split.npz     the train/validation split
  plots/        maps.png and curves.png, drawn from result.npz after the fit
_uz_exp1_k256.npy and _uz_exp2_k256.npy are the thickness-marginal quantiles the four ff_
runs consumed (fitted once per preset by get_independent_quantiles and cached).
Paths hardcoded inside overnight.py and queue.sh point at the session scratchpad the batch
was run from, not at this folder.
