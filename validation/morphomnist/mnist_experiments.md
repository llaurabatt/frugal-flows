# MorphoMNIST experiment plan (ICLR)

What to run, what to put in the paper, and what it can honestly claim.

Run everything from `validation/morphomnist/`.

---

## The runs

### Stage 0 — baselines 

```bash
python baselines.py --all --size 8 --seeds 1 2 3 4 5
```

Seconds. Writes `runs/baselines/baselines_k64_poly3.csv`. Re-run with
`--basis linear` and `--basis poly5` for an appendix sensitivity check — if OLS
collapses under `linear`, that is worth reporting, since it shows the baseline's
strength depends on getting the covariate basis right whereas the flow learns it.

### Stage 0b — the τ(u) baseline ⛔ **not implemented, blocking**

Needs writing before the main figure can be defended. IPW-weighted quantile
contrast per pixel, scored against `TAU_MARGINAL` (and `TAU_ANALYTIC` on E5)
using the same 40-bin grid as `frugal_flows.interventions.tau_curve`.

### Stage 1 — resolve the transformer (~1 h)

```bash
for p in exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate; do python exp_ate_recovery.py --preset $p --arm flexible_continuous --conditioner transformer --size 8 --learning-rate 5e-3 --max-patience 60 --max-epochs 200; done
```

In the last sweep 4 of 6 transformer cells never trained (`best_val_loss` +82,
−24.6, −34.1, +7.5 against a healthy band of −48 to −53), all early-stopping at
38–66 epochs. The two that converged were competitive with the mlp. Until this
is settled the transformer cannot appear in a results table.

Gate: if ≥2 cells still fail, the transformer goes in the appendix as a
negative result on optimisation stability, not in the main comparison.

### Stage 1b — hyperparameter tuning ⛔ **must precede Stage 2**

The Stage 0 comparison is **not yet fair**. OLS is closed-form and effectively
hyperparameter-free; the flow ran entirely at defaults nobody tuned. Some of the
2–4× gap is likely a tuning artefact rather than a property of the method.

#### The trap: what do you tune *on*?

**Not `ate_mae` on the reported runs** — that is computed from the ground truth,
i.e. the test metric. Tuning on it and then reporting it is selecting on the
answer.

**And the standard legitimate criterion is actively misleading here.** We
measured it: in 6 of 9 cells a *better* validation log-likelihood came with
*worse* ATE recovery (E3's transformer improved 5.7 nats while its MAE tripled).
So tuning on held-out likelihood may tune us away from the estimand.

That dissociation is itself a result worth reporting — likelihood-based model
selection is not aligned with causal recovery for this class of model — but it
means tuning needs a protocol, not just a grid.

#### Protocol

1. **Tune on disjoint DGP instances.** Select on `ate_mae` using seeds 101–105
   of E2 and E4, freeze the settings, then report on seeds 1–5 across all six
   presets. Using the truth for selection is legitimate provided the selection
   and reporting instances are disjoint — standard for synthetic benchmarks, and
   honest as long as it is stated.
2. **Tune the baselines on the same instances and criterion.** OLS currently
   uses `poly3` because it was picked by hand; its basis should be selected the
   same way IPW's regularisation is. Otherwise the unfairness has just moved.
3. **Report both selection criteria.** Show what you get tuning on validation
   likelihood versus on held-out `ate_mae`. If they disagree — and the evidence
   says they will — that gap is a finding, not an embarrassment.

#### Highest-leverage knobs, ranked by evidence

**`ate_init` (loctrans only).** Currently 0.5, while the on-support truth is 1.0
and off-support is 0. The failure signature matches exactly: `mae_ON` ≫
`mae_OFF` (0.245 vs 0.108 on E3) with correlation still 0.87–0.97 — it finds
*where* the effect is and undershoots *how big*. A parameter that has not
travelled far enough from its start. Prime suspect for loctrans's whole gap.

**Learning rate.** Decisive and **arm-dependent**: the transformer diverges at
1e-2 and trains at 5e-3; loctrans under-trains at 5e-3 and does better at 1e-2.
There is no single good value, so it must be tuned per arm and stated in the
paper.

**Epoch budget / patience.** 5 of 9 cells in the first sweep hit
`max_epochs=100` with the loss still falling, so those numbers are lower bounds
on achievable accuracy, not converged results.

**Capacity** (`nn_width`, `flow_layers`, `rqs_knots`, `nn_depth`) — untested,
probably second-order relative to the above. Worth only a coarse pass.

#### Commands — `location_translation`

Three things established:

- **Initialising at the true effect is the worst choice** (`ate_init=1.0` →
  0.535). 52 of 64 pixels have a true effect of exactly zero, so starting every
  pixel at 0 begins 81% of them already correct. That also makes 0.0 the
  principled *agnostic* default — a sparse-effect prior — rather than a value
  tuned toward the answer.
- **Validation loss cannot rank these.** The three `lr=1e-2` rows sit within
  1.2 nats of each other (−51.81, −50.61, −50.63) while `ate_mae` spans 0.0105
  to 0.5352 — a 50× range — and the *second-best* val loss belongs to the single
  worst estimator. This is the sharpest evidence for the selection problem above.
- **The two knobs interact.** `lr=3e-2` is best at `ate_init=0.5` but
  destabilises at `ate_init=0.0` (val −7.95). Coordinate descent would miss
  this, so these two must be gridded **jointly**.

**Joint grid.** 4 × 4 × 2 presets × 3 tuning seeds = 96 runs. At ~10 min/run
(200 epochs) that is ~16 h; drop to 2 seeds or one preset for a first pass.

```bash
for s in 101 102 103; do for p in exp2_confounded_homogeneous exp4_covariate_cate; do for ai in 0.0 0.25 0.5 1.0; do for lr in 3e-3 1e-2 2e-2 3e-2; do python exp_ate_recovery.py --preset $p --arm location_translation --size 8 --ate-init $ai --learning-rate $lr --max-epochs 200 --max-patience 60 --seed-data $s --seed-fit $s; done; done; done; done
```

A cheaper first pass — one preset, two seeds, 32 runs, ~5 h:

```bash
for s in 101 102; do for ai in 0.0 0.25 0.5 1.0; do for lr in 3e-3 1e-2 2e-2 3e-2; do python exp_ate_recovery.py --preset exp4_covariate_cate --arm location_translation --size 8 --ate-init $ai --learning-rate $lr --max-epochs 200 --max-patience 60 --seed-data $s --seed-fit $s; done; done; done
```

⚠️ The probe above is **one seed on one preset**. Nothing should be claimed from
it until replicated — it justifies running the grid, it is not itself a result.

#### Commands — `flexible_continuous / mlp`

Cheap enough to grid directly at K=64 (~3.5 min/run at 100 epochs). Coordinate
descent, not a full grid: fix everything, move one axis, keep the winner.

**A. Learning rate** — the decisive axis. 4 values × 2 presets × 3 tuning seeds
= 24 runs ≈ 1.5 h.

```bash
for s in 101 102 103; do for p in exp2_confounded_homogeneous exp4_covariate_cate; do for lr in 3e-3 5e-3 1e-2 2e-2; do python exp_ate_recovery.py --preset $p --arm flexible_continuous --conditioner mlp --size 8 --learning-rate $lr --max-epochs 200 --max-patience 60 --seed-data $s --seed-fit $s; done; done; done
```

**B. Capacity**, at whichever `lr` won A (substitute below). 4 combos × 2 presets
× 3 seeds = 24 runs ≈ 1.5 h.

```bash
for s in 101 102 103; do for p in exp2_confounded_homogeneous exp4_covariate_cate; do for fl in 4 8; do for w in 48 96; do python exp_ate_recovery.py --preset $p --arm flexible_continuous --conditioner mlp --size 8 --learning-rate LR_FROM_A --flow-layers $fl --nn-width $w --max-epochs 200 --max-patience 60 --seed-data $s --seed-fit $s; done; done; done; done
```

**C. Spline resolution**, a cheap third axis worth one pass (`--rqs-knots 8 12 16`).

#### Commands — `flexible_continuous / transformer`

**This arm cannot be gridded at full budget.** A K=64 run at 200 epochs is
~55 min, so the mlp grid above would be ~22 h for the transformer. It needs a
reduced design.

**A. Learning-rate scan at a truncated budget.** 60 epochs with patience 60 (so
nothing early-stops) is enough to see which rates are descending healthily —
recall the failures had `best_val_loss` above −35 while healthy runs sit near
−50. One preset, 2 seeds, 5 rates = 10 runs ≈ 3 h.

```bash
for s in 101 102; do for lr in 1e-3 3e-3 5e-3 1e-2 2e-2; do python exp_ate_recovery.py --preset exp4_covariate_cate --arm flexible_continuous --conditioner transformer --size 8 --learning-rate $lr --max-epochs 60 --max-patience 60 --n-mc 1000 --seed-data $s --seed-fit $s; done; done
```

Judge on `best_val_loss`, **not** on `ate_mae` at 60 epochs — the run is
deliberately truncated, so the recovery number is not yet meaningful. Keep the
two rates whose val loss is descending fastest and has not diverged.

**B. Architecture, at the two surviving rates.** The transformer has knobs the
mlp does not; `nn_width` must stay divisible by `nn_heads`. 8 combos × 2 seeds
= 16 runs ≈ 5 h.

```bash
for s in 101 102; do for lr in LR1_FROM_A LR2_FROM_A; do for d in 1 2; do for w in 48 96; do python exp_ate_recovery.py --preset exp4_covariate_cate --arm flexible_continuous --conditioner transformer --size 8 --learning-rate $lr --nn-depth $d --nn-width $w --nn-heads 4 --max-epochs 60 --max-patience 60 --n-mc 1000 --seed-data $s --seed-fit $s; done; done; done; done
```

**C. Confirm the finalist at full budget** on both tuning presets, 3 seeds
= 6 runs ≈ 5.5 h.

```bash
for s in 101 102 103; do for p in exp2_confounded_homogeneous exp4_covariate_cate; do python exp_ate_recovery.py --preset $p --arm flexible_continuous --conditioner transformer --size 8 --learning-rate LR_BEST --nn-depth D_BEST --nn-width W_BEST --max-epochs 200 --max-patience 60 --seed-data $s --seed-fit $s; done; done
```

Two economies worth knowing, and one that does **not** work.

`--n-mc 1000` during tuning cuts the read-out ~5×, but at 200 epochs the
read-out is only ~13% of a transformer run, so the fit budget is what matters.
Truncating epochs is the real saving — hence scanning at 60 and confirming at
200 rather than scanning at full length.

**Dropping to K=16 does not buy much.** Measured: a K=16 transformer run at 200
epochs takes 13.7 min against ~55 min at K=64 — only ~4×, because the epoch
count dominates rather than the pixel count. Combined with the open question of
whether the optimum transfers across `K`, tuning at low resolution is not worth
it for this arm.

⚠️ **`best_val_loss` is not comparable across `K`.** It is a log-density over
`K + dim(Z)` dimensions, so the healthy band moves with resolution: ~−50 at
K=64, but ~−7 at K=16 (measured: a K=16 transformer run at lr 5e-3 reached
−6.99 over 192 epochs and recovered `ate_mae` 0.029 — healthy, despite a number
that would look catastrophic at K=64). Always compare against **the mlp's val
loss on the same preset at the same K**, never against an absolute threshold.

⚠️ **Gate before spending stage C.** If no learning rate in A gets within ~10%
of the mlp's `best_val_loss` on the same preset and `K`, the transformer is not
merely mistuned and further architecture search is unlikely to help. At that
point it becomes an appendix result on optimisation stability, and the compute
is better spent on seeds for the mlp.

#### Cost and expectation

A full grid over everything × 6 presets × 5 seeds is unaffordable. Coordinate
descent over `{lr, ate_init, epochs}` on two presets × 3 seeds is ~60–80 runs,
so 4–6 h for the mlp and loctrans arms and considerably more with the
transformer. One overnight job — but it must run **before** Stage 2, or Stage 2
gets repeated.

Expect tuning to close much of the gap, especially for loctrans. It is unlikely
to overturn the OLS result: OLS is close to correctly specified for an
additive-in-logit effect and already sits at the oracle floor, so there is
little room above it to win. The Stage 0 reframing should survive, with the flow
landing nearer parity — which is a stronger sentence for the paper anyway
("competitive after tuning, plus τ(u)" beats "2–4× worse at defaults, plus τ(u)").

### Stage 2 — main results, at the Stage 1b settings (~7–8 h, overnight)

Vary **both** seeds together, so each replicate is a fresh dataset *and* a fresh
initialisation and the error bars cover total variance.

```bash
for s in 1 2 3 4 5; do for p in exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate; do python exp_ate_recovery.py --preset $p --arm flexible_continuous --conditioner mlp --size 8 --seed-data $s --seed-fit $s; done; done
```

```bash
for s in 1 2 3 4 5; do for p in exp1_rct_homogeneous exp2_confounded_homogeneous exp3_confounded_heterogeneous exp4_covariate_cate exp5_quantile_effect exp6_spatial_cate; do python exp_ate_recovery.py --preset $p --arm location_translation --size 8 --seed-data $s --seed-fit $s; done; done
```

30 runs each: mlp ≈ 70 min, loctrans ≈ 100 min. Add the transformer loop with
whatever Stage 1 settles on (≈ 5 h).

### Stage 3 — ablations (appendix, 3 seeds)

CATE shape — does a non-monotone or discontinuous CATE break ATE recovery?

```bash
for s in 1 2 3; do for h in linear cubic quadratic sine step; do python exp_ate_recovery.py --preset exp4_covariate_cate --h-shape $h --arm flexible_continuous --size 8 --seed-data $s --seed-fit $s; done; done
```

Confounding strength — the overlap curve and where it breaks:

```bash
for s in 1 2 3; do for b in 0.0 0.6 1.2 1.8 2.4; do python exp_ate_recovery.py --preset exp4_covariate_cate --ps-slope $b --arm flexible_continuous --size 8 --seed-data $s --seed-fit $s; done; done
```

Dimensionality — the scaling claim:

```bash
for s in 1 2 3; do for k in 4 8 16; do python exp_ate_recovery.py --preset exp4_covariate_cate --size $k --arm flexible_continuous --seed-data $s --seed-fit $s; done; done
```

---

## Paper outputs

### Main paper

**Table 1 — ATE recovery.** Rows = E1–E6, columns = {naive, IPW, OLS, AIPW,
FF-loctrans, FF-flexcont}, cells = `ate_mae` mean ± sd over 5 seeds, with the
oracle-IPW row as the floor. Framed as *competitiveness*, not victory.

**Figure 1 — what only the flow provides.** Two panels:
- (a) true ATE map / recovered / error, one preset (E4 or E6), showing the
  method works qualitatively;
- (b) **estimated `τ(u)` against the closed-form truth on E5**, with the
  IPW-quantile baseline overlaid once Stage 0b exists.

Panel (b) is the figure that earns the paper. E5 is the only experiment where
the target is analytic (`TAU_ANALYTIC`, matching the realised
`Q1(u) − Q0(u)` to 5e-15).

### Appendix

- **Design diagnostics table** — confounding strength, ATT−ATE separation,
  τ(u) paired-vs-marginal gap, noise floor per preset. This is what makes the
  benchmark credible as a *benchmark*; it belongs in the paper even though it
  contains no results.
- **On/off-support error split.** ~80% of the pooled MAE's weight is on pixels
  whose true effect is exactly zero, so "recovered the magnitude" and "kept the
  zeros at zero" are separate failures and should be shown separately.
- Per-preset τ(u) figures for E2 (truth exactly flat — the specification test)
  and E3/E6.
- The three ablations.
- Timing (`total_s` breakdown), and the transformer optimisation-failure table
  if unresolved.
- The **modulation-rank argument** (rank 1 for E4, 2 for E6, full for E3/E5) as
  a structural proof that a covariate-driven effect differs from an
  outcome-coupled one.
- The **covariate-status verification** — attributes are causal parents of the
  image, confirmed against DeepSCM's distributions, with the interventional
  invariance check.

---

## Open issues to fix before submission

1. **The floor quoted in the README and `experimental_setup.tex` is wrong.**
   Those say ~0.065, which is a *max-abs*; the like-for-like MAE floor is
   **~0.010–0.019**. Every "how close to optimal" statement needs rescaling.
2. **The τ(u) baseline (Stage 0b)** — blocking the main figure.
2b. **The flow is untuned (Stage 1b)** — the Stage 0 table compares tuned-by-
   construction baselines against flow defaults. No ATE comparison should be
   published before this is done, in either direction.
3. **Single seeds.** Nothing in the current archive is replicated. Stage 2 fixes
   this; no ranking should be quoted until it lands.
4. **The transformer is chaotically sensitive.** A 8e-5 *relative* perturbation
   to E3's data moved it from 0.018 to 0.167 while mlp and loctrans barely
   moved. Either it stabilises under Stage 1 or it is an appendix finding.
5. **E4/E6 do not demonstrate CATE recovery.** `train_frugal_flow` still refuses
   `u_z_hetero` outside the scalar gaussian arm, so those presets test whether
   the *marginal* ATE survives in the presence of CATE structure. State this
   plainly rather than letting a reviewer infer more.
6. **Ground truth is sample-relative** — the ATE *value* is stable across
   samples but the CATE *function* shifts, since the modulators use within-sample
   ranks. Fine for ATE recovery; needs the population-centering option before any
   CATE claim.

---

## Suggested reframing of the contribution

Given Stage 0, the MorphoMNIST section supports these claims and not more:

1. **A benchmark with an exactly-known, tunable ATE** on a high-dimensional
   image outcome — six experiments isolating confounding, covariate
   heterogeneity, outcome coupling, quantile effects and spatial structure. The
   truth is imposed to ~1e-15, not estimated.
2. **The frugal flow recovers the ATE competitively** with classical estimators
   across all six, and is insensitive to which mechanism generates the
   heterogeneity (0.025–0.029 across E3–E6) — as the theory predicts, since the
   causal margin targets the marginal effect regardless.
3. **It additionally recovers the quantile-resolved effect** against a
   closed-form target, which weighting and regression baselines do not provide
   (modulo the IPW-quantile comparison in Stage 0b).
4. **A cautionary result**: a flexible density model buys nothing for a pure ATE
   and costs 2–4× the error of OLS. Worth saying out loud — it is a more useful
   contribution than pretending otherwise, and it tells practitioners when to
   reach for this machinery.
