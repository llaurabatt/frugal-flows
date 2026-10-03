"""Halo ladder: datasets, preprocessing variants, pixel classes, templates and noise floors.

Everything here reuses Laura's generator (``prepare_morphomnist_exps.build_preset``) by
import. The only intervention is a TEMPORARY replacement of the module-global
``prepare_morphomnist_exps.dequantize_and_logit`` (``build_experiment`` resolves it by
name at call time), which (a) records the raw pooled pixels before dequantisation and
(b) swaps the pixel->model-space map for the P2/P3/P4 variants. The replacement draws the
dequantisation noise with exactly the same ``rng`` call as the original, so the image
order, the noise and the treatment assignment are identical across variants of one seed.

Preprocessing variants (``Preproc.kind``):
    P0  raw logit, alpha=0.05 (the runner's status quo; no standardisation)
    P1  P0 then per-column standardise (``OutcomeTransform("standardize")``, fitted on the
        fitting data); estimand-preserving because samples are inverted before contrasts
    P2  logit with alpha=0.3 (dataset rebuilt)                       NOT estimand-preserving
    P3  no logit: dequantised pixel x in [0,1] mapped affinely to [-0.9, 0.9]   NOT e.p.
    P4  no dequantisation noise (exact atoms at logit(alpha/2)); demonstration only
    P5  (Amendment A1) P0 then the frengression comparator's floored per-pixel scaling
        (``exp_frengression_recovery.prepare_inputs``, y_scaling=per_pixel, y_sd_floor=0.25):
        Z_k = (Y_k - mean_k) / max(sd_k, 0.25 * (sd(Y_all) or 1.0)); fitted on the fitting
        data, inverted on samples; affine, so estimand-preserving
For P2/P3/P4 ``build_experiment`` adds the ITE AFTER the transform, so ``ATE`` is still the
+base_shift disc map but in a different space: never compare E_tau across P0 and P2/P3/P4.

Synthetic images (S1 controls, ``synthetic`` in {"smooth", "zinf"}): G ~ N(mean, cov) of
the real logit Y (P0, E1, base_shift 0, same seed_data), n = 5923 rows.
    smooth  Y = G. Equivalent to drawing in pixel space x = (sigmoid(G) - a/2)/(1 - a) and
            applying the squeeze+logit with no clipping and no quantisation: continuous,
            no atoms, the data's first two moments in logit space.
    zinf    the same x, clipped to [0,1], quantised down to the 1/256 grid
            (min(floor(256 x), 255)/256, so the background becomes exact-0 atoms), then
            Laura's ``dequantize_and_logit`` (U(0,1/256) noise, squeeze, logit): atoms
            restored as the pure-floor sliver, exactly as in the real data.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys

import numpy as np

HALO_DIR = os.path.dirname(os.path.abspath(__file__))
MM_DIR = os.path.abspath(os.path.join(HALO_DIR, "..", ".."))
for _p in (MM_DIR, HALO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prepare_data  # noqa: E402
import prepare_morphomnist_exps as pme  # noqa: E402

SIZE, RADIUS, DIGIT, N_DIGIT0 = 8, 2, 0, 5923
SEEDS_DATA = tuple(range(31, 41))
CORPUS_B_SEED = 1000   # one master permutation of all 60,000 train images (Corpus B)
CACHE_DIR = os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/_cache")
PRESETS = {"E1": "exp1_rct_homogeneous", "E2": "exp2_confounded_homogeneous"}
PREPROCS = ("P0", "P1", "P2", "P3", "P4", "P5")
P5_SD_FLOOR = 0.25   # frengression frozen tuning setting y_sd_floor
ALPHA = 0.05


def _logit(p: float) -> float:
    return float(np.log(p) - np.log1p(-p))


# Closed form of the pure-background pixel. A pooled pixel that is exactly 0 in [0,1]
# becomes x = 0 + U/256 (U ~ U[0,1)), squeezed to a/2 + (1-a) x, then logit. So
# Y = logit(0.025 + 0.95 U/256) has support [logit(0.025), logit(0.025 + 0.95/256)]
# = [-3.66356, -3.52134]: width 0.142, sd ~0.041.
QUIET_LO, QUIET_HI = _logit(ALPHA / 2), _logit(ALPHA / 2 + (1 - ALPHA) / 256)
FLOOR_Y = -3.52  # the prereg's floor threshold (P0 space), 0.94% of the sliver width above QUIET_HI
_FLOOR_MARGIN = (FLOOR_Y - QUIET_HI) / (QUIET_HI - QUIET_LO)


def quiet_bounds(preproc: str) -> tuple[float, float, float]:
    """(lo, hi, floor_threshold) of the pure-floor support in the cell's DATA space.
    P0/P1/P4 share P0's space (P4's atom sits at lo); P2 and P3 have their own sliver.
    The floor threshold keeps P0's relative margin above hi, so P0 gives exactly -3.52."""
    if preproc == "P2":
        lo, hi = _logit(0.15), _logit(0.15 + 0.7 / 256)
    elif preproc == "P3":
        lo, hi = -0.9, -0.9 + 1.8 / 256
    else:
        return QUIET_LO, QUIET_HI, FLOOR_Y
    return lo, hi, hi + _FLOOR_MARGIN * (hi - lo)


def quiet_truth(n: int = 1_000_000, seed: int = 0) -> dict:
    """Closed-form pure-floor density in P0 space, summarised by a large MC draw."""
    u = np.random.default_rng(seed).uniform(0, 1, n)
    y = np.log(ALPHA / 2 + (1 - ALPHA) * u / 256) - np.log1p(-(ALPHA / 2 + (1 - ALPHA) * u / 256))
    return {"lo": QUIET_LO, "hi": QUIET_HI, "width": QUIET_HI - QUIET_LO,
            "mean": float(y.mean()), "sd": float(y.std())}


# ----------------------------------------------------------------- transform variants
def _variant(kind: str, store: dict):
    """A stand-in for ``prepare_data.dequantize_and_logit`` (same rng consumption) that
    records the raw pooled pixel in [0,1] into ``store["RAW"]``."""
    orig = prepare_data.dequantize_and_logit

    def fn(images_flat, rng, alpha=ALPHA):
        store["RAW"] = (np.asarray(images_flat) + 1.0) / 2.0
        if kind in ("P0", "P1"):
            return orig(images_flat, rng, alpha=alpha)
        if kind == "P2":
            return orig(images_flat, rng, alpha=0.3)
        x = (np.asarray(images_flat) + 1.0) / 2.0
        noise = rng.uniform(0.0, 1.0 / 256.0, size=x.shape)   # drawn in every variant
        if kind == "P3":
            return -0.9 + 1.8 * np.clip(x + noise, 0.0, 1.0)
        if kind == "P4":
            x = alpha / 2 + (1 - alpha) * np.clip(x, 0.0, 1.0)
            return np.log(x) - np.log1p(-x)
        raise ValueError(kind)
    return fn


@contextlib.contextmanager
def _patched(kind: str, store: dict):
    saved = pme.dequantize_and_logit
    pme.dequantize_and_logit = _variant(kind, store)
    try:
        yield
    finally:
        pme.dequantize_and_logit = saved


def disc_mask_geometric() -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    c = (SIZE - 1) / 2
    return (((xx - c) ** 2 + (yy - c) ** 2) <= RADIUS ** 2).ravel()


def ring_mask() -> np.ndarray:
    """Laura's geometric ring (``exp_ate_recovery.region_masks``), recomputed here so S0
    need not import the runner; ``test_halo`` asserts equality with hers."""
    disc = disc_mask_geometric().reshape(SIZE, SIZE)
    ring = np.zeros_like(disc)
    for i in range(SIZE):
        for j in range(SIZE):
            if not disc[i, j]:
                ring[i, j] = any(0 <= i + a < SIZE and 0 <= j + b < SIZE and disc[i + a, j + b]
                                 for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    return ring.ravel()


ROW_KEYS = ("Y", "X", "Y0", "Y1", "ITE", "THICKNESS", "PROPENSITY", "z_cont", "RAW")


def _build(preset: str, base_shift: float, seed: int, preproc: str, digit, ps_slope=None) -> dict:
    store: dict = {}
    extra = {} if ps_slope is None else {"ps_slope": float(ps_slope)}   # S10 (A5) override only
    with _patched(preproc, store):
        d = pme.build_preset(PRESETS.get(preset, preset), size=SIZE, radius=RADIUS, digit=digit,
                             n=None, seed=seed, base_shift=float(base_shift), **extra)
    out = {k: np.asarray(d[k], dtype=np.float64) for k in ROW_KEYS[:-1] + ("ATE",)}
    out["RAW"] = store["RAW"]
    out["ps_slope"] = float(d["config"]["ps_slope"])
    out["dataset_id"], out["data_hash"] = d["dataset_id"], d["data_hash"]
    return out


def _corpus_b_master(preset: str, base_shift: float, preproc: str) -> dict:
    """All ten digits, n = 60,000, ONE master permutation (seed 1000), cached on disk.
    ``build_experiment``'s digit=None branch appends a digit one-hot to Z; ``z_cont`` is
    still thickness alone (z_cat_idx), which is all this ladder uses. Propensity z-scores
    thickness over the 60,000 rows."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"corpusB_{preset}_bs{float(base_shift)}_{preproc}_s{CORPUS_B_SEED}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
    d = _build(preset, base_shift, CORPUS_B_SEED, preproc, None)
    assert len(d["Y"]) == 60000, len(d["Y"])
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)                      # atomic: concurrent cells never see a partial file
    return d


def build_real(preset: str, base_shift: float, seed_data: int, preproc: str = "P0",
               corpus: str = "A", ps_slope: float | None = None) -> dict:
    """Laura's ``build_preset`` (size 8, preset's own ps_slope) with the transform variant
    of ``preproc``. Corpus A: digit 0, all 5,923 images, seed = seed_data (the same images
    in every seed; order, noise and assignment differ). Corpus B: rows
    [i*5923, (i+1)*5923) of the all-digit master build, i = seed_data - 31: ten genuinely
    disjoint sets of 5,923 independent units. Returns numpy arrays plus RAW pooled pixels."""
    if corpus == "A":
        out = _build(preset, base_shift, seed_data, preproc, DIGIT, ps_slope)
        assert len(out["Y"]) == N_DIGIT0, len(out["Y"])
    elif corpus == "B":
        assert ps_slope is None, "ps_slope override is Corpus-A only (S10)"
        m = _corpus_b_master(preset, base_shift, preproc)
        i = seed_data - SEEDS_DATA[0]
        assert 0 <= i < 10, seed_data
        rows = slice(i * N_DIGIT0, (i + 1) * N_DIGIT0)
        out = {k: (m[k][rows] if k in ROW_KEYS else m[k]) for k in m}
        out["dataset_id"] = f"{m['dataset_id']}_block{i}"
    else:
        raise ValueError(corpus)
    out["disc"] = (out["ATE"] != 0) if np.any(out["ATE"] != 0) else disc_mask_geometric()
    return out


@functools.lru_cache(maxsize=8)
def class_reference(seed_data: int, corpus: str = "A") -> dict:
    """The per-(seed_data, corpus) reference for pixel sets and templates: REAL data, E1,
    base_shift 0, P0 (Y == Y0). Synthetic cells use the Corpus-A reference of their seed."""
    return build_real("E1", 0.0, seed_data, "P0", corpus)


def build_synthetic(kind: str, seed_data: int) -> dict:
    """See the module docstring. X/THICKNESS are carried over from the reference (unused
    by S1, which fits p(Y) only); ATE is zero; Y0 = Y1 = Y."""
    ref = class_reference(seed_data)
    Yr = ref["Y"]
    rng = np.random.default_rng([seed_data, 7717])
    G = rng.multivariate_normal(Yr.mean(0), np.cov(Yr, rowvar=False), size=len(Yr), method="eigh")
    x = (1.0 / (1.0 + np.exp(-G)) - ALPHA / 2) / (1 - ALPHA)
    if kind == "smooth":
        Y, raw = G, x
    elif kind == "zinf":
        q = np.minimum(np.floor(np.clip(x, 0.0, 1.0) * 256), 255) / 256
        Y, raw = prepare_data.dequantize_and_logit(2 * q - 1, rng, alpha=ALPHA), q
    else:
        raise ValueError(kind)
    out = dict(ref)
    out.update(Y=Y, Y0=Y, Y1=Y, ITE=np.zeros_like(Y), ATE=np.zeros(Y.shape[1]), RAW=raw,
               disc=disc_mask_geometric(), dataset_id=f"synthetic_{kind}_{seed_data}", data_hash="")
    return out


def build_dataset(cfg: dict) -> dict:
    """The cell's dataset from its config (preset, base_shift, seed_data, preproc, synthetic)."""
    if cfg.get("synthetic"):
        return build_synthetic(cfg["synthetic"], cfg["seed_data"])
    pp = cfg["preproc"] if cfg["preproc"] in ("P2", "P3", "P4") else "P0"
    # S10 (Amendment A5): the cell's ps_slope is passed to the generator (1.2 = E2's own value,
    # 2.4 = the stronger-confounding cells). Earlier stages keep the preset's own ps_slope.
    ps = cfg["ps_slope"] if cfg.get("stage") == "S10" else None
    out = build_real(cfg["preset"], cfg["base_shift"], cfg["seed_data"], pp, cfg.get("corpus", "A"), ps)
    if ps is not None:
        assert out["ps_slope"] == float(ps), (out["ps_slope"], ps)
    return out


def placebo_permutation(seed_data: int, n: int) -> tuple[np.ndarray, list[int]]:
    """S10 Anchor B (Amendment A5): a seeded permutation of the n units, applied to the
    covariate ranks so the covariate is independent of (T, Y). Returns (perm, rng seed)."""
    seed = [int(seed_data), 5150]
    return np.random.default_rng(seed).permutation(n), seed


class FlooredStandardize:
    """P5: the frengression comparator's ``y_scaling="per_pixel"`` scaling, replicated
    from ``exp_frengression_recovery.prepare_inputs`` (frengression worktree, L314-322):

        y_sd_global = float(Y.std())                       # all n x K entries, ddof 0
        y_mean      = Y.mean(axis=0)
        y_scale     = max(Y.std(axis=0), floor * (y_sd_global or 1.0))
        Z           = (Y - y_mean) / y_scale ;   inverse: Z * y_scale + y_mean

    Duck-typed ``forward``/``inverse`` in numpy float64. It is NOT an ``OutcomeTransform``
    (``as_outcome_transform`` accepts only its own class), so ff_full sampling passes
    ``outcome_transform=None`` and applies ``inverse`` to the returned draws."""

    def __init__(self, floor: float = P5_SD_FLOOR):
        self.floor = float(floor)
        self.y_mean = self.y_scale = self.y_sd_global = None

    def fit(self, Y) -> "FlooredStandardize":
        Y = np.asarray(Y, dtype=np.float64)
        self.y_sd_global = float(Y.std())
        self.y_mean = Y.mean(axis=0)
        self.y_scale = np.maximum(Y.std(axis=0), self.floor * (self.y_sd_global or 1.0))
        return self

    @property
    def floor_value(self) -> float:
        return self.floor * (self.y_sd_global or 1.0)

    def forward(self, Y) -> np.ndarray:
        return (np.asarray(Y, dtype=np.float64) - self.y_mean) / self.y_scale

    def inverse(self, Z) -> np.ndarray:
        return np.asarray(Z, dtype=np.float64) * self.y_scale + self.y_mean

    def info(self) -> dict:
        sd = self.y_scale
        return {"y_sd_floor": self.floor, "y_sd_global": self.y_sd_global, "floor_value": self.floor_value,
                "n_floored": int(np.sum(sd == self.floor_value)), "y_scale": sd.tolist(),
                "y_mean": self.y_mean.tolist()}


class Preproc:
    """Fit-time preprocessing. P1 (standardise per column) and P5 (frengression floored
    per-pixel scaling) act here, fitted on the fitting data and inverted on samples;
    P0/P2/P3/P4 are identity at fit time because their transform was applied when the
    dataset was built."""

    def __init__(self, kind: str):
        assert kind in PREPROCS, kind
        self.kind, self.transform = kind, None

    def fit(self, Y) -> "Preproc":
        if self.kind == "P1":
            from frugal_flows.outcome_transforms import OutcomeTransform
            self.transform = OutcomeTransform("standardize").fit(np.asarray(Y))
        elif self.kind == "P5":
            self.transform = FlooredStandardize().fit(Y)
        return self

    def info(self) -> dict:
        d = {"preproc": self.kind}
        if self.kind == "P5":
            d.update(self.transform.info())
        return d

    def forward(self, Y) -> np.ndarray:
        return np.asarray(Y) if self.transform is None else np.asarray(self.transform.forward(Y))

    def inverse(self, Z) -> np.ndarray:
        return np.asarray(Z) if self.transform is None else np.asarray(self.transform.inverse(Z))


# ----------------------------------------------------------------- classes / templates
def pixel_classes(Y: np.ndarray, disc: np.ndarray, raw: np.ndarray | None = None) -> dict:
    """Boolean (64,) masks. quiet: sd(Y) < 0.3; disc: the effect support (ATE != 0, or the
    geometric r=2 disc when ATE == 0); active_off: neither. Floor classes from
    P(raw pooled pixel == 0): exact_floor == 1, pure_floor >= 0.95, mixture in (0.05, 0.95),
    ink <= 0.05. Regions reg_disc/reg_ring/reg_far are Laura's ``region_masks``.
    Without ``raw`` the floor indicator falls back to P(Y < FLOOR_Y)."""
    sd = np.asarray(Y).std(0)
    quiet = sd < 0.3
    disc = np.asarray(disc, bool)
    pf = (np.asarray(raw) == 0).mean(0) if raw is not None else (np.asarray(Y) < FLOOR_Y).mean(0)
    d = disc_mask_geometric()
    ring = ring_mask()
    return {"quiet": quiet, "disc": disc, "active_off": ~quiet & ~disc,
            "exact_floor": pf == 1.0, "pure_floor": pf >= 0.95, "mixture": (pf > 0.05) & (pf < 0.95), "ink": pf <= 0.05,
            "active": ~quiet, "reg_disc": d, "reg_ring": ring, "reg_far": ~d & ~ring,
            "p_floor": pf}


def _std(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


TEMPLATE_NAMES = ("t_mix", "t_sd", "t_imb", "t_thick", "t_ring")


def naive_diff(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    t = np.asarray(X)[:, 0].astype(bool)
    return Y[t].mean(0) - Y[~t].mean(0)


def templates(data: dict, ref: dict | None = None) -> dict:
    """Standardised (over 64 px) templates. t_mix = P(floor)(1-P(floor)), t_sd = sd(Y),
    t_thick = corr(Y_k, thickness), t_ring = Laura's ring: from ``ref`` (the seed's
    class reference; defaults to ``data``). t_imb = naive T-difference - ATE of ``data``."""
    ref = data if ref is None else ref
    pf = (ref["RAW"] == 0).mean(0)
    Yr = ref["Y"]
    th = ref["THICKNESS"]
    corr = ((Yr - Yr.mean(0)) * (th - th.mean())[:, None]).mean(0) / (Yr.std(0) * th.std() + 1e-300)
    return {"t_mix": _std(pf * (1 - pf)), "t_sd": _std(Yr.std(0)),
            "t_imb": _std(naive_diff(data["Y"], data["X"]) - data["ATE"]),
            "t_thick": _std(corr), "t_ring": _std(ring_mask().astype(float))}


def template_corr(tpl: dict) -> np.ndarray:
    """5x5 correlation matrix of the templates (TEMPLATE_NAMES order) over the 64 pixels."""
    return np.corrcoef(np.vstack([tpl[n] for n in TEMPLATE_NAMES]))


def floors(Y: np.ndarray, X: np.ndarray, Y0: np.ndarray, B: int = 200, seed: int = 0) -> dict:
    """Data noise floors, per pixel: 200-row-bootstrap SE of the mean, the sd and the naive
    T-difference; split-half |delta mean| and |delta sd|; the null naive difference on Y0
    (what an E1 tau_hat error looks like with no model at all)."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    t = np.asarray(X)[:, 0].astype(bool)
    bm, bs, bd = [], [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        Yb, tb = Y[i], t[i]
        bm.append(Yb.mean(0)), bs.append(Yb.std(0)), bd.append(Yb[tb].mean(0) - Yb[~tb].mean(0))
    p = rng.permutation(n)
    a, b = p[: n // 2], p[n // 2: 2 * (n // 2)]
    return {"se_mean": np.std(bm, 0, ddof=1), "se_sd": np.std(bs, 0, ddof=1),
            "se_naive": np.std(bd, 0, ddof=1),
            "split_dmean": np.abs(Y[a].mean(0) - Y[b].mean(0)),
            "split_dsd": np.abs(Y[a].std(0) - Y[b].std(0)),
            "null_naive_Y0": naive_diff(Y0, X)}


# ----------------------------------------------------------------- S0
def run_s0(out: str, seeds=SEEDS_DATA, corpus: str = "A") -> dict:
    """Model-free stage: per seed_data classes, templates (+ their 5x5 correlation),
    floors, the E1 imbalance map (base_shift 1.0 and 0.0) and the E2 unadjusted-bias map."""
    os.makedirs(out, exist_ok=True)
    summary = {"corpus": corpus, "quiet_truth": quiet_truth(), "seeds": {}}
    tag = "" if corpus == "A" else "_B"
    for s in seeds:
        ref = class_reference(s, corpus)
        e1 = build_real("E1", 1.0, s, "P0", corpus)
        e2 = build_real("E2", 1.0, s, "P0", corpus)
        cls = pixel_classes(ref["Y"], e1["disc"], ref["RAW"])
        tpl = templates(e1, ref)
        fl = floors(e1["Y"], e1["X"], e1["Y0"], seed=s)
        maps = {"imb_E1_bs1": naive_diff(e1["Y"], e1["X"]) - e1["ATE"],
                "imb_E1_bs0": naive_diff(ref["Y"], ref["X"]) - ref["ATE"],
                "bias_E2": naive_diff(e2["Y"], e2["X"]) - e2["ATE"],
                "t_imb_E2": templates(e2, ref)["t_imb"]}
        np.savez(os.path.join(out, f"s0{tag}_sd{s}.npz"), **{f"cls_{k}": v for k, v in cls.items()},
                 **tpl, template_corr=template_corr(tpl), **{f"floor_{k}": v for k, v in fl.items()}, **maps)
        nq = int(cls["quiet"].sum())
        summary["seeds"][str(s)] = {
            "n": int(len(ref["Y"])), "quiet": nq, "quiet_flag": corpus == "A" and not 30 <= nq <= 34,
            **{c: int(cls[c].sum()) for c in ("disc", "active_off", "exact_floor", "pure_floor", "mixture", "ink")},
            "treated_frac_E1": float(e1["X"].mean()), "treated_frac_E2": float(e2["X"].mean()),
            "split_dmean_max_quiet": float(fl["split_dmean"][cls["quiet"]].max()),
            "split_dmean_max_active": float(fl["split_dmean"][~cls["quiet"]].max()),
            "split_dsd_max_quiet": float(fl["split_dsd"][cls["quiet"]].max()),
            "split_dsd_max_active": float(fl["split_dsd"][~cls["quiet"]].max()),
            "null_naive_sd_active": float(fl["null_naive_Y0"][~cls["quiet"]].std()),
            "null_naive_maxabs_active": float(np.abs(fl["null_naive_Y0"][~cls["quiet"]]).max()),
            "bias_E2_mean_active_off": float(maps["bias_E2"][cls["active_off"]].mean()),
            "bias_E2_mean_disc": float(maps["bias_E2"][cls["disc"]].mean()),
            "frac_ref_outside_sliver_pure_floor": float(
                ((ref["Y"] < QUIET_LO) | (ref["Y"] > QUIET_HI))[:, cls["pure_floor"]].mean()),
        }
        print(f"S0 seed {s}: {summary['seeds'][str(s)]}", flush=True)
    with open(os.path.join(out, f"s0{tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S0", choices=["S0"])
    ap.add_argument("--out", default=os.path.expanduser(os.environ.get("FF_RUNS_LOG", "~/work/halo-runs") + "/S0"))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS_DATA))
    ap.add_argument("--corpus", default="A", choices=["A", "B"])
    a = ap.parse_args(argv)
    run_s0(os.path.expanduser(a.out), a.seeds, a.corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
