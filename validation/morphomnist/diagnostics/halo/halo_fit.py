"""Halo ladder: run ONE cell.  ``python halo_fit.py --config cell.json [--smoke]``

config -> dataset -> preproc -> fit -> paired draws -> inverse preproc -> maps, template
regression, class summaries -> <out_dir>/{config.json, metrics.json, maps.npz, losses.npz,
log.txt}.  Any exception writes <out_dir>/FAILED (traceback) and exits 1.  float32 only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HALO_DIR = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HALO_DIR, "..", "..", "..", ".."))
sys.path.insert(0, HALO_DIR)

IDENTITY_KEYS = ("stage", "corpus", "arm", "preproc", "task", "preset", "base_shift", "ps_slope", "synthetic",
                 "seed_data", "seed_fit", "seed_mc", "seed_mc2", "width", "depth", "layers", "knots",
                 "rank_mode", "lr", "max_epochs", "patience", "batch", "n_mc", "x64", "xla_flags")
# Amendment A3 (S7): identity keys that enter a cell's identity ONLY when the cell carries them,
# so every S0-S6 identity (and run_id hash) is unchanged by their introduction.
OPTIONAL_IDENTITY_KEYS = ("copula_rank_rule", "copula_width", "paper_setting")
# Amendment A5 (S10): carried by S10 cells only (same rule: absent keys leave old identities alone)
OPTIONAL_IDENTITY_KEYS += ("shift_init", "placebo_covariate")
SHIFT_INITS = ("zero", "naive", "plus2")


def identity_of(cell: dict) -> dict:
    ident = {k: cell.get(k) for k in IDENTITY_KEYS}
    ident.update({k: cell[k] for k in OPTIONAL_IDENTITY_KEYS if k in cell})
    return ident


class _Tee:
    def __init__(self, path):
        self.fh, self.out = open(path, "a", buffering=1), sys.__stdout__

    def write(self, s):
        self.fh.write(s), self.out.write(s)

    def flush(self):
        self.fh.flush(), self.out.flush()


def _versions() -> dict:
    import importlib.metadata as md
    return {p: md.version(p) for p in ("jax", "jaxlib", "flowjax", "equinox", "numpy", "scipy",
                                       "torch", "zuko")}


def _git() -> dict:
    run = lambda *a: subprocess.run(["git", "-C", WORKTREE, *a], capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "dirty_tracked": bool(run("status", "--porcelain", "-uno"))}


def _drop_nonfinite(y0, y1):
    keep = np.isfinite(y0).all(1) if y1 is None else np.isfinite(y0).all(1) & np.isfinite(y1).all(1)
    if not keep.any():
        raise RuntimeError("every draw non-finite: degenerate fit")
    return y0[keep], (None if y1 is None else y1[keep]), int((~keep).sum())


LT_ARMS = ("lt", "lt_n", "gff_shift")     # location-translation arms: tau_hat = fitted LocCond ate
GFF_ARMS = ("gff_flex", "gff_shift")      # S9 (Amendment A4): Gaussian-scale FF via the package


def ate_to_data_scale(pre, ate) -> np.ndarray:
    """A LocCond ``ate`` fitted on the preprocessed scale, mapped to logit (data) units.
    Every fit-time preprocessing is affine per column (P1: z = (y - mean)/sd), so a shift
    of ``ate`` in z is a shift of ``inverse(ate) - inverse(0)`` = ``ate * sd`` in y. P0 (no
    fit-time transform) returns ``ate`` unchanged, bit for bit."""
    ate = np.asarray(ate, np.float64)
    if pre.transform is None:
        return ate
    z0 = np.zeros((1, ate.shape[0]))
    return np.asarray(pre.inverse(ate[None, :]), np.float64)[0] - np.asarray(pre.inverse(z0), np.float64)[0]


def shift_init_vector(kind: str, Y_fit: np.ndarray, X: np.ndarray, pre):
    """S10 (A5) starting value of the shift, on the PREPROCESSED (fitted) scale, or None for the
    package default (zero). "naive": the per-pixel treated-minus-untreated difference of the
    fitted outcome (all n rows). "plus2": +2 logit units per pixel, i.e. 2 / sd_k under P1
    (sd_k = the data-scale size of a unit shift on the fitted scale, from ``ate_to_data_scale``)."""
    import halo_data as hd
    if kind == "zero":
        return None
    if kind == "naive":
        return np.asarray(hd.naive_diff(np.asarray(Y_fit, np.float64), X), np.float64)
    if kind == "plus2":
        unit = ate_to_data_scale(pre, np.ones(np.asarray(Y_fit).shape[1]))
        return 2.0 / unit
    raise ValueError(kind)


def _fit_and_sample(cfg: dict, data: dict, pre):
    """Returns (draw function seed_mc -> (y0, y1, n_clamped) on the DATA scale, fit info,
    loss dicts, extra metrics)."""
    import jax.numpy as jnp
    import jax.random as jr
    Y, X = pre.forward(data["Y"]), data["X"]
    arm, extra = cfg["arm"], {}
    if arm.startswith("zuko"):
        import halo_zuko as hz
        flow, losses = hz.fit_with_fallback(cfg, Y, X if cfg["task"] == "cond" else None)

        def draw(seed):
            y0, y1 = hz.sample_arms(flow, cfg["n_mc"], seed, cfg["task"], Y.shape[1])
            return pre.inverse(y0), (None if y1 is None else pre.inverse(y1)), 0
        return draw, [losses], extra
    if arm == "ff_full":
        from frugal_flows.causal_flows import train_frugal_flow
        from frugal_flows.interventions import interventional_samples
        from scipy.stats import rankdata
        zc = np.asarray(data["z_cont"], np.float64)
        u_z = rankdata(zc, axis=0) / (zc.shape[0] + 1)     # ECDF midranks (runner's "ecdf")
        key = jr.PRNGKey(cfg["seed_fit"])
        key, _ = jr.split(key)                              # stage-1 slot (not fitted: ECDF)
        key, sub = jr.split(key)
        from halo_ranks import copula_rank_rule, fitted_copula_reach
        rule = cfg.get("copula_rank_rule", "new")       # S7 (A3); absent = the package rule
        cop_w = cfg.get("copula_width", 50)             # S7 (A3); absent = S4/S5 width 50
        # The copula masks are built inside train_frugal_flow (-> train_frugal_flow_flexible_
        # continuous -> masked_autoregressive_flow_first_uniform -> MaskedAutoregressiveFirst
        # Uniform.__init__), so the whole call runs under the rule.
        with copula_rank_rule(rule):
            flow, losses = train_frugal_flow(
                key=sub, y=jnp.asarray(Y), u_z=jnp.asarray(u_z), condition=jnp.asarray(X),
                causal_model="flexible_continuous", RQS_knots=8, nn_depth=1, nn_width=cop_w, flow_layers=4,
                learning_rate=cfg["lr"], max_epochs=cfg["max_epochs"], max_patience=cfg["patience"],
                batch_size=cfg["batch"], show_progress=False,
                fit_kwargs={"ema_decay": None, "wall_cap_s": None},
                causal_model_args={"RQS_knots": cfg["knots"], "nn_depth": cfg["depth"],
                                   "nn_width": cfg["width"], "flow_layers": cfg["layers"],
                                   "conditioner": "mlp"})
        # proof of which rule ran: reachability from the FITTED flow's own copula masks
        extra.update(copula_rank_rule=rule, copula_width=cop_w, **fitted_copula_reach(flow, Y.shape[1]))
        if extra["copula_mask_width"] != cop_w:
            raise RuntimeError(f"copula hidden width {extra['copula_mask_width']} != requested {cop_w}")

        # P1 passes its OutcomeTransform into the sampler (unchanged S4 path). P5's transform
        # is not an OutcomeTransform (as_outcome_transform rejects it), so sample on the
        # fitting scale (outcome_transform=None -> identity) and invert here: the package
        # applies the inverse pointwise to the same draws before any statistic, so this is
        # the same operation. P0's inverse is the identity.
        ot = pre.transform if pre.kind == "P1" else None

        def draw(seed):
            r = interventional_samples(jr.key(seed), flow, cond_dim=1, n_mc=cfg["n_mc"],
                                       outcome_transform=ot, dim_y=Y.shape[1])
            y0, y1 = np.asarray(r["y0"]), np.asarray(r["y1"])
            if pre.kind == "P5":
                y0, y1 = pre.inverse(y0), pre.inverse(y1)
            return y0, y1, int(r["n_clamped"])
        return draw, [losses], extra
    import halo_models as hm
    if arm in GFF_ARMS:                                 # S9 (Amendment A4)
        import frugal_flows.gaussian_scale as gs
        from frugal_flows.interventions import interventional_samples
        from scipy.stats import rankdata
        zc = np.asarray(data["z_cont"], np.float64)
        u_z = rankdata(zc, axis=0) / (zc.shape[0] + 1)     # ECDF midranks of thickness, as ff_full
        if cfg.get("placebo_covariate"):                    # S10 Anchor B (A5): uninformative covariate
            import halo_data as hd
            perm, pseed = hd.placebo_permutation(cfg["seed_data"], u_z.shape[0])
            u_z = u_z[perm]
            extra.update(placebo_perm_seed=pseed, placebo_perm_fixed_points=int(np.sum(perm == np.arange(len(perm)))),
                         placebo_corr_uz_thickness=float(np.corrcoef(u_z[:, 0], zc[:, 0])[0, 1]))
        init = shift_init_vector(cfg.get("shift_init", "zero"), Y, X, pre)
        extra["init_fit_scale"] = np.zeros(Y.shape[1]) if init is None else init
        extra["init_vector"] = ate_to_data_scale(pre, extra["init_fit_scale"])
        flow, losses = hm.fit_gff(cfg, Y, X, u_z, ate_init=init)
        if arm == "gff_shift":
            extra["lt_ate_fit_scale"] = np.asarray(gs.shift_vector(flow), np.float64)
            extra["lt_ate"] = ate_to_data_scale(pre, extra["lt_ate_fit_scale"])
        extra["calibration"] = hm.gff_calibration(flow, Y, X, u_z, losses["info"]["val_idx"],
                                                  seed=cfg["seed_mc"] + 1000)
        ot = pre.transform if pre.kind == "P1" else None

        def draw(seed):
            r = interventional_samples(jr.key(seed), flow, cond_dim=1, n_mc=cfg["n_mc"],
                                       outcome_transform=ot, dim_y=Y.shape[1])
            y0, y1 = np.asarray(r["y0"]), np.asarray(r["y1"])
            if pre.kind == "P5":
                y0, y1 = pre.inverse(y0), pre.inverse(y1)
            return y0, y1, int(r["n_clamped"])
        return draw, [losses], extra
    if cfg.get("shift_init", "zero") != "zero" or cfg.get("placebo_covariate"):
        raise ValueError(f"shift_init / placebo_covariate are implemented for the GFF arms only, not {arm}")
    dist, losses = hm.fit_arm(cfg, Y, X)
    if arm in LT_ARMS:
        extra["init_vector"] = np.zeros(Y.shape[1])        # LocCond(ate=0)
        extra["lt_ate_fit_scale"] = hm.loccond_ate(dist)
        extra["lt_ate"] = ate_to_data_scale(pre, extra["lt_ate_fit_scale"])
    if cfg.get("stage") == "S8":                        # S8 (Amendment A4): latent calibration
        extra["calibration"] = hm.latent_calibration(dist, Y, X if cfg["task"] == "cond" else None,
                                                     losses[0]["info"]["val_idx"])

    def draw(seed):
        y0, y1, c = hm.sample_arms(seed, dist, cfg["n_mc"], "cond" if arm == "sep" else cfg["task"])
        return pre.inverse(y0), (None if y1 is None else pre.inverse(y1)), int(c)
    return draw, losses, extra


def run(cfg: dict) -> dict:
    import jax
    assert os.environ.get("JAX_ENABLE_X64") == "0", "JAX_ENABLE_X64=0 must be set"
    assert not jax.config.jax_enable_x64, "float64 active"
    import frugal_flows  # noqa: F401  (its precision default must not override the env)
    assert not jax.config.jax_enable_x64, "frugal_flows switched on float64"
    # feat/gaussian-scale worktree: the package must be THIS worktree's (PYTHONPATH override of
    # the env's editable install, which points at another worktree)
    ff_file = os.path.abspath(frugal_flows.__file__)
    assert ff_file.startswith(WORKTREE + os.sep), f"frugal_flows imported from {ff_file}, not {WORKTREE}"
    if cfg.get("xla_flags") is not None:
        assert os.environ.get("XLA_FLAGS", "") == cfg["xla_flags"], "XLA flag string mismatch"
    import halo_data as hd
    import halo_metrics as hmx

    t0 = time.monotonic()
    data = hd.build_dataset(cfg)
    ref = hd.class_reference(cfg["seed_data"], cfg.get("corpus", "A"))
    classes = hd.pixel_classes(ref["Y"], data["disc"], ref["RAW"])
    tpl = hd.templates(data, ref)
    lo, hi, floor_thr = hd.quiet_bounds(cfg["preproc"])
    pre = hd.Preproc(cfg["preproc"]).fit(data["Y"])
    t_data = time.monotonic() - t0

    t0 = time.monotonic()
    draw, losses, extra = _fit_and_sample(cfg, data, pre)
    t_fit = time.monotonic() - t0

    t0 = time.monotonic()
    y0, y1, n_clamped = draw(cfg["seed_mc"])
    y0, y1, n_nonfinite = _drop_nonfinite(y0, y1)
    t_sample = time.monotonic() - t0

    t0 = time.monotonic()
    paired = y1 is not None
    if paired:
        ref0, ref1 = data["Y0"], data["Y1"]
    else:
        ref0, ref1 = data["Y"], None
    leak = cfg.get("synthetic") != "smooth"           # LEAK undefined without atoms
    m = hmx.maps(y0, ref0, y1, ref1, data["ATE"], extra.get("lt_ate"), lo=lo, hi=hi,
                 floor_thr=floor_thr, leak=leak)
    if cfg.get("seed_mc2") is not None:
        a0, a1, _ = draw(cfg["seed_mc2"])
        a0, a1, _ = _drop_nonfinite(a0, a1)
        m2 = hmx.maps(a0, ref0, a1, ref1, data["ATE"], extra.get("lt_ate"), lo=lo, hi=hi,
                      floor_thr=floor_thr, leak=leak)
        m.update({f"mc2_{k}": m2[k] for k in ("E_mu0", "E_sd0", "R_sd0", "LEAK_X0", "E_mu1", "E_sd1",
                                               "E_tau") if k in m2})
    imb = hd.naive_diff(data["Y"], data["X"]) - data["ATE"]
    # template OLS is DESCRIPTIVE only (prereg v1): coefficients + R2 + the 5x5 template correlation
    tmpl = {k: hmx.template_regression(m[k], tpl) for k in ("E_mu0", "E_sd0", "E_tau", "LEAK_X0")
            if k in m and np.isfinite(m[k]).any()}
    metrics = {
        "classes": hmx.class_summaries(m, hmx.summary_classes(classes)),
        "templates": tmpl, "template_corr": hd.template_corr(tpl).tolist(),
        "template_names": list(hd.TEMPLATE_NAMES),
        "n_mc": cfg["n_mc"], "n_used": int(len(y0)), "n_nonfinite": n_nonfinite,
        "n_clamped": n_clamped, "frac_nonfinite": n_nonfinite / cfg["n_mc"],
        "frac_clamped_coords": n_clamped / (cfg["n_mc"] * y0.shape[1] * (2 if paired else 1)),
        "fit_info": [{k: v for k, v in l["info"].items() if not k.endswith("_idx")} for l in losses],
        "best_val": [float(np.nanmin(l["val"])) for l in losses],
        "final_val": [float(l["val"][-1]) for l in losses],
        "class_counts": {k: int(v.sum()) for k, v in classes.items() if v.dtype == bool},
        "quiet_flag": cfg.get("corpus", "A") == "A" and not 30 <= int(classes["quiet"].sum()) <= 34,
    }
    if paired:
        off = ~classes["disc"]
        metrics["slope_imb_offsupport"] = hmx.slope_on(m["E_tau"], imb, off)
        metrics["slope_imb_all"] = hmx.slope_on(m["E_tau"], imb, np.ones(64, bool))
    if "lt_ate" in extra:
        metrics["lt_ate"] = extra["lt_ate"].tolist()
        metrics["lt_ate_minus_crn_maxabs"] = float(np.abs(extra["lt_ate"] - m["tau_hat_crn"]).max())
        # S6 (Amendment A2): both tau_hats kept; analysis uses the ate-vector one (as for lt)
        metrics["lt_ate_fit_scale"] = extra["lt_ate_fit_scale"].tolist()
        metrics["tau_hat_ate_vector"] = extra["lt_ate"].tolist()
        metrics["tau_hat_sampled"] = np.asarray(m["tau_hat_crn"]).tolist()
        metrics["lt_ate_minus_crn_meanabs"] = float(np.abs(extra["lt_ate"] - m["tau_hat_crn"]).mean())
    floors = hmx.oracle_floors(np.asarray(ref0), None if ref1 is None else np.asarray(ref1),
                               np.random.default_rng(cfg["seed_data"] + 9901))
    t_metrics = time.monotonic() - t0
    emu_max = max(float(np.nanmax(np.abs(m[k]))) for k in ("E_mu0", "E_mu1") if k in m)
    metrics["E_mu_maxabs"] = emu_max
    metrics["diverged"] = bool(not np.isfinite(emu_max) or emu_max > 10.0)   # prereg v1.1 guard
    metrics["timings_s"] = {"data": t_data, "fit": t_fit, "sample": t_sample, "metrics": t_metrics}
    metrics["dataset_id"], metrics["data_hash"] = data["dataset_id"], data["data_hash"]
    metrics["ps_slope_data"] = data["ps_slope"]
    metrics["preproc"] = pre.info()                  # P5: fitted floor value, n floored, scales
    metrics["frugal_flows_file"] = ff_file
    if "calibration" in extra:                       # S8/S9 (Amendment A4)
        metrics["calibration"] = extra["calibration"]
    if cfg.get("stage") == "S10":                    # Amendment A5 per-cell endpoints
        naive = hd.naive_diff(data["Y"], data["X"])       # ORIGINAL logit scale
        metrics["s10"] = {**hmx.s10_endpoints(m["tau_hat"], data["ATE"], naive, classes["disc"],
                                              classes["active_off"], classes["quiet"], extra.get("init_vector")),
                          "naive_map": naive.tolist(), "truth_map": np.asarray(data["ATE"]).tolist(),
                          "tau_hat": np.asarray(m["tau_hat"]).tolist(),
                          "init_vector": np.asarray(extra.get("init_vector")).tolist(),
                          "init_fit_scale": np.asarray(extra.get("init_fit_scale", np.zeros(64))).tolist(),
                          "shift_init": cfg.get("shift_init"), "placebo_covariate": cfg.get("placebo_covariate"),
                          "ps_slope": data["ps_slope"]}
        for k in ("placebo_perm_seed", "placebo_perm_fixed_points", "placebo_corr_uz_thickness"):
            if k in extra:
                metrics["s10"][k] = extra[k]
    for k in ("copula_rank_rule", "copula_width", "copula_reachable", "copula_blind",
              "copula_reachable_per_layer", "copula_mask_width", "copula_dim", "copula_nvars"):
        if k in extra:
            metrics[k] = extra[k]
    return {"metrics": metrics, "maps": m, "classes": classes, "templates": tpl, "floors": floors,
            "imb": imb, "ATE": data["ATE"], "losses": losses}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="max_epochs 5, n_mc 2000")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    if a.smoke:
        cfg.update(max_epochs=5, n_mc=2000)
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    for f in ("FAILED", "metrics.json"):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    sys.stdout = sys.stderr = _Tee(os.path.join(out, "log.txt"))
    t_start = time.monotonic()
    rec = {"config": identity_of(cfg), "run_id": cfg.get("run_id"),
           "identity_sha": cfg.get("identity_sha"), "git": _git(), "versions": _versions(),
           "x64": False, "xla_flags": os.environ.get("XLA_FLAGS", ""), "pid": os.getpid()}
    json.dump(rec, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"cell {cfg.get('run_id')}: {rec['config']}", flush=True)
    try:
        r = run(cfg)
        rec["wall_s"] = time.monotonic() - t_start
        info = r["metrics"]["fit_info"]
        rec["epochs_run"] = [i["n_epochs"] for i in info]
        rec["best_epoch"] = [i["best_epoch"] for i in info]
        rec["best_val"], rec["final_val"] = r["metrics"]["best_val"], r["metrics"]["final_val"]
        json.dump(rec, open(os.path.join(out, "config.json"), "w"), indent=1)
        np.savez(os.path.join(out, "maps.npz"), **r["maps"], ATE=r["ATE"], imb=r["imb"],
                 **{f"cls_{k}": v for k, v in r["classes"].items()}, **r["templates"], **r["floors"])
        np.savez(os.path.join(out, "losses.npz"),
                 **{f"{k}{i}": np.asarray(l[k]) for i, l in enumerate(r["losses"]) for k in ("train", "val")})
        r["metrics"]["complete"] = True
        r["metrics"]["wall_s"] = rec["wall_s"]
        json.dump(r["metrics"], open(os.path.join(out, "metrics.json"), "w"), indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"done in {rec['wall_s']:.1f}s  epochs {rec['epochs_run']}", flush=True)
        return 0
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        open(os.path.join(out, "FAILED"), "w").write(tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())
