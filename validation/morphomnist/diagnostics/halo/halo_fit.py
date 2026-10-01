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
        flow, losses = train_frugal_flow(
            key=sub, y=jnp.asarray(Y), u_z=jnp.asarray(u_z), condition=jnp.asarray(X),
            causal_model="flexible_continuous", RQS_knots=8, nn_depth=1, nn_width=50, flow_layers=4,
            learning_rate=cfg["lr"], max_epochs=cfg["max_epochs"], max_patience=cfg["patience"],
            batch_size=cfg["batch"], show_progress=False,
            fit_kwargs={"ema_decay": None, "wall_cap_s": None},
            causal_model_args={"RQS_knots": cfg["knots"], "nn_depth": cfg["depth"],
                               "nn_width": cfg["width"], "flow_layers": cfg["layers"],
                               "conditioner": "mlp"})

        def draw(seed):
            r = interventional_samples(jr.key(seed), flow, cond_dim=1, n_mc=cfg["n_mc"],
                                       outcome_transform=pre.transform, dim_y=Y.shape[1])
            return np.asarray(r["y0"]), np.asarray(r["y1"]), int(r["n_clamped"])
        return draw, [losses], extra
    import halo_models as hm
    dist, losses = hm.fit_arm(cfg, Y, X)
    if arm == "lt":
        extra["lt_ate"] = hm.loccond_ate(dist)

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
    floors = hmx.oracle_floors(np.asarray(ref0), None if ref1 is None else np.asarray(ref1),
                               np.random.default_rng(cfg["seed_data"] + 9901))
    t_metrics = time.monotonic() - t0
    metrics["timings_s"] = {"data": t_data, "fit": t_fit, "sample": t_sample, "metrics": t_metrics}
    metrics["dataset_id"], metrics["data_hash"] = data["dataset_id"], data["data_hash"]
    metrics["ps_slope_data"] = data["ps_slope"]
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
    rec = {"config": {k: cfg.get(k) for k in IDENTITY_KEYS}, "run_id": cfg.get("run_id"),
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
