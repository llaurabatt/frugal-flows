"""Copy one dataset's fits of every model into examples/ (weights, config, metrics, arrays, W&B link).

Default: experiment E2, dataset k = 1 of the n = 5000, 8x8 all-digit grid (runs/sub5k and
runs/sub5k_frengression), the Gaussian spline's best E2 dataset with every model present. The
Gaussian-spline fit also gets a model_spec.json so ``frugal_flows.load_gaussian_flow`` loads it
without the MorphoMNIST runner; the script checks the reload matches the runner's log-density.

  python make_examples.py [--exp exp2_confounded_homogeneous] [--k 1] [--runs-root DIR] [--out ../../../../examples/morphomnist_8x8_n5000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, MM)
CODES = {("flexible_continuous", "none", "zero"): "U-flex-raw", ("flexible_continuous", "standardize", "zero"): "U-flex-std",
         ("location_translation", "none", "zero"): "U-LT-raw",
         ("flexible_continuous_gaussian", "standardize", "zero"): "G-flex-std",
         ("location_translation_gaussian", "standardize", "scalar"): "G-LT-std",
         ("location_translation_gaussian", "standardize", "naive"): "G-LT-head"}
FILES = ("model.eqx", "model.pt", "config.json", "metrics.json", "wandb.json", "arrays.npz")


def find(exp, k, runs_root):
    out = {}
    for root, kind in (("sub5k", "flow"), ("sub5k_frengression", "freng")):
        for d in sorted(glob.glob(os.path.join(runs_root, root, "*"))):
            cp = os.path.join(d, "config.json")
            if not (os.path.exists(cp) and os.path.exists(os.path.join(d, "metrics.json"))):
                continue
            c = json.load(open(cp))["config"]
            if c.get("preset") != exp or int(c["seed_assign"]) != k or int(c["seed_fit"]) != k or c.get("n") != 5000:
                continue
            code = "frengression" if kind == "freng" else CODES.get((c["arm"], c.get("y_scaling", "none"), c.get("shift_init", "zero")))
            if code and any(os.path.exists(os.path.join(d, f)) for f in ("model.eqx", "model.pt")):
                out[code] = d
    return out


def gaussian_spec(d, dest):
    import exp_ate_recovery as E
    import jax.numpy as jnp
    import dataset_store
    from frugal_flows.gaussian_scale import load_gaussian_flow, normal_scores_from_uniform, save_gaussian_flow
    c = json.load(open(os.path.join(d, "config.json")))["config"]
    cfg = E.Config(**{k: v for k, v in c.items() if k in {f.name for f in E.fields(E.Config)}})
    data = dataset_store.build_for_run(d)
    flow = E.load_model(d)
    ot = E.outcome_transform_for(cfg, data)
    u_z = np.load(os.path.join(d, "arrays.npz"))["u_z"]
    build = dict(dim_y=cfg.size ** 2, nvars=int(u_z.shape[1]), cond_dim=1, margin="flexible",
                 RQS_knots=cfg.copula_rqs_knots, nn_depth=cfg.copula_nn_depth, nn_width=cfg.copula_nn_width,
                 flow_layers=cfg.copula_flow_layers,
                 causal_model_args={"RQS_knots": cfg.rqs_knots, "nn_depth": cfg.nn_depth, "nn_width": cfg.nn_width,
                                    "flow_layers": cfg.flow_layers, "interval": 5.0})
    save_gaussian_flow(dest, flow, build, outcome_transform=ot)
    flow2, ot2 = load_gaussian_flow(dest)
    x = jnp.hstack([ot.forward(jnp.asarray(data["Y"])), normal_scores_from_uniform(jnp.asarray(u_z))])
    c_ = jnp.asarray(data["X"]).reshape(-1, 1)
    assert np.allclose(flow.log_prob(x, c_), flow2.log_prob(x, c_), atol=1e-5), "reload mismatch"
    assert np.allclose(ot2.forward(jnp.asarray(data["Y"])), ot.forward(jnp.asarray(data["Y"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp2_confounded_homogeneous")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--runs-root", default=os.path.join(MM, "runs"), help="folder holding sub5k/ and sub5k_frengression/")
    ap.add_argument("--out", default=os.path.abspath(os.path.join(MM, "..", "..", "examples", "morphomnist_8x8_n5000")))
    a = ap.parse_args()
    runs = find(a.exp, a.k, a.runs_root)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for code, d in sorted(runs.items()):
        dest = os.path.join(a.out, code)
        os.makedirs(dest, exist_ok=True)
        for f in FILES:
            if os.path.exists(os.path.join(d, f)):
                shutil.copy2(os.path.join(d, f), dest)
        if code == "G-flex-std":
            gaussian_spec(d, dest)
        m = json.load(open(os.path.join(d, "metrics.json")))
        rows.append({"model": code, "ate_mae": round(m["ate_mae"], 4), "source_run": os.path.basename(d)})
    json.dump(rows, open(os.path.join(a.out, "models.json"), "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
