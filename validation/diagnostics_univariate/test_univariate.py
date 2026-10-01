"""S11 unit checks. Run in env frugal-flows-halo with PYTHONPATH=<frugal-flows-gauss worktree>:

  PYTHONPATH=$PWD micromamba run -n frugal-flows-halo python -m pytest validation/diagnostics_univariate/test_univariate.py -q
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from npz_io import load_npz, save_npz  # noqa: E402


def _synthetic(n=500, seed=0, disc=True):
    """Small confounded dataset with the causl output contract (no R needed)."""
    rng = np.random.default_rng(seed)
    zc = rng.gamma(2.0, 1.0, size=(n, 2))
    zd = rng.integers(0, 2, size=(n, 2)) if disc else None
    lin = -0.5 + 0.4 * zc[:, 0] + (0.5 * zd[:, 0] if disc else 0)
    x = (rng.uniform(size=n) < 1 / (1 + np.exp(-lin))).astype(float)[:, None]
    y = (1.0 + 1.0 * x[:, 0] + 0.5 * zc[:, 0] + rng.normal(size=n))[:, None]
    naive = float(y[x[:, 0] == 1, 0].mean() - y[x[:, 0] == 0, 0].mean())
    meta = dict(model="synthetic", generator="test", causal_params=[1.0, 1.0], true_ate=1.0, ate_arg=1.0,
                n=n, seed=seed, naive=naive, ols=float("nan"))
    return dict(Y=y, X=x, Z_cont=zc, Z_disc=zd, meta=meta)


@pytest.mark.parametrize("disc", [True, False])
def test_npz_round_trip(tmp_path, disc):
    d = _synthetic(disc=disc)
    p = str(tmp_path / "d.npz")
    save_npz(p, d)
    r = load_npz(p)
    for k in ("Y", "X", "Z_cont"):
        np.testing.assert_array_equal(r[k], d[k])
    if disc:
        np.testing.assert_array_equal(r["Z_disc"], d["Z_disc"])
        assert np.issubdtype(r["Z_disc"].dtype, np.integer)
    else:
        assert r["Z_disc"] is None
    assert r["meta"]["true_ate"] == 1.0 and r["meta"]["causal_params"] == [1.0, 1.0]


def test_shift_to_data_scale():
    import fit_one
    t = SimpleNamespace(_sd=np.array([2.5]))
    assert fit_one.to_data_scale([0.4], t) == [pytest.approx(1.0)]
    assert fit_one.to_data_scale(None, t) is None


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("s11") / "syn.npz")
    save_npz(p, _synthetic(n=500, seed=1, disc=True))
    return p


@pytest.mark.parametrize("arm", ["gaussian", "location_translation", "flexible_continuous",
                                 "location_translation_gaussian", "flexible_continuous_gaussian"])
def test_arm_dispatch_finite_ate(tmp_path, data_path, arm):
    import fit_one
    out = str(tmp_path / arm)
    rc = fit_one.main(["--data", data_path, "--arm", arm, "--seed-fit", "101", "--out", out, "--max-epochs", "3"])
    assert rc == 0, open(os.path.join(out, "FAILED")).read() if os.path.exists(os.path.join(out, "FAILED")) else ""
    r = json.load(open(os.path.join(out, "result.json")))
    assert np.isfinite(r["ate_hat"])
    assert r["precision"]["jax_enable_x64"] is True and r["precision"]["y_fit_dtype"] == "float64"
    assert r["epochs_run"] == 3
    if arm in fit_one.SHIFT_ARMS:
        # a location model's paired CRN contrast is exactly the shift (on the Y scale)
        assert abs(r["shift_data_scale"][0] - r["ate_hat"]) < 1e-6
    else:
        assert r["shift_data_scale"] is None
