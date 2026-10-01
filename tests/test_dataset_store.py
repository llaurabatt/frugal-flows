"""validation/morphomnist/dataset_store (2026-10-01): runs no longer save Y / ITE; the dataset is rebuilt
from config.json and checked against the recorded dataset_id / data_hash."""
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import pytest

MM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "validation", "morphomnist")
sys.path.insert(0, os.path.abspath(MM))

E = pytest.importorskip("exp_ate_recovery")
import dataset_store as DS  # noqa: E402


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A fake run folder in the post-2026-10-01 layout: config.json + arrays.npz without Y / ITE."""
    monkeypatch.setattr(DS, "CACHE_DIR", str(tmp_path / "cache"))
    cfg = E.Config(preset="exp2_confounded_homogeneous", size=4, digit=0, n=300, seed_data=1, seed_assign=1)
    data = E.build_data(cfg)
    d = tmp_path / "run"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"config": asdict(cfg), "dataset_id": data["dataset_id"],
                                               "data_hash": data["data_hash"]}, default=str))
    np.savez(d / "arrays.npz", tau_hat=np.zeros(16), X=np.asarray(data["X"]), ATE=np.asarray(data["ATE"]))
    return str(d), data


def test_rebuild_matches_the_recorded_dataset(run_dir):
    d, data = run_dir
    rebuilt = DS.build_for_run(d)
    assert rebuilt["data_hash"] == data["data_hash"]
    assert np.array_equal(rebuilt["Y"], data["Y"]) and np.array_equal(rebuilt["ITE"], data["ITE"])


def test_run_arrays_fills_y_and_ite_and_caches(run_dir):
    d, data = run_dir
    a = DS.run_arrays(d)
    assert "Y" in a.files and "ITE" in a.files and "tau_hat" in a.files
    assert np.array_equal(a["Y"], data["Y"]) and np.array_equal(a["ITE"], data["ITE"])
    assert os.path.exists(os.path.join(DS.CACHE_DIR, f"{data['dataset_id']}.npz"))
    b = DS.run_arrays(d)                                   # second call reads the cache
    assert np.array_equal(b["Y"], a["Y"])


def test_wrong_hash_is_refused(run_dir):
    d, _ = run_dir
    rec = json.load(open(os.path.join(d, "config.json")))
    rec["data_hash"] = "000000000000"
    json.dump(rec, open(os.path.join(d, "config.json"), "w"), default=str)
    with pytest.raises(ValueError, match="data_hash"):
        DS.build_for_run(d)
