"""Fail-closed comparison of MorphoMNIST estimator archives.

The command accepts only complete, one-to-one preset x seed grids with matching
data-generating configurations. It never averages a partial or duplicated cell.

Usage:
    python compare_frengression_ff.py --size 8 --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from frugal_flows.interventions import tau_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from prepare_morphomnist_exps import PRESETS

EFFECT_SCORE_KEYS = (
    "ate_mae",
    "ate_rmse",
    "ate_max_abs_err",
    "ate_mae_on_support",
    "ate_mae_off_support",
    "ate_corr",
    "att_mae",
    "atc_mae",
)
QTE_SCORE_KEYS = (
    "tau_u_rmse_vs_marginal",
    "tau_u_rmse_on_support",
    "tau_u_sd_on_support",
    "tau_u_sd_off_support",
    "true_tau_u_sd_on_support",
)
SCIENTIFIC_SCORE_KEYS = EFFECT_SCORE_KEYS + QTE_SCORE_KEYS
GENERATOR_OVERRIDE_KEYS = (
    "base_shift", "effect_mode", "a_cov", "a_bright", "a_inter",
    "h_shape", "g_shape", "b_quant", "ps_slope", "ps_intercept", "effect",
)
ARM_LABEL = {
    ("location_translation", "mlp"): "ff_loctrans",
    ("location_translation", "n/a"): "ff_loctrans",
    ("flexible_continuous", "mlp"): "ff_flexcont_mlp",
    ("flexible_continuous", "transformer"): "ff_flexcont_transformer",
}
DEFAULT_METHODS = (
    "naive", "ipw", "ols", "aipw", "oracle_ipw",
    "ff_loctrans", "ff_flexcont_mlp", "ff_flexcont_transformer",
    "frengression",
)
METHOD_ORDER = DEFAULT_METHODS


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _require_scores(metrics: dict, source: str) -> dict:
    missing = [key for key in EFFECT_SCORE_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"{source}: missing shared metrics {missing}")
    out = {key: float(metrics[key]) for key in EFFECT_SCORE_KEYS}
    bad = [key for key, value in out.items() if not np.isfinite(value)]
    if bad:
        raise ValueError(f"{source}: non-finite shared metrics {bad}")
    for key in QTE_SCORE_KEYS:
        if key in metrics:
            value = float(metrics[key])
            if not np.isfinite(value):
                raise ValueError(f"{source}: non-finite optional metric {key}")
            out[key] = value
    return out


def _override_signature(config: dict) -> str:
    values = {key: config.get(key) for key in GENERATOR_OVERRIDE_KEYS}
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _backfill_tau_u(metrics: dict, run_dir: str, method: str) -> None:
    """Recompute the existing FF tau(u) metrics when raw samples are available."""
    if "tau_u_rmse_vs_marginal" in metrics:
        return
    arrays_path = os.path.join(run_dir, "arrays.npz")
    if not os.path.exists(arrays_path):
        return
    with np.load(arrays_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if not {"ATE", "TAU_MARGINAL"} <= set(arrays):
        return

    if method == "frengression":
        samples_path = os.path.join(run_dir, "samples.npz")
        if not os.path.exists(samples_path):
            return
        with np.load(samples_path) as samples:
            y0, y1 = samples["y0"], samples["y1"]
    elif {"mc_y0", "mc_y1"} <= set(arrays):
        y0, y1 = arrays["mc_y0"], arrays["mc_y1"]
    else:
        return

    _, curves = tau_curve(y0, y1)
    support = arrays["ATE"] != 0
    true_marg = np.asarray(arrays["TAU_MARGINAL"])
    curve_err = np.asarray(curves) - true_marg
    flat = np.asarray(curves).std(axis=0)
    metrics.update({
        "tau_u_rmse_vs_marginal": float(np.sqrt((curve_err**2).mean())),
        "tau_u_rmse_on_support": float(
            np.sqrt((curve_err[:, support] ** 2).mean())
        ),
        "tau_u_sd_on_support": float(flat[support].mean()),
        "tau_u_sd_off_support": float(flat[~support].mean()),
        "true_tau_u_sd_on_support": float(
            true_marg[:, support].std(axis=0).mean()
        ),
    })


def _model_row(method: str, metrics: dict, record: dict, run: str) -> dict:
    config = record.get("config", {})
    required = ("preset", "size", "seed_data")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{run}: config missing {missing}")
    radius = record.get("effective_radius")
    if radius is None or metrics.get("n_units") is None:
        raise ValueError(f"{run}: missing effective radius or n_units")
    return {
        "preset": config["preset"],
        "method": method,
        "seed_data": int(config["seed_data"]),
        "seed_fit": config.get("seed_fit"),
        "size": int(config["size"]),
        "radius": int(radius),
        "n_units": int(metrics["n_units"]),
        "digit": config.get("digit"),
        "dgp_overrides": _override_signature(config),
        "run": run,
        **_require_scores(metrics, run),
    }


def load_frengression(root: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        run_dir = os.path.join(root, name)
        metrics_path = os.path.join(run_dir, "metrics.json")
        config_path = os.path.join(run_dir, "config.json")
        if not (os.path.exists(metrics_path) and os.path.exists(config_path)):
            continue
        metrics = _read_json(metrics_path)
        if metrics.get("status") != "ok":
            continue
        _backfill_tau_u(metrics, run_dir, "frengression")
        rows.append(_model_row(
            "frengression", metrics, _read_json(config_path), name
        ))
    return rows


def load_ff(root: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        run_dir = os.path.join(root, name)
        metrics_path = os.path.join(run_dir, "metrics.json")
        config_path = os.path.join(run_dir, "config.json")
        if not (os.path.exists(metrics_path) and os.path.exists(config_path)):
            continue
        metrics = _read_json(metrics_path)
        if "ate_mae" not in metrics:
            continue
        record = _read_json(config_path)
        config = record.get("config", {})
        arm = (config.get("arm"), config.get("conditioner", "n/a"))
        method = ARM_LABEL.get(arm)
        if method is None:
            continue
        _backfill_tau_u(metrics, run_dir, method)
        rows.append(_model_row(method, metrics, record, name))
    return rows


def load_baselines(root: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not name.endswith(".csv"):
            continue
        path = os.path.join(root, name)
        with open(path, encoding="utf-8", newline="") as handle:
            for line_number, raw in enumerate(csv.DictReader(handle), start=2):
                source = f"{path}:{line_number}"
                required = {"preset", "method", "n_pixels", "n_units", *EFFECT_SCORE_KEYS}
                missing = sorted(required - set(raw))
                if missing:
                    raise ValueError(f"{source}: baseline CSV lacks {missing}")
                seed_key = "seed_data" if "seed_data" in raw else "seed"
                if seed_key not in raw:
                    raise ValueError(f"{source}: baseline CSV lacks seed")
                n_pixels = int(float(raw["n_pixels"]))
                size = int(round(np.sqrt(n_pixels)))
                if size**2 != n_pixels:
                    raise ValueError(f"{source}: n_pixels={n_pixels} is not square")
                size = int(float(raw["size"])) if raw.get("size") else size
                radius = (
                    int(float(raw["radius"]))
                    if raw.get("radius") else max(1, round(size / 4))
                )
                digit = (
                    None if raw.get("digit") in ("", "None")
                    else int(float(raw["digit"]))
                    if raw.get("digit") is not None
                    else 0
                )
                metrics = _require_scores(raw, source)
                rows.append({
                    "preset": raw["preset"],
                    "method": raw["method"],
                    "seed_data": int(float(raw[seed_key])),
                    "seed_fit": None,
                    "size": size,
                    "radius": radius,
                    "n_units": int(float(raw["n_units"])),
                    "digit": digit,
                    "dgp_overrides": _override_signature({}),
                    "run": name,
                    **metrics,
                })
    return rows


def validate_grid(
    rows: list[dict], presets: list[str], seeds: list[int], methods: list[str]
) -> None:
    """Require one matching row for every requested method/preset/seed."""
    identity = [
        (row["method"], row["preset"], row["seed_data"])
        for row in rows
    ]
    duplicates = sorted(key for key, count in Counter(identity).items() if count != 1)
    if duplicates:
        raise ValueError(f"duplicate comparison identities: {duplicates}")

    present = set(identity)
    expected = {
        (method, preset, seed)
        for method in methods for preset in presets for seed in seeds
    }
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing:
        raise ValueError(f"incomplete comparison grid; missing {missing}")
    if unexpected:
        raise ValueError(f"comparison grid contains unexpected cells: {unexpected}")

    for preset in presets:
        for seed in seeds:
            cell = [
                row for row in rows
                if row["preset"] == preset and row["seed_data"] == seed
            ]
            signatures = {
                (row["size"], row["radius"], row["n_units"], row["digit"],
                 row["dgp_overrides"])
                for row in cell
            }
            if len(signatures) != 1:
                detail = {row["method"]: (
                    row["size"], row["radius"], row["n_units"], row["digit"],
                    row["dgp_overrides"]
                ) for row in cell}
                raise ValueError(
                    f"DGP mismatch for {preset} seed {seed}: {detail}"
                )


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for preset in sorted({row["preset"] for row in rows}):
        for method in METHOD_ORDER:
            cell = [
                row for row in rows
                if row["preset"] == preset and row["method"] == method
            ]
            if not cell:
                continue
            entry = {
                "preset": preset,
                "method": method,
                "n_seeds": len(cell),
                "seeds": ",".join(str(row["seed_data"]) for row in cell),
            }
            for key in SCIENTIFIC_SCORE_KEYS:
                values = [row[key] for row in cell if key in row]
                if not values:
                    continue
                entry[f"{key}_mean"] = float(np.mean(values))
                entry[f"{key}_sd"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
                )
            output.append(entry)
    return output


def print_table(aggregates: list[dict], metric: str) -> None:
    presets = sorted({row["preset"] for row in aggregates})
    methods = [method for method in METHOD_ORDER if any(
        row["method"] == method for row in aggregates
    )]
    width = max((len(preset) for preset in presets), default=20) + 2
    print(f"\n{metric} (mean +/- sd over seeds)\n")
    print("preset".ljust(width) + "".join(method.rjust(24) for method in methods))
    for preset in presets:
        line = preset.ljust(width)
        for method in methods:
            cell = next(
                row for row in aggregates
                if row["preset"] == preset and row["method"] == method
            )
            mean = cell.get(f"{metric}_mean")
            sd = cell.get(f"{metric}_sd")
            value = "-" if mean is None else (
                f"{mean:.4f}+/-{sd:.4f} [{cell['n_seeds']}]"
                if sd is not None and np.isfinite(sd)
                else f"{mean:.4f} [{cell['n_seeds']}]"
            )
            line += value.rjust(24)
        print(line)


def _write_csv(path: str, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(rows: list[dict], aggregates: list[dict], out_dir: str, metric: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "runs.csv"), rows)
    _write_csv(os.path.join(out_dir, "summary.csv"), aggregates)

    methods = [method for method in METHOD_ORDER if any(
        row["method"] == method for row in aggregates
    )]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(f"# MorphoMNIST recovery: {metric}\n\n")
        handle.write("Complete one-to-one grid; mean +/- sample SD over seeds.\n\n")
        handle.write("| preset | " + " | ".join(methods) + " |\n")
        handle.write("|" + "---|" * (len(methods) + 1) + "\n")
        for preset in sorted({row["preset"] for row in aggregates}):
            cells = []
            for method in methods:
                cell = next(row for row in aggregates if (
                    row["preset"] == preset and row["method"] == method
                ))
                mean = cell.get(f"{metric}_mean")
                sd = cell.get(f"{metric}_sd")
                cells.append("-" if mean is None else (
                    f"{mean:.4f} +/- {sd:.4f} ({cell['n_seeds']})"
                    if sd is not None and np.isfinite(sd)
                    else f"{mean:.4f} ({cell['n_seeds']})"
                ))
            handle.write(f"| {preset} | " + " | ".join(cells) + " |\n")


def plot_maps(rows: list[dict], freng_root: str, ff_root: str, out_dir: str, size: int) -> None:
    def mean_map(method: str, preset: str, root: str):
        maps, truth = [], None
        for row in rows:
            if row["method"] != method or row["preset"] != preset:
                continue
            path = os.path.join(root, row["run"], "arrays.npz")
            if not os.path.exists(path):
                continue
            with np.load(path) as arrays:
                maps.append(arrays["tau_hat"])
                truth = arrays["ATE"]
        return (np.mean(maps, axis=0) if maps else None), truth

    for preset in sorted({row["preset"] for row in rows}):
        estimates = []
        truth = None
        for method, label, root in (
            ("ff_loctrans", "FF location translation", ff_root),
            ("ff_flexcont_mlp", "FF flexible / MLP", ff_root),
            ("ff_flexcont_transformer", "FF flexible / transformer", ff_root),
            ("frengression", "Frengression", freng_root),
        ):
            estimate, method_truth = mean_map(method, preset, root)
            if truth is None and method_truth is not None:
                truth = method_truth
            if estimate is not None:
                estimates.append((estimate, label))
        if truth is None:
            continue
        panels = [(truth, "true ATE"), *estimates]
        figure, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.4))
        axes = np.atleast_1d(axes)
        limit = max(np.abs(image).max() for image, _ in panels) or 1.0
        for axis, (image, title) in zip(axes, panels):
            shown = axis.imshow(np.asarray(image).reshape(size, size), cmap="RdBu_r",
                                vmin=-limit, vmax=limit)
            axis.set_title(title)
            axis.axis("off")
            figure.colorbar(shown, ax=axis, fraction=0.046)
        figure.suptitle(f"{preset} (seed averaged)")
        figure.tight_layout()
        figure.savefig(os.path.join(out_dir, f"ate_maps_{preset}.png"), dpi=110)
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    runs = os.path.join(SCRIPT_DIR, "runs")
    parser.add_argument("--frengression-root", default=os.path.join(runs, "frengression"))
    parser.add_argument("--ff-root", default=os.path.join(runs, "exp_ate_recovery"))
    parser.add_argument("--baselines-root", default=os.path.join(runs, "baselines"))
    parser.add_argument("--out", default=os.path.join(runs, "comparison"))
    parser.add_argument("--metric", default="ate_mae", choices=SCIENTIFIC_SCORE_KEYS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--presets", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    rows = (
        load_frengression(args.frengression_root)
        + load_ff(args.ff_root)
        + load_baselines(args.baselines_root)
    )
    methods = list(dict.fromkeys(args.methods))
    rows = [row for row in rows if (
        row["size"] == args.size
        and row["seed_data"] in args.seeds
        and row["preset"] in args.presets
        and row["method"] in methods
    )]
    validate_grid(rows, args.presets, args.seeds, methods)
    missing_metric = [
        (row["method"], row["preset"], row["seed_data"])
        for row in rows if args.metric not in row
    ]
    if missing_metric:
        raise ValueError(
            f"requested metric {args.metric!r} is unavailable for {missing_metric}"
        )
    aggregates = aggregate(rows)
    print_table(aggregates, args.metric)
    write_outputs(rows, aggregates, args.out, args.metric)
    if not args.no_plots:
        plot_maps(rows, args.frengression_root, args.ff_root, args.out, args.size)


if __name__ == "__main__":
    main()
