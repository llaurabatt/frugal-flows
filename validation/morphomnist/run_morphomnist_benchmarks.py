"""Run the complete MorphoMNIST benchmark matrix from one command.

The established implementations remain authoritative:

* ``baselines.py`` supplies naive, IPW, OLS, AIPW, and oracle IPW;
* ``exp_ate_recovery.py`` supplies all three Frugal Flow cells;
* ``exp_frengression_recovery.py`` supplies the one additional comparator.

This file only coordinates those runners and then invokes the fail-closed
comparison. It deliberately contains no estimator implementation.

Usage (from ``validation/morphomnist``)::

    python run_morphomnist_benchmarks.py --smoke
    python run_morphomnist_benchmarks.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

from prepare_morphomnist_exps import PRESETS

CLASSICAL_METHODS = ("naive", "ipw", "ols", "aipw", "oracle_ipw")
FF_CELLS = {
    "ff_loctrans": ("location_translation", "mlp"),
    "ff_flexcont_mlp": ("flexible_continuous", "mlp"),
    "ff_flexcont_transformer": ("flexible_continuous", "transformer"),
}
ALL_METHODS = (*CLASSICAL_METHODS, *FF_CELLS, "frengression")
DISTRIBUTIONAL_METHODS = (
    "ff_flexcont_mlp", "ff_flexcont_transformer", "frengression",
)
DEFAULT_REPORTS = ("ate_mae", "tau_u_rmse_vs_marginal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", default=list(PRESETS),
                        choices=list(PRESETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS),
                        choices=list(ALL_METHODS))
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument("--all-digits", action="store_true")
    parser.add_argument("--basis", default="poly3", choices=("linear", "poly3", "poly5"))
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "runs" / "benchmark"))
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_REPORTS),
                        choices=list(DEFAULT_REPORTS))
    parser.add_argument("--resume", action="store_true",
                        help="skip completed cells in an existing matching benchmark")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="all methods and presets at tiny non-reporting budgets (seed 0, K=16)",
    )

    # Optional budget overrides make the driver testable without changing the
    # scientifically chosen defaults in either established runner.
    parser.add_argument("--ff-max-epochs", type=int, default=None)
    parser.add_argument("--ff-marginal-max-epochs", type=int, default=None)
    parser.add_argument("--ff-n-mc", type=int, default=None)
    parser.add_argument("--frengression-num-iters", type=int, default=None)
    parser.add_argument("--frengression-n-mc", type=int, default=None)
    parser.add_argument("--frengression-threads", type=int, default=4)
    return parser


def _apply_smoke(args: argparse.Namespace) -> None:
    """Use every code path at budgets that assert plumbing, not recovery."""
    args.presets = list(PRESETS)
    args.seeds = [0]
    args.methods = list(ALL_METHODS)
    args.size = 4
    args.n = 300
    args.ff_max_epochs = 2
    args.ff_marginal_max_epochs = 2
    args.ff_n_mc = 200
    args.frengression_num_iters = 20
    args.frengression_n_mc = 800
    args.frengression_threads = 1


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _plan(args: argparse.Namespace, digit: int | None) -> dict:
    return {
        "schema": 1,
        "git_commit": _git_commit(),
        "presets": list(args.presets),
        "seeds": list(args.seeds),
        "methods": list(args.methods),
        "size": args.size,
        "n": args.n,
        "digit": digit,
        "basis": args.basis,
        "ff_max_epochs": args.ff_max_epochs,
        "ff_marginal_max_epochs": args.ff_marginal_max_epochs,
        "ff_n_mc": args.ff_n_mc,
        "frengression_num_iters": args.frengression_num_iters,
        "frengression_n_mc": args.frengression_n_mc,
    }


def _prepare_output(root: Path, plan: dict, resume: bool) -> None:
    manifest = root / "manifest.json"
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError(f"{root} is not empty; use --resume or a new --output-root")
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError(
                f"{root} belongs to a different benchmark configuration; "
                "use a new --output-root"
            )
    root.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def _run_baselines(args: argparse.Namespace, root: Path, digit: int | None) -> None:
    wanted = set(args.methods) & set(CLASSICAL_METHODS)
    if not wanted:
        return

    import baselines

    for preset in args.presets:
        for seed in args.seeds:
            path = root / f"{preset}_s{seed}_{args.basis}.csv"
            if args.resume and path.exists():
                print(f"already completed: classical baselines | {preset} | seed {seed}")
                continue
            print(f"running: classical baselines | {preset} | seed {seed}")
            rows = baselines.run_one(preset, args.size, seed, args.basis, args.n, digit)
            selected = [
                {
                    **row,
                    "seed_data": seed,
                    "size": args.size,
                    "radius": max(1, round(args.size / 4)),
                    "digit": digit,
                }
                for row in rows if row["method"] in wanted
            ]
            _write_rows(path, selected)


def _ff_kwargs(args: argparse.Namespace) -> dict:
    values = {}
    for arg, field in (
        ("ff_max_epochs", "max_epochs"),
        ("ff_marginal_max_epochs", "marginal_max_epochs"),
        ("ff_n_mc", "n_mc"),
    ):
        value = getattr(args, arg)
        if value is not None:
            values[field] = value
    return values


def _run_ff(args: argparse.Namespace, root: Path, digit: int | None) -> None:
    wanted = [method for method in FF_CELLS if method in args.methods]
    if not wanted:
        return

    import exp_ate_recovery as ff

    root.mkdir(parents=True, exist_ok=True)
    done = ff.completed_cells(str(root)) if args.resume else set()
    for preset in args.presets:
        for seed in args.seeds:
            for method in wanted:
                arm, conditioner = FF_CELLS[method]
                cfg = ff.Config(
                    preset=preset, arm=arm, conditioner=conditioner,
                    size=args.size, n=args.n, digit=digit,
                    seed_data=seed, seed_fit=seed, **_ff_kwargs(args),
                )
                identity = tuple(asdict(cfg).get(key) for key in ff.CELL_IDENTITY)
                if identity in done:
                    print(f"already completed: {method} | {preset} | seed {seed}")
                    continue
                print(f"running: {method} | {preset} | seed {seed}")
                ff.jax.config.update("jax_enable_x64", cfg.x64)
                ff.run_one(cfg, runs_root=str(root))


def _frengression_kwargs(args: argparse.Namespace) -> dict:
    values = {"threads": args.frengression_threads}
    if args.frengression_num_iters is not None:
        values["num_iters"] = args.frengression_num_iters
    if args.frengression_n_mc is not None:
        values["n_mc"] = args.frengression_n_mc
    return values


def _run_frengression(args: argparse.Namespace, root: Path, digit: int | None) -> None:
    if "frengression" not in args.methods:
        return

    import exp_frengression_recovery as freng

    root.mkdir(parents=True, exist_ok=True)
    done = freng.completed_cells(str(root)) if args.resume else set()
    for preset in args.presets:
        for seed in args.seeds:
            cfg = freng.Config(
                preset=preset, size=args.size, n=args.n, digit=digit,
                seed_data=seed, seed_fit=seed, **_frengression_kwargs(args),
            )
            if freng.cell_identity(cfg) in done:
                print(f"already completed: frengression | {preset} | seed {seed}")
                continue
            print(f"running: frengression | {preset} | seed {seed}")
            freng.run_one(cfg, runs_root=str(root), plots=not args.no_plots)


def _compare(args: argparse.Namespace, root: Path) -> None:
    import compare_frengression_ff as compare

    for metric in args.metrics:
        methods = list(args.methods)
        if metric == "tau_u_rmse_vs_marginal":
            methods = [method for method in methods if method in DISTRIBUTIONAL_METHODS]
        if not methods:
            print(f"no compatible methods selected for {metric}; skipping report")
            continue
        compare_args = [
            "--ff-root", str(root / "ff"),
            "--frengression-root", str(root / "frengression"),
            "--baselines-root", str(root / "baselines"),
            "--out", str(root / "comparison" / metric),
            "--metric", metric,
            "--size", str(args.size),
            "--seeds", *(str(seed) for seed in args.seeds),
            "--presets", *args.presets,
            "--methods", *methods,
        ]
        if args.no_plots or metric != "ate_mae":
            compare_args.append("--no-plots")
        compare.main(compare_args)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke:
        _apply_smoke(args)
    args.methods = list(dict.fromkeys(args.methods))
    digit = None if args.all_digits else args.digit
    root = Path(args.output_root).resolve()
    _prepare_output(root, _plan(args, digit), args.resume)

    _run_baselines(args, root / "baselines", digit)
    _run_ff(args, root / "ff", digit)
    _run_frengression(args, root / "frengression", digit)
    _compare(args, root)
    print(f"\nbenchmark complete: {root / 'comparison'}")


if __name__ == "__main__":
    main()
