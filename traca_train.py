"""
Train-only pass for decoupled train/eval workflow.

For each (fold, radius), learns τ and saves to disk.  Never evaluates against
target — that happens in ``traca_run_evaluation.py``.

The saved artifact is a pickle dict keyed by ``fold_<k>`` → ``{axis}_{value}``
→ per-entry dict with ``tau``, ``opt_result``, ``test_indices``, ``shift_spec``,
``config_snapshot``, and ``training_metadata``.

Usage
-----
    python traca_train.py --config configs/atce/gaussian_z_entrywise_full.yaml \
        --sweep_axis eps --values 0.0 0.1 0.2 0.3 \
        --shift_type noise_mean --shift_magnitude 0.3 --shift_node 0

    python traca_train.py --config configs/atce/gaussian_z_entrywise_full.yaml \
        --sweep_axis eta --values 0.0 0.1 0.2 0.3

Saves
-----
    {out_dir}/traca_cv_results.pkl  — full results dict (joblib)
    {out_dir}/bundle.pkl            — source bundle (for eval to load)
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lan_scm  # noqa: E402 — pickle resolution
from experiments.run import (
    _build_mechanism_set,
    _build_environment_set,
    _build_constructive_class,
    _build_optim_config,
    _load_bundle,
    _git_provenance,
)
from traca.optim import fit_empirical, fit_gaussian, OptimConfig
from traca.query import query_family_from_config


# ---------------------------------------------------------------------------
# Helpers inlined from removed scripts (traca_cv, traca_radius_sweep)
# ---------------------------------------------------------------------------

def _split_bundle(bundle, train_idx: np.ndarray, val_idx: np.ndarray):
    """Return two bundles with noise_samples split by train/val indices.

    Both bundles share the same SCM structure (W, A, etc.) — only the sample
    arrays differ. We build lightweight wrapper objects rather than re-running
    lan_scm to avoid re-sampling.
    """
    train_bundle = copy.copy(bundle)
    val_bundle = copy.copy(bundle)

    n_iota = bundle.n_interventions()
    ns = bundle.noise_samples
    xs = bundle.endogenous_samples
    ns_list = [ns[i] for i in range(n_iota)] if isinstance(ns, dict) else list(ns)
    xs_list = [xs[i] for i in range(n_iota)] if isinstance(xs, dict) else list(xs)
    train_noise = [U[train_idx] for U in ns_list]
    val_noise   = [U[val_idx]   for U in ns_list]
    train_X     = [X[train_idx] for X in xs_list]
    val_X       = [X[val_idx]   for X in xs_list]

    train_bundle.noise_samples       = train_noise
    val_bundle.noise_samples         = val_noise
    train_bundle.endogenous_samples  = train_X
    val_bundle.endogenous_samples    = val_X
    train_bundle.n                   = len(train_idx)
    val_bundle.n                     = len(val_idx)

    return train_bundle, val_bundle


def _substitute_eps(cfg: dict, eps: float) -> dict:
    """Return a deep copy of cfg with environment eps set to *eps*."""
    cfg = copy.deepcopy(cfg)
    cfg["ambiguity"]["environment"]["eps"] = eps
    return cfg


def _substitute_eta(cfg: dict, eta: float) -> dict:
    """Return a deep copy of cfg with mechanism budget set to *eta*."""
    cfg = copy.deepcopy(cfg)
    mech = cfg["ambiguity"]["mechanism"]
    t = mech["type"]

    if t == "FrobeniusBall":
        mech["eta"] = eta
    elif t == "EntrywiseBox":
        B = np.array(mech["B"], dtype=float)
        B[B != 0] = eta
        mech["B"] = B.tolist()
    elif t == "RowBudget":
        mech["rho"] = {k: eta for k in mech["rho"]}
    elif t == "ColumnBudget":
        if isinstance(mech["c"], dict):
            mech["c"] = {k: eta for k in mech["c"]}
        else:
            mech["c"] = eta
    else:
        raise ValueError(f"Unknown mechanism type for eta sweep: {t!r}")

    return cfg


def _substitute_eps_eta(cfg: dict, eps: float, eta: float) -> dict:
    """Return a deep copy of cfg with both eps and eta set."""
    cfg = _substitute_eps(cfg, eps)
    mech = cfg["ambiguity"]["mechanism"]
    t = mech["type"]
    if t == "FrobeniusBall":
        mech["eta"] = eta
    elif t == "EntrywiseBox":
        B = np.array(mech["B"], dtype=float)
        B[B != 0] = eta
        mech["B"] = B.tolist()
    elif t == "RowBudget":
        mech["rho"] = {k: eta for k in mech["rho"]}
    elif t == "ColumnBudget":
        if isinstance(mech["c"], dict):
            mech["c"] = {k: eta for k in mech["c"]}
        else:
            mech["c"] = eta
    else:
        raise ValueError(f"Unknown mechanism type for eta sweep: {t!r}")
    return cfg


# ---------------------------------------------------------------------------
# Train one (fold, radius)
# ---------------------------------------------------------------------------

def _train_one_fold_radius(
    base_cfg: dict,
    train_bundle,
    test_indices: np.ndarray,
    sweep_axis: str,
    radius_value: float,
    objective: str,
    shift_spec: dict | None,
    show_progress: bool = False,
    progress_desc: str = "",
) -> dict:
    """Train τ for one (fold, radius).  Returns dict for pickle storage.

    Parameters
    ----------
    base_cfg : experiment YAML dict (radius will be substituted)
    train_bundle : training-fold SCMBundle
    test_indices : held-out sample indices (stored, never used here)
    sweep_axis : "eps" or "eta"
    radius_value : absolute radius value to substitute
    objective : "gaussian" or "empirical"
    shift_spec : target identity dict (stored, never used during training)
    show_progress : show live tqdm progress bar for the optimizer
    progress_desc : label for the progress bar

    Returns
    -------
    dict with tau, opt_result, test_indices, shift_spec, config_snapshot,
    training_metadata
    """
    # Substitute radius into config
    if sweep_axis == "eps":
        cfg = _substitute_eps(base_cfg, radius_value)
    else:
        cfg = _substitute_eta(base_cfg, radius_value)

    d = train_bundle.d
    N_train = train_bundle.n

    # For Gaussian objective, estimate exogenous moments from this fold's
    # training samples so each fold learns a genuinely different τ (real CV).
    # _split_bundle gives train_bundle.noise_samples containing only train_idx
    # rows — no held-out leakage.  Empirical path already uses per-fold samples.
    if objective == "gaussian":
        ns = train_bundle.noise_samples
        U_train_obs = ns[0] if isinstance(ns, dict) else ns[0]
        train_bundle.noise_mean = np.mean(U_train_obs, axis=0)
        train_bundle.noise_cov = np.cov(U_train_obs, rowvar=False)

    aw = _build_mechanism_set(cfg["ambiguity"]["mechanism"], d)
    ae = _build_environment_set(
        cfg["ambiguity"]["environment"],
        train_bundle.noise_mean, train_bundle.noise_cov, N_train,
    )
    cc = _build_constructive_class(cfg["constructive_class"], d)
    optim_cfg = _build_optim_config(cfg.get("training", {}))
    optim_cfg.show_progress = show_progress
    optim_cfg.progress_desc = progress_desc
    query_family = query_family_from_config(cfg)

    if objective == "empirical":
        result = fit_empirical(
            train_bundle, aw, ae, cc, optim_cfg, query_family=query_family
        )
    else:
        result = fit_gaussian(
            train_bundle, aw, ae, cc, optim_cfg, query_family=query_family
        )

    return {
        "tau": result.tau,
        "opt_result": result,
        "test_indices": test_indices,
        "shift_spec": shift_spec,
        "config_snapshot": cfg,
        "training_metadata": {
            "converged": bool(result.converged),
            "n_iters": int(result.n_iters),
            "final_loss": float(result.final_loss),
            "initial_loss": float(result.initial_loss) if result.initial_loss is not None else None,
            "objective": objective,
        },
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _mech_radius_label(mech_cfg: dict) -> str:
    """Extract a human-readable η label from the mechanism ambiguity config."""
    t = mech_cfg.get("type", "?")
    if t == "FrobeniusBall":
        return str(mech_cfg.get("eta", "?"))
    if t == "EntrywiseBox":
        B = np.array(mech_cfg.get("B", []), dtype=float)
        nonzero = B[B != 0]
        if len(nonzero) > 0 and np.all(nonzero == nonzero[0]):
            return str(float(nonzero[0]))
        return "box"
    return "?"


def train_cv(
    config_path: Path,
    sweep_axis: str,
    radius_values: list[float],
    out_dir: Path,
    k_folds: int = 5,
    include_baselines: bool = True,
    shift_spec: dict | None = None,
    show_progress: bool = False,
) -> Path:
    """K-fold training pass.  Returns path to saved results pickle.

    Parameters
    ----------
    config_path : experiment YAML
    sweep_axis : "eps" or "eta"
    radius_values : absolute radius values to sweep
    out_dir : output directory for pickle + bundle
    k_folds : number of CV folds
    include_baselines : if True, store tau=I at radius=0 for each fold
    shift_spec : target identity dict — stored with each entry for eval to
        consume.  NEVER used during training.  For synthetic evaluation, this
        is the shift that eval will apply to build the pseudo-target from the
        held-out fold.  None for Portland (real target bundle).

        Note: tau=I (baseline_identity) is the weak floor.  The DiRoCA-
        comparable baseline is the eps=0-learned-tau (optimizer trained with
        no robustness), which is simply the radius=0.0 entry in the sweep.

    Note on fold variance: for the Gaussian objective, training estimates
    exogenous moments from each fold's training samples, so each fold learns
    a genuinely different τ.  Eval-side holdout moment estimation (see
    traca_run_evaluation) adds a second layer of fold variance to the
    certificate and Phi_pushed.
    """
    if sweep_axis not in ("eps", "eta"):
        raise ValueError(f"sweep_axis must be 'eps' or 'eta', got {sweep_axis!r}")

    with config_path.open() as f:
        base_cfg = yaml.safe_load(f)

    name = base_cfg["name"]
    objective = base_cfg.get("objective", "empirical")
    training_cfg = base_cfg.get("training", {})

    # Load full source bundle
    bundle = _load_bundle(base_cfg["data"], training_cfg)
    d = bundle.d
    N = bundle.n

    exp_out = out_dir / name
    exp_out.mkdir(parents=True, exist_ok=True)

    print(f"[train] experiment: {name}  d={d}  n={N}  k={k_folds}")
    print(f"[train] sweep_axis={sweep_axis}  values={radius_values}")
    if shift_spec:
        print(f"[train] shift_spec: {shift_spec}")
    if objective == "gaussian":
        print("[train] NOTE: gaussian training estimates moments per-fold "
              "from training samples; each fold learns a different τ.")

    # K-fold split
    indices = np.arange(N)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(indices)
    folds = np.array_split(shuffled, k_folds)

    results: dict[str, Any] = {}

    # Extract the fixed radius (the one NOT swept) for progress labels.
    amb = base_cfg.get("ambiguity", {})
    if sweep_axis == "eps":
        fixed_eta = _mech_radius_label(amb.get("mechanism", {}))
        _make_desc = lambda fi, rv: (
            "fold %d/%d \u03b5=%.2f \u03b7=%s" % (fi + 1, k_folds, rv, fixed_eta))
    else:
        fixed_eps = amb.get("environment", {}).get("eps", "?")
        _make_desc = lambda fi, rv: (
            "fold %d/%d \u03b5=%s \u03b7=%.2f" % (fi + 1, k_folds, fixed_eps, rv))

    for fold_idx in range(k_folds):
        fold_key = f"fold_{fold_idx}"
        results[fold_key] = {}

        val_idx = folds[fold_idx]
        train_idx = np.concatenate(
            [folds[j] for j in range(k_folds) if j != fold_idx]
        )
        train_bundle, _ = _split_bundle(bundle, train_idx, val_idx)

        for rv in radius_values:
            radius_key = f"{sweep_axis}_{rv:.2f}"
            desc = _make_desc(fold_idx, rv)
            if not show_progress:
                print(f"[train] {desc}  ...", end=" ", flush=True)

            entry = _train_one_fold_radius(
                base_cfg, train_bundle, val_idx,
                sweep_axis, rv, objective, shift_spec,
                show_progress=show_progress,
                progress_desc=desc,
            )
            results[fold_key][radius_key] = entry

            md = entry["training_metadata"]
            if not show_progress:
                print(f"loss={md['final_loss']:.4f}  converged={md['converged']}")
            else:
                print(f"[train] {desc}  loss={md['final_loss']:.4f}  "
                      f"converged={md['converged']}  n_iters={md['n_iters']}")

        # Baseline: tau=I at radius=0
        if include_baselines:
            results[fold_key]["baseline_identity"] = {
                "tau": np.eye(d),
                "opt_result": None,
                "test_indices": val_idx,
                "shift_spec": shift_spec,
                "config_snapshot": base_cfg,
                "training_metadata": {
                    "converged": True,
                    "n_iters": 0,
                    "final_loss": None,
                    "initial_loss": None,
                    "objective": objective,
                },
            }

    # Metadata
    prov = _git_provenance()
    results["__metadata__"] = {
        "experiment_name": name,
        "config_path": str(config_path),
        "sweep_axis": sweep_axis,
        "radius_values": [float(v) for v in radius_values],
        "k_folds": k_folds,
        "d": d,
        "n_samples": N,
        "n_interventions": bundle.n_interventions(),
        "objective": objective,
        "shift_spec": shift_spec,
        "git_commit": prov["git_commit"],
        "git_dirty": prov["git_dirty"],
        "timestamp": datetime.datetime.now().isoformat(),
        "source_bundle_path": str(exp_out / "bundle.pkl"),
    }

    # Save
    pkl_path = exp_out / "traca_cv_results.pkl"
    joblib.dump(results, pkl_path)
    joblib.dump(bundle, exp_out / "bundle.pkl")
    with (exp_out / "config.yaml").open("w") as f:
        yaml.dump(base_cfg, f)

    print(f"\n[train] saved {pkl_path}")
    print(f"[train] saved {exp_out / 'bundle.pkl'}")
    return pkl_path


def train_cv_grid(
    config_path: Path,
    eps_values: list[float],
    eta_values: list[float],
    grid_mode: str,
    out_dir: Path,
    k_folds: int = 5,
    include_baselines: bool = True,
    shift_spec: dict | None = None,
    show_progress: bool = False,
) -> Path:
    """K-fold training pass over 2D (ε,η) grid.  Returns path to saved pickle.

    Parameters
    ----------
    config_path : experiment YAML
    eps_values : ε values to sweep
    eta_values : η values to sweep
    grid_mode : "cross" for Cartesian product, "zip" for paired diagonal (ε=η)
    out_dir : output directory
    k_folds : number of CV folds
    include_baselines : if True, store tau=I baseline for each fold
    shift_spec : target identity dict (stored, not used during training)
    show_progress : show live tqdm progress bars
    """
    if grid_mode not in ("cross", "zip"):
        raise ValueError(f"grid_mode must be 'cross' or 'zip', got {grid_mode!r}")
    if grid_mode == "zip" and len(eps_values) != len(eta_values):
        raise ValueError(
            f"grid_mode='zip' requires equal-length eps_values and eta_values, "
            f"got {len(eps_values)} and {len(eta_values)}"
        )

    with config_path.open() as f:
        base_cfg = yaml.safe_load(f)

    name = base_cfg["name"]
    objective = base_cfg.get("objective", "empirical")
    training_cfg = base_cfg.get("training", {})

    bundle = _load_bundle(base_cfg["data"], training_cfg)
    d = bundle.d
    N = bundle.n

    exp_out = out_dir / name
    exp_out.mkdir(parents=True, exist_ok=True)

    # Build grid pairs
    if grid_mode == "cross":
        import itertools
        grid_pairs = list(itertools.product(eps_values, eta_values))
    else:  # zip
        grid_pairs = list(zip(eps_values, eta_values))

    print(f"[train] experiment: {name}  d={d}  n={N}  k={k_folds}")
    print(f"[train] grid_mode={grid_mode}  {len(grid_pairs)} (ε,η) pairs")
    if shift_spec:
        print(f"[train] shift_spec: {shift_spec}")
    if objective == "gaussian":
        print("[train] NOTE: gaussian training estimates moments per-fold "
              "from training samples; each fold learns a different τ.")

    # K-fold split
    indices = np.arange(N)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(indices)
    folds = np.array_split(shuffled, k_folds)

    results: dict[str, Any] = {}

    for fold_idx in range(k_folds):
        fold_key = f"fold_{fold_idx}"
        results[fold_key] = {}

        val_idx = folds[fold_idx]
        train_idx = np.concatenate(
            [folds[j] for j in range(k_folds) if j != fold_idx]
        )
        train_bundle, _ = _split_bundle(bundle, train_idx, val_idx)

        for eps_val, eta_val in grid_pairs:
            radius_key = f"eps_{eps_val:.2f}_eta_{eta_val:.2f}"
            desc = f"fold {fold_idx + 1}/{k_folds} ε={eps_val:.2f} η={eta_val:.2f}"
            if not show_progress:
                print(f"[train] {desc}  ...", end=" ", flush=True)

            # Substitute both eps and eta
            cfg = _substitute_eps_eta(base_cfg, eps_val, eta_val)

            entry = _train_one_fold_radius(
                cfg, train_bundle, val_idx,
                "eps", eps_val, objective, shift_spec,
                show_progress=show_progress,
                progress_desc=desc,
            )
            # Override config_snapshot with the fully-substituted cfg
            entry["config_snapshot"] = cfg
            results[fold_key][radius_key] = entry

            md = entry["training_metadata"]
            if not show_progress:
                print(f"loss={md['final_loss']:.4f}  converged={md['converged']}")
            else:
                print(f"[train] {desc}  loss={md['final_loss']:.4f}  "
                      f"converged={md['converged']}  n_iters={md['n_iters']}")

        if include_baselines:
            results[fold_key]["baseline_identity"] = {
                "tau": np.eye(d),
                "opt_result": None,
                "test_indices": val_idx,
                "shift_spec": shift_spec,
                "config_snapshot": base_cfg,
                "training_metadata": {
                    "converged": True,
                    "n_iters": 0,
                    "final_loss": None,
                    "initial_loss": None,
                    "objective": objective,
                },
            }

    # Metadata
    prov = _git_provenance()
    results["__metadata__"] = {
        "experiment_name": name,
        "config_path": str(config_path),
        "sweep_axis": "grid",
        "grid_mode": grid_mode,
        "eps_values": [float(v) for v in eps_values],
        "eta_values": [float(v) for v in eta_values],
        "radius_values": [],  # backward compat: empty for grid mode
        "k_folds": k_folds,
        "d": d,
        "n_samples": N,
        "n_interventions": bundle.n_interventions(),
        "objective": objective,
        "shift_spec": shift_spec,
        "git_commit": prov["git_commit"],
        "git_dirty": prov["git_dirty"],
        "timestamp": datetime.datetime.now().isoformat(),
        "source_bundle_path": str(exp_out / "bundle.pkl"),
    }

    # Save
    pkl_path = exp_out / "traca_cv_results.pkl"
    joblib.dump(results, pkl_path)
    joblib.dump(bundle, exp_out / "bundle.pkl")
    with (exp_out / "config.yaml").open("w") as f:
        yaml.dump(base_cfg, f)

    print(f"\n[train] saved {pkl_path}")
    print(f"[train] saved {exp_out / 'bundle.pkl'}")
    return pkl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_shift_spec(args) -> dict | None:
    """Build shift_spec dict from CLI args, or None if no shift specified."""
    if args.shift_type is None or args.shift_type == "none":
        return None
    spec: dict[str, Any] = {
        "shift_type": args.shift_type,
        "magnitude": args.shift_magnitude,
    }
    if args.shift_type == "mechanism_edge":
        if args.shift_edge is None:
            raise ValueError("--shift_edge required for mechanism_edge shift")
        spec["edge"] = tuple(args.shift_edge)
    else:
        if args.shift_node is None:
            raise ValueError(f"--shift_node required for {args.shift_type} shift")
        spec["node"] = args.shift_node
    return spec


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="K-fold training pass for decoupled TraCA workflow."
    )
    p.add_argument("--config", required=True, help="Path to experiment YAML.")
    # 1D sweep (existing)
    p.add_argument(
        "--sweep_axis", default=None, choices=["eps", "eta"],
        help="Which radius to sweep (1D mode).",
    )
    p.add_argument(
        "--values", nargs="+", type=float, default=None,
        help="Absolute radius values (1D mode).",
    )
    # 2D grid (new)
    p.add_argument(
        "--eps_values", nargs="+", type=float, default=None,
        help="ε values for 2D grid mode.",
    )
    p.add_argument(
        "--eta_values", nargs="+", type=float, default=None,
        help="η values for 2D grid mode.",
    )
    p.add_argument(
        "--grid_mode", default="cross", choices=["cross", "zip"],
        help="Grid mode: 'cross' for Cartesian product, 'zip' for paired diagonal.",
    )
    p.add_argument("--out_dir", default="results", help="Root output directory.")
    p.add_argument("--k_folds", type=int, default=5, help="Number of CV folds.")
    p.add_argument(
        "--no_baselines", action="store_true",
        help="Skip tau=I baseline entries.",
    )
    p.add_argument(
        "--progress", action="store_true",
        help="Show live tqdm progress bars during optimization.",
    )
    # Shift spec args (stored, not used during training)
    p.add_argument(
        "--shift_type", default=None,
        choices=["none", "mechanism_edge", "noise_mean", "noise_std", "noise_cov"],
        help="Shift type for synthetic target identity.",
    )
    p.add_argument("--shift_magnitude", type=float, default=0.0)
    p.add_argument("--shift_node", type=int, default=None)
    p.add_argument("--shift_edge", nargs=2, type=int, default=None)
    p.add_argument(
        "--shift_spec_json", type=str, default=None,
        help="JSON string for compound shift_spec (list of dicts). "
             "Overrides --shift_type/--shift_magnitude/--shift_node/--shift_edge.",
    )
    return p


if __name__ == "__main__":
    import json as _json
    args = _build_arg_parser().parse_args()
    if args.shift_spec_json is not None:
        shift_spec = _json.loads(args.shift_spec_json)
    else:
        shift_spec = _parse_shift_spec(args)

    is_1d = args.sweep_axis is not None and args.values is not None
    is_2d = args.eps_values is not None and args.eta_values is not None

    if is_1d and is_2d:
        raise ValueError(
            "Cannot use both 1D (--sweep_axis/--values) and 2D (--eps_values/--eta_values) modes."
        )
    if not is_1d and not is_2d:
        raise ValueError(
            "Must specify either 1D mode (--sweep_axis + --values) or "
            "2D mode (--eps_values + --eta_values)."
        )

    if is_2d:
        train_cv_grid(
            config_path=Path(args.config),
            eps_values=args.eps_values,
            eta_values=args.eta_values,
            grid_mode=args.grid_mode,
            out_dir=Path(args.out_dir),
            k_folds=args.k_folds,
            include_baselines=not args.no_baselines,
            shift_spec=shift_spec,
            show_progress=args.progress,
        )
    else:
        train_cv(
            config_path=Path(args.config),
            sweep_axis=args.sweep_axis,
            radius_values=args.values,
            out_dir=Path(args.out_dir),
            k_folds=args.k_folds,
            include_baselines=not args.no_baselines,
            shift_spec=shift_spec,
        show_progress=args.progress,
    )
