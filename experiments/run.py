"""
CLI entry point: python -m experiments.run --config <yaml> [--out_dir <dir>]

Loads source bundle, builds ambiguity + constructive objects from the experiment
YAML, runs the optimizer, and saves:
    <out_dir>/<name>/result.pkl      — OptResult (full object)
    <out_dir>/<name>/summary.json    — scalar diagnostics + tau diagonal

Usage
-----
    python -m experiments.run --config configs/atce/gaussian_z_entrywise_full.yaml
    python -m experiments.run --config configs/atce/gaussian_z_entrywise_full.yaml \\
        --out_dir results/custom
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

# Ensure lan_scm is importable when invoked from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lan_scm  # noqa: E402  (needed before joblib.load to resolve pickle refs)
from lan_scm import _scm_from_config, LANSCM, SCMBundle  # noqa: E402

from traca.ambiguity import (
    FrobeniusBall, RowBudget, ColumnBudget, EntrywiseBox,
    GelbrichBall, FrobeniusEmpirical,
)
from traca.constructive import ConstructiveClass
from traca.optim import OptimConfig, fit_empirical, fit_gaussian
from traca.query import query_family_from_config


# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------

def _build_mechanism_set(cfg: dict, d: int):
    """Build a mechanism ambiguity set from YAML dict."""
    t = cfg["type"]
    shifted = tuple(int(i) for i in cfg["shifted_rows"])
    if t == "FrobeniusBall":
        entry_mask = None
        if "entry_mask" in cfg:
            entry_mask = np.array(cfg["entry_mask"], dtype=float)
        return FrobeniusBall(eta=float(cfg["eta"]), shifted_rows=shifted, d=d,
                             entry_mask=entry_mask)
    if t == "RowBudget":
        rho = {int(k): float(v) for k, v in cfg["rho"].items()}
        return RowBudget(rho=rho, shifted_rows=shifted, d=d)
    if t == "ColumnBudget":
        c = cfg["c"]
        c_val = {int(k): float(v) for k, v in c.items()} if isinstance(c, dict) else float(c)
        return ColumnBudget(c=c_val, shifted_rows=shifted, d=d)
    if t == "EntrywiseBox":
        B = np.array(cfg["B"], dtype=float)
        delta = np.array(cfg["delta"], dtype=float) if "delta" in cfg else None
        return EntrywiseBox(B=B, shifted_rows=shifted, d=d, delta=delta)
    raise ValueError(f"Unknown mechanism ambiguity type: {t!r}")


def _build_environment_set(cfg: dict, mu_s: np.ndarray, Sigma_s: np.ndarray, N: int):
    """Build an environment ambiguity set from YAML dict."""
    t = cfg["type"]
    shifted = tuple(int(i) for i in cfg["shifted_rows"])
    if t == "FrobeniusEmpirical":
        return FrobeniusEmpirical(eps=float(cfg["eps"]), N=N, shifted_rows=shifted)
    if t == "GelbrichBall":
        return GelbrichBall(
            mu_s=mu_s, Sigma_s=Sigma_s,
            eps=float(cfg["eps"]),
            shifted_rows=shifted,
        )
    raise ValueError(f"Unknown environment ambiguity type: {t!r}")


def _build_constructive_class(cfg: dict, d: int) -> ConstructiveClass:
    """Build a ConstructiveClass from YAML dict."""
    t = cfg["type"]
    shifted = [int(i) for i in cfg["shifted_nodes"]]
    if t == "markovian":
        return ConstructiveClass.markovian(d=d, shifted=shifted)
    if t == "from_districts":
        districts = [list(map(int, district)) for district in cfg["districts"]]
        return ConstructiveClass.from_districts(d=d, districts=districts, shifted=shifted)
    raise ValueError(f"Unknown constructive class type: {t!r}")


def _load_bundle(data_cfg: dict, training_cfg: dict) -> SCMBundle:
    """Load or build a source bundle.

    Two modes:
    - ``data.bundle_path``: load a pre-built pkl directly (e.g. Portland).
      Optional ``data.districts`` / ``data.sel_nodes`` keys attach metadata.
    - ``data.source``: build from a data_configs YAML (all other benchmarks).
    """
    if "bundle_path" in data_cfg:
        bundle_path = Path(data_cfg["bundle_path"])
        bundle = joblib.load(bundle_path)
        # Allow the experiment YAML to override/set metadata
        if "districts" in data_cfg:
            bundle.districts = data_cfg["districts"]
        if "sel_nodes" in data_cfg:
            bundle.sel_nodes = [int(i) for i in data_cfg["sel_nodes"]]
        return bundle

    source_yaml = Path(data_cfg["source"])
    with source_yaml.open() as f:
        scm_config = yaml.safe_load(f)

    scm = _scm_from_config(scm_config)
    interventions = list(scm_config["interventions"])
    n = int(training_cfg.get("n_samples", scm_config.get("n_samples", 1000)))
    seed = training_cfg.get("seed", scm_config.get("seed", 0))
    bundle = scm.bundle(interventions=interventions, n=n, seed=seed)

    # Attach semi-Markovian metadata from the source YAML if present
    scm_inner = scm_config.get("scm", scm_config)
    bundle.districts = scm_inner.get("districts", None)
    bundle.sel_nodes = scm_inner.get("sel_nodes", None)
    return bundle


def _build_optim_config(training_cfg: dict) -> OptimConfig:
    """Build OptimConfig from training section of experiment YAML.

    Reads ``k_tau`` and ``k_adv`` directly.  The legacy keys ``k_min``
    and ``k_max`` are deprecated; a DeprecationWarning is raised if they
    appear, and the fallback values are used so old configs still run.
    """
    import warnings
    if "k_max" in training_cfg or "k_min" in training_cfg:
        warnings.warn(
            "YAML keys k_min and k_max are deprecated; rename to k_tau and k_adv.",
            DeprecationWarning,
            stacklevel=2,
        )
    k_adv = int(training_cfg.get("k_adv", training_cfg.get("k_max", 5)))
    k_tau = int(training_cfg.get("k_tau", training_cfg.get("k_min", 1)))
    return OptimConfig(
        eta_tau=float(training_cfg.get("eta_tau", 1e-2)),
        eta_adv=float(training_cfg.get("eta_adv", 1e-2)),
        k_tau=k_tau,
        k_adv=k_adv,
        n_iters=int(training_cfg.get("n_iters", 1000)),
        tol=float(training_cfg.get("tol", 1e-6)),
        conv_window=int(training_cfg.get("conv_window", 20)),
        grad_mode=training_cfg.get("grad_mode", "analytic"),
        grad_backend=training_cfg.get("grad_backend", "autograd"),
        covprox_aggregation=training_cfg.get("covprox_aggregation", "norm_weighted"),
        seed=training_cfg.get("seed", None),
        tau_init=training_cfg.get("tau_init", "identity"),
        tau_seed=training_cfg.get("tau_seed", None),
    )


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _git_provenance() -> dict:
    """Return git commit hash and dirty flag.  Degrades gracefully if git unavailable."""
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip())
    except Exception:
        commit = "unknown"
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _write_meta(exp_out: Path, optim_cfg, objective: str) -> None:
    """Write meta.json: git provenance + resolved optimizer config.

    The resolved config records the actual flag values used (post-defaults),
    not what the YAML specified.  This is the positive confirmation that
    covprox_aggregation and grad_backend are what was actually computed with.
    """
    meta: dict[str, Any] = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "objective": objective,
    }
    meta.update(_git_provenance())
    meta["resolved_config"] = dataclasses.asdict(optim_cfg)
    with (exp_out / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_experiment(config_path: Path, out_dir: Path) -> Path:
    """Run one experiment from a YAML config, save result.pkl + summary.json.

    Parameters
    ----------
    config_path : Path to experiment YAML
    out_dir : Root output directory (results are saved to out_dir/<name>/)

    Returns
    -------
    Path to the output directory
    """
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    name = cfg["name"]
    exp_out = out_dir / name
    exp_out.mkdir(parents=True, exist_ok=True)
    print(f"[run] experiment: {name}")
    print(f"[run] output:     {exp_out}")

    # Build bundle
    bundle = _load_bundle(cfg["data"], cfg.get("training", {}))
    d = bundle.d
    N = bundle.n
    print(f"[run] bundle: d={d}, n={N}, n_interventions={bundle.n_interventions()}")

    # Build ambiguity sets
    aw = _build_mechanism_set(cfg["ambiguity"]["mechanism"], d)
    ae = _build_environment_set(
        cfg["ambiguity"]["environment"],
        bundle.noise_mean, bundle.noise_cov, N
    )

    # Build constructive class
    cc = _build_constructive_class(cfg["constructive_class"], d)
    print(f"[run] constructive class: {cc.districts}, shifted={cc.shifted_nodes}")

    # Build optimizer config + query family
    training_cfg = cfg.get("training", {})
    optim_cfg = _build_optim_config(training_cfg)
    query_family = query_family_from_config(cfg)

    # Run optimizer
    objective = cfg.get("objective", "empirical")
    print(f"[run] fitting ({objective}) ...")

    if objective == "empirical":
        result = fit_empirical(bundle, aw, ae, cc, optim_cfg, query_family=query_family)
    elif objective == "gaussian":
        result = fit_gaussian(bundle, aw, ae, cc, optim_cfg, query_family=query_family)
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    print(f"[run] done — converged={result.converged}, n_iters={result.n_iters}")
    print(f"[run] initial_loss={result.initial_loss:.4f}, final_loss={result.final_loss:.4f}")

    # Save result.pkl
    result_path = exp_out / "result.pkl"
    joblib.dump(result, result_path)
    print(f"[run] saved result to {result_path}")

    # Save meta.json — git commit + resolved config (positive provenance stamp)
    _write_meta(exp_out, optim_cfg, objective)
    print(f"[run] saved provenance to {exp_out / 'meta.json'}")

    # Build summary.json (plain scalars only)
    summary: dict[str, Any] = {
        "name": name,
        "objective": objective,
        "d": d,
        "n_samples": N,
        "n_interventions": bundle.n_interventions(),
        "converged": bool(result.converged),
        "n_iters": int(result.n_iters),
        "initial_loss": float(result.initial_loss),
        "final_loss": float(result.final_loss),
        "tau_diagonal": [float(x) for x in np.diag(result.tau)],
        "tau_full": result.tau.tolist(),
        "dW_norms": [float(np.linalg.norm(dW, "fro")) for dW in result.dW_iota],
        "history_best_loss": [float(x) for x in result.history_best_loss],
    }

    summary_path = exp_out / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] saved summary to {summary_path}")

    # Also save the bundle and config for downstream evaluate.py
    joblib.dump(bundle, exp_out / "bundle.pkl")
    with (exp_out / "config.yaml").open("w") as f:
        yaml.dump(cfg, f)

    return exp_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a TraCA experiment from a YAML config."
    )
    p.add_argument("--config", required=True, help="Path to experiment YAML.")
    p.add_argument(
        "--out_dir", default="results",
        help="Root output directory (default: results/)."
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_experiment(Path(args.config), Path(args.out_dir))
