"""
Eval pass for decoupled train/eval workflow.

Loads saved τ's from ``traca_train.py``, reveals targets, scores uniformly,
and emits a long-format CSV.

The shift is NOT optional for synthetic evaluation — it's the whole point.
A raw held-out source fold has no domain shift; scoring against it tests
within-source generalization, not transport robustness (ε* would favor ε≈0).

Usage
-----
    python traca_run_evaluation.py --results results/atce_gaussian_z_entrywise_full/traca_cv_results.pkl

    # Override shift at eval time:
    python traca_run_evaluation.py --results <pkl> \
        --shift_type noise_mean --shift_magnitude 0.5 --shift_node 0

    # Portland (real target bundle, no synthetic shift):
    python traca_run_evaluation.py --results <pkl> \
        --target_config data_configs/portland_target.yaml

Saves
-----
    {output_dir}/evaluation.csv     — long-format results
    {output_dir}/summary.json       — aggregate statistics
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
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lan_scm  # noqa: E402 — pickle resolution
from lan_scm import _scm_from_config  # noqa: E402

from traca.shifts import apply_shift, SHIFT_TYPES
from traca.stability import gamma, alpha_polynomial, gating_matrix
from traca.certificates import (
    full_joint_certificate,
    single_query_certificate,
    query_interval,
)
from traca.losses import GaussianLoss, EmpiricalLoss
from traca.utils import interventional_exo_mean, bundle_exo_means, build_U_effs
from traca.directional_certificates import (
    directional_certificate_gaussian, directional_certificate_empirical,
)
from experiments.run import (
    _build_mechanism_set,
    _build_environment_set,
)
from experiments.evaluate import (
    _build_alpha_iotas,
    _compute_target_loss_per_iota,
    _iv_to_nodes,
)


# ---------------------------------------------------------------------------
# Observational pushed distance (R1.1 selection signal)
# ---------------------------------------------------------------------------

def _observational_pushed_distance(
    tau: np.ndarray,
    eval_bundle,
    target_bundle,
    objective: str,
) -> float:
    """W₂²(τ_# P_s^obs, P_t^obs) at intervention index 0 (observational).

    This is the R1.1 selection signal: how close the pushed source
    observational distribution is to the target observational distribution.
    Only meaningful when a real external target exists (ATE, Portland).

    Uses Gaussian W₂² for Gaussian objectives (the only R1.1 benchmarks).
    """
    d = eval_bundle.d
    W_s = eval_bundle.W
    A_s_0 = eval_bundle.intervened_scms[0].A  # observational propagator
    R_0 = np.eye(d)  # no intervention
    dW_zero = np.zeros((d, d))

    if objective == "gaussian":
        loss_fn = GaussianLoss()
        mu_s = eval_bundle.noise_mean
        Sigma_s = eval_bundle.noise_cov
        A_t_0 = target_bundle.intervened_scms[0].A
        mu_t_obs = target_bundle.noise_mean @ A_t_0
        Sigma_t_obs = A_t_0.T @ target_bundle.noise_cov @ A_t_0
        return float(loss_fn.value(
            tau, dW_zero, W_s, A_s_0, R_0,
            mu_s, Sigma_s, mu_t_obs, Sigma_t_obs,
        ))
    else:
        # Empirical branch (dead code for R1.1 benchmarks — ATE/Portland are Gaussian)
        loss_fn = EmpiricalLoss()
        U_s_0 = eval_bundle.noise_samples[0]
        N = U_s_0.shape[0]
        W_t = target_bundle.W
        dW_actual = W_t - W_s
        Theta_zero = np.zeros((N, d))
        return float(loss_fn.value(
            tau, dW_actual, Theta_zero, W_s, A_s_0, R_0, U_s_0,
        ))


# ---------------------------------------------------------------------------
# Load + flatten
# ---------------------------------------------------------------------------

def load_training_results(results_pkl: Path) -> tuple[dict, dict]:
    """Load the saved training results and return (results, metadata).

    Returns
    -------
    results : dict with fold_key → radius_key → entry
    metadata : the __metadata__ dict
    """
    data = joblib.load(results_pkl)
    metadata = data.pop("__metadata__")
    return data, metadata


def flatten_to_methods(
    results: dict,
    sweep_axis: str,
) -> list[dict]:
    """Unroll nested fold→radius into flat method entries.

    Each entry: {
        'method': "TraCA (eps_0.10)" or "TraCA (eps_0.10_eta_0.30)",
        'radius': 0.1,          # 1D compat; 0.0 for baselines
        'eps': 0.1,             # parsed from key (None if not present)
        'eta': 0.3,             # parsed from key (None if not present)
        'fold': "fold_0",
        'tau': np.ndarray,
        'test_indices': np.ndarray,
        'shift_spec': dict | None,
        'config_snapshot': dict,
        'training_metadata': dict,
    }

    Baselines flatten as: "TraCA_baseline_identity" with radius=0.
    Handles both 1D keys ("eps_0.10") and 2D keys ("eps_0.10_eta_0.30").
    """
    entries = []
    for fold_key, fold_dict in sorted(results.items()):
        for radius_key, entry in fold_dict.items():
            if radius_key == "baseline_identity":
                method = "TraCA_baseline_identity"
                radius = 0.0
                eps_val = None
                eta_val = None
            elif "_eta_" in radius_key and radius_key.startswith("eps_"):
                # 2D key: "eps_0.10_eta_0.30"
                method = f"TraCA ({radius_key})"
                parts = radius_key.split("_")  # ['eps', '0.10', 'eta', '0.30']
                eps_val = float(parts[1])
                eta_val = float(parts[3])
                radius = 0.0  # no single radius in 2D mode
            else:
                # 1D key: "eps_0.10" or "eta_0.30"
                method = f"TraCA ({radius_key})"
                radius = float(radius_key.split("_", 1)[1])
                if radius_key.startswith("eps_"):
                    eps_val = radius
                    eta_val = None
                else:
                    eps_val = None
                    eta_val = radius

            entries.append({
                "method": method,
                "radius": radius,
                "eps": eps_val,
                "eta": eta_val,
                "fold": fold_key,
                "tau": entry["tau"],
                "test_indices": entry["test_indices"],
                "shift_spec": entry.get("shift_spec"),
                "config_snapshot": entry["config_snapshot"],
                "training_metadata": entry["training_metadata"],
            })
    return entries


def _expand_baseline_across_grid(
    flat: list[dict],
    metadata: dict,
) -> list[dict]:
    """Replicate baseline_identity entries across the full radius grid.

    The identity map (τ=I) is fixed — it doesn't depend on the training
    radius.  But its *certificate* depends on ε (and η), so it must be
    scored at every grid point to enable like-for-like comparison with
    the TraCA rows.  This function expands each baseline entry into one
    entry per grid point, patching the config_snapshot's ε (and η) so
    that _reveal_target_and_score uses the correct ambiguity radii.

    Non-baseline entries pass through unchanged.
    """
    sweep_axis = metadata.get("sweep_axis", "")
    baselines = [e for e in flat if e["method"] == "TraCA_baseline_identity"]
    non_baselines = [e for e in flat if e["method"] != "TraCA_baseline_identity"]

    if not baselines:
        return flat

    # Determine the grid points to expand over
    if sweep_axis == "grid":
        eps_values = metadata.get("eps_values", [])
        eta_values = metadata.get("eta_values", [])
        grid_mode = metadata.get("grid_mode", "cross")
        if grid_mode == "cross":
            grid_points = [(e, h) for e in eps_values for h in eta_values]
        else:  # zip
            grid_points = list(zip(eps_values, eta_values))
    elif sweep_axis in ("eps", "eta"):
        radius_values = metadata.get("radius_values", [])
        if sweep_axis == "eps":
            grid_points = [(rv, None) for rv in radius_values]
        else:
            grid_points = [(None, rv) for rv in radius_values]
    else:
        return flat  # unknown sweep axis — don't expand

    expanded = []
    for base_entry in baselines:
        for eps_val, eta_val in grid_points:
            entry = dict(base_entry)  # shallow copy
            # Deep-copy and patch the config_snapshot
            cfg = copy.deepcopy(base_entry["config_snapshot"])
            if eps_val is not None:
                cfg.setdefault("ambiguity", {}).setdefault("environment", {})["eps"] = eps_val
            if eta_val is not None:
                # Patch mechanism eta (FrobeniusBall/EntrywiseBox)
                mech = cfg.setdefault("ambiguity", {}).setdefault("mechanism", {})
                if "eta" in mech:
                    mech["eta"] = eta_val
                if "B" in mech:
                    # EntrywiseBox: scale nonzero entries to eta_val
                    B = mech["B"]
                    if isinstance(B, list):
                        for i in range(len(B)):
                            if isinstance(B[i], list):
                                for j in range(len(B[i])):
                                    if B[i][j] != 0:
                                        B[i][j] = eta_val

            entry["config_snapshot"] = cfg
            # Set method name and radius fields for CSV alignment
            if sweep_axis == "grid":
                tag = f"eps_{eps_val:.2f}_eta_{eta_val:.2f}"
                entry["method"] = f"TraCA_baseline_identity ({tag})"
                entry["eps"] = eps_val
                entry["eta"] = eta_val
                entry["radius"] = 0.0
            elif sweep_axis == "eps":
                entry["method"] = f"TraCA_baseline_identity (eps_{eps_val:.2f})"
                entry["eps"] = eps_val
                entry["radius"] = eps_val
            else:  # eta
                entry["method"] = f"TraCA_baseline_identity (eta_{eta_val:.2f})"
                entry["eta"] = eta_val
                entry["radius"] = eta_val

            expanded.append(entry)

    return non_baselines + expanded


# ---------------------------------------------------------------------------
# Target construction + scoring
# ---------------------------------------------------------------------------

def _apply_shift_spec(source_scm, shift_spec, *, verbose: bool = False):
    """Apply one or more shifts to a source SCM in sequence.

    Parameters
    ----------
    source_scm : LANSCM to shift (not modified)
    shift_spec : dict (single shift, backward compat) or list[dict] (compound)
    verbose : if True, log before/after for each sub-shift

    Returns
    -------
    LANSCM with all shifts applied in sequence
    """
    if shift_spec is None:
        return source_scm
    specs = [shift_spec] if isinstance(shift_spec, dict) else list(shift_spec)
    scm = source_scm
    for spec in specs:
        kwargs: dict = {}
        if "edge" in spec:
            kwargs["edge"] = tuple(int(x) for x in spec["edge"])
        if "node" in spec:
            kwargs["node"] = int(spec["node"])
        if verbose:
            _log_shift_before(scm, spec)
        scm = apply_shift(scm, spec["shift_type"], float(spec["magnitude"]), **kwargs)
        if verbose:
            _log_shift_after(source_scm, scm, spec)
    return scm


def _log_shift_before(scm, spec):
    shift_type = spec["shift_type"]
    if shift_type == "mechanism_edge":
        i, j = (int(x) for x in spec["edge"])
        print(f"  [shift] mechanism_edge W[{i},{j}]: before={scm.W[i,j]:.6f}")
    elif shift_type in ("noise_std", "noise_cov", "noise_mean"):
        k = int(spec["node"])
        print(f"  [shift] {shift_type} node {k}: before σ={float(np.sqrt(scm.noise_cov[k,k])):.6f}")


def _log_shift_after(original_scm, shifted_scm, spec):
    shift_type = spec["shift_type"]
    if shift_type == "mechanism_edge":
        i, j = (int(x) for x in spec["edge"])
        print(f"  [shift] mechanism_edge W[{i},{j}]: after={shifted_scm.W[i,j]:.6f}  "
              f"(Δ={shifted_scm.W[i,j]-original_scm.W[i,j]:+.6f})")
    elif shift_type in ("noise_std", "noise_cov", "noise_mean"):
        k = int(spec["node"])
        old_s = float(np.sqrt(original_scm.noise_cov[k, k]))
        new_s = float(np.sqrt(shifted_scm.noise_cov[k, k]))
        print(f"  [shift] {shift_type} node {k}: after σ={new_s:.6f}  (×{new_s/old_s:.4f})")


_PAIRED_SHIFT_TYPES = {"mechanism_edge", "noise_mean"}


def _build_paired_target_bundle(
    source_bundle,
    shift_spec,
    test_indices: np.ndarray,
):
    """Build target bundle by propagating held-out source U through shifted SCM.

    Paired construction: reuses the exact held-out U rows from the source
    bundle, applies noise transforms (mean shift only), and propagates
    through the shifted propagator A'.  No fresh sampling — source and
    target share the same exogenous draws, so sampling noise cancels.

    Supported shift types: mechanism_edge (U unchanged, A changes) and
    noise_mean (U[:, node] += delta).  Scale shifts (noise_std, noise_cov)
    are rejected — they require distributional resampling.

    Parameters
    ----------
    source_bundle : full source SCMBundle (noise_samples must cover test_indices)
    shift_spec : dict (single) or list[dict] (compound) — shift specifications
    test_indices : held-out sample indices

    Returns
    -------
    target_bundle : SCMBundle with paired samples and sample-estimated moments
    """
    from lan_scm import SCMBundle

    specs = [shift_spec] if isinstance(shift_spec, dict) else list(shift_spec)
    for spec in specs:
        st = spec["shift_type"]
        if st not in _PAIRED_SHIFT_TYPES:
            raise ValueError(
                f"Paired target construction does not support shift_type={st!r}. "
                f"Supported: {sorted(_PAIRED_SHIFT_TYPES)}. "
                f"Use --pairing unpaired for scale/covariance shifts."
            )

    shifted_scm = _apply_shift_spec(source_bundle.scm, shift_spec)

    # Observational held-out U (intervention 0 = {}, no zeroed columns)
    U_obs_holdout = source_bundle.noise_samples[0][test_indices].copy()

    # Apply noise_mean transforms to U
    for spec in specs:
        if spec["shift_type"] == "noise_mean":
            node = int(spec["node"])
            U_obs_holdout[:, node] += float(spec["magnitude"])

    # Per-intervention: propagate paired U through shifted+intervened SCM
    intervened_scms: dict[int, Any] = {}
    endogenous_samples: dict[int, np.ndarray] = {}
    noise_samples: dict[int, np.ndarray] = {}

    for i, iv in enumerate(source_bundle.interventions):
        scm_do = shifted_scm.intervene(iv)
        U_i = U_obs_holdout.copy()
        if scm_do._J:
            U_i[:, scm_do._J] = 0.0
        X_target = U_i @ scm_do.A + scm_do._fixed @ scm_do.A
        intervened_scms[i] = scm_do
        endogenous_samples[i] = X_target
        noise_samples[i] = U_i

    # Estimate moments from observational paired U (sample-then-estimate)
    noise_mean = np.mean(U_obs_holdout, axis=0)
    noise_cov = np.cov(U_obs_holdout, rowvar=False)

    return SCMBundle(
        scm=shifted_scm,
        interventions=list(source_bundle.interventions),
        intervened_scms=intervened_scms,
        noise_mean=noise_mean,
        noise_cov=noise_cov,
        endogenous_samples=endogenous_samples,
        noise_samples=noise_samples,
        n=len(test_indices),
        seed=None,
    )


def _build_shifted_target_bundle(
    source_bundle,
    shift_spec,
    test_indices: np.ndarray,
    seed: int = 99999,
):
    """Apply shift(s) to source SCM and build a target bundle.

    Parameters
    ----------
    source_bundle : full source SCMBundle
    shift_spec : dict (single) or list[dict] (compound) — shift specifications
    test_indices : held-out sample indices (determines n for target)
    seed : random seed for target sampling

    Returns
    -------
    target_bundle : SCMBundle from the shifted SCM
    """
    shifted_scm = _apply_shift_spec(source_bundle.scm, shift_spec)

    target_bundle = shifted_scm.bundle(
        interventions=source_bundle.interventions,
        n=len(test_indices),
        seed=seed,
    )
    return target_bundle


def _reveal_target_and_score(
    tau: np.ndarray,
    source_bundle,
    test_indices: np.ndarray,
    shift_spec: dict | None,
    objective: str,
    config: dict,
    target_bundle=None,
    seed: int = 99999,
    pairing: str = "paired",
    is_external_target: bool = False,
    config_name: str | None = None,
) -> dict:
    """Reveal held-out target and compute scores.

    For synthetics (shift_spec is not None):
        If pairing == "paired": reuse held-out U rows through shifted SCM
        If pairing == "unpaired": draw fresh target samples

    For Portland (shift_spec is None, target_bundle provided):
        Score τ against pre-built target_bundle directly (pairing ignored).

    Returns
    -------
    dict with: target_loss, certificate, gap, coverage_fraction, all_covered,
               per_query (dict of query intervals + coverage)
    """
    if shift_spec is not None:
        if pairing == "paired":
            target_bundle = _build_paired_target_bundle(
                source_bundle, shift_spec, test_indices,
            )
        else:
            target_bundle = _build_shifted_target_bundle(
                source_bundle, shift_spec, test_indices, seed=seed,
            )
    elif target_bundle is None:
        raise ValueError(
            "Either shift_spec (for synthetics) or target_bundle (for Portland) "
            "must be provided.  Scoring against unshifted source folds is "
            "meaningless — ε* would favor ε≈0."
        )

    d = source_bundle.d
    N = source_bundle.n

    # --- Holdout subsetting ---
    # U_s_list uses the SAME observational U as training (build_U_effs),
    # preserving paired variance-cancellation across source/target/training.
    use_holdout = len(test_indices) < N
    holdout_idx = test_indices if use_holdout else None
    U_s_list = build_U_effs(source_bundle, indices=holdout_idx)
    N_eval = len(test_indices) if use_holdout else N

    # Source moments from held-out observational U (for BOTH objectives —
    # needed for paired target construction and Gaussian W₂² fallback).
    U_obs_test = source_bundle.noise_samples[0][test_indices] if use_holdout \
        else source_bundle.noise_samples[0]
    noise_mean_eval = np.mean(U_obs_test, axis=0) if use_holdout \
        else source_bundle.noise_mean
    noise_cov_eval = np.cov(U_obs_test, rowvar=False) if use_holdout \
        else source_bundle.noise_cov

    # eval_bundle: shallow copy with corrected noise data for downstream
    # consumers (directional certificates, obs_distance) that expect a bundle.
    eval_bundle = copy.copy(source_bundle)
    eval_bundle.noise_mean = noise_mean_eval
    eval_bundle.noise_cov = noise_cov_eval
    eval_bundle.n = N_eval
    # Store obs-derived U_effs as noise_samples so any consumer that reads
    # eval_bundle.noise_samples[i] gets the paired data.
    eval_bundle.noise_samples = {i: U_s_list[i] for i in range(source_bundle.n_interventions())}

    # Build ambiguity sets (for certificate computation)
    aw = _build_mechanism_set(config["ambiguity"]["mechanism"], d)
    ae = _build_environment_set(
        config["ambiguity"]["environment"],
        noise_mean_eval, noise_cov_eval, N_eval,
    )

    # Stability moduli
    A_iotas, alpha_iotas = _build_alpha_iotas(source_bundle, aw)
    n_iota = source_bundle.n_interventions()

    # mu_s_effs: per-intervention effective means from holdout moments
    mu_s_effs = [
        interventional_exo_mean(
            noise_mean_eval,
            source_bundle.intervened_scms[i]._fixed,
            source_bundle.intervened_scms[i]._J,
        )
        for i in range(n_iota)
    ]

    # Certificate
    eps = float(config["ambiguity"]["environment"]["eps"])

    certificate = full_joint_certificate(
        tau, alpha_iotas, A_iotas,
        mu_s_effs, noise_cov_eval,
        eps=eps, mode=objective,
        U_s_list=U_s_list if objective == "empirical" else None,
        N=N_eval if objective == "empirical" else None,
    )

    # Target loss — pass U_s_list for empirical mode to preserve pairing.
    # For Gaussian mode, use source_bundle's population moments.
    # For empirical mode, pass holdout-estimated source moments so the
    # Gaussian W₂² fallback (pure noise shift) uses the same estimation
    # frame as the target — at zero shift, both sides are identical.
    loss_source = source_bundle
    per_iota_losses = _compute_target_loss_per_iota(
        tau, loss_source, target_bundle, objective,
        shift_spec=shift_spec,
        U_s_list=U_s_list if objective == "empirical" else None,
        source_noise_mean=noise_mean_eval if objective == "empirical" else None,
        source_noise_cov=noise_cov_eval if objective == "empirical" else None,
    )
    target_loss = float(np.mean(per_iota_losses))
    gap = float(certificate - target_loss)

    # Sub×full cross-eval: compute err_subfamily if config_name resolves
    err_subfamily = float("nan")
    if config_name is not None:
        # Lazy import to avoid circular dependency with traca_radius_eval
        from traca_radius_eval import (
            _compute_restricted_loss,
            _CONFIG_NAME_MAP,
            _cross_eval_columns,
        )
        resolved = _CONFIG_NAME_MAP.get(config_name)
        if resolved is not None:
            family, trained_on = resolved
            cross_cols = _cross_eval_columns(trained_on, family)
            subfamily_qf = cross_cols.get("err_subfamily")
            if subfamily_qf is not None:
                err_subfamily = _compute_restricted_loss(
                    tau, source_bundle, target_bundle,
                    subfamily_qf, objective,
                    U_s_list=U_s_list if objective == "empirical" else None,
                    source_noise_mean=noise_mean_eval,
                    source_noise_cov=noise_cov_eval,
                )

    # Per-query intervals + coverage
    var_names = source_bundle.scm.var_names
    queries: dict[str, dict] = {}
    for i, iv in enumerate(source_bundle.interventions):
        A_t = target_bundle.intervened_scms[i].A
        mu_s_i = mu_s_effs[i]
        for k in range(d):
            key = f"({i},{k})"
            cert_q = float(single_query_certificate(
                tau, alpha_iotas[i], A_iotas[i],
                mu_s_i, noise_cov_eval,
                eps=eps, O=[k], d=d, mode=objective,
                U_s=U_s_list[i] if objective == "empirical" else None,
                N=N_eval if objective == "empirical" else None,
            ))
            if objective == "empirical":
                # U_s_list[i] is the obs-derived U_eff (= U_obs with intervened
                # cols zeroed + fixed).  Its sample mean already includes the
                # noise-mean component.  Phi_pushed = mean(U_eff @ A @ τ)_k.
                Phi_pushed = float(np.mean(U_s_list[i] @ A_iotas[i] @ tau, axis=0)[k])
            elif use_holdout:
                X_test_i = source_bundle.endogenous_samples[i][test_indices]
                Phi_pushed = float(np.mean(X_test_i @ tau, axis=0)[k])
            else:
                Phi_pushed = float((mu_s_i @ A_iotas[i] @ tau)[k])

            lo, hi = query_interval(Phi_pushed, 1.0, cert_q)

            # True target value — include intervention fixed values
            if objective == "gaussian":
                scm_t_i = target_bundle.intervened_scms[i]
                target_val = float((interventional_exo_mean(
                    target_bundle.noise_mean, scm_t_i._fixed, scm_t_i._J
                ) @ A_t)[k])
            else:
                X_t = target_bundle.endogenous_samples[i]
                target_val = float(np.mean(X_t[:, k]))

            queries[key] = {
                "lower": float(lo),
                "upper": float(hi),
                "width": float(hi - lo),
                "Phi_pushed": Phi_pushed,
                "delta_sq": cert_q,
                "target_value": target_val,
                "covered": bool(lo <= target_val <= hi),
            }

    n_queries = len(queries)
    n_covered = sum(1 for q in queries.values() if q["covered"])

    # Directional certificates (per-query tighter bounds)
    dir_certs: dict[str, dict] = {}
    for i, iv in enumerate(source_bundle.interventions):
        R_i = gating_matrix(d, _iv_to_nodes(iv, var_names))
        g_i = gamma(A_iotas[i], R_i, aw)
        for k in range(d):
            key = f"({i},{k})"
            q_vec = np.zeros(d); q_vec[k] = 1.0
            try:
                if objective == "empirical":
                    dc = directional_certificate_empirical(
                        tau, q_vec, [k], i, eval_bundle,
                        aw, ae, alpha_iotas[i], g_i,
                        U_eff=U_s_list[i],
                    )
                else:
                    dc = directional_certificate_gaussian(
                        tau, q_vec, [k], i, eval_bundle,
                        aw, ae, alpha_iotas[i], g_i,
                        noise_mean=noise_mean_eval,
                    )
                phi = queries[key]["Phi_pushed"]
                dir_cert_val = float(dc["certificate"])
                dir_certs[key] = {
                    "certificate": dir_cert_val,
                    "lower": float(phi - dir_cert_val),
                    "upper": float(phi + dir_cert_val),
                    "env_method": dc.get("env_method", "unknown"),
                }
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"[traca_run_evaluation] directional cert failed for ι={i}, node={k}: {exc}",
                    RuntimeWarning, stacklevel=2,
                )
                dir_certs[key] = {"error": str(exc), "certificate": float("nan")}

    mean_dir_cert = float(np.nanmean([
        v["certificate"] for v in dir_certs.values() if "error" not in v
    ])) if dir_certs else float("nan")

    # Observational pushed distance (R1.1 only — external real target)
    obs_distance = None
    if is_external_target:
        obs_distance = _observational_pushed_distance(
            tau, eval_bundle, target_bundle, objective,
        )

    return {
        "target_loss": target_loss,
        "err_subfamily": err_subfamily,
        "certificate": float(certificate),
        "gap": gap,
        "coverage_fraction": n_covered / n_queries if n_queries > 0 else None,
        "all_covered": n_covered == n_queries,
        "per_query": queries,
        "dir_certificate": mean_dir_cert,
        "per_query_dir": dir_certs,
        "obs_distance": obs_distance,
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def _shift_spec_labels(shift_spec) -> tuple[str, float]:
    """CSV labels for a shift spec (single dict or compound list)."""
    if shift_spec is None:
        return "none", 0.0
    if isinstance(shift_spec, list):
        return ("+".join(s["shift_type"] for s in shift_spec),
                float(shift_spec[0]["magnitude"]))
    return shift_spec["shift_type"], float(shift_spec["magnitude"])


def run_evaluation(
    results_pkl: Path,
    target_config: Path | None = None,
    shift_spec_override: dict | None = None,
    num_trials: int = 1,
    output_dir: Path | None = None,
    target_seed: int = 99999,
    shift_grid: list | None = None,
    pairing: str = "paired",
) -> pd.DataFrame:
    """Main eval loop.  Loads saved τ's, reveals targets, scores uniformly.

    Parameters
    ----------
    results_pkl : path to traca_cv_results.pkl from traca_train.py
    target_config : path to target data_configs YAML (for Portland)
    shift_spec_override : override stored shift_spec (e.g., to try different magnitudes)
    num_trials : number of trials per method (for stochastic shifts; a no-op
        for the Gaussian objective, which scores analytically)
    output_dir : where to save CSV + summary (defaults to results_pkl parent)
    target_seed : seed for target bundle sampling (must match across paths for parity)
    shift_grid : optional list of shift specs (each a dict or a compound
        list of dicts).  When provided, every stored map is scored against
        every grid shift — the (prior × test-time shift) surface.  Output
        goes to ``evaluation_grid.csv`` with columns
        [prior_axis, prior_value, shift_type, shift_magnitude, trial, ...].
        Training is never touched (target-blind preserved).
    pairing : "paired" or "unpaired".  Paired reuses held-out U rows through
        the shifted SCM (both source and target moments estimated from same U);
        unpaired draws fresh target samples.  Default "paired".

    Returns
    -------
    pd.DataFrame in long format
    """
    results, metadata = load_training_results(results_pkl)
    sweep_axis = metadata["sweep_axis"]
    objective = metadata["objective"]

    # Load source bundle
    bundle_path = Path(metadata["source_bundle_path"])
    source_bundle = joblib.load(bundle_path)

    # Load external target bundle if provided (Portland)
    external_target = None
    if target_config is not None:
        with target_config.open() as f:
            tcfg = yaml.safe_load(f)
        target_scm = _scm_from_config(tcfg)
        target_interventions = list(tcfg["interventions"])
        n_target = int(tcfg.get("n_samples", source_bundle.n))
        external_target = target_scm.bundle(
            target_interventions, n=n_target,
            seed=int(tcfg.get("seed", 0)) + 9999,
        )

    # Fallback: target_bundle_path in the saved experiment config (Portland)
    if external_target is None:
        saved_cfg_path = results_pkl.parent / "config.yaml"
        if saved_cfg_path.exists():
            with saved_cfg_path.open() as f:
                saved_cfg = yaml.safe_load(f)
            tbp = saved_cfg.get("data", {}).get("target_bundle_path")
            if tbp is not None and Path(tbp).exists():
                external_target = joblib.load(Path(tbp))
                print(f"[eval] target bundle: {tbp} (from config.yaml)")

    # Flatten to methods
    flat = flatten_to_methods(results, sweep_axis)

    # Expand baseline_identity across the full ε (and η) grid so that
    # the identity map is scored at every radius the TraCA rows use.
    flat = _expand_baseline_across_grid(flat, metadata)

    agg_axis = metadata.get("aggregation_axis", "fold")
    n_groups = len({e["fold"] for e in flat})
    if sweep_axis == "grid":
        grid_mode = metadata.get("grid_mode", "?")
        n_eps = len(metadata.get("eps_values", []))
        n_eta = len(metadata.get("eta_values", []))
        n_pts = n_eps * n_eta if grid_mode == "cross" else min(n_eps, n_eta)
        print(f"[eval] {len(flat)} method entries ({n_groups} {agg_axis}s × "
              f"{n_pts} grid points + baselines)")
    else:
        print(f"[eval] {len(flat)} method entries ({n_groups} {agg_axis}s × "
              f"{len(metadata['radius_values'])} radii + baselines)")

    records: list[dict] = []
    query_records: list[dict] = []
    config_name = metadata.get("experiment_name")

    if shift_grid is not None:
        # ------------------------------------------------------------------
        # (Prior × test-time shift) grid: score every stored map against
        # every grid shift.  Long-form output; no Huber alpha axis (the
        # selection diagram fixes WHICH nodes shift — the grid asks only
        # BY HOW MUCH).  certificate = δ² (squared); dir_certificate =
        # half-width (unsquared).  Separate columns — never divide them.
        # ------------------------------------------------------------------
        print(f"[eval] shift grid: {len(shift_grid)} settings × {len(flat)} entries")
        for entry in flat:
            for g_spec in shift_grid:
                scores = _reveal_target_and_score(
                    entry["tau"], source_bundle, entry["test_indices"],
                    shift_spec=g_spec, objective=objective,
                    config=entry["config_snapshot"],
                    seed=target_seed,
                    pairing=pairing,
                    config_name=config_name,
                )
                shift_type_str, shift_mag = _shift_spec_labels(g_spec)
                records.append({
                    "prior_axis": sweep_axis,
                    "prior_value": entry["radius"],
                    "method": entry["method"],
                    "shift_type": shift_type_str,
                    "shift_magnitude": shift_mag,
                    "trial": 0,
                    "fold": entry["fold"],
                    "target_loss": scores["target_loss"],
                    "err_subfamily": scores["err_subfamily"],
                    "certificate": scores["certificate"],
                    "dir_certificate": scores["dir_certificate"],
                    "coverage_fraction": scores["coverage_fraction"],
                    "gap": scores["gap"],
                    "all_covered": scores["all_covered"],
                })
            print(f"[eval] {entry['method']:30s} {entry['fold']}  "
                  f"({len(shift_grid)} shifts scored)")
    else:
        for trial in range(num_trials):
            for entry in flat:
                # Determine shift_spec: override > stored > None
                shift_spec = shift_spec_override or entry["shift_spec"]

                # Determine target
                if external_target is not None:
                    # Portland/ATE: use pre-built target (pairing ignored — no shift_spec)
                    scores = _reveal_target_and_score(
                        entry["tau"], source_bundle, entry["test_indices"],
                        shift_spec=None, objective=objective,
                        config=entry["config_snapshot"],
                        target_bundle=external_target,
                        seed=target_seed,
                        pairing=pairing,
                        is_external_target=True,
                        config_name=config_name,
                    )
                elif shift_spec is not None:
                    # Synthetic: apply shift to build pseudo-target
                    scores = _reveal_target_and_score(
                        entry["tau"], source_bundle, entry["test_indices"],
                        shift_spec=shift_spec, objective=objective,
                        config=entry["config_snapshot"],
                        seed=target_seed + trial,
                        pairing=pairing,
                        config_name=config_name,
                    )
                else:
                    print(f"[eval] WARNING: no shift_spec and no target_config for "
                          f"{entry['method']} — skipping (would score against unshifted source)")
                    continue

                shift_type_str, shift_mag = _shift_spec_labels(shift_spec)

                rec = {
                    "method": entry["method"],
                    "radius": entry["radius"],
                    "eps": entry.get("eps"),
                    "eta": entry.get("eta"),
                    "sweep_axis": sweep_axis,
                    "shift_type": shift_type_str,
                    "shift_magnitude": shift_mag,
                    "fold": entry["fold"],
                    "trial": trial,
                    "coverage_fraction": scores["coverage_fraction"],
                    "target_loss": scores["target_loss"],
                    "err_subfamily": scores["err_subfamily"],
                    "certificate": scores["certificate"],
                    "gap": scores["gap"],
                    "all_covered": scores["all_covered"],
                    "dir_certificate": scores["dir_certificate"],
                }
                if scores.get("obs_distance") is not None:
                    rec["obs_distance"] = scores["obs_distance"]
                records.append(rec)

                # Per-query records (evaluation_queries.csv)
                obs_dist = scores.get("obs_distance", float("nan"))
                for qkey, qdata in scores["per_query"].items():
                    iota_str, node_str = qkey.strip("()").split(",")
                    dir_data = scores["per_query_dir"].get(qkey, {})
                    dir_lo = dir_data.get("lower", float("nan"))
                    dir_hi = dir_data.get("upper", float("nan"))
                    dir_width = float(dir_hi - dir_lo) if not (
                        np.isnan(dir_lo) or np.isnan(dir_hi)) else float("nan")
                    target_val = qdata["target_value"]
                    dir_covered = (
                        bool(dir_lo <= target_val <= dir_hi)
                        if not (np.isnan(dir_lo) or np.isnan(dir_hi))
                        else None
                    )
                    query_records.append({
                        "method": entry["method"],
                        "eps": entry.get("eps"),
                        "eta": entry.get("eta"),
                        "fold": entry["fold"],
                        "iota": int(iota_str),
                        "node": int(node_str),
                        "obs_distance": obs_dist if obs_dist is not None else float("nan"),
                        "Phi_pushed": qdata["Phi_pushed"],
                        "target_value": target_val,
                        "std_lo": qdata["lower"],
                        "std_hi": qdata["upper"],
                        "std_width": qdata["width"],
                        "std_covered": qdata["covered"],
                        "dir_lo": dir_lo,
                        "dir_hi": dir_hi,
                        "dir_width": dir_width,
                        "dir_covered": dir_covered,
                    })

                print(f"[eval] {entry['method']:30s} fold={entry['fold']}  "
                      f"cert={scores['certificate']:.4f}  "
                      f"loss={scores['target_loss']:.4f}  "
                      f"cov={scores['coverage_fraction']:.2f}")

    df = pd.DataFrame(records)

    # Save
    if output_dir is None:
        output_dir = results_pkl.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "evaluation_grid.csv" if shift_grid is not None else "evaluation.csv"
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"\n[eval] saved {csv_path}  ({len(df)} rows)")

    # Per-query CSV (non-shift_grid path only)
    if query_records:
        df_q = pd.DataFrame(query_records)
        q_csv_path = output_dir / "evaluation_queries.csv"
        df_q.to_csv(q_csv_path, index=False)
        print(f"[eval] saved {q_csv_path}  ({len(df_q)} rows)")

    # Summary
    summary: dict[str, Any] = {
        "experiment_name": metadata["experiment_name"],
        "n_rows": len(df),
        "n_methods": df["method"].nunique() if len(df) > 0 else 0,
        "aggregation_axis": agg_axis,
        "error_bar_semantics": (
            "source_holdout_moments" if objective == "gaussian" else "fold_samples"
        ),
        "n_trials": num_trials,
        "sweep_axis": sweep_axis,
        "objective": objective,
        "grid_mode": shift_grid is not None,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if shift_grid is None and len(df) > 0:
        summary["n_folds"] = df["fold"].nunique()
    if len(df) > 0:
        summary["mean_certificate"] = float(df["certificate"].mean())
        summary["mean_target_loss"] = float(df["target_loss"].mean())
        summary["mean_coverage"] = float(df["coverage_fraction"].mean())
    summary_name = "summary_grid.json" if shift_grid is not None else "summary.json"
    with (output_dir / summary_name).open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] saved {output_dir / summary_name}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eval pass for decoupled TraCA workflow."
    )
    p.add_argument(
        "--results", required=True,
        help="Path to traca_cv_results.pkl from traca_train.py.",
    )
    p.add_argument("--target_config", default=None, help="Target data_configs YAML (Portland).")
    p.add_argument("--output_dir", default=None, help="Output directory for CSV + summary.")
    p.add_argument("--num_trials", type=int, default=1, help="Number of trials per method.")
    p.add_argument("--target_seed", type=int, default=99999, help="Seed for target bundle sampling.")
    # Shift override
    p.add_argument(
        "--shift_type", default=None,
        choices=["none", "mechanism_edge", "noise_mean", "noise_std", "noise_cov"],
        help="Override shift type for all entries.",
    )
    p.add_argument("--shift_magnitude", type=float, default=0.0)
    p.add_argument("--shift_node", type=int, default=None)
    p.add_argument("--shift_edge", nargs=2, type=int, default=None)
    # Pairing
    p.add_argument(
        "--pairing", default="paired", choices=["paired", "unpaired"],
        help="Target construction: 'paired' reuses held-out U rows through "
             "shifted SCM; 'unpaired' draws fresh target samples. Default: paired.",
    )
    # (Prior × test-time shift) grid
    p.add_argument(
        "--shift_grid_json", type=str, default=None,
        help="JSON list of shift specs (each a dict or compound list of "
             "dicts). Scores every stored map against every grid shift; "
             "writes evaluation_grid.csv. Overrides --shift_type.",
    )
    return p


def _parse_shift_override(args) -> dict | None:
    """Build shift_spec override from CLI args."""
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


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    shift_override = _parse_shift_override(args)
    shift_grid = (
        json.loads(args.shift_grid_json) if args.shift_grid_json else None
    )
    run_evaluation(
        results_pkl=Path(args.results),
        target_config=Path(args.target_config) if args.target_config else None,
        shift_spec_override=shift_override,
        num_trials=args.num_trials,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        target_seed=args.target_seed,
        shift_grid=shift_grid,
        pairing=args.pairing,
    )
