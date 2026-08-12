"""
Radius-sampling evaluation for TraCA.

Loads already-trained τ from ``traca_train.py``'s saved artifacts
(``traca_cv_results.pkl``), samples structurally admissible targets from
ambiguity balls at a test radius ρ_test, and scores each variant with
**zero retraining**.

``ρ_train`` (baked into the saved τ) and ``ρ_test`` (eval-time sampling
radius) are independent knobs — that independence is what makes the
robustness curve well-defined.

Sampling semantics
------------------
- **EntrywiseBox**: uniform-in-box at radius ρ_test.
- **FrobeniusBall / FrobeniusEmpirical**: uniform-on-sphere at radius ρ_test.
- **GelbrichBall**: projected-to-boundary at radius ρ_test.

Seeds are fold-independent: ``seed = hash((rho_test_idx, k))`` so the
same K targets are shared across all folds/variants.

Intervals are read from the trained artifact (built at ρ_train),
**never rebuilt at ρ_test**.

Usage
-----
    python traca_radius_eval.py \\
        --results results/ate/traca_cv_results.pkl \\
        --rho_test 0.0 0.1 0.2 0.3 0.5 \\
        --K 50

Saves
-----
    {output_dir}/radius_eval.csv      — raw per-instance rows
    {output_dir}/radius_eval_agg.csv  — aggregated error/coverage curves
"""
from __future__ import annotations

import argparse
import copy
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
from lan_scm import LANSCM, SCMBundle  # noqa: E402

from traca.ambiguity import (
    FrobeniusBall, EntrywiseBox, GelbrichBall, FrobeniusEmpirical,
    MechanismAmbiguitySet,
)
from traca.stability import gamma, alpha_polynomial, gating_matrix, perturbed_propagator
from traca.certificates import (
    full_joint_certificate,
    single_query_certificate,
    query_interval,
)
from traca.losses import GaussianLoss, EmpiricalLoss
from traca.directional_certificates import (
    directional_certificate_gaussian,
    directional_certificate_empirical,
)
from traca.utils import interventional_exo_mean, bundle_exo_means, build_U_effs
from traca.query import F_iota_O_rho, F_iota_O_U
from experiments.run import (
    _build_mechanism_set,
    _build_environment_set,
)
from experiments.evaluate import (
    _build_alpha_iotas,
    _compute_target_loss_per_iota,
    _iv_to_nodes,
)
from traca_run_evaluation import (
    load_training_results,
    flatten_to_methods,
    _expand_baseline_across_grid,
)


# Uniform containment tolerance for coverage checks.  Absorbs machine-
# precision non-associativity (e.g. (U @ A) @ τ  vs  U @ (A @ τ)) so that
# zero-width intervals don't register as misses.
COVERAGE_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Geometry guards
# ---------------------------------------------------------------------------

_MECHANISM_TYPE_MAP = {
    "EntrywiseBox": EntrywiseBox,
    "FrobeniusBall": FrobeniusBall,
}

_ENVIRONMENT_TYPE_MAP = {
    "GelbrichBall": GelbrichBall,
    "FrobeniusEmpirical": FrobeniusEmpirical,
}


def _mechanism_type_name(cfg: dict) -> str:
    return cfg["ambiguity"]["mechanism"]["type"]


def _environment_type_name(cfg: dict) -> str:
    return cfg["ambiguity"]["environment"]["type"]


# ---------------------------------------------------------------------------
# Q-restricted query-family registry
# ---------------------------------------------------------------------------
#
# Per-benchmark-family registry of query-set definitions.  These are the
# (ι, O) pairs that define the "subfamily" query set for each benchmark
# family.  Validated at runtime against the corresponding YAML configs
# to catch staleness.
#
# Cross-eval matrix (which err_* columns get populated):
#   trained on full      → err, err_subfamily
#   trained on subfamily → err, err_subfamily

_QUERY_FAMILY_REGISTRY: dict[str, dict[str, list[tuple[int, list[int]]]]] = {
    "ate": {
        "subfamily": [(1, [1]), (2, [1])],
    },
    "atce": {
        "subfamily": [(1, [2]), (2, [2])],
    },
    "lilucas_frob": {
        "subfamily": [(1, [3]), (2, [3]), (3, [3]), (4, [3])],
    },
    "lilucas_ew": {
        "subfamily": [(1, [3]), (2, [3]), (3, [3]), (4, [3])],
    },
    "portland": {
        "subfamily": [(1, [3]), (2, [3]), (3, [3]), (4, [3])],
    },
}

# Map from config name → (benchmark_family, query_type).
# query_type is "full" or "subfamily" — what this config trained on.
_CONFIG_NAME_MAP: dict[str, tuple[str, str]] = {
    "ate_gaussian_entrywise_subfamily":            ("ate",          "subfamily"),
    "ate_gaussian_entrywise_subfamily_directional": ("ate",          "subfamily"),
    "atce_gaussian_z_entrywise_full":              ("atce",         "full"),
    "atce_gaussian_z_entrywise_subfamily":         ("atce",         "subfamily"),
    "lilucas_light_frobenius_full":                ("lilucas_frob", "full"),
    "lilucas_light_frobenius_subfamily":           ("lilucas_frob", "subfamily"),
    "lilucas_light_entrywise_full":                ("lilucas_ew",   "full"),
    "lilucas_light_entrywise_subfamily":           ("lilucas_ew",   "subfamily"),
    "lilucas_light_entrywise_full_directional":    ("lilucas_ew",   "full"),
    "lilucas_light_entrywise_subfamily_directional": ("lilucas_ew", "subfamily"),
    "lilucas_light_gaussian_entrywise_subfamily":  ("lilucas_ew",   "subfamily"),
    "portland_backdoor_gaussian_qrestricted":      ("portland",     "subfamily"),
    "portland_env_subfamily":                      ("portland",     "subfamily"),
    "portland_env_full":                            ("portland",     "full"),
}

# Config files that can be used to validate the hardcoded registry entries.
# Keys = (benchmark_family, query_type), values = config file paths.
_VALIDATION_CONFIGS: dict[tuple[str, str], str] = {
    ("ate",          "subfamily"): "configs/ate/gaussian_entrywise_subfamily.yaml",
    ("atce",         "subfamily"): "configs/atce/gaussian_z_entrywise_subfamily.yaml",
    ("lilucas_frob", "subfamily"): "configs/lilucas/light_empirical_frobenius_subfamily.yaml",
    ("lilucas_ew",   "subfamily"): "configs/lilucas/light_empirical_entrywise_subfamily.yaml",
    ("portland",     "subfamily"): "experiments/configs/portland_backdoor_gaussian_qrestricted.yaml",
}


def _validate_query_registry() -> None:
    """Assert hardcoded registry matches the corresponding YAML configs.

    Called once at eval startup.  Fails loudly on any mismatch.
    """
    from traca.query import query_family_from_config

    for (family, qtype), cfg_path in _VALIDATION_CONFIGS.items():
        path = Path(cfg_path)
        if not path.exists():
            continue
        with path.open() as f:
            cfg = yaml.safe_load(f)
        from_config = query_family_from_config(cfg)
        from_registry = _QUERY_FAMILY_REGISTRY[family].get(qtype)
        assert from_config == from_registry, (
            f"Query registry mismatch for ({family}, {qtype}):\n"
            f"  config {cfg_path}: {from_config}\n"
            f"  registry:          {from_registry}\n"
            f"Update _QUERY_FAMILY_REGISTRY to match the config."
        )


def _resolve_benchmark_family(config_name: str) -> tuple[str, str] | None:
    """Return (benchmark_family, trained_query_type) for a config name, or None."""
    return _CONFIG_NAME_MAP.get(config_name)


def _cross_eval_columns(
    trained_on: str,
    family: str,
) -> dict[str, list[tuple[int, list[int]]] | None]:
    """Return which Q-restricted losses to compute for this variant.

    Returns dict of column_name → query_family_pairs (or None to skip).
    Implements the cross-eval matrix.
    """
    reg = _QUERY_FAMILY_REGISTRY.get(family, {})
    subfamily_qf = reg.get("subfamily")

    result: dict[str, list[tuple[int, list[int]]] | None] = {}

    if trained_on in ("full", "subfamily"):
        result["err_subfamily"] = subfamily_qf
    else:
        result["err_subfamily"] = None

    return result


def _compute_restricted_loss(
    tau: np.ndarray,
    source_bundle: SCMBundle,
    target_bundle: SCMBundle,
    query_pairs: list[tuple[int, list[int]]],
    objective: str,
    U_s_list: list[np.ndarray] | None = None,
    source_noise_mean: np.ndarray | None = None,
    source_noise_cov: np.ndarray | None = None,
) -> float:
    """Compute Q-restricted transport loss averaged over the given (ι, O) pairs.

    Uses F_iota_O_rho (Gaussian) or F_iota_O_U (empirical) from traca.query.

    Parameters
    ----------
    tau : (d, d) transport map
    source_bundle, target_bundle : SCMBundle
    query_pairs : list of (intervention_idx, output_nodes)
    objective : "gaussian" or "empirical"
    U_s_list : per-intervention U_s arrays (empirical mode)
    source_noise_mean, source_noise_cov : holdout-estimated moments (empirical Gaussian fallback)
    """
    W_s = source_bundle.W
    W_t = target_bundle.W
    dW_actual = W_t - W_s
    d = source_bundle.d
    var_names = source_bundle.scm.var_names
    pure_noise_shift = float(np.linalg.norm(dW_actual, "fro")) < 1e-10

    losses = []
    for iota_idx, O in query_pairs:
        A_s = source_bundle.intervened_scms[iota_idx].A
        R_i = gating_matrix(d, _iv_to_nodes(
            source_bundle.interventions[iota_idx], var_names))

        if objective == "empirical" and not pure_noise_shift:
            U_s = (U_s_list[iota_idx] if U_s_list is not None
                   else source_bundle.noise_samples[iota_idx])
            N = U_s.shape[0]
            from experiments.evaluate import _extract_noise_mean_deltas
            noise_deltas = _extract_noise_mean_deltas(None, d)
            Theta = np.tile(noise_deltas, (N, 1))
            losses.append(float(F_iota_O_U(
                tau, dW_actual, Theta, W_s, A_s, R_i, U_s, O)))
        else:
            # Gaussian W₂²
            mu_s = (source_noise_mean if source_noise_mean is not None
                    else source_bundle.noise_mean)
            Sigma_s = (source_noise_cov if source_noise_cov is not None
                       else source_bundle.noise_cov)
            scm_s_i = source_bundle.intervened_scms[iota_idx]
            mu_s_eff = interventional_exo_mean(mu_s, scm_s_i._fixed, scm_s_i._J)
            A_t = target_bundle.intervened_scms[iota_idx].A
            scm_t_i = target_bundle.intervened_scms[iota_idx]
            mu_t_obs = (interventional_exo_mean(
                target_bundle.noise_mean, scm_t_i._fixed, scm_t_i._J) @ A_t)
            Sigma_t_obs = A_t.T @ target_bundle.noise_cov @ A_t
            losses.append(float(F_iota_O_rho(
                tau, np.zeros((d, d)), W_s, A_s, R_i,
                mu_s_eff, Sigma_s, mu_t_obs, Sigma_t_obs, O)))

    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Build target bundle from sampled perturbations
# ---------------------------------------------------------------------------

def _build_sampled_target_bundle_gaussian(
    source_bundle: SCMBundle,
    dW: np.ndarray,
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    test_indices: np.ndarray,
) -> SCMBundle:
    """Build a target bundle from sampled (dW, mu_t, Sigma_t).

    Uses the paired variance-cancellation trick: source and target share
    the same held-out observational U.

    Parameters
    ----------
    source_bundle : full source SCMBundle
    dW : (d, d) mechanism perturbation
    mu_t : (d,) target exogenous mean
    Sigma_t : (d, d) target exogenous covariance
    test_indices : held-out sample indices
    """
    d = source_bundle.d
    scm_s = source_bundle.scm

    # Build perturbed SCM: W_t = W_s + dW
    shifted_scm = scm_s.perturb(dW)

    # Override noise parameters to the sampled target
    shifted_scm.noise_mean = mu_t.copy()
    shifted_scm.noise_cov = Sigma_t.copy()

    # Observational held-out U
    U_obs_holdout = source_bundle.noise_samples[0][test_indices].copy()

    # Per-intervention: propagate paired U through shifted+intervened SCM
    intervened_scms: dict[int, Any] = {}
    endogenous_samples: dict[int, np.ndarray] = {}
    noise_samples: dict[int, np.ndarray] = {}

    for i, iv in enumerate(source_bundle.interventions):
        scm_do = shifted_scm.intervene(iv)
        U_i = U_obs_holdout.copy()
        if scm_do._J:
            U_i[:, scm_do._J] = 0.0
        # Target endogenous: (U_zeroed + fixed) @ A_t_do
        # But for Gaussian scoring we need the moments, not samples.
        # We build endogenous samples for completeness (empirical fallback).
        X_target = U_i @ scm_do.A + scm_do._fixed @ scm_do.A
        intervened_scms[i] = scm_do
        endogenous_samples[i] = X_target
        noise_samples[i] = U_i

    return SCMBundle(
        scm=shifted_scm,
        interventions=list(source_bundle.interventions),
        intervened_scms=intervened_scms,
        noise_mean=mu_t.copy(),
        noise_cov=Sigma_t.copy(),
        endogenous_samples=endogenous_samples,
        noise_samples=noise_samples,
        n=len(test_indices),
        seed=None,
    )


def _build_sampled_target_bundle_empirical(
    source_bundle: SCMBundle,
    dW: np.ndarray,
    Theta: np.ndarray,
    test_indices: np.ndarray,
) -> SCMBundle:
    """Build a target bundle from sampled (dW, Theta).

    Empirical mode: target uses shared observational U + noise shift Theta,
    propagated through the perturbed mechanism W_s + dW.

    Parameters
    ----------
    source_bundle : full source SCMBundle
    dW : (d, d) mechanism perturbation
    Theta : (N_test, d) additive noise shift
    test_indices : held-out sample indices
    """
    d = source_bundle.d
    scm_s = source_bundle.scm

    shifted_scm = scm_s.perturb(dW)

    U_obs_holdout = source_bundle.noise_samples[0][test_indices].copy()
    # Apply noise shift
    U_shifted = U_obs_holdout + Theta

    intervened_scms: dict[int, Any] = {}
    endogenous_samples: dict[int, np.ndarray] = {}
    noise_samples: dict[int, np.ndarray] = {}

    for i, iv in enumerate(source_bundle.interventions):
        scm_do = shifted_scm.intervene(iv)
        U_i = U_shifted.copy()
        if scm_do._J:
            U_i[:, scm_do._J] = 0.0
        X_target = U_i @ scm_do.A + scm_do._fixed @ scm_do.A
        intervened_scms[i] = scm_do
        endogenous_samples[i] = X_target
        noise_samples[i] = U_i

    # Estimate moments from the shifted noise
    noise_mean = np.mean(U_shifted, axis=0)
    noise_cov = np.cov(U_shifted, rowvar=False)

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


# ---------------------------------------------------------------------------
# Per-variant cert precomputation (source-dependent, target-independent)
# ---------------------------------------------------------------------------

def _precompute_query_certs(
    tau: np.ndarray,
    source_bundle: SCMBundle,
    test_indices: np.ndarray,
    config: dict,
    objective: str,
) -> dict:
    """Precompute per-query certificate data that depends only on the source.

    Returns a dict with:
      - certificate: float (full-joint standard cert)
      - query_certs: {key: {Phi_pushed, std_cert_q, std_lo, std_hi,
                            dir_cert, dir_lo, dir_hi}}

    These are fixed for a given (tau, fold, config) and can be reused
    across all K target samples and rho_test values.
    """
    d = source_bundle.d
    N = source_bundle.n
    var_names = source_bundle.scm.var_names

    use_holdout = len(test_indices) < N
    holdout_idx = test_indices if use_holdout else None
    U_s_list = build_U_effs(source_bundle, indices=holdout_idx)
    N_eval = len(test_indices) if use_holdout else N

    U_obs_test = source_bundle.noise_samples[0][test_indices] if use_holdout \
        else source_bundle.noise_samples[0]
    noise_mean_eval = np.mean(U_obs_test, axis=0) if use_holdout \
        else source_bundle.noise_mean
    noise_cov_eval = np.cov(U_obs_test, rowvar=False) if use_holdout \
        else source_bundle.noise_cov

    aw = _build_mechanism_set(config["ambiguity"]["mechanism"], d)
    ae = _build_environment_set(
        config["ambiguity"]["environment"],
        noise_mean_eval, noise_cov_eval, N_eval,
    )

    A_iotas, alpha_iotas = _build_alpha_iotas(source_bundle, aw)
    n_iota = source_bundle.n_interventions()
    eps = float(config["ambiguity"]["environment"]["eps"])

    # Holdout effective means — used for standard cert width terms.
    mu_s_effs = [
        interventional_exo_mean(
            noise_mean_eval,
            source_bundle.intervened_scms[i]._fixed,
            source_bundle.intervened_scms[i]._J,
        )
        for i in range(n_iota)
    ]

    # Population effective means — used for Gaussian Phi_pushed and
    # directional cert.  The Gaussian certificate is a population
    # statement: both Q_{τs} and Q_t are functionals of probability
    # measures, not sample estimates.
    if objective == "gaussian":
        mu_s_pop_effs = [
            interventional_exo_mean(
                source_bundle.noise_mean,       # population, not holdout
                source_bundle.intervened_scms[i]._fixed,
                source_bundle.intervened_scms[i]._J,
            )
            for i in range(n_iota)
        ]

    certificate = full_joint_certificate(
        tau, alpha_iotas, A_iotas,
        mu_s_effs, noise_cov_eval,
        eps=eps, mode=objective,
        U_s_list=U_s_list if objective == "empirical" else None,
        N=N_eval if objective == "empirical" else None,
    )

    query_certs: dict[str, dict] = {}

    for i, iv in enumerate(source_bundle.interventions):
        J_i = set(_iv_to_nodes(iv, var_names))
        R_i = gating_matrix(d, list(J_i))
        g_i = gamma(A_iotas[i], R_i, aw)

        for k in range(d):
            if k in J_i:
                continue  # skip intervened nodes — O ⊆ [d] \ J_ι
            key = f"({i},{k})"

            # Standard certificate (width — holdout moments are fine here,
            # the standard cert's slack absorbs O(1/√N) estimation error)
            cert_q = float(single_query_certificate(
                tau, alpha_iotas[i], A_iotas[i],
                mu_s_effs[i], noise_cov_eval,
                eps=eps, O=[k], d=d, mode=objective,
                U_s=U_s_list[i] if objective == "empirical" else None,
                N=N_eval if objective == "empirical" else None,
            ))

            # Phi_pushed — the interval CENTER.
            # Gaussian: population formula (paper Thm: Q_{τs} = Φ(τ_# P_s),
            #   a functional of the population measure).
            # Empirical: holdout sample mean (paper Thm: Q_{τs} = q^T τ A Ū^s,
            #   a functional of the same fixed N rows of U^s).
            if objective == "empirical":
                Phi_pushed = float(np.mean(
                    U_s_list[i] @ A_iotas[i] @ tau, axis=0
                )[k])
            else:
                Phi_pushed = float(
                    (mu_s_pop_effs[i] @ A_iotas[i] @ tau)[k]
                )

            lo_std, hi_std = query_interval(Phi_pushed, 1.0, cert_q)

            # Directional certificate.
            # Gaussian: v = μ_s (population) — pass None so the function
            #   reads bundle.noise_mean (population default).
            # Empirical: v = Ū^s (holdout sample mean) — pass U_eff so
            #   the function uses the same N holdout rows.
            q_vec = np.zeros(d)
            q_vec[k] = 1.0
            try:
                if objective == "empirical":
                    dc = directional_certificate_empirical(
                        tau, q_vec, [k], i, source_bundle,
                        aw, ae, alpha_iotas[i], g_i,
                        U_eff=U_s_list[i],
                    )
                else:
                    dc = directional_certificate_gaussian(
                        tau, q_vec, [k], i, source_bundle,
                        aw, ae, alpha_iotas[i], g_i,
                    )
                dir_cert_val = float(dc["certificate"])
            except Exception:
                dir_cert_val = float("nan")

            dir_lo = Phi_pushed - dir_cert_val
            dir_hi = Phi_pushed + dir_cert_val

            query_certs[key] = {
                "Phi_pushed": Phi_pushed,
                "std_cert_q": cert_q,
                "std_lo": float(lo_std),
                "std_hi": float(hi_std),
                "dir_cert": dir_cert_val,
                "dir_lo": dir_lo,
                "dir_hi": dir_hi,
            }

    return {
        "certificate": float(certificate),
        "query_certs": query_certs,
    }


# ---------------------------------------------------------------------------
# Per-variant scoring (no retraining)
# ---------------------------------------------------------------------------

def _score_variant(
    tau: np.ndarray,
    source_bundle: SCMBundle,
    target_bundle: SCMBundle,
    test_indices: np.ndarray,
    config: dict,
    objective: str,
    precomputed_certs: dict | None = None,
) -> dict:
    """Score a single variant's τ against a sampled target.

    Computes:
    - target_loss: transport error
    - per_query: {Phi_pushed, target_value, std/dir intervals + coverage}

    When *precomputed_certs* is provided (from ``_precompute_query_certs``),
    the certificate, Phi_pushed, and intervals are read from the cache — only
    target_val and target_loss are computed fresh.  This avoids redundant
    cert computation across K target samples.

    The certificate and intervals use the variant's own ρ_train (from
    config_snapshot), NOT ρ_test.
    """
    d = source_bundle.d
    N = source_bundle.n
    n_iota = source_bundle.n_interventions()

    # --- Source data (needed for target_loss even when certs are cached) ---
    use_holdout = len(test_indices) < N
    holdout_idx = test_indices if use_holdout else None
    U_s_list = build_U_effs(source_bundle, indices=holdout_idx)

    # Holdout-estimated source moments for the Gaussian W₂² fallback
    # (empirical mode, pure noise shift).  Both source and target moments
    # are sample-estimated from the same held-out U rows, so at zero shift
    # they are identical by construction → err = 0 exactly.
    if objective == "empirical" and use_holdout:
        U_obs_test = source_bundle.noise_samples[0][test_indices]
        noise_mean_eval = np.mean(U_obs_test, axis=0)
        noise_cov_eval = np.cov(U_obs_test, rowvar=False)
    else:
        noise_mean_eval = None
        noise_cov_eval = None

    # Target loss
    per_iota_losses = _compute_target_loss_per_iota(
        tau, source_bundle, target_bundle, objective,
        U_s_list=U_s_list if objective == "empirical" else None,
        source_noise_mean=noise_mean_eval,
        source_noise_cov=noise_cov_eval,
    )
    target_loss = float(np.mean(per_iota_losses))

    # --- Certs: use precomputed if available ---
    if precomputed_certs is not None:
        certificate = precomputed_certs["certificate"]
        qc_cache = precomputed_certs["query_certs"]
    else:
        # Fallback: compute inline (backward-compat for callers without cache)
        qc_cache = _precompute_query_certs(
            tau, source_bundle, test_indices, config, objective,
        )
        certificate = qc_cache["certificate"]
        qc_cache = qc_cache["query_certs"]

    # --- Per-query: read cached intervals, compute target_val, check coverage ---
    var_names = source_bundle.scm.var_names
    queries: dict[str, dict] = {}
    for i, iv in enumerate(source_bundle.interventions):
        J_i = set(_iv_to_nodes(iv, var_names))
        A_t = target_bundle.intervened_scms[i].A
        for k in range(d):
            if k in J_i:
                continue  # skip intervened nodes — O ⊆ [d] \ J_ι
            key = f"({i},{k})"
            qc = qc_cache[key]

            # Target value (target-dependent, always computed fresh)
            if objective == "gaussian":
                scm_t_i = target_bundle.intervened_scms[i]
                target_val = float((interventional_exo_mean(
                    target_bundle.noise_mean, scm_t_i._fixed, scm_t_i._J
                ) @ A_t)[k])
            else:
                X_t = target_bundle.endogenous_samples[i]
                target_val = float(np.mean(X_t[:, k]))

            # Coverage with uniform tolerance
            std_covered = bool(
                qc["std_lo"] - COVERAGE_ATOL <= target_val
                <= qc["std_hi"] + COVERAGE_ATOL
            )
            dir_covered = bool(
                qc["dir_lo"] - COVERAGE_ATOL <= target_val
                <= qc["dir_hi"] + COVERAGE_ATOL
            )

            queries[key] = {
                "Phi_pushed": qc["Phi_pushed"],
                "target_value": target_val,
                "std_cert_q": qc["std_cert_q"],
                "lower": qc["std_lo"],
                "upper": qc["std_hi"],
                "width": qc["std_hi"] - qc["std_lo"],
                "covered": std_covered,
                "dir_cert": qc["dir_cert"],
                "dir_lower": qc["dir_lo"],
                "dir_upper": qc["dir_hi"],
                "dir_width": qc["dir_hi"] - qc["dir_lo"],
                "dir_covered": dir_covered,
            }

    n_queries = len(queries)
    n_covered = sum(1 for q in queries.values() if q["covered"])
    n_dir_covered = sum(1 for q in queries.values() if q["dir_covered"])

    return {
        "target_loss": target_loss,
        "certificate": float(certificate),
        "gap": float(certificate - target_loss),
        "coverage_fraction": n_covered / n_queries if n_queries > 0 else None,
        "dir_coverage_fraction": n_dir_covered / n_queries if n_queries > 0 else None,
        "all_covered": n_covered == n_queries,
        "per_query": queries,
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def radius_sampling_eval(
    results_pkl: Path,
    rho_test_values: list[float],
    K: int,
    output_dir: Path | None = None,
    base_seed: int = 2026,
) -> pd.DataFrame:
    """Run the radius-sampling evaluation.

    Parameters
    ----------
    results_pkl : path to traca_cv_results.pkl from traca_train.py
    rho_test_values : list of test radii to evaluate
    K : number of random targets per ρ_test
    output_dir : where to save CSV (defaults to results_pkl parent)
    base_seed : base seed for target sampling

    Returns
    -------
    DataFrame with per-instance rows
    """
    results, metadata = load_training_results(results_pkl)
    sweep_axis = metadata.get("sweep_axis", "eps")
    objective = metadata["objective"]
    d = metadata["d"]

    # Load source bundle
    bundle_path = Path(metadata["source_bundle_path"])
    source_bundle = joblib.load(bundle_path)

    # Build pristine ambiguity sets from metadata config
    # (these are the TRAINING geometry — never mutated)
    sample_cfg_path = Path(metadata["config_path"])
    with sample_cfg_path.open() as f:
        base_cfg = yaml.safe_load(f)

    pristine_mechanism = _build_mechanism_set(base_cfg["ambiguity"]["mechanism"], d)
    pristine_environment = _build_environment_set(
        base_cfg["ambiguity"]["environment"],
        source_bundle.noise_mean, source_bundle.noise_cov, source_bundle.n,
    )
    mechanism_type = _mechanism_type_name(base_cfg)
    environment_type = _environment_type_name(base_cfg)

    # Flatten training results
    flat = flatten_to_methods(results, sweep_axis)
    flat = _expand_baseline_across_grid(flat, metadata)

    # Geometry guard: verify all variants use same geometry as base config
    for entry in flat:
        entry_mech = _mechanism_type_name(entry["config_snapshot"])
        entry_env = _environment_type_name(entry["config_snapshot"])
        if entry_mech != mechanism_type:
            raise ValueError(
                f"Geometry mismatch: variant {entry['method']} uses mechanism "
                f"{entry_mech}, but base config uses {mechanism_type}. "
                f"Cannot score across geometries."
            )
        if entry_env != environment_type:
            raise ValueError(
                f"Geometry mismatch: variant {entry['method']} uses environment "
                f"{entry_env}, but base config uses {environment_type}. "
                f"Cannot score across geometries."
            )

    # --- Q-restricted query-family resolution ---
    _validate_query_registry()
    config_name = base_cfg.get("name", "")
    resolved = _resolve_benchmark_family(config_name)
    if resolved is not None:
        benchmark_family, trained_qtype = resolved
        reg = _QUERY_FAMILY_REGISTRY.get(benchmark_family, {})
        print(f"[radius_eval] benchmark family: {benchmark_family}")
        print(f"[radius_eval]   trained query type: {trained_qtype}")
        print(f"[radius_eval]   subfamily (ι,O) = {reg.get('subfamily')}")
        cross_cols = _cross_eval_columns(trained_qtype, benchmark_family)
        print(f"[radius_eval]   cross-eval columns: "
              + ", ".join(f"{k}={'populated' if v else 'NaN'}" for k, v in cross_cols.items()))
    else:
        benchmark_family = None
        cross_cols = {"err_subfamily": None}
        print(f"[radius_eval] config '{config_name}' not in registry — "
              f"err_subfamily will be NaN")

    # --- Precompute per-variant cert data (source-dependent only) -----------
    # Each directional cert call is O(1) (closed-form) or O(scipy) for
    # EntrywiseBox env with ≤ 4 free entries.  Precomputing once per variant
    # and reusing across all K × rho_test avoids ~1000× redundant work.
    print(f"[radius_eval] precomputing certs for {len(flat)} variants ...")
    variant_certs: list[dict] = []
    for idx, entry in enumerate(flat):
        vc = _precompute_query_certs(
            entry["tau"], source_bundle, entry["test_indices"],
            entry["config_snapshot"], objective,
        )
        variant_certs.append(vc)
    print(f"[radius_eval] cert precomputation done")

    records: list[dict] = []
    query_records: list[dict] = []

    for rho_idx, rho_test in enumerate(rho_test_values):
        for k in range(K):
            # Fold-independent seed
            seed_k = hash((rho_idx, k, base_seed)) % (2**31)
            rng = np.random.default_rng(seed_k)

            # Sample mechanism perturbation
            if rho_test == 0.0:
                dW = np.zeros((d, d))
            else:
                mech_at_rho = _mechanism_at_rho_test(
                    pristine_mechanism, rho_test
                )
                dW = mech_at_rho.sample(rng)

            # Sample environment perturbation
            if rho_test == 0.0:
                if objective == "gaussian":
                    mu_t = source_bundle.noise_mean.copy()
                    Sigma_t = source_bundle.noise_cov.copy()
                else:
                    N_test_max = max(len(e["test_indices"]) for e in flat)
                    Theta = np.zeros((N_test_max, d))
            else:
                base_env_eps = _base_environment_radius(pristine_environment)
                env_at_rho = pristine_environment.scale(
                    rho_test / base_env_eps if base_env_eps > 0 else 0.0
                )
                if objective == "gaussian":
                    mu_t, Sigma_t = env_at_rho.sample(rng)
                else:
                    N_test_max = max(len(e["test_indices"]) for e in flat)
                    Theta = env_at_rho.sample(rng, N=N_test_max, d=d)

            # Score each variant
            for idx, entry in enumerate(flat):
                tau = entry["tau"]
                test_indices = entry["test_indices"]
                config_snapshot = entry["config_snapshot"]
                fold = entry["fold"]

                # Build target for this fold's holdout
                if objective == "gaussian":
                    target_bundle = _build_sampled_target_bundle_gaussian(
                        source_bundle, dW, mu_t, Sigma_t, test_indices,
                    )
                else:
                    n_test = len(test_indices)
                    Theta_fold = Theta[:n_test]
                    target_bundle = _build_sampled_target_bundle_empirical(
                        source_bundle, dW, Theta_fold, test_indices,
                    )

                scores = _score_variant(
                    tau, source_bundle, target_bundle,
                    test_indices, config_snapshot, objective,
                    precomputed_certs=variant_certs[idx],
                )

                all_covered = scores["all_covered"]
                mean_width = float(np.mean([
                    q["width"] for q in scores["per_query"].values()
                ]))
                mean_dir_width = float(np.mean([
                    q["dir_width"] for q in scores["per_query"].values()
                ]))

                # Q-restricted losses
                # Reuse the holdout-estimated source moments from _score_variant
                use_holdout = len(test_indices) < source_bundle.n
                holdout_idx = test_indices if use_holdout else None
                if objective == "empirical" and use_holdout:
                    U_obs_test = source_bundle.noise_samples[0][test_indices]
                    _noise_mean_eval = np.mean(U_obs_test, axis=0)
                    _noise_cov_eval = np.cov(U_obs_test, rowvar=False)
                else:
                    _noise_mean_eval = None
                    _noise_cov_eval = None
                _U_s_list = (build_U_effs(source_bundle, indices=holdout_idx)
                             if objective == "empirical" else None)

                restricted_errs: dict[str, float | None] = {}
                for col_name, qf_pairs in cross_cols.items():
                    if qf_pairs is not None:
                        restricted_errs[col_name] = _compute_restricted_loss(
                            tau, source_bundle, target_bundle,
                            qf_pairs, objective,
                            U_s_list=_U_s_list,
                            source_noise_mean=_noise_mean_eval,
                            source_noise_cov=_noise_cov_eval,
                        )
                    else:
                        restricted_errs[col_name] = float("nan")

                records.append({
                    "rho_test": rho_test,
                    "k": k,
                    "fold": fold,
                    "variant": entry["method"],
                    "rho_train": entry["radius"],
                    "eps_train": entry.get("eps"),
                    "eta_train": entry.get("eta"),
                    "err": scores["target_loss"],
                    "err_subfamily": restricted_errs.get("err_subfamily", float("nan")),
                    "certificate": scores["certificate"],
                    "gap": scores["gap"],
                    "coverage_fraction": scores["coverage_fraction"],
                    "dir_coverage_fraction": scores["dir_coverage_fraction"],
                    "all_covered": all_covered,
                    "mean_width": mean_width,
                    "mean_dir_width": mean_dir_width,
                })

                # Per-query rows for re-scorability
                for qkey, qdata in scores["per_query"].items():
                    # Parse "(i,k)" key
                    iota_str, node_str = qkey.strip("()").split(",")
                    query_records.append({
                        "rho_test": rho_test,
                        "k": k,
                        "fold": fold,
                        "variant": entry["method"],
                        "iota": int(iota_str),
                        "node": int(node_str),
                        "Phi_pushed": qdata["Phi_pushed"],
                        "target_val": qdata["target_value"],
                        "std_cert_q": qdata["std_cert_q"],
                        "std_covered": qdata["covered"],
                        "dir_cert": qdata["dir_cert"],
                        "dir_covered": qdata["dir_covered"],
                    })

        print(f"[radius_eval] ρ_test={rho_test:.4f}: {K} samples × {len(flat)} variants done")

    # Invariant: pristine sets unchanged
    fresh_mech = _build_mechanism_set(base_cfg["ambiguity"]["mechanism"], d)
    assert _base_mechanism_radius(pristine_mechanism) == _base_mechanism_radius(fresh_mech), \
        "Pristine mechanism set mutated during eval loop!"

    df = pd.DataFrame(records)
    df_queries = pd.DataFrame(query_records)

    # Save
    if output_dir is None:
        output_dir = results_pkl.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "radius_eval.csv"
    df.to_csv(raw_path, index=False)
    print(f"[radius_eval] raw rows saved to {raw_path} ({len(df)} rows)")

    queries_path = output_dir / "radius_eval_queries.csv"
    df_queries.to_csv(queries_path, index=False)
    print(f"[radius_eval] per-query rows saved to {queries_path} ({len(df_queries)} rows)")

    # Aggregations
    agg = _aggregate(df)
    agg_path = output_dir / "radius_eval_agg.csv"
    agg.to_csv(agg_path, index=False)
    print(f"[radius_eval] aggregated saved to {agg_path}")

    return df


def _mechanism_at_rho_test(pristine: MechanismAmbiguitySet,
                          rho_test: float) -> MechanismAmbiguitySet:
    """Build mechanism set at test radius *rho_test*.

    Mirrors ``_substitute_eta``: for EntrywiseBox with directional prior
    (delta is not None), sets all nonzero B entries to *rho_test* with
    delta **fixed** — the box becomes [delta - rho_test, delta + rho_test].
    This matches training's η-substitution semantics exactly.

    For delta=None (symmetric) or non-EntrywiseBox types, falls through to
    proportional ``scale()`` — preserving the existing code path bit-identically.
    """
    if isinstance(pristine, EntrywiseBox) and pristine.delta is not None:
        B_new = np.zeros_like(pristine.B)
        B_new[pristine.B != 0] = rho_test
        return EntrywiseBox(
            B=B_new, shifted_rows=pristine.shifted_rows,
            d=pristine.d, delta=pristine.delta.copy(),
        )
    # delta=None or non-EntrywiseBox: existing proportional scale() path
    base = _base_mechanism_radius(pristine)
    return pristine.scale(rho_test / base if base > 0 else 0.0)


def _base_mechanism_radius(mech_set) -> float:
    """Extract the base scalar radius from a mechanism set."""
    if isinstance(mech_set, FrobeniusBall):
        return mech_set.eta
    elif isinstance(mech_set, EntrywiseBox):
        # Use the max nonzero entry of effective_bound as the "radius"
        # (= |delta|+B per entry; when delta=None, just B)
        return float(np.max(mech_set.effective_bound))
    else:
        raise ValueError(f"Unsupported mechanism type: {type(mech_set)}")


def _base_environment_radius(env_set) -> float:
    """Extract the base scalar radius from an environment set."""
    if isinstance(env_set, GelbrichBall):
        return env_set.eps
    elif isinstance(env_set, FrobeniusEmpirical):
        return env_set.eps
    else:
        raise ValueError(f"Unsupported environment type: {type(env_set)}")


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-instance rows.

    Error curve: mean(err) over K per (rho_test, variant), ±std.
    Coverage: fraction(all_covered) over K per (rho_test, variant, fold),
              then mean across folds.  Both standard and directional.
    """
    # Error curve: mean over K per (rho_test, variant)
    agg_dict = {
        "err_mean": ("err", "mean"),
        "err_std": ("err", "std"),
        "cert_mean": ("certificate", "mean"),
        "width_mean": ("mean_width", "mean"),
        "dir_width_mean": ("mean_dir_width", "mean"),
    }
    # Add Q-restricted error columns if present
    if "err_subfamily" in df.columns:
        agg_dict["err_subfamily_mean"] = ("err_subfamily", "mean")
        agg_dict["err_subfamily_std"] = ("err_subfamily", "std")
    err_agg = df.groupby(["rho_test", "variant"]).agg(**agg_dict).reset_index()

    # Standard coverage: per (rho_test, variant, fold), fraction over K
    cov_per_fold = df.groupby(["rho_test", "variant", "fold"]).agg(
        coverage_K=("all_covered", "mean"),
    ).reset_index()

    cov_agg = cov_per_fold.groupby(["rho_test", "variant"]).agg(
        coverage_mean=("coverage_K", "mean"),
        coverage_std=("coverage_K", "std"),
    ).reset_index()

    # Directional coverage: mean of per-query dir_coverage_fraction
    dir_cov_per_fold = df.groupby(["rho_test", "variant", "fold"]).agg(
        dir_coverage_K=("dir_coverage_fraction", "mean"),
    ).reset_index()

    dir_cov_agg = dir_cov_per_fold.groupby(["rho_test", "variant"]).agg(
        dir_coverage_mean=("dir_coverage_K", "mean"),
        dir_coverage_std=("dir_coverage_K", "std"),
    ).reset_index()

    # Merge
    agg = err_agg.merge(cov_agg, on=["rho_test", "variant"])
    agg = agg.merge(dir_cov_agg, on=["rho_test", "variant"])
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Radius-sampling evaluation for TraCA."
    )
    p.add_argument(
        "--results", required=True,
        help="Path to traca_cv_results.pkl"
    )
    p.add_argument(
        "--rho_test", nargs="+", type=float, required=True,
        help="Test radii to evaluate"
    )
    p.add_argument(
        "--K", type=int, default=50,
        help="Number of random targets per rho_test (default: 50)"
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Output directory (default: same as results pkl)"
    )
    p.add_argument(
        "--seed", type=int, default=2026,
        help="Base seed for target sampling"
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    radius_sampling_eval(
        results_pkl=Path(args.results),
        rho_test_values=args.rho_test,
        K=args.K,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        base_seed=args.seed,
    )
