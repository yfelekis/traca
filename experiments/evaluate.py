"""
Standardized evaluation: target_loss, certificate, gap, per-intervention
breakdown, and query-level intervals.

Usage
-----
    python -m experiments.evaluate results/atce_gaussian_z_entrywise_full [--target_config data_configs/atce_target.yaml]

Reads
-----
    <exp_dir>/result.pkl    — OptResult
    <exp_dir>/bundle.pkl    — source SCMBundle (used for certificates)
    <exp_dir>/config.yaml   — experiment config (for ambiguity + eval params)

Writes
------
    <exp_dir>/eval.json     — EvalResult serialized to JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lan_scm  # noqa: E402
from lan_scm import _scm_from_config  # noqa: E402

from traca.stability import gamma, alpha_polynomial, gating_matrix
from traca.certificates import (
    delta_iota_U, delta_iota_rho, full_joint_certificate,
    single_query_certificate, query_interval,
)
from traca.losses import EmpiricalLoss, GaussianLoss
from traca.stability import perturbed_propagator
from traca.ambiguity import GelbrichBall
from traca.utils import gelbrich_distance, interventional_exo_mean, bundle_exo_means
from traca.directional_certificates import (
    directional_certificate_gaussian, directional_certificate_empirical,
)
from experiments.run import (
    _build_mechanism_set, _build_environment_set, _build_constructive_class,
)


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Standardized evaluation result.

    All fields are plain Python scalars, dicts, lists, and tuples — no NumPy
    arrays — so EvalResult serializes cleanly to JSON via eval.json.

    Fields
    ------
    target_loss : empirical model-level loss on held-out target bundle
    certificate : δ²(τ) full-joint certificate (bound on target_loss)
    gap : certificate - target_loss  (≥ 0 guarantees certificate validity)
    per_intervention : {iota_idx: {"target_loss": float, "certificate": float, "gap": float}}
    queries : {str(iota_idx,O): {"lower": float, "upper": float,
                                 "Phi_pushed": float, "L_Phi": float}}
    adversary_sup_estimate : empirical sup of loss over random adversaries;
        must satisfy adversary_sup_estimate ≤ certificate
    certificate_valid : adversary_sup_estimate ≤ certificate (or None if not computed)
    """
    target_loss: float
    certificate: float
    gap: float
    per_intervention: dict
    queries: dict
    adversary_sup_estimate: float | None = None
    certificate_valid: bool | None = None
    directional_queries: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iv_to_nodes(iv: dict, var_names: list[str]) -> list[int]:
    if not iv:
        return []
    return [var_names.index(k) if isinstance(k, str) else int(k) for k in iv.keys()]


def _build_alpha_iotas(
    bundle,
    mechanism_set,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (A_iotas, alpha_iotas) for all interventions in bundle."""
    d = bundle.d
    var_names = bundle.scm.var_names
    A_iotas, alpha_iotas = [], []
    for i, iv in enumerate(bundle.interventions):
        A_i = bundle.intervened_scms[i].A
        R_i = gating_matrix(d, _iv_to_nodes(iv, var_names))
        g = gamma(A_i, R_i, mechanism_set)
        alpha_iotas.append(alpha_polynomial(A_i, g, d))
        A_iotas.append(A_i)
    return A_iotas, alpha_iotas


def _extract_noise_mean_deltas(shift_spec, d: int) -> np.ndarray:
    """Extract noise_mean deltas from shift_spec into a (d,) vector.

    Returns a zero vector if shift_spec is None or contains no noise_mean
    sub-shifts.  For compound shift_spec (list of dicts), accumulates all
    noise_mean entries.
    """
    deltas = np.zeros(d)
    if shift_spec is None:
        return deltas
    specs = [shift_spec] if isinstance(shift_spec, dict) else list(shift_spec)
    for spec in specs:
        if spec.get("shift_type") == "noise_mean":
            node = int(spec["node"])
            deltas[node] += float(spec["magnitude"])
    return deltas


def _compute_target_loss_per_iota(
    tau: np.ndarray,
    source_bundle,
    target_bundle,
    objective: str,
    shift_spec=None,
    U_s_list: list | None = None,
    source_noise_mean: np.ndarray | None = None,
    source_noise_cov: np.ndarray | None = None,
) -> list[float]:
    """Compute per-intervention transport loss: τ_# source vs actual target.

    Empirical mode (ΔW_actual ≠ 0):
        (1/N) ‖U_s A_s^ι τ − (U_s + Θ) A_t^ι‖_F²
        Evaluates EmpiricalLoss at (τ, ΔW_actual, Θ) with source samples.
        Θ encodes any noise_mean sub-shifts from shift_spec; zero if no noise
        component (mechanism-only shift) or shift_spec not provided.

    When ``U_s_list`` is provided, uses those arrays as U_s instead of
    ``source_bundle.noise_samples[i]``.  Callers that need paired variance-
    cancellation (radius eval, decoupled eval) should pass obs-derived
    U_effs built via ``build_U_effs`` so that training/scoring/certificate
    all share the same observational U.

    Empirical mode fallback (ΔW_actual ≈ 0, pure noise shift):
        Falls back to Gaussian W₂² using bundle noise moments. When ΔW = 0 the
        empirical formula degenerates: (1/N)‖U_s A_s(τ−I)‖²=0 for τ=I regardless
        of the noise distribution shift, so it cannot capture environmental shifts
        (different root-node distribution, e.g. Portland). The Gaussian surrogate
        measures the actual distributional gap via source/target noise moments.

        When ``source_noise_mean`` / ``source_noise_cov`` are provided, the
        fallback uses those instead of ``source_bundle.noise_mean`` /
        ``source_bundle.noise_cov``.  Callers evaluating empirical objectives
        on holdout folds should pass holdout-estimated moments so that at
        zero shift the source and target moments are identical by
        construction (both sample-estimated from the same U rows).

    Gaussian mode:
        W₂²(τ_#P_s^ι, P_t^ι) evaluated with source params + actual target moments.

    This is NOT the same as the training objective (which maximizes over adversary).
    It measures the actual transport error to the held-out target.
    """
    losses = []
    W_s = source_bundle.W
    W_t = target_bundle.W
    dW_actual = W_t - W_s
    d = source_bundle.d
    var_names = source_bundle.scm.var_names
    pure_noise_shift = float(np.linalg.norm(dW_actual, "fro")) < 1e-10

    if objective == "empirical" and not pure_noise_shift:
        loss_fn = EmpiricalLoss()
        # Build Theta from noise_mean sub-shifts in shift_spec (if any)
        noise_deltas = _extract_noise_mean_deltas(shift_spec, d)
        for i, iv in enumerate(source_bundle.interventions):
            A_s = source_bundle.intervened_scms[i].A
            R_i = gating_matrix(d, _iv_to_nodes(iv, var_names))
            U_s = U_s_list[i] if U_s_list is not None else source_bundle.noise_samples[i]
            N = U_s.shape[0]
            Theta = np.tile(noise_deltas, (N, 1))  # (N, d), broadcast delta per row
            losses.append(float(loss_fn.value(tau, dW_actual, Theta, W_s, A_s, R_i, U_s)))
    else:
        # Gaussian W₂² approximation: works for both Gaussian training objective and
        # empirical-with-pure-noise-shift (dW_actual=0) where sample formula degenerates.
        loss_fn = GaussianLoss()
        mu_s = source_noise_mean if source_noise_mean is not None else source_bundle.noise_mean
        Sigma_s = source_noise_cov if source_noise_cov is not None else source_bundle.noise_cov
        mu_t = target_bundle.noise_mean
        Sigma_t = target_bundle.noise_cov
        dW_zero = np.zeros((d, d))
        for i, iv in enumerate(source_bundle.interventions):
            A_s = source_bundle.intervened_scms[i].A
            R_i = gating_matrix(d, _iv_to_nodes(iv, var_names))
            # Target observed moments: propagate target noise through target mechanism.
            # Include intervention fixed values: E[X^(ι)] = (μ_U + fixed) @ A_ι.
            A_t = target_bundle.intervened_scms[i].A
            scm_s_i = source_bundle.intervened_scms[i]
            mu_s_eff = interventional_exo_mean(mu_s, scm_s_i._fixed, scm_s_i._J)
            scm_t_i = target_bundle.intervened_scms[i]
            mu_t_obs = interventional_exo_mean(mu_t, scm_t_i._fixed, scm_t_i._J) @ A_t
            Sigma_t_obs = A_t.T @ Sigma_t @ A_t
            losses.append(float(loss_fn.value(tau, dW_zero, W_s, A_s, R_i,
                                               mu_s_eff, Sigma_s, mu_t_obs, Sigma_t_obs)))
    return losses


def _sample_gelbrich_random(
    mu_s: np.ndarray,
    Sigma_s: np.ndarray,
    env_set: GelbrichBall,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random point in the Gelbrich ball (may be in interior)."""
    d = len(mu_s)
    eps = env_set.eps
    shifted_rows = env_set.shifted_rows

    # Random mean perturbation
    mu_t = mu_s.copy()
    for j in shifted_rows:
        mu_t[j] = mu_s[j] + rng.standard_normal() * eps

    # Random PSD perturbation to covariance
    H = rng.standard_normal((d, d))
    H = H + H.T
    Sigma_t = Sigma_s + eps * 0.5 * H
    eigvals, eigvecs = np.linalg.eigh(Sigma_t)
    Sigma_t = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T

    # Project onto ball
    return env_set.project(mu_t, Sigma_t)


def _empirical_adversary_sup(
    tau: np.ndarray,
    source_bundle,
    mechanism_set,
    environment_set,
    objective: str,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Estimate sup_{ΔW, ξ} F by sampling random adversaries and taking the max.

    For empirical objective: samples ΔW and Θ independently.
    For gaussian objective: samples ΔW and (μ_t^exo, Σ_t^exo) independently from
    the Gelbrich ball.  The previous implementation only sampled ΔW and derived
    (μ_t, Σ_t) from it, which ignored independent noise shifts and systematically
    understated the adversary sup.
    """
    d = source_bundle.d
    W = source_bundle.W
    var_names = source_bundle.scm.var_names
    n_iota = source_bundle.n_interventions()
    mu_s_effs = bundle_exo_means(source_bundle)  # per-intervention effective means
    Sigma_s = source_bundle.noise_cov

    if objective == "empirical":
        loss_fn = EmpiricalLoss()
    else:
        loss_fn = GaussianLoss()
        is_gelbrich = isinstance(environment_set, GelbrichBall)

    best = 0.0
    for _ in range(n_samples):
        total = 0.0
        dW = mechanism_set.project(rng.standard_normal((d, d)))

        # Sample noise adversary once per sample (shared across interventions
        # as a joint shift, consistent with the certificate)
        if objective == "gaussian" and is_gelbrich:
            mu_t_exo, Sigma_t_exo = _sample_gelbrich_random(
                source_bundle.noise_mean, Sigma_s, environment_set, rng
            )

        for i, iv in enumerate(source_bundle.interventions):
            A_i = source_bundle.intervened_scms[i].A
            R_i = gating_matrix(d, _iv_to_nodes(iv, var_names))
            mu_s_i = mu_s_effs[i]
            if objective == "empirical":
                U_s = source_bundle.noise_samples[i]
                N = U_s.shape[0]
                Theta = environment_set.project(rng.standard_normal((N, d)))
                total += float(loss_fn.value(tau, dW, Theta, W, A_i, R_i, U_s))
            else:
                if is_gelbrich:
                    # Convert exogenous adversary to observed space
                    A_prime = perturbed_propagator(W, dW, R_i)
                    mu_t_obs = mu_t_exo @ A_prime
                    Sigma_t_obs = A_prime.T @ Sigma_t_exo @ A_prime
                    total += float(loss_fn.value(
                        tau, dW, W, A_i, R_i, mu_s_i, Sigma_s, mu_t_obs, Sigma_t_obs
                    ))
                else:
                    # Fallback: derive target from ΔW only
                    total += float(loss_fn.value(tau, dW, W, A_i, R_i, mu_s_i, Sigma_s))
        best = max(best, total / n_iota)
    return best


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    exp_dir: Path,
    target_config: Path | None = None,
) -> EvalResult:
    """Evaluate an experiment result.

    Parameters
    ----------
    exp_dir : directory containing result.pkl, bundle.pkl, config.yaml
    target_config : optional override for target data_configs YAML

    Returns
    -------
    EvalResult (also written to exp_dir/eval.json)
    """
    # Load artifacts
    result = joblib.load(exp_dir / "result.pkl")
    bundle = joblib.load(exp_dir / "bundle.pkl")
    with (exp_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)

    tau = result.tau
    d = bundle.d
    N = bundle.n
    objective = cfg.get("objective", "empirical")

    # Build ambiguity sets (for certificate + adversary sup)
    mechanism_set = _build_mechanism_set(cfg["ambiguity"]["mechanism"], d)
    environment_set = _build_environment_set(
        cfg["ambiguity"]["environment"],
        bundle.noise_mean, bundle.noise_cov, N
    )

    # Build per-intervention stability moduli
    A_iotas, alpha_iotas = _build_alpha_iotas(bundle, mechanism_set)
    mu_s_effs = bundle_exo_means(bundle)  # per-intervention effective exogenous means

    # -----------------------------------------------------------------------
    # Certificate
    # -----------------------------------------------------------------------
    eps = float(cfg["ambiguity"]["environment"].get("eps", 0.1))
    # noise_samples may be a dict {iota_idx: array} or a list — normalise to list
    ns = bundle.noise_samples
    U_s_list = [ns[i] for i in range(bundle.n_interventions())] if isinstance(ns, dict) else list(ns)

    if objective == "empirical":
        certificate = full_joint_certificate(
            tau, alpha_iotas, A_iotas,
            mu_s_effs, bundle.noise_cov,
            eps=eps, mode="empirical",
            U_s_list=U_s_list, N=N,
        )
    else:
        certificate = full_joint_certificate(
            tau, alpha_iotas, A_iotas,
            mu_s_effs, bundle.noise_cov,
            eps=eps, mode="gaussian",
        )

    # Per-intervention certificates
    per_iota_cert: list[float] = []
    for i in range(bundle.n_interventions()):
        if objective == "empirical":
            # delta_iota_U returns unsquared δ^U; certificate is (δ^U)²
            c_i = float(delta_iota_U(tau, alpha_iotas[i], A_iotas[i], U_s_list[i], eps=eps, N=N)) ** 2
        else:
            # delta_iota_rho_sq already returns the squared value δ^ρ²
            c_i = float(delta_iota_rho(tau, alpha_iotas[i], A_iotas[i],
                                        mu_s_effs[i], bundle.noise_cov, eps=eps))
        per_iota_cert.append(c_i)

    # -----------------------------------------------------------------------
    # Target loss (requires target bundle)
    # -----------------------------------------------------------------------
    data_cfg = cfg.get("data", {})
    target_bundle_path_str = data_cfg.get("target_bundle_path", None)
    target_yaml_str = data_cfg.get("target", None) if target_config is None else str(target_config)

    if target_bundle_path_str:
        target_bundle = joblib.load(Path(target_bundle_path_str))
        per_iota_losses = _compute_target_loss_per_iota(tau, bundle, target_bundle, objective)
    elif target_yaml_str and Path(target_yaml_str).exists():
        with Path(target_yaml_str).open() as f:
            target_scm_cfg = yaml.safe_load(f)
        target_scm = _scm_from_config(target_scm_cfg)
        target_interventions = list(target_scm_cfg["interventions"])
        target_n = int(cfg.get("training", {}).get("n_samples", target_scm_cfg.get("n_samples", N)))
        target_seed = int(cfg.get("training", {}).get("seed", target_scm_cfg.get("seed", 0))) + 9999
        target_bundle = target_scm.bundle(target_interventions, n=target_n, seed=target_seed)
        per_iota_losses = _compute_target_loss_per_iota(tau, bundle, target_bundle, objective)
    else:
        print("[evaluate] warning: no target bundle available, using source bundle for target_loss")
        per_iota_losses = _compute_target_loss_per_iota(tau, bundle, bundle, objective)
    target_loss = float(np.mean(per_iota_losses))

    gap = float(certificate - target_loss)

    # Per-intervention dict
    per_intervention = {
        i: {
            "target_loss": float(per_iota_losses[i]),
            "certificate": float(per_iota_cert[i]),
            "gap": float(per_iota_cert[i] - per_iota_losses[i]),
        }
        for i in range(bundle.n_interventions())
    }

    # -----------------------------------------------------------------------
    # Query intervals (for each (iota, output_node) pair)
    #
    # Φ = E[V_k] (mean functional).
    # L_Phi = 1 for both modes:
    #   Gaussian: |E_P[V_k] - E_Q[V_k]| ≤ W_2(P, Q), so L_Φ = 1 w.r.t. W₂.
    #   Empirical: |mean(X[:,k]) - mean(Y[:,k])| ≤ ||X[:,k]-Y[:,k]||_F/√N
    #              = 1 · √(cert_q), since cert_q bounds ||res_O||_F²/N.
    # -----------------------------------------------------------------------
    queries: dict[str, dict] = {}
    for i, iv in enumerate(bundle.interventions):
        for out_node in range(d):
            mu_s_i = mu_s_effs[i]
            if objective == "empirical":
                cert_q = float(single_query_certificate(
                    tau, alpha_iotas[i], A_iotas[i],
                    mu_s_i, bundle.noise_cov,
                    eps=eps, O=[out_node], d=d, mode="empirical",
                    U_s=U_s_list[i], N=N,
                ))
                pushed_X = U_s_list[i] @ A_iotas[i] @ tau
                Phi_pushed = float(np.mean(pushed_X[:, out_node])) + float((mu_s_i @ A_iotas[i] @ tau)[out_node])
            else:
                cert_q = float(single_query_certificate(
                    tau, alpha_iotas[i], A_iotas[i],
                    mu_s_i, bundle.noise_cov,
                    eps=eps, O=[out_node], d=d, mode="gaussian",
                ))
                Phi_pushed = float(
                    (mu_s_i @ A_iotas[i] @ tau)[out_node]
                )
            L_Phi = 1.0
            lo, hi = query_interval(Phi_pushed, L_Phi, cert_q)
            key = f"({i},{out_node})"
            queries[key] = {
                "lower": float(lo),
                "upper": float(hi),
                "Phi_pushed": float(Phi_pushed),
                "L_Phi": float(L_Phi),
                "delta_sq": float(cert_q),
            }

    # -----------------------------------------------------------------------
    # Directional certificates (per-query tighter bounds)
    # -----------------------------------------------------------------------
    var_names = bundle.scm.var_names
    directional_queries: dict[str, dict] = {}
    for i, iv in enumerate(bundle.interventions):
        R_i = gating_matrix(d, _iv_to_nodes(iv, var_names))
        g_i = gamma(A_iotas[i], R_i, mechanism_set)
        for out_node in range(d):
            q = np.zeros(d); q[out_node] = 1.0
            try:
                if objective == "empirical":
                    dc = directional_certificate_empirical(
                        tau, q, [out_node], i, bundle,
                        mechanism_set, environment_set, alpha_iotas[i], g_i,
                    )
                else:
                    dc = directional_certificate_gaussian(
                        tau, q, [out_node], i, bundle,
                        mechanism_set, environment_set, alpha_iotas[i], g_i,
                    )
                Phi_pushed = queries[f"({i},{out_node})"]["Phi_pushed"]
                dir_cert = float(dc["certificate"])
                lo = float(Phi_pushed - dir_cert)
                hi = float(Phi_pushed + dir_cert)
                directional_queries[f"({i},{out_node})"] = {
                    "certificate": dir_cert,
                    "lower": lo,
                    "upper": hi,
                    "Phi_pushed": float(Phi_pushed),
                    "env_method": dc.get("env_method", "unknown"),
                    "transport": float(dc.get("transport", 0.0)),
                    "mechanism": float(dc.get("mechanism", 0.0)),
                    "environment": float(dc.get("environment", 0.0)),
                }
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"[evaluate] directional cert failed for ι={i}, node={out_node}: {exc}",
                    RuntimeWarning, stacklevel=2,
                )
                directional_queries[f"({i},{out_node})"] = {"error": str(exc)}

    # -----------------------------------------------------------------------
    # Adversary sup estimate
    # -----------------------------------------------------------------------
    eval_cfg = cfg.get("evaluation", {})
    n_adv_samples = int(eval_cfg.get("adversary_samples", 200))
    rng = np.random.default_rng(12345)
    adv_sup = _empirical_adversary_sup(
        tau, bundle, mechanism_set, environment_set,
        objective, n_adv_samples, rng,
    )
    cert_valid = bool(adv_sup <= certificate + 1e-9)

    eval_result = EvalResult(
        target_loss=target_loss,
        certificate=float(certificate),
        gap=gap,
        per_intervention=per_intervention,
        queries=queries,
        adversary_sup_estimate=float(adv_sup),
        certificate_valid=cert_valid,
        directional_queries=directional_queries,
    )

    # Serialize to JSON
    eval_dict = {
        "target_loss": eval_result.target_loss,
        "certificate": eval_result.certificate,
        "gap": eval_result.gap,
        "per_intervention": eval_result.per_intervention,
        "queries": eval_result.queries,
        "adversary_sup_estimate": eval_result.adversary_sup_estimate,
        "certificate_valid": eval_result.certificate_valid,
        "directional_queries": eval_result.directional_queries,
    }
    eval_path = exp_dir / "eval.json"
    with eval_path.open("w") as f:
        json.dump(eval_dict, f, indent=2)
    print(f"[evaluate] saved eval to {eval_path}")
    print(f"[evaluate] target_loss={target_loss:.4f}, certificate={certificate:.4f}, gap={gap:.4f}")
    print(f"[evaluate] adversary_sup={adv_sup:.4f}, certificate_valid={cert_valid}")

    return eval_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a TraCA experiment result.")
    p.add_argument("exp_dir", help="Path to experiment output directory.")
    p.add_argument(
        "--target_config", default=None,
        help="Override target data_configs YAML path."
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    evaluate(
        Path(args.exp_dir),
        target_config=Path(args.target_config) if args.target_config else None,
    )
