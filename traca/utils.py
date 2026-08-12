"""
Small numerical helpers shared across modules.

Row-vector convention: X = U @ A.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm as _scipy_sqrtm


# ---------------------------------------------------------------------------
# Matrix square-root (symmetric positive semi-definite)
# ---------------------------------------------------------------------------

def bures_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric matrix square-root of a PSD matrix via SVD.

    Parameters
    ----------
    M : (n, n) symmetric PSD array

    Returns
    -------
    M_sqrt : (n, n) array satisfying M_sqrt @ M_sqrt ≈ M, M_sqrt symmetric PSD
    """
    M = np.asarray(M, dtype=float)
    # Symmetrize to guard against floating-point asymmetry
    M = 0.5 * (M + M.T)
    # SVD-based sqrt: M = U S U^T  =>  M^{1/2} = U S^{1/2} U^T
    U, s, _ = np.linalg.svd(M, hermitian=True)
    s_sqrt = np.sqrt(np.maximum(s, 0.0))
    return (U * s_sqrt) @ U.T


# ---------------------------------------------------------------------------
# Gelbrich (Bures–Wasserstein) distance
# ---------------------------------------------------------------------------

def gelbrich_distance(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """Gelbrich (Bures–Wasserstein) distance between two Gaussians.

    W_2^2(N(mu1, Sigma1), N(mu2, Sigma2))
      = ||mu1 - mu2||^2
        + Tr(Sigma1 + Sigma2 - 2 * (Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})

    Parameters
    ----------
    mu1, mu2 : (d,) arrays
    Sigma1, Sigma2 : (d, d) symmetric PSD arrays

    Returns
    -------
    float : W_2^2 value (squared distance)
    """
    mu1, mu2 = np.asarray(mu1, dtype=float), np.asarray(mu2, dtype=float)
    S1, S2 = np.asarray(Sigma1, dtype=float), np.asarray(Sigma2, dtype=float)
    mean_term = float(np.dot(mu1 - mu2, mu1 - mu2))
    S1_sqrt = bures_sqrt(S1)
    M = S1_sqrt @ S2 @ S1_sqrt
    cross = bures_sqrt(M)
    cov_term = float(np.trace(S1) + np.trace(S2) - 2.0 * np.trace(cross))
    return mean_term + max(cov_term, 0.0)  # clip numerical negatives


# ---------------------------------------------------------------------------
# L1 (Euclidean simplex) projection — per row / column
# ---------------------------------------------------------------------------

def project_l1(v: np.ndarray, radius: float) -> np.ndarray:
    """Project vector v onto the L1 ball of given radius.

    Algorithm: Duchi et al. (2008) linear-time L1 projection.

    Parameters
    ----------
    v : (n,) array
    radius : float >= 0

    Returns
    -------
    (n,) array with ||result||_1 <= radius
    """
    v = np.asarray(v, dtype=float)
    if np.sum(np.abs(v)) <= radius:
        return v.copy()
    u = np.sort(np.abs(v))[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - radius) / (np.arange(len(u)) + 1))[0][-1]
    theta = (cssv[rho] - radius) / (rho + 1)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0.0)


# ---------------------------------------------------------------------------
# Interventional exogenous mean
# ---------------------------------------------------------------------------

def interventional_exo_mean(
    noise_mean: np.ndarray,
    fixed: np.ndarray,
    intervened_nodes: list | tuple,
) -> np.ndarray:
    """Effective exogenous mean under a hard intervention.

    Replicates gaussian_joint()'s mean logic (lan_scm.py:314-320):
    1. Copy noise_mean
    2. Zero entries at intervened nodes (_J)
    3. Add _fixed vector

    Parameters
    ----------
    noise_mean : (d,) source exogenous noise mean
    fixed : (d,) intervention fixed-value vector (nonzero at intervened nodes)
    intervened_nodes : list of node indices that are intervened on

    Returns
    -------
    (d,) effective exogenous mean μ_eff
    """
    mu = np.asarray(noise_mean, dtype=float).copy()
    if intervened_nodes:
        mu[list(intervened_nodes)] = 0.0
    return mu + np.asarray(fixed, dtype=float)


def bundle_exo_means(bundle) -> list[np.ndarray]:
    """Return list of per-intervention effective exogenous means.

    mu_s_effs[i] = interventional_exo_mean(bundle.noise_mean,
                                            bundle.intervened_scms[i]._fixed,
                                            bundle.intervened_scms[i]._J)
    """
    return [
        interventional_exo_mean(
            bundle.noise_mean,
            bundle.intervened_scms[i]._fixed,
            bundle.intervened_scms[i]._J,
        )
        for i in range(bundle.n_interventions())
    ]


def build_U_effs(bundle, indices: np.ndarray | None = None) -> list[np.ndarray]:
    """Build per-intervention effective noise samples from shared observational U.

    Replicates the optimizer's U_eff construction (optim.py:645-652):
        U_eff = U_obs.copy();  U_eff[:, J] = 0;  U_eff += fixed

    This ensures that source, certificate, Phi_pushed, and target all share
    the same observational U, preserving paired variance-cancellation.

    Parameters
    ----------
    bundle : SCMBundle
    indices : optional index array to subset rows (held-out fold)

    Returns
    -------
    list of (N, d) arrays, one per intervention
    """
    ns = bundle.noise_samples
    obs_idx = bundle.interventions.index({})
    U_obs = ns[obs_idx] if isinstance(ns, dict) else ns[obs_idx]
    if indices is not None:
        U_obs = U_obs[indices]

    U_effs = []
    for i in range(bundle.n_interventions()):
        scm_i = bundle.intervened_scms[i]
        U_eff = U_obs.copy()
        if scm_i._J:
            U_eff[:, list(scm_i._J)] = 0.0
        U_eff += scm_i._fixed[np.newaxis, :]
        U_effs.append(U_eff)
    return U_effs


# ---------------------------------------------------------------------------
# Random generator helper
# ---------------------------------------------------------------------------

def as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Resolve seed/Generator to a Generator instance."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
