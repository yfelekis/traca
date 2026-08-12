"""
Certificate computation: δ_ι^ρ², δ_ι^U, full-joint, single-query, query interval.

Row-vector convention: the code uses X = U @ A (row vectors) with covariance
Σ_X = A^T Σ_U A.  Certificate values are transpose-invariant (Frobenius and
spectral norms), so row-vector and column-vector forms are numerically identical.

Squaring convention:
* δ^ρ_{ι,O}(T)² (Gaussian) is returned ALREADY SQUARED by `delta_iota_rho_sq`
  (the superscript ² is part of the notation).
* δ^U_{ι,O}(T) (empirical) is returned UNSQUARED by `delta_iota_U`.
  `full_joint_certificate` squares internally; callers that use
  `delta_iota_U` directly must square the result.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_SO(d: int, O: list[int] | None) -> np.ndarray | None:
    """Return the (|O|, d) selection matrix for output set O, or None."""
    if O is None:
        return None
    return selection_matrix(d, O)


# ---------------------------------------------------------------------------
# Gaussian per-intervention certificate  δ^ρ_{ι,O}(T)²
# ---------------------------------------------------------------------------

def delta_iota_rho_sq(
    tau: np.ndarray,
    alpha_iota: float,
    A_iota: np.ndarray,
    mu_s: np.ndarray,
    Sigma_s: np.ndarray,
    eps: float,
    S_O: np.ndarray | None = None,
) -> float:
    """Gaussian per-intervention certificate δ^ρ_{ι,O}(T)² (ALREADY SQUARED).

    δ^ρ_{ι,O}(T)² = 4 ‖S_O(T−I)A_ι μ_s‖²₂
                  + 4 ‖Σ_s‖₂  ‖S_O(T−I)A_ι‖²_F
                  + 4 α_ι²   ‖μ_s‖²₂
                  + 4 |O|    ‖Σ_s‖₂  α_ι²
                  + 2 (‖S_O A_ι‖₂ + α_ι)² ε²

    Row-vector translation (code convention):
        ‖S_O(T−I)A_ι μ_s‖₂  =  ‖μ_s @ A_iota @ (τ−I) @ S_O^T‖₂
        ‖S_O(T−I)A_ι‖_F     =  ‖A_iota @ (τ−I) @ S_O^T‖_F
        ‖S_O A_ι‖₂          =  ‖A_iota @ S_O^T‖₂

    Parameters
    ----------
    tau : (d, d) transport map
    alpha_iota : float — α_ι stability modulus
    A_iota : (d, d) interventional propagator (code convention: X = U @ A)
    mu_s : (d,) source noise mean
    Sigma_s : (d, d) source noise covariance
    eps : float — Gelbrich/W₂ environment radius ε
    S_O : (|O|, d) selection matrix or None (full post-interventional → |O|=d)

    Returns
    -------
    float : δ^ρ_{ι,O}(T)² — the certificate value (squared per paper notation)
    """
    tau = np.asarray(tau, dtype=float)
    A = np.asarray(A_iota, dtype=float)
    mu_s = np.asarray(mu_s, dtype=float)
    Sigma_s = np.asarray(Sigma_s, dtype=float)
    d = A.shape[0]
    I_d = np.eye(d)

    # A_iota @ (tau - I) — transport-mechanism product, (d, d)
    A_tau_I = A @ (tau - I_d)

    if S_O is not None:
        S = np.asarray(S_O, dtype=float)    # (|O|, d)
        n_O = S.shape[0]
        # ‖S_O(T−I)A_ι μ_s‖₂ = ‖μ_s @ A_tau_I @ S^T‖₂
        mu_transport = mu_s @ A_tau_I @ S.T  # (|O|,)
        # ‖S_O(T−I)A_ι‖_F = ‖A_tau_I @ S^T‖_F
        F_fro = float(np.linalg.norm(A_tau_I @ S.T, "fro"))
        # ‖S_O A_ι‖₂ = ‖A @ S^T‖₂
        A_S_norm2 = float(np.linalg.norm(A @ S.T, ord=2))
    else:
        n_O = d
        mu_transport = mu_s @ A_tau_I         # (d,)
        F_fro = float(np.linalg.norm(A_tau_I, "fro"))
        A_S_norm2 = float(np.linalg.norm(A, ord=2))

    Sigma_s_norm2 = float(np.linalg.norm(Sigma_s, ord=2))   # spectral norm ‖Σ_s‖₂
    mu_transport_sq = float(np.dot(mu_transport, mu_transport))
    mu_s_sq = float(np.dot(mu_s, mu_s))

    term1 = 4.0 * mu_transport_sq                          # transport × mean
    term2 = 4.0 * Sigma_s_norm2 * F_fro ** 2              # transport × covariance
    term3 = 4.0 * alpha_iota ** 2 * mu_s_sq               # mechanism × mean
    term4 = 4.0 * n_O * Sigma_s_norm2 * alpha_iota ** 2   # mechanism × covariance
    term5 = 2.0 * (A_S_norm2 + alpha_iota) ** 2 * eps ** 2  # environment

    return float(term1 + term2 + term3 + term4 + term5)


# Keep the old name as an alias so existing call-sites don't break immediately.
delta_iota_rho = delta_iota_rho_sq


# ---------------------------------------------------------------------------
# Empirical per-intervention certificate  δ^U_{ι,O}(T)
# ---------------------------------------------------------------------------

def delta_iota_U(
    tau: np.ndarray,
    alpha_iota: float,
    A_iota: np.ndarray,
    U_s: np.ndarray,
    eps: float,
    N: int,
    S_O: np.ndarray | None = None,
) -> float:
    """Empirical per-intervention certificate δ^U_{ι,O}(T)  **UNSQUARED, NORMALIZED**.

    Returns the square-root of the 1/N-normalized certificate, so that
    `delta_iota_U(...)**2` directly bounds `‖residual‖_F² / N`.

    The un-normalized certificate is:
        δ^U = ‖S_O(T−I)A_ι U_s‖_F + α_ι ‖U_s‖_F + ε√N (‖S_O A_ι‖₂ + α_ι)

    The code normalizes by √N:
        δ^U_code = (‖S_O(T−I)A_ι U_s‖_F + α_ι ‖U_s‖_F) / √N
                   + ε (‖S_O A_ι‖₂ + α_ι)

    Square the result when calling directly; `full_joint_certificate` and
    `single_query_certificate` square internally.

    Row-vector translation (code convention X = U @ A):
        (‖S_O(T−I)A_ι U_s‖_F) / √N  =  ‖U_s @ A_iota @ (τ−I) @ S_O^T‖_F / √N
        α_ι ‖U_s‖_F / √N              =  α_ι * rms_norm(U_s)
        ε (‖S_O A_ι‖₂ + α_ι)          =  ε * (‖A_iota @ S_O^T‖₂ + α_ι)

    Parameters
    ----------
    tau : (d, d)
    alpha_iota : float
    A_iota : (d, d)
    U_s : (N, d) source noise samples
    eps : float — per-sample environment radius (total Frobenius radius = ε√N)
    N : int — number of samples
    S_O : (|O|, d) selection matrix or None

    Returns
    -------
    float : δ^U_{ι,O}(T) normalized — bounds code's 1/N-normalized loss when squared
    """
    tau = np.asarray(tau, dtype=float)
    A = np.asarray(A_iota, dtype=float)
    U_s = np.asarray(U_s, dtype=float)
    d = A.shape[0]
    I_d = np.eye(d)
    sqrtN = float(np.sqrt(N))

    A_tau_I = A @ (tau - I_d)  # (d, d)

    if S_O is not None:
        S = np.asarray(S_O, dtype=float)    # (|O|, d)
        # transport/√N: ‖U_s @ A_tau_I @ S^T‖_F / √N
        transport = float(np.linalg.norm(U_s @ A_tau_I @ S.T, "fro")) / sqrtN
        A_S_norm2 = float(np.linalg.norm(A @ S.T, ord=2))
    else:
        transport = float(np.linalg.norm(U_s @ A_tau_I, "fro")) / sqrtN
        A_S_norm2 = float(np.linalg.norm(A, ord=2))

    # mech/√N: α_ι ‖U_s‖_F / √N  (= α_ι * RMS norm of U_s)
    mech = alpha_iota * float(np.linalg.norm(U_s, "fro")) / sqrtN
    # env: ε√N * (‖A‖₂ + α_ι) / √N = ε * (‖A‖₂ + α_ι)
    env = eps * (A_S_norm2 + alpha_iota)

    return float(transport + mech + env)


# ---------------------------------------------------------------------------
# Full-joint averages
# ---------------------------------------------------------------------------

def E_rho_joint(
    tau: np.ndarray,
    deltas_sq: list[float],
) -> float:
    """Full-joint Gaussian certificate: average δ^ρ_{ι,O}² over queries.

    Parameters
    ----------
    tau : (d, d) — unused, kept for uniform signature
    deltas_sq : list of δ^ρ_{ι,O}² values (one per query)

    Returns
    -------
    float
    """
    return float(np.mean(deltas_sq))


def E_U_joint(
    tau: np.ndarray,
    deltas: list[float],
) -> float:
    """Full-joint empirical certificate: average (δ^U_{ι,O})² over queries.

    `deltas` must be the UNSQUARED values from `delta_iota_U`; the function
    squares each before averaging.

    Parameters
    ----------
    tau : (d, d) — unused, kept for uniform signature
    deltas : list of δ^U_{ι,O} values (unsquared, from delta_iota_U)

    Returns
    -------
    float
    """
    return float(np.mean([d ** 2 for d in deltas]))


# ---------------------------------------------------------------------------
# Full-joint certificate
# ---------------------------------------------------------------------------

def full_joint_certificate(
    tau: np.ndarray,
    alpha_iotas: list[float],
    A_iotas: list[np.ndarray],
    mu_s: np.ndarray | list[np.ndarray],
    Sigma_s: np.ndarray,
    eps: float,
    mode: str = "gaussian",
    U_s_list: list[np.ndarray] | None = None,
    N: int | None = None,
    query_family: list[tuple[int, list[int]]] | None = None,
) -> float:
    """Full-joint certificate: average δ² over all queries.

    For Gaussian mode: averages δ^ρ_{ι,O}² (already squared).
    For empirical mode: averages (δ^U_{ι,O})² (squares internally).

    Parameters
    ----------
    tau : (d, d)
    alpha_iotas : list of α_ι per intervention
    A_iotas : list of (d, d) propagators
    mu_s : (d,) single mean for all interventions, OR list of (d,) per-intervention
        effective exogenous means (from bundle_exo_means). Per-intervention means
        account for intervention fixed values and noise zeroing.
    Sigma_s : (d, d)
    eps : float
    mode : "gaussian" or "empirical"
    U_s_list : list of (N, d) arrays — required for empirical mode
    N : int — required for empirical mode
    query_family : list of (iota_idx, O) pairs, or None (full post-interventional).
        If provided, each pair specifies which intervention index and which output
        coordinates to certify. Used for Q-restricted certificates.

    Returns
    -------
    float : full-joint certificate (always a squared δ² quantity)
    """
    d = A_iotas[0].shape[0]

    # Accept list of per-intervention means or single array (backward-compat)
    if isinstance(mu_s, list):
        mu_s_list = mu_s
    else:
        mu_s_list = [np.asarray(mu_s, dtype=float)] * len(A_iotas)

    if query_family is not None:
        # Q-restricted: one (ι, O) pair per entry
        deltas_sq: list[float] = []
        for iota_idx, O in query_family:
            S_O = selection_matrix(d, list(O))
            alpha = alpha_iotas[iota_idx]
            A = A_iotas[iota_idx]
            if mode == "gaussian":
                deltas_sq.append(delta_iota_rho_sq(tau, alpha, A, mu_s_list[iota_idx], Sigma_s, eps, S_O))
            else:
                if U_s_list is None or N is None:
                    raise ValueError("U_s_list and N required for empirical mode.")
                deltas_sq.append(
                    delta_iota_U(tau, alpha, A, U_s_list[iota_idx], eps, N, S_O) ** 2
                )
        return float(np.mean(deltas_sq))

    # Full post-interventional (no selector)
    deltas_sq = []
    for i, (alpha, A) in enumerate(zip(alpha_iotas, A_iotas)):
        if mode == "gaussian":
            deltas_sq.append(delta_iota_rho_sq(tau, alpha, A, mu_s_list[i], Sigma_s, eps))
        else:
            if U_s_list is None or N is None:
                raise ValueError("U_s_list and N required for empirical mode.")
            deltas_sq.append(delta_iota_U(tau, alpha, A, U_s_list[i], eps, N) ** 2)
    return float(np.mean(deltas_sq))


# ---------------------------------------------------------------------------
# Single-query certificate
# ---------------------------------------------------------------------------

def single_query_certificate(
    tau: np.ndarray,
    alpha_iota: float,
    A_iota: np.ndarray,
    mu_s: np.ndarray,
    Sigma_s: np.ndarray,
    eps: float,
    O: list[int],
    d: int,
    mode: str = "gaussian",
    U_s: np.ndarray | None = None,
    N: int | None = None,
) -> float:
    """Single-query certificate: δ² for one (ι, O) query.

    Returns the SQUARED certificate value (δ^ρ² or (δ^U)²) for direct use
    as the `delta_sq` argument of `query_interval`.

    Parameters
    ----------
    O : list of int — output coordinate indices
    d : int — ambient dimension

    Returns
    -------
    float : squared certificate δ²
    """
    S_O = selection_matrix(d, O)
    if mode == "gaussian":
        return delta_iota_rho_sq(tau, alpha_iota, A_iota, mu_s, Sigma_s, eps, S_O)
    else:
        if U_s is None or N is None:
            raise ValueError("U_s and N required for empirical mode.")
        return delta_iota_U(tau, alpha_iota, A_iota, U_s, eps, N, S_O) ** 2


# ---------------------------------------------------------------------------
# Query interval
# ---------------------------------------------------------------------------

def query_interval(
    Phi_pushed: float,
    L_Phi: float,
    delta_sq: float,
) -> tuple[float, float]:
    """Lipschitz-transfer query interval.

    If Φ is L_Φ-Lipschitz w.r.t. W₂ / Frobenius distance, then:
        Φ(P_t^ι) ∈ [Φ_pushed − L_Φ √δ², Φ_pushed + L_Φ √δ²]

    where Φ_pushed = Φ(τ_# P_s^ι) and δ² is the certificate (already squared).

    Parameters
    ----------
    Phi_pushed : float — Φ evaluated on pushed source distribution τ_# P_s^ι
    L_Phi : float — Lipschitz constant of Φ
    delta_sq : float — squared certificate δ²(τ) from single_query_certificate

    Returns
    -------
    (lower, upper) : certified interval for Φ(P_t^ι)
    """
    radius = L_Phi * float(delta_sq) ** 0.5
    lower = float(Phi_pushed) - radius
    upper = float(Phi_pushed) + radius
    return lower, upper


# ---------------------------------------------------------------------------
# Selection matrix helper
# ---------------------------------------------------------------------------

def selection_matrix(d: int, O: list[int]) -> np.ndarray:
    """Build selection matrix S_O of shape (|O|, d).

    S_O[k, O[k]] = 1, all other entries 0.

    Parameters
    ----------
    d : int — ambient dimension
    O : list of int — output coordinate indices

    Returns
    -------
    (|O|, d) array
    """
    S = np.zeros((len(O), d), dtype=float)
    for k, o in enumerate(O):
        S[k, o] = 1.0
    return S
