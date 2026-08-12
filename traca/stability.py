"""
Amplification factor γ_ι and propagator stability modulus α_ι.

Bounds how much a mechanism perturbation ΔW propagates through the DAG.

IMPORTANT: Row-wise bounds use ||·||_∞; column-wise use ||·||_1.
    ||M||_2 <= sqrt(d) * ||M||_∞   (row-wise case)
    ||M||_2 <= sqrt(d) * ||M||_1   (column-wise case)
Code follows the plug-in proposition statement (not the proof, which
has swapped row/column inequalities).
"""
from __future__ import annotations

import numpy as np

from traca.ambiguity import (
    FrobeniusBall, RowBudget, ColumnBudget, EntrywiseBox,
    MechanismAmbiguitySet,
)


# ---------------------------------------------------------------------------
# Gating matrix
# ---------------------------------------------------------------------------

def gating_matrix(d: int, intervened_nodes: Sequence[int]) -> np.ndarray:
    """Diagonal gating matrix R_ι: zeros on intervened nodes, ones elsewhere.

    R_ι[j,j] = 0 if j ∈ J_ι, else 1.

    Parameters
    ----------
    d : int
    intervened_nodes : list of int
        J_ι — the set of nodes directly intervened on.

    Returns
    -------
    (d, d) diagonal array
    """
    R = np.eye(d, dtype=float)
    for j in intervened_nodes:
        R[j, j] = 0.0
    return R


# ---------------------------------------------------------------------------
# Amplification factor γ_ι
# ---------------------------------------------------------------------------

def gamma(
    A_iota: np.ndarray,
    R_iota: np.ndarray,
    ambiguity_set: MechanismAmbiguitySet,
) -> float:
    """Closed-form amplification factor γ_ι = sup_{ΔW ∈ A_W} ||A_ι (ΔW R_ι)||_2.

    Right-mult convention: the resolvent identity is
        A'_ι − A_ι = A_ι (ΔW R_ι) A'_ι
    so gamma bounds ||A_ι (ΔW R_ι)||, with R_ι gating the *columns* of ΔW
    (zeroing intervened-node columns), NOT the columns of A_ι.

    Plug-in bound:
        FrobeniusBall:  γ_ι = σ_max(A_ι[:, K]) · η
                        (R does not tighten: ||ΔW R||_F ≤ ||ΔW||_F)
        RowBudget:      γ_ι = √d · max_i Σ_{j∈K} |A_ι[i,j]| · ρ_j
                        (R does not tighten row-∞ bound)
        ColumnBudget:   γ_ι = √d · max_k ||A_ι[:, K]||_1 · c_k · R[k,k]
                        (R zeros budget at intervened columns)
        EntrywiseBox:   γ_ι = || |A_ι| (M_K · B · diag(R)) ||_F
                        (R zeros budget at intervened columns)

    Parameters
    ----------
    A_iota : (d, d) array
        Interventional propagator (I - (W R_ι))^{-1}.
    R_iota : (d, d) diagonal array
        Gating matrix.
    ambiguity_set : MechanismAmbiguitySet

    Returns
    -------
    float : γ_ι upper bound
    """
    A = np.asarray(A_iota, dtype=float)
    R = np.asarray(R_iota, dtype=float)
    d = A.shape[0]
    r_diag = np.diag(R)  # (d,) — 0 at intervened nodes, 1 elsewhere

    if isinstance(ambiguity_set, FrobeniusBall):
        # sup_{||ΔW_K||_F <= η} ||A (ΔW_K R)||_2
        # Since ||ΔW_K R||_F ≤ ||ΔW_K||_F ≤ η (R is a projection),
        # the bound is σ_max(A restricted to shifted-row columns) · η.
        # R does not tighten the Frobenius ball.
        shifted_cols = list(ambiguity_set.shifted_rows)
        A_sub = A[:, shifted_cols]  # (d, |K|)
        return float(np.linalg.norm(A_sub, ord=2)) * ambiguity_set.eta

    elif isinstance(ambiguity_set, RowBudget):
        # Row budget: ||ΔW[j,:]||_1 ≤ ρ_j.
        # (ΔW R)[j,:] has ||·||_1 = Σ_k |ΔW[j,k]| R[k,k] ≤ Σ_k |ΔW[j,k]| ≤ ρ_j.
        # So ||A (ΔW R)||_∞ ≤ max_i Σ_{j∈K} |A[i,j]| · ρ_j.
        # R does not tighten the row-budget ∞-norm bound.
        val = 0.0
        for i in range(d):
            row_sum = sum(
                abs(A[i, j]) * ambiguity_set.rho.get(j, 0.0)
                for j in ambiguity_set.shifted_rows
            )
            val = max(val, row_sum)
        return float(d) ** 0.5 * val

    elif isinstance(ambiguity_set, ColumnBudget):
        # Column budget: Σ_{j∈K} |ΔW[j,k]| ≤ c_k.
        # (ΔW R)[:,k] = ΔW[:,k] · R[k,k], so
        # ||(ΔW R)[:,k]||_1 = R[k,k] · Σ_{j∈K} |ΔW[j,k]| ≤ c_k · R[k,k].
        # ||A (ΔW R)||_1 ≤ max_k ||A[:,:]||_1 · c_k · R[k,k].
        # More precisely: max_k (Σ_i |[A (ΔW R)]_{ik}|) but the 1-norm
        # bound is max_k (col-l1 of A restricted to K) · c_k · R[k,k].
        val = 0.0
        shifted_set = set(ambiguity_set.shifted_rows)
        for k in range(d):
            ck = ambiguity_set._budget(k) * r_diag[k]
            col_l1 = float(np.sum(np.abs(A[:, list(shifted_set)])))
            val = max(val, col_l1 * ck)
        return float(d) ** 0.5 * val

    elif isinstance(ambiguity_set, EntrywiseBox):
        # Entrywise: ΔW[j,k] ∈ [delta-B, delta+B].
        # sup |ΔW[j,k]| over the box = max(|delta-B|, |delta+B|) = |delta|+B
        # = effective_bound.  When delta=None, effective_bound returns B.
        # |(ΔW R)[j,k]| ≤ effective_bound[j,k] · R[k,k].
        # sup ||A (ΔW R)||_F = || |A| @ (M_K · effective_bound · diag(R)_col) ||_F
        absA = np.abs(A)
        B_sup = ambiguity_set.effective_bound
        M_K = np.zeros_like(B_sup)
        for j in ambiguity_set.shifted_rows:
            M_K[j, :] = 1.0
        B_eff = M_K * B_sup * r_diag[np.newaxis, :]  # R gates columns
        return float(np.linalg.norm(absA @ B_eff, "fro"))

    else:
        raise TypeError(f"Unknown ambiguity set type: {type(ambiguity_set)}")


# ---------------------------------------------------------------------------
# Propagator stability modulus α_ι
# ---------------------------------------------------------------------------

def alpha_polynomial(
    A_iota: np.ndarray,
    gamma_val: float,
    d: int,
) -> float:
    """Polynomial-expansion bound on α_ι = sup ||A'_ι(ΔW) - A_ι||_2.

    α_ι <= ||A_ι||_2 * sum_{k=1}^{d-1} gamma^k

    Valid for any gamma >= 0 (nilpotency guarantees the sum terminates).

    Parameters
    ----------
    A_iota : (d, d) array
    gamma_val : float
        γ_ι from :func:`gamma`.
    d : int
        Ambient dimension (truncates the sum at d-1).

    Returns
    -------
    float : α_ι upper bound
    """
    A_norm = float(np.linalg.norm(A_iota, ord=2))
    series = sum(gamma_val ** k for k in range(1, d))
    return A_norm * series


def alpha_neumann(
    A_iota: np.ndarray,
    gamma_val: float,
    sup_R_dW: float,
) -> float:
    """Neumann-regime bound on α_ι (valid only when gamma_val < 1).

    α_ι <= ||A_ι||_2^2 / (1 - gamma_val) * sup ||R_ι ΔW||_2

    Parameters
    ----------
    A_iota : (d, d) array
    gamma_val : float
        γ_ι; must be < 1.
    sup_R_dW : float
        sup_{ΔW ∈ A_W} ||R_ι ΔW||_2.

    Returns
    -------
    float : α_ι upper bound

    Raises
    ------
    ValueError if gamma_val >= 1.
    """
    if gamma_val >= 1.0:
        raise ValueError(
            f"Neumann bound requires gamma < 1; got gamma = {gamma_val:.4f}. "
            "Use alpha_polynomial instead."
        )
    A_norm = float(np.linalg.norm(A_iota, ord=2))
    return A_norm ** 2 / (1.0 - gamma_val) * sup_R_dW


# ---------------------------------------------------------------------------
# Perturbed propagator
# ---------------------------------------------------------------------------

def perturbed_propagator(
    W: np.ndarray,
    dW: np.ndarray,
    R_iota: np.ndarray,
) -> np.ndarray:
    """Compute A'_ι(ΔW) = (I - (W + ΔW) R_ι)^{-1}.

    Right-multiplication convention: matches lan_scm.py's column-zeroing.
    do(X_k=v) zeroes column k of W, which is equivalent to right-multiplying
    by R_ι where (R_ι)_{kk} = 0 for intervened node k, else 1.

    Uses the exact inverse (no approximation) since d is small.

    Parameters
    ----------
    W : (d, d) strictly upper-triangular
    dW : (d, d) mechanism perturbation
    R_iota : (d, d) diagonal gating matrix

    Returns
    -------
    (d, d) array
    """
    d = W.shape[0]
    W_new = W + dW
    return np.linalg.inv(np.eye(d) - W_new @ R_iota)


# typing import for sequence
from typing import Sequence
