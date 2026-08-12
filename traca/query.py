"""
Q-restricted losses and selection matrix utilities.

Q-restricted training restricts the transport objective to a designated
set of output coordinates per intervention. When O = full post-interventional,
this recovers the standard full-joint loss.

"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from traca.certificates import selection_matrix
from traca.losses import GaussianLoss, EmpiricalLoss
from traca.stability import perturbed_propagator
from traca.utils import gelbrich_distance


def S_O(d: int, O: Sequence[int]) -> np.ndarray:
    """Selection matrix of shape (|O|, d).

    Parameters
    ----------
    d : int
    O : sequence of int — output coordinate indices

    Returns
    -------
    (|O|, d) array
    """
    return selection_matrix(d, list(O))


def F_iota_O_rho(
    tau: np.ndarray,
    dW: np.ndarray,
    W: np.ndarray,
    A_iota: np.ndarray,
    R_iota: np.ndarray,
    mu_s: np.ndarray,
    Sigma_s: np.ndarray,
    mu_t: np.ndarray | None,
    Sigma_t: np.ndarray | None,
    O: Sequence[int],
) -> float:
    """Q-restricted Gaussian W_2^2 loss restricted to output coordinates O.

    F_ι^{O,ρ}(τ, ΔW) = W_2^2(S_O τ_# N(μ_s A_ι, A_ι.T Σ_s A_ι),
                               S_O N(μ_t, Σ_t))

    When O covers all dimensions, this equals the full-joint loss.

    Parameters
    ----------
    tau : (d, d)
    dW : (d, d)
    W : (d, d)
    A_iota : (d, d)
    R_iota : (d, d)
    mu_s : (d,)
    Sigma_s : (d, d)
    mu_t : (d,) or None (derived from ΔW if None)
    Sigma_t : (d, d) or None
    O : output coordinate indices

    Returns
    -------
    float
    """
    tau = np.asarray(tau, dtype=float)
    d = tau.shape[0]
    SO = S_O(d, O)  # (|O|, d)

    # Pushed source (restricted)
    mu_pushed = mu_s @ A_iota @ tau        # (d,)
    Sigma_pushed = tau.T @ A_iota.T @ Sigma_s @ A_iota @ tau  # (d, d)
    mu_pushed_O = SO @ mu_pushed           # (|O|,)
    Sigma_pushed_O = SO @ Sigma_pushed @ SO.T  # (|O|, |O|)

    # Target (restricted)
    if mu_t is None or Sigma_t is None:
        A_prime = perturbed_propagator(W, dW, R_iota)
        mu_t_use = mu_s @ A_prime
        Sigma_t_use = A_prime.T @ Sigma_s @ A_prime
    else:
        mu_t_use = np.asarray(mu_t)
        Sigma_t_use = np.asarray(Sigma_t)
    mu_t_O = SO @ mu_t_use
    Sigma_t_O = SO @ Sigma_t_use @ SO.T

    return gelbrich_distance(mu_pushed_O, Sigma_pushed_O, mu_t_O, Sigma_t_O)


def F_iota_O_U(
    tau: np.ndarray,
    dW: np.ndarray,
    Theta: np.ndarray,
    W: np.ndarray,
    A_iota: np.ndarray,
    R_iota: np.ndarray,
    U_s: np.ndarray,
    O: Sequence[int],
) -> float:
    """Q-restricted empirical Frobenius^2 loss restricted to output coordinates O.

    F_ι^{O,U}(τ, ΔW, Θ) = (1/N) ||S_O (U_s A_ι τ - (U_s+Θ) A'_ι)||_F^2

    When O covers all dimensions, equals the full-joint empirical loss.

    Parameters
    ----------
    tau : (d, d)
    dW : (d, d)
    Theta : (N, d)
    W : (d, d)
    A_iota : (d, d)
    R_iota : (d, d)
    U_s : (N, d)
    O : output coordinate indices

    Returns
    -------
    float
    """
    tau = np.asarray(tau, dtype=float)
    dW = np.asarray(dW, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    U_s = np.asarray(U_s, dtype=float)
    d = tau.shape[0]
    N = U_s.shape[0]
    SO = S_O(d, O)  # (|O|, d)

    A_prime = perturbed_propagator(W, dW, R_iota)
    pushed = U_s @ A_iota @ tau           # (N, d)
    target = (U_s + Theta) @ A_prime      # (N, d)
    residual = pushed - target            # (N, d)

    # Project residual onto O coordinates
    residual_O = residual @ SO.T          # (N, |O|)
    return float(np.linalg.norm(residual_O, "fro") ** 2) / N


def query_family_from_config(config: dict) -> list[tuple[int, list[int]]] | None:
    """Parse query family from a YAML config dict.

    Expected YAML format:
        training:
          query_family: null   # or:
          query_family:
            - {intervention_idx: 0, O: [2, 3]}
            - {intervention_idx: 1, O: [2]}

    Parameters
    ----------
    config : dict parsed from YAML

    Returns
    -------
    list of (intervention_idx, O) pairs, or None if query_family is null
    """
    training = config.get("training", {})
    qf = training.get("query_family", None)
    if qf is None:
        return None
    result = []
    for entry in qf:
        iota_idx = int(entry["intervention_idx"])
        O = [int(x) for x in entry["O"]]
        result.append((iota_idx, O))
    return result
