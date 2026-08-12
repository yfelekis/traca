"""
Loss functions for the adversarial minimax objective.

Two loss families:
    GaussianLoss  — Wasserstein-2 squared distance between Gaussians.
    EmpiricalLoss — Frobenius-squared residual on noise samples.

Plus the Bures surrogate for GaussianLoss.

Row-vector convention: X = U @ A.
Transported source mean/cov under τ:
    μ_pushed = μ_s @ A_ι @ τ
    Σ_pushed = τ.T @ A_ι.T @ Σ_s @ A_ι @ τ

Under ΔW the target propagator is A'_ι = (I - R_ι(W+ΔW))^{-1}.
Target Gaussian: N(μ_s @ A'_ι, A'_ι.T @ Σ_s @ A'_ι).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traca.utils import bures_sqrt, gelbrich_distance
from traca.stability import perturbed_propagator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pushed_mean(mu_s: np.ndarray, A_iota: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Transported source mean: μ_s @ A_ι @ τ.  Shape: (d,)."""
    return mu_s @ A_iota @ tau


def _pushed_cov(
    Sigma_s: np.ndarray, A_iota: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """Transported source cov: τ.T @ A_ι.T @ Σ_s @ A_ι @ τ.  Shape: (d, d)."""
    B = A_iota @ tau  # (d, d)
    return B.T @ Sigma_s @ B


def _target_gaussian(
    mu_s: np.ndarray,
    Sigma_s: np.ndarray,
    W: np.ndarray,
    dW: np.ndarray,
    R_iota: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Adversarial target Gaussian parameters under ΔW.

    μ_t^adv = μ_s @ A'_ι(ΔW)
    Σ_t^adv = A'_ι(ΔW).T @ Σ_s @ A'_ι(ΔW)
    """
    A_prime = perturbed_propagator(W, dW, R_iota)
    mu_t = mu_s @ A_prime
    Sigma_t = A_prime.T @ Sigma_s @ A_prime
    return mu_t, Sigma_t


# ---------------------------------------------------------------------------
# Gaussian loss (W_2^2)
# ---------------------------------------------------------------------------

class GaussianLoss:
    """Wasserstein-2 squared loss for the Gaussian objective.

    F_ι^ρ(τ, ΔW; μ_s, Σ_s) = W_2^2(τ_# N(μ_s A_ι, A_ι.T Σ_s A_ι),
                                      N(μ_s A'_ι, A'_ι.T Σ_s A'_ι))
    """

    def value(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        mu_s: np.ndarray,
        Sigma_s: np.ndarray,
        mu_t: np.ndarray | None = None,
        Sigma_t: np.ndarray | None = None,
    ) -> float:
        """Evaluate F_ι^ρ.

        If mu_t / Sigma_t are provided, they are used as the adversarial target
        (for the alternating adversary step). Otherwise they are derived from ΔW.

        Parameters
        ----------
        tau : (d, d)
        dW : (d, d) mechanism perturbation (used to build A'_ι if mu_t/Sigma_t absent)
        W : (d, d) base structural matrix
        A_iota : (d, d) base interventional propagator
        R_iota : (d, d) gating matrix
        mu_s : (d,)
        Sigma_s : (d, d)
        mu_t : (d,) or None
        Sigma_t : (d, d) or None

        Returns
        -------
        float : W_2^2 value
        """
        tau = np.asarray(tau, dtype=float)
        dW = np.asarray(dW, dtype=float)

        # Pushed source
        mu_pushed = _pushed_mean(mu_s, A_iota, tau)
        Sigma_pushed = _pushed_cov(Sigma_s, A_iota, tau)

        # Target
        if mu_t is None or Sigma_t is None:
            mu_t_use, Sigma_t_use = _target_gaussian(mu_s, Sigma_s, W, dW, R_iota)
        else:
            mu_t_use, Sigma_t_use = np.asarray(mu_t), np.asarray(Sigma_t)

        return gelbrich_distance(mu_pushed, Sigma_pushed, mu_t_use, Sigma_t_use)

    def surrogate(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        mu_s: np.ndarray,
        Sigma_s: np.ndarray,
        mu_t: np.ndarray | None = None,
        Sigma_t: np.ndarray | None = None,
    ) -> float:
        """Bures surrogate F̃_ι^ρ — a differentiable lower bound of value().

        Replaces the cross-term Tr((C_s^{1/2} C_t C_s^{1/2})^{1/2}) with the
        Cauchy–Schwarz lower bound ||C_s^{1/2}||_F ||C_t^{1/2}||_F.

        F̃ = ||μ_pushed - μ_t||^2
            + ||Σ_pushed^{1/2}||_F^2 + ||Σ_t^{1/2}||_F^2
            - 2 ||Σ_pushed^{1/2}||_F ||Σ_t^{1/2}||_F

        Note: F̃ <= F (surrogate <= exact).
        """
        tau = np.asarray(tau, dtype=float)
        dW = np.asarray(dW, dtype=float)

        mu_pushed = _pushed_mean(mu_s, A_iota, tau)
        Sigma_pushed = _pushed_cov(Sigma_s, A_iota, tau)

        if mu_t is None or Sigma_t is None:
            mu_t_use, Sigma_t_use = _target_gaussian(mu_s, Sigma_s, W, dW, R_iota)
        else:
            mu_t_use, Sigma_t_use = np.asarray(mu_t), np.asarray(Sigma_t)

        mean_term = float(np.dot(mu_pushed - mu_t_use, mu_pushed - mu_t_use))
        S_push_sqrt = bures_sqrt(Sigma_pushed)
        S_t_sqrt = bures_sqrt(Sigma_t_use)
        f_push = float(np.linalg.norm(S_push_sqrt, "fro"))
        f_t = float(np.linalg.norm(S_t_sqrt, "fro"))
        cov_term = f_push ** 2 + f_t ** 2 - 2.0 * f_push * f_t
        return mean_term + max(cov_term, 0.0)

    # ------------------------------------------------------------------
    # Analytic gradients (implemented in Phase 5.5)
    # ------------------------------------------------------------------

    def grad_tau(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        mu_s: np.ndarray,
        Sigma_s: np.ndarray,
        mu_t: np.ndarray | None = None,
        Sigma_t: np.ndarray | None = None,
    ) -> np.ndarray:
        """Gradient of the surrogate F̃_ι^ρ w.r.t. τ.

        Returns
        -------
        (d, d) array : ∂F̃/∂τ
        """
        tau = np.asarray(tau, dtype=float)
        dW = np.asarray(dW, dtype=float)

        mu_pushed = _pushed_mean(mu_s, A_iota, tau)       # μ_s A_ι τ
        Sigma_pushed = _pushed_cov(Sigma_s, A_iota, tau)  # τ.T A_ι.T Σ_s A_ι τ

        if mu_t is None or Sigma_t is None:
            mu_t_use, Sigma_t_use = _target_gaussian(mu_s, Sigma_s, W, dW, R_iota)
        else:
            mu_t_use, Sigma_t_use = np.asarray(mu_t), np.asarray(Sigma_t)

        # Gradient of mean term: ||μ_pushed - μ_t||^2 w.r.t. τ
        # μ_pushed = μ_s @ A_ι @ τ  =>  ∂/∂τ = A_ι.T @ μ_s.T (μ_pushed - μ_t) outer
        mu_diff = mu_pushed - mu_t_use  # (d,)
        # d||μ_pushed - μ_t||^2/dτ = 2 * (A_ι.T μ_s.T)^T (μ_pushed - μ_t)^T
        # Since μ_pushed = μ_s A_ι τ, the Jacobian ∂μ_pushed/∂τ_{ij} = (μ_s A_ι)_i δ_{jk}
        # So ∂loss/∂τ_{ij} = 2 sum_k μ_diff_k * (μ_s A_ι)_i δ_{kj}
        #                   = 2 * (μ_s A_ι)_i * μ_diff_j
        B = mu_s @ A_iota  # (d,)
        grad_mean = 2.0 * np.outer(B, mu_diff)  # (d, d)

        # Gradient of covariance surrogate term
        # Σ_pushed = τ.T A_ι.T Σ_s A_ι τ  = C.T τ where C = A_ι.T Σ_s A_ι
        # F_push = ||Σ_pushed^{1/2}||_F = sqrt(Tr(Σ_pushed))
        # ∂Tr(Σ_pushed)/∂τ = 2 A_ι.T Σ_s A_ι τ = 2 C τ  (where C = A_ι.T Σ_s A_ι)
        S_push_sqrt = bures_sqrt(Sigma_pushed)
        S_t_sqrt = bures_sqrt(Sigma_t_use)
        f_push = float(np.linalg.norm(S_push_sqrt, "fro"))
        f_t = float(np.linalg.norm(S_t_sqrt, "fro"))

        C = A_iota.T @ Sigma_s @ A_iota  # (d, d)

        # d/dτ [f_push^2 - 2 f_push f_t] = d/dτ [Tr(Σ_pushed)] * (1 - f_t/f_push)
        # where d/dτ Tr(Σ_pushed) = 2 C τ
        if f_push > 1e-12:
            coeff = 1.0 - f_t / f_push
        else:
            coeff = 1.0
        grad_cov = 2.0 * C @ tau * coeff  # (d, d)

        return grad_mean + grad_cov

    def grad_dW(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        mu_s: np.ndarray,
        Sigma_s: np.ndarray,
        mu_t: np.ndarray | None = None,
        Sigma_t: np.ndarray | None = None,
    ) -> np.ndarray:
        """Gradient of the Bures surrogate F̃ w.r.t. ΔW via resolvent identity.

        Uses F̃ (not exact W₂²) for consistency with grad_tau and grad_Sigma_t.
        The exact W₂² gradient requires analytic matrix-sqrt derivatives; the
        surrogate gives a fast analytic ascent direction sufficient in practice.

        Resolvent identity: ∂A'_ι[j,k]/∂(ΔW)_{pq} = A'_ι[j,p] · R_ι[q,q] · A'_ι[q,k]

        Vectorized chain rule (mu_t, Sigma_t in OBSERVED space):
            G_mu[p,q]   = μ_t_obs[p] · R_ι[q,q] · (A'_ι g_mu)[q]
                        = outer(μ_t_obs, R_ι @ (A'_ι g_mu))[p,q]
            G_Sigma[p,q] = 2 · (Σ_t_obs g_Sigma_sym A'_ι.T R_ι)[p,q]
                         where g_Sigma_sym = 0.5 * (g_Sigma + g_Sigma.T)

        mu_t and Sigma_t must be the adversary params in OBSERVED space
        (i.e. already transformed by A'_ι, not in exogenous space).

        Returns
        -------
        (d, d) array : ∂F̃/∂(ΔW)
        """
        tau = np.asarray(tau, dtype=float)
        dW = np.asarray(dW, dtype=float)

        A_prime = perturbed_propagator(W, dW, R_iota)

        if mu_t is None or Sigma_t is None:
            mu_t_use = mu_s @ A_prime
            Sigma_t_use = A_prime.T @ Sigma_s @ A_prime
        else:
            mu_t_use = np.asarray(mu_t, dtype=float)
            Sigma_t_use = np.asarray(Sigma_t, dtype=float)

        # ∂F̃/∂μ_t = -2(μ_pushed - μ_t)
        mu_pushed = _pushed_mean(mu_s, A_iota, tau)
        g_mu = -2.0 * (mu_pushed - mu_t_use)  # (d,)

        # ∂F̃/∂Σ_t (surrogate, numerical)
        g_Sigma = self.grad_Sigma_t(
            tau, dW, W, A_iota, R_iota, mu_s, Sigma_s, mu_t_use, Sigma_t_use
        )  # (d, d)

        # Mean contribution: G_mu[p,q] = μ_t_obs[p] · R[q,q] · (A' g_mu)[q]
        G_mu = np.outer(mu_t_use, R_iota @ (A_prime @ g_mu))  # (d, d)

        # Covariance contribution: G_Sigma[p,q] = 2 (Σ_t_obs g_Sigma_sym A'.T R)[p,q]
        g_Sigma_sym = 0.5 * (g_Sigma + g_Sigma.T)
        G_Sigma = 2.0 * (Sigma_t_use @ g_Sigma_sym @ A_prime.T @ R_iota)

        return G_mu + G_Sigma

    def grad_mu_t(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        mu_s: np.ndarray,
        Sigma_s: np.ndarray,
        mu_t: np.ndarray,
        Sigma_t: np.ndarray,
    ) -> np.ndarray:
        """Gradient of F_ι^ρ w.r.t. μ_t (adversary step for Gaussian environment).

        ∂W_2^2/∂μ_t = -2 (μ_pushed - μ_t)

        Returns
        -------
        (d,) array
        """
        mu_pushed = _pushed_mean(mu_s, A_iota, tau)
        return -2.0 * (mu_pushed - np.asarray(mu_t))

    def grad_Sigma_t(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        mu_s: np.ndarray,
        Sigma_s: np.ndarray,
        mu_t: np.ndarray,
        Sigma_t: np.ndarray,
    ) -> np.ndarray:
        """Gradient of the Bures surrogate F̃_ι^ρ w.r.t. Σ_t (closed form).

        F̃ = ||μ_pushed - μ_t||² + (f_push - f_t)²
        where f_t = √Tr(Σ_t).

        ∂F̃/∂Σ_t = 2(f_push - f_t) · (-1/(2f_t)) · I = (1 - f_push/f_t) · I

        Returns
        -------
        (d, d) array
        """
        d = Sigma_t.shape[0]
        Sigma_t = np.asarray(Sigma_t, dtype=float)

        Sigma_pushed = _pushed_cov(Sigma_s, A_iota, tau)
        f_push = float(np.sqrt(max(np.trace(Sigma_pushed), 0.0)))
        f_t = float(np.sqrt(max(np.trace(Sigma_t), 0.0)))

        if f_t < 1e-12:
            return np.zeros((d, d))

        return (1.0 - f_push / f_t) * np.eye(d)


# ---------------------------------------------------------------------------
# Empirical loss (Frobenius^2)
# ---------------------------------------------------------------------------

class EmpiricalLoss:
    """Frobenius-squared residual for the empirical objective.

    F_ι^U(τ, ΔW, Θ; U_s) = ||U_s A_ι τ - (U_s + Θ) A'_ι(ΔW)||_F^2 / N
    """

    def value(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        Theta: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        U_s: np.ndarray,
    ) -> float:
        """Evaluate F_ι^U.

        Parameters
        ----------
        tau : (d, d) transport map
        dW : (d, d) mechanism perturbation
        Theta : (N, d) additive noise shift (Θ_ι in the paper)
        W : (d, d) base structural matrix
        A_iota : (d, d) base propagator
        R_iota : (d, d) gating matrix
        U_s : (N, d) source noise samples

        Returns
        -------
        float : F_ι^U value
        """
        tau = np.asarray(tau, dtype=float)
        dW = np.asarray(dW, dtype=float)
        Theta = np.asarray(Theta, dtype=float)
        U_s = np.asarray(U_s, dtype=float)
        N = U_s.shape[0]

        A_prime = perturbed_propagator(W, dW, R_iota)
        pushed = U_s @ A_iota @ tau           # (N, d) transported source
        target = (U_s + Theta) @ A_prime      # (N, d) adversarial target
        residual = pushed - target            # (N, d)
        return float(np.linalg.norm(residual, "fro") ** 2) / N

    # ------------------------------------------------------------------
    # Analytic gradients
    # ------------------------------------------------------------------

    def grad_tau(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        Theta: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        U_s: np.ndarray,
    ) -> np.ndarray:
        """Gradient of F_ι^U w.r.t. τ.

        ∂F/∂τ = (2/N) (A_ι.T U_s.T) (U_s A_ι τ - (U_s+Θ) A'_ι)

        Returns
        -------
        (d, d) array
        """
        tau = np.asarray(tau, dtype=float)
        dW = np.asarray(dW, dtype=float)
        Theta = np.asarray(Theta, dtype=float)
        U_s = np.asarray(U_s, dtype=float)
        N = U_s.shape[0]

        A_prime = perturbed_propagator(W, dW, R_iota)
        residual = U_s @ A_iota @ tau - (U_s + Theta) @ A_prime  # (N, d)
        # ∂/∂τ = (2/N) * (A_ι.T U_s.T residual)
        return (2.0 / N) * A_iota.T @ (U_s.T @ residual)

    def grad_Theta(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        Theta: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        U_s: np.ndarray,
    ) -> np.ndarray:
        """Gradient of F_ι^U w.r.t. Θ (adversary step).

        ∂F/∂Θ = -(2/N) residual @ A'_ι.T

        Returns
        -------
        (N, d) array
        """
        tau = np.asarray(tau, dtype=float)
        dW = np.asarray(dW, dtype=float)
        Theta = np.asarray(Theta, dtype=float)
        U_s = np.asarray(U_s, dtype=float)
        N = U_s.shape[0]

        A_prime = perturbed_propagator(W, dW, R_iota)
        residual = U_s @ A_iota @ tau - (U_s + Theta) @ A_prime  # (N, d)
        return -(2.0 / N) * residual @ A_prime.T

    def grad_dW(
        self,
        tau: np.ndarray,
        dW: np.ndarray,
        Theta: np.ndarray,
        W: np.ndarray,
        A_iota: np.ndarray,
        R_iota: np.ndarray,
        U_s: np.ndarray,
    ) -> np.ndarray:
        """Gradient of F_ι^U w.r.t. ΔW via the resolvent identity.

        ∂A'_ι(ΔW)[H] = A'_ι @ R_ι @ H @ A'_ι   (resolvent identity)

        ∂F/∂(ΔW) = -(2/N) A'_ι.T ((U_s+Θ).T residual) A'_ι.T

        Returns
        -------
        (d, d) array
        """
        tau = np.asarray(tau, dtype=float)
        dW = np.asarray(dW, dtype=float)
        Theta = np.asarray(Theta, dtype=float)
        U_s = np.asarray(U_s, dtype=float)
        N = U_s.shape[0]

        A_prime = perturbed_propagator(W, dW, R_iota)
        residual = U_s @ A_iota @ tau - (U_s + Theta) @ A_prime  # (N, d)
        # Chain rule: ∂F/∂(ΔW_{jk}) = -2/N * sum_n residual[n,:] @ ∂A'[H_jk]/∂Θ
        # where ∂A'[H] = A' R H A'
        # So: ∂F/∂(ΔW) = -(2/N) * (A' R).T @ (U_s+Θ).T @ residual @ A'.T
        UpT = U_s + Theta  # (N, d)
        # sum_n (U_s+Θ)[n,:].T residual[n,:] = (U_s+Θ).T @ residual  -> (d,d)
        M = UpT.T @ residual  # (d, d)
        # ∂F/∂ΔW_{jk} = -(2/N) sum_{i,l} (A' R)_{ij} M_{il} A'_{lk}
        # = -(2/N) (A' R).T M A'.T
        AR = A_prime @ R_iota  # (d, d)
        return -(2.0 / N) * AR.T @ M @ A_prime.T


# ---------------------------------------------------------------------------
# Numerical gradient helper (for fallback / gradient checking)
# ---------------------------------------------------------------------------

def numerical_gradient(
    f,
    x: np.ndarray,
    eps: float = 1e-6,
    *args,
    **kwargs,
) -> np.ndarray:
    """Central-difference numerical gradient of scalar f w.r.t. x.

    Parameters
    ----------
    f : callable(x, *args, **kwargs) -> float
    x : array of any shape
    eps : float, perturbation size

    Returns
    -------
    array of same shape as x
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = float(f(x, *args, **kwargs))
        x[idx] = orig - eps
        fm = float(f(x, *args, **kwargs))
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def _numerical_grad_dW_gaussian(
    loss: GaussianLoss,
    tau, dW, W, A_iota, R_iota, mu_s, Sigma_s, mu_t, Sigma_t
) -> np.ndarray:
    """Numerical gradient of GaussianLoss.value w.r.t. ΔW."""
    dW = np.asarray(dW, dtype=float)
    def f(dw):
        return loss.value(tau, dw, W, A_iota, R_iota, mu_s, Sigma_s, mu_t, Sigma_t)
    return numerical_gradient(f, dW)
