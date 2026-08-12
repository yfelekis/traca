"""
PyTorch twins of the core Gaussian loss functions, enabling autograd-based
gradient computation as an alternative to the hand-coded analytic gradients
in traca.losses.

These functions are the *only* entry points into PyTorch from the TraCA
package.  The analytic path in losses.py / optim.py is completely unaffected.

Row-vector convention (matches losses.py):
    mu_pushed = mu_s @ A_iota @ tau
    Sigma_pushed = tau.T @ A_iota.T @ Sigma_s @ A_iota @ tau
    A'_iota = (I - (W+dW) R_iota)^{-1}   (right-mult, matches stability.py)

All functions accept torch.Tensor inputs; the caller controls which tensors
carry requires_grad=True.  Each returns a 0-dim torch.Tensor so .backward()
is directly available.

Optional import -- torch is NOT a hard dependency of the traca package.
"""
from __future__ import annotations

try:
    import torch
except ImportError as _e:
    raise ImportError(
        "traca.losses_torch requires PyTorch. "
        "Install it with: pip install torch"
    ) from _e

import numpy as np


# ---------------------------------------------------------------------------
# Internal conversion helper
# ---------------------------------------------------------------------------

def _to_tensor(x, *, requires_grad: bool = False) -> "torch.Tensor":
    """Convert a numpy array or tensor to a float64 torch tensor."""
    if isinstance(x, torch.Tensor):
        t = x.double()
    else:
        t = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64)
    if requires_grad:
        t.requires_grad_(True)
    return t


# ---------------------------------------------------------------------------
# Differentiable matrix square-root
# ---------------------------------------------------------------------------

def bures_sqrt_torch(M: "torch.Tensor") -> "torch.Tensor":
    """Differentiable symmetric matrix square-root via eigh.

    Uses torch.linalg.eigh, which supports autograd through the symmetric
    eigenvector decomposition.  Eigenvalues are clipped to [0, inf) to
    handle numerical near-zero negatives.

    Zero eigenvalues (from e.g. zeroed covariance rows/cols at intervened
    nodes) produce sqrt(0)=0 with **zero gradient**, not the infinite
    d/dx sqrt(x)|_{x=0} that torch.sqrt would give.  This is correct:
    zero-variance directions contribute zero to the Bures distance and
    zero to its gradient w.r.t. τ.

    Matches traca.utils.bures_sqrt exactly for PSD inputs.

    Parameters
    ----------
    M : (n, n) symmetric PSD tensor

    Returns
    -------
    (n, n) tensor  s.t.  result @ result = M,  result symmetric PSD
    """
    M = 0.5 * (M + M.T)
    L, V = torch.linalg.eigh(M)          # L ascending eigenvalues, V eigenvectors
    L = torch.clamp(L, min=0.0)
    # Guard: torch.sqrt(0) has infinite gradient (d/dx √x = 1/(2√x) → ∞).
    # This arises whenever a covariance is singular — e.g. intervened nodes
    # have their rows/cols zeroed, producing exact-zero eigenvalues.
    # Fix: clamp to a tiny floor so sqrt never sees exact 0, then mask out
    # the near-zero eigenvalue contributions so they don't affect the result.
    # Both clamp (grad=0 below floor) and mask (hard zero) independently
    # kill gradient flow through zero eigenvalues — the correct behavior,
    # since zero-variance directions contribute nothing to the Bures distance.
    active = (L > 1e-15).to(L.dtype)                   # 1 for real eigenvalues, 0 for zero
    sqrt_L = torch.sqrt(torch.clamp(L, min=1e-30)) * active
    return V @ torch.diag(sqrt_L) @ V.T


# ---------------------------------------------------------------------------
# Differentiable propagator
# ---------------------------------------------------------------------------

def perturbed_propagator_torch(
    W: "torch.Tensor",
    dW: "torch.Tensor",
    R_iota: "torch.Tensor",
) -> "torch.Tensor":
    """Perturbed propagator A'_iota = (I - (W+dW) R_iota)^{-1}.

    Right-multiplication convention -- matches traca.stability.perturbed_propagator.

    Returns
    -------
    (d, d) tensor
    """
    d = W.shape[0]
    I = torch.eye(d, dtype=W.dtype, device=W.device)
    F = I - (W + dW) @ R_iota
    return torch.linalg.solve(F, I)


# ---------------------------------------------------------------------------
# Gelbrich (Bures-Wasserstein) distance -- exact
# ---------------------------------------------------------------------------

def gelbrich_distance_torch(
    mu1: "torch.Tensor",
    Sigma1: "torch.Tensor",
    mu2: "torch.Tensor",
    Sigma2: "torch.Tensor",
) -> "torch.Tensor":
    """Exact Gelbrich W2 distance squared between two Gaussians, differentiable.

    W2^2(N(mu1, Sigma1), N(mu2, Sigma2))
      = ||mu1 - mu2||^2
        + Tr(Sigma1 + Sigma2 - 2*(Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})

    Matches traca.utils.gelbrich_distance exactly.

    Returns
    -------
    scalar tensor (0-dim)
    """
    diff = mu1 - mu2
    mean_term = torch.dot(diff, diff)
    S1_sqrt = bures_sqrt_torch(Sigma1)
    M = S1_sqrt @ Sigma2 @ S1_sqrt
    cross = bures_sqrt_torch(M)
    cov_term = torch.trace(Sigma1) + torch.trace(Sigma2) - 2.0 * torch.trace(cross)
    return mean_term + torch.clamp(cov_term, min=0.0)


# ---------------------------------------------------------------------------
# Bures surrogate F_tilde  (matches GaussianLoss.surrogate)
# ---------------------------------------------------------------------------

def surrogate_torch(
    tau: "torch.Tensor",
    dW: "torch.Tensor",
    W: "torch.Tensor",
    A_iota: "torch.Tensor",
    R_iota: "torch.Tensor",
    mu_s: "torch.Tensor",
    Sigma_s: "torch.Tensor",
    mu_t: "torch.Tensor",
    Sigma_t: "torch.Tensor",
) -> "torch.Tensor":
    """Bures surrogate F_tilde_iota -- differentiable PyTorch twin.

    F_tilde = ||mu_pushed - mu_t||^2
              + (||Sigma_pushed^{1/2}||_F - ||Sigma_t^{1/2}||_F)^2

    Matches GaussianLoss.surrogate exactly.  mu_t and Sigma_t must be in
    OBSERVED (endogenous) space -- same convention as losses.py.

    The dW parameter is accepted for interface symmetry with exact_loss_torch
    but is not used in the surrogate computation (tau drives the push-forward).

    Returns
    -------
    scalar tensor (0-dim)
    """
    mu_pushed = mu_s @ A_iota @ tau         # (d,)
    B = A_iota @ tau                         # (d, d)
    Sigma_pushed = B.T @ Sigma_s @ B        # (d, d)

    mean_term = torch.dot(mu_pushed - mu_t, mu_pushed - mu_t)
    S_push = bures_sqrt_torch(Sigma_pushed)
    S_t = bures_sqrt_torch(Sigma_t)
    f_push = torch.linalg.norm(S_push, ord="fro")
    f_t = torch.linalg.norm(S_t, ord="fro")
    cov_term = (f_push - f_t) ** 2

    return mean_term + torch.clamp(cov_term, min=0.0)


# ---------------------------------------------------------------------------
# Exact W2 loss F  (matches GaussianLoss.value)
# ---------------------------------------------------------------------------

def exact_loss_torch(
    tau: "torch.Tensor",
    dW: "torch.Tensor",
    W: "torch.Tensor",
    A_iota: "torch.Tensor",
    R_iota: "torch.Tensor",
    mu_s: "torch.Tensor",
    Sigma_s: "torch.Tensor",
    mu_t: "torch.Tensor",
    Sigma_t: "torch.Tensor",
) -> "torch.Tensor":
    """Exact W2 squared loss F_iota -- differentiable PyTorch twin.

    Matches GaussianLoss.value exactly.  mu_t and Sigma_t must be in
    OBSERVED (endogenous) space.

    Returns
    -------
    scalar tensor (0-dim)
    """
    mu_pushed = mu_s @ A_iota @ tau         # (d,)
    B = A_iota @ tau                         # (d, d)
    Sigma_pushed = B.T @ Sigma_s @ B        # (d, d)

    return gelbrich_distance_torch(mu_pushed, Sigma_pushed, mu_t, Sigma_t)
