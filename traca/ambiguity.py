"""
Ambiguity sets for mechanism shifts (A_W) and environment shifts (A_U / A_rho).

Four mechanism geometries:
    FrobeniusBall   — ||ΔW||_F <= eta
    RowBudget       — ||ΔW[j,:]||_1 <= rho[j]  for each shifted row j
    ColumnBudget    — ||ΔW[:,k]||_1 <= c[k]     for each column k
    EntrywiseBox    — |ΔW[j,k]| <= B[j,k]

Two environment-side sets:
    GelbrichBall         — W_2(target, source) <= eps  (Gaussian)
    FrobeniusEmpirical   — ||Theta||_F <= eps * sqrt(N)

Convention: the shifted-node mask M_K is applied *before* projecting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np

from traca.utils import project_l1, bures_sqrt, as_rng


# ---------------------------------------------------------------------------
# Shared mask helper
# ---------------------------------------------------------------------------

def _apply_shift_mask(dW: np.ndarray, shifted_rows: Sequence[int]) -> np.ndarray:
    """Zero out all rows not in shifted_rows AND enforce strict upper-triangular.

    ΔW must preserve the causal ordering (DAG structure), so W' = W + ΔW
    must remain strictly upper-triangular.  Entries ΔW[j, k] with k <= j
    are zeroed to enforce this.
    """
    out = np.zeros_like(dW)
    for j in shifted_rows:
        out[j] = dW[j]
    # Enforce strictly upper-triangular: zero diagonal and lower triangle
    return np.triu(out, k=1)


# ---------------------------------------------------------------------------
# Mechanism ambiguity sets
# ---------------------------------------------------------------------------

@dataclass
class FrobeniusBall:
    """Frobenius-norm ball on ΔW restricted to shifted rows.

    Constraint: ||M_K ΔW||_F <= eta

    Parameters
    ----------
    eta : float
        Radius.
    shifted_rows : tuple[int, ...]
        Indices of shifted nodes K. Only these rows of ΔW are non-zero.
    d : int
        Ambient dimension.
    entry_mask : (d, d) array or None
        Optional binary mask applied after shift-row masking. Entries where
        entry_mask[j,k] == 0 are pinned to zero regardless of shifted_rows.
        Use this to restrict the adversary to a column-aligned subset of the
        shifted rows (e.g. only the mechanism of a specific shifted node).
        If None, no additional masking is applied.
    """
    eta: float
    shifted_rows: tuple[int, ...]
    d: int
    entry_mask: np.ndarray | None = None

    def __post_init__(self):
        if self.entry_mask is not None:
            self.entry_mask = np.asarray(self.entry_mask, dtype=float)

    def project(self, dW: np.ndarray) -> np.ndarray:
        """Project ΔW onto the Frobenius ball (mask first, then project).

        Parameters
        ----------
        dW : (d, d) array

        Returns
        -------
        (d, d) array in the constraint set
        """
        dW = np.asarray(dW, dtype=float)
        dW = _apply_shift_mask(dW, self.shifted_rows)
        if self.entry_mask is not None:
            dW = dW * self.entry_mask
        norm = float(np.linalg.norm(dW, "fro"))
        if norm <= self.eta:
            return dW
        return dW * (self.eta / norm)

    def scale(self, multiplier: float) -> "FrobeniusBall":
        """Return a new ball with radius scaled by multiplier."""
        return FrobeniusBall(eta=self.eta * multiplier,
                             shifted_rows=self.shifted_rows, d=self.d,
                             entry_mask=self.entry_mask)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample uniformly on the Frobenius sphere at radius eta.

        Frobenius → uniform-on-sphere: random direction over free entries,
        normalized to ||dW||_F = eta.  Mask applied before normalization
        so magnitude is measured over free entries only.

        ``entry_mask`` is read-only — never written through.
        """
        dW = rng.standard_normal((self.d, self.d))
        dW = _apply_shift_mask(dW, self.shifted_rows)
        if self.entry_mask is not None:
            dW = dW * self.entry_mask
        norm = float(np.linalg.norm(dW, "fro"))
        if norm < 1e-15:
            return np.zeros((self.d, self.d))
        return dW * (self.eta / norm)


@dataclass
class RowBudget:
    """Per-row L1 budget on ΔW restricted to shifted rows.

    Constraint: ||ΔW[j,:]||_1 <= rho[j]  for j in shifted_rows

    Parameters
    ----------
    rho : dict[int, float]
        Budget per shifted row index.
    shifted_rows : tuple[int, ...]
        Indices of shifted nodes K.
    d : int
        Ambient dimension.
    """
    rho: dict[int, float]
    shifted_rows: tuple[int, ...]
    d: int

    def project(self, dW: np.ndarray) -> np.ndarray:
        """Project ΔW onto the row-budget set (mask first, then per-row L1 project).

        Parameters
        ----------
        dW : (d, d) array

        Returns
        -------
        (d, d) array in the constraint set
        """
        dW = np.asarray(dW, dtype=float)
        dW = _apply_shift_mask(dW, self.shifted_rows)
        out = np.zeros_like(dW)
        for j in self.shifted_rows:
            budget = self.rho.get(j, 0.0)
            out[j] = project_l1(dW[j], budget)
        return out

    def scale(self, multiplier: float) -> "RowBudget":
        return RowBudget(
            rho={j: v * multiplier for j, v in self.rho.items()},
            shifted_rows=self.shifted_rows, d=self.d
        )


@dataclass
class ColumnBudget:
    """Per-column L1 budget on ΔW restricted to shifted rows.

    Constraint: ||M_K ΔW[:,k]||_1 <= c[k]  for each column k

    Parameters
    ----------
    c : dict[int, float] or float
        Budget per column. If float, all columns share the same budget.
    shifted_rows : tuple[int, ...]
        Indices of shifted nodes K.
    d : int
        Ambient dimension.
    """
    c: dict[int, float] | float
    shifted_rows: tuple[int, ...]
    d: int

    def _budget(self, k: int) -> float:
        if isinstance(self.c, dict):
            return self.c.get(k, 0.0)
        return float(self.c)

    def project(self, dW: np.ndarray) -> np.ndarray:
        """Project ΔW onto the column-budget set (mask first, then per-column L1 project).

        Parameters
        ----------
        dW : (d, d) array

        Returns
        -------
        (d, d) array in the constraint set
        """
        dW = np.asarray(dW, dtype=float)
        dW = _apply_shift_mask(dW, self.shifted_rows)
        out = np.zeros_like(dW)
        for k in range(self.d):
            col_restricted = dW[:, k].copy()
            # Only shifted rows are non-zero; project those
            shifted_vals = col_restricted[list(self.shifted_rows)]
            projected = project_l1(shifted_vals, self._budget(k))
            for idx, j in enumerate(self.shifted_rows):
                out[j, k] = projected[idx]
        return out

    def scale(self, multiplier: float) -> "ColumnBudget":
        if isinstance(self.c, dict):
            new_c: dict[int, float] | float = {k: v * multiplier for k, v in self.c.items()}
        else:
            new_c = float(self.c) * multiplier
        return ColumnBudget(c=new_c, shifted_rows=self.shifted_rows, d=self.d)


@dataclass
class EntrywiseBox:
    """Entrywise box constraint on ΔW restricted to shifted rows.

    Constraint: ΔW[j,k] ∈ [delta[j,k] - B[j,k], delta[j,k] + B[j,k]]
    for j in shifted_rows.  When delta is None, this reduces to the
    symmetric box |ΔW[j,k]| <= B[j,k].

    Parameters
    ----------
    B : (d, d) array
        Element-wise half-width. Entries for non-shifted rows are ignored
        (zeroed during projection).
    shifted_rows : tuple[int, ...]
        Indices of shifted nodes K.
    d : int
        Ambient dimension.
    delta : (d, d) array or None
        Box center (directional prior). When None, the box is symmetric
        around zero — all code paths are bit-identical to the pre-delta
        implementation.
    """
    B: np.ndarray
    shifted_rows: tuple[int, ...]
    d: int
    delta: np.ndarray | None = None

    def __post_init__(self):
        self.B = np.asarray(self.B, dtype=float)
        if self.delta is not None:
            self.delta = np.asarray(self.delta, dtype=float)

    @property
    def effective_bound(self) -> np.ndarray:
        """Per-entry worst-case absolute value over the box.

        Returns max(|delta-B|, |delta+B|) = |delta| + B per entry.
        When delta is None, returns B (same object — bit-identical).
        """
        if self.delta is None:
            return self.B
        return np.abs(self.delta) + self.B

    def project(self, dW: np.ndarray) -> np.ndarray:
        """Project ΔW onto the entrywise box (mask first, then clip).

        Parameters
        ----------
        dW : (d, d) array

        Returns
        -------
        (d, d) array in the constraint set
        """
        dW = np.asarray(dW, dtype=float)
        dW = _apply_shift_mask(dW, self.shifted_rows)
        if self.delta is None:
            return np.clip(dW, -self.B, self.B)
        return np.clip(dW, self.delta - self.B, self.delta + self.B)

    def scale(self, multiplier: float) -> "EntrywiseBox":
        return EntrywiseBox(
            B=self.B * multiplier,
            shifted_rows=self.shifted_rows, d=self.d,
            delta=self.delta * multiplier if self.delta is not None else None,
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample uniformly within the entrywise box.

        Box → uniform-in-box: each free entry drawn independently from
        Uniform(delta-B, delta+B) (or Uniform(-B, +B) when delta is None).
        Structural zeros in B produce zeros automatically.
        ``_apply_shift_mask`` enforces row selection and strict
        upper-triangularity.
        """
        dW = rng.uniform(-1.0, 1.0, size=(self.d, self.d)) * self.B
        if self.delta is not None:
            dW = dW + self.delta
        return _apply_shift_mask(dW, self.shifted_rows)


# ---------------------------------------------------------------------------
# Environment ambiguity sets
# ---------------------------------------------------------------------------

@dataclass
class GelbrichBall:
    """Gelbrich (Bures–Wasserstein) W_2 ball around the source Gaussian.

    Constraint: W_2(N(mu_t, Sigma_t), N(mu_s, Sigma_s)) <= eps
    with invariance pinning: non-shifted variables are fixed to their source
    parameters.

    Parameters
    ----------
    mu_s : (d,) array
        Source noise mean.
    Sigma_s : (d, d) array
        Source noise covariance.
    eps : float
        Radius.
    shifted_rows : tuple[int, ...]
        Indices of shifted nodes K (only these coordinates of mu/Sigma may differ).
    """
    mu_s: np.ndarray
    Sigma_s: np.ndarray
    eps: float
    shifted_rows: tuple[int, ...]

    def __post_init__(self):
        self.mu_s = np.asarray(self.mu_s, dtype=float)
        self.Sigma_s = np.asarray(self.Sigma_s, dtype=float)

    def project(
        self,
        mu_t: np.ndarray,
        Sigma_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project (mu_t, Sigma_t) onto the Gelbrich ball with invariance pinning.

        Non-shifted coordinates are snapped back to source values (Pi_inv).
        The shifted-coordinate subvector is then projected onto the W_2 ball
        via bisection on the Lagrange multiplier.

        Parameters
        ----------
        mu_t : (d,) array
        Sigma_t : (d, d) array

        Returns
        -------
        (mu_t_proj, Sigma_t_proj) : projected parameters
        """
        from traca.utils import gelbrich_distance
        mu_t = np.asarray(mu_t, dtype=float).copy()
        Sigma_t = np.asarray(Sigma_t, dtype=float).copy()

        # Step 1: invariance pinning — fix non-shifted coordinates to source
        mu_t, Sigma_t = _pi_inv(mu_t, Sigma_t, self.mu_s, self.Sigma_s,
                                 self.shifted_rows)

        # PSD clip after pinning: splicing source rows/cols can break PSD
        # for multi-node districts (Schur complement may become indefinite).
        eigvals, eigvecs = np.linalg.eigh(Sigma_t)
        Sigma_t = eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T

        # Step 2: check if already inside ball
        dist2 = gelbrich_distance(mu_t, Sigma_t, self.mu_s, self.Sigma_s)
        if dist2 <= self.eps ** 2 + 1e-12:
            return mu_t, Sigma_t

        # Step 3: project onto the boundary via bisection (shrink toward source)
        # Simple bisection: interpolate (mu_t, Sigma_t) toward (mu_s, Sigma_s)
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            mu_m = (1 - mid) * mu_t + mid * self.mu_s
            # Covariance interpolation via geodesic is complex;
            # use convex combination as a tractable approximation
            Sigma_m = (1 - mid) * Sigma_t + mid * self.Sigma_s
            if gelbrich_distance(mu_m, Sigma_m, self.mu_s, self.Sigma_s) <= self.eps ** 2:
                hi = mid
            else:
                lo = mid
        alpha = 0.5 * (lo + hi)
        mu_proj = (1 - alpha) * mu_t + alpha * self.mu_s
        Sigma_proj = (1 - alpha) * Sigma_t + alpha * self.Sigma_s
        return mu_proj, Sigma_proj

    def scale(self, multiplier: float) -> "GelbrichBall":
        return GelbrichBall(mu_s=self.mu_s, Sigma_s=self.Sigma_s,
                            eps=self.eps * multiplier,
                            shifted_rows=self.shifted_rows)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random (mu_t, Sigma_t) on the Gelbrich ball boundary.

        Generates a random direction in shifted-coordinate subspace,
        overshoots, then uses ``.project()`` to snap onto the ball boundary.
        ``self.mu_s`` and ``self.Sigma_s`` are read-only — never written through.
        """
        d = len(self.mu_s)
        # Random mean direction (shifted coords only)
        delta_mu = np.zeros(d)
        for j in self.shifted_rows:
            delta_mu[j] = rng.standard_normal()
        mu_norm = float(np.linalg.norm(delta_mu))
        if mu_norm < 1e-15:
            delta_mu = np.zeros(d)
        else:
            delta_mu = delta_mu / mu_norm

        # Random PSD covariance direction (full d×d, pinning handles invariance)
        Z = rng.standard_normal((d, d))
        delta_Sigma = Z @ Z.T
        sig_norm = float(np.linalg.norm(delta_Sigma, "fro"))
        if sig_norm < 1e-15:
            delta_Sigma = np.zeros((d, d))
        else:
            delta_Sigma = delta_Sigma / sig_norm

        # Overshoot: push well beyond the ball so .project() lands on boundary
        overshoot = self.eps * 3.0
        mu_raw = self.mu_s + overshoot * delta_mu
        Sigma_raw = self.Sigma_s + overshoot * delta_Sigma
        return self.project(mu_raw, Sigma_raw)


def _pi_inv(
    mu: np.ndarray,
    Sigma: np.ndarray,
    mu_s: np.ndarray,
    Sigma_s: np.ndarray,
    shifted_rows: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Invariance pinning: fix non-shifted coordinates to source values.

    For i not in shifted_rows:
        mu[i] = mu_s[i]
        Sigma[i, :] = Sigma_s[i, :]  and  Sigma[:, i] = Sigma_s[:, i]
    """
    d = len(mu)
    inv_nodes = [i for i in range(d) if i not in shifted_rows]
    mu = mu.copy()
    Sigma = Sigma.copy()
    for i in inv_nodes:
        mu[i] = mu_s[i]
        Sigma[i, :] = Sigma_s[i, :]
        Sigma[:, i] = Sigma_s[:, i]
    return mu, Sigma


@dataclass
class FrobeniusEmpirical:
    """Frobenius-norm ball on noise shift matrix Theta.

    Constraint: ||Theta||_F <= eps * sqrt(N)
    where Theta is the (d, N) additive noise shift applied to U_s.

    Parameters
    ----------
    eps : float
        Per-sample radius (radius = eps * sqrt(N)).
    N : int
        Number of samples (used to compute the actual radius).
    shifted_rows : tuple[int, ...]
        Indices of shifted nodes K.
    """
    eps: float
    N: int
    shifted_rows: tuple[int, ...]

    def radius(self) -> float:
        """Total Frobenius radius: eps * sqrt(N)."""
        return self.eps * float(self.N) ** 0.5

    def project(self, Theta: np.ndarray) -> np.ndarray:
        """Project Theta onto the Frobenius ball (mask first, then scale).

        Parameters
        ----------
        Theta : (d, N) or (N, d) array — we follow row-vector convention
                so Theta has shape (N, d) where Theta[i, j] is the shift
                for sample i, variable j.

        Returns
        -------
        (N, d) array in the constraint set
        """
        Theta = np.asarray(Theta, dtype=float)
        # Zero out non-shifted columns
        mask = np.zeros(Theta.shape[1] if Theta.ndim == 2 else len(Theta))
        for j in self.shifted_rows:
            mask[j] = 1.0
        Theta = Theta * mask[np.newaxis, :]
        r = self.radius()
        norm = float(np.linalg.norm(Theta, "fro"))
        if norm <= r:
            return Theta
        return Theta * (r / norm)

    def scale(self, multiplier: float) -> "FrobeniusEmpirical":
        return FrobeniusEmpirical(eps=self.eps * multiplier, N=self.N,
                                  shifted_rows=self.shifted_rows)

    def sample(self, rng: np.random.Generator, N: int | None = None,
               d: int | None = None) -> np.ndarray:
        """Sample uniformly on the Frobenius sphere at radius eps * sqrt(n).

        Frobenius → uniform-on-sphere: random direction over shifted columns,
        normalized to ||Theta||_F = eps * sqrt(n) where n is the actual row
        count being sampled.  Non-shifted columns are exactly zero.

        Parameters
        ----------
        rng : numpy random Generator
        N : override sample count (e.g. held-out fold size). If None, uses self.N.
        d : ambient dimension (number of variables). Required because
            FrobeniusEmpirical does not store d.
        """
        n = N if N is not None else self.N
        if d is None:
            raise ValueError("FrobeniusEmpirical.sample() requires d (ambient dimension)")
        Theta = rng.standard_normal((n, d))
        # Zero non-shifted columns
        col_mask = np.zeros(d)
        for j in self.shifted_rows:
            col_mask[j] = 1.0
        Theta = Theta * col_mask[np.newaxis, :]
        norm = float(np.linalg.norm(Theta, "fro"))
        if norm < 1e-15:
            return np.zeros((n, d))
        target_radius = self.eps * float(n) ** 0.5
        return Theta * (target_radius / norm)


# ---------------------------------------------------------------------------
# Type alias for mechanism sets
# ---------------------------------------------------------------------------

MechanismAmbiguitySet = Union[FrobeniusBall, RowBudget, ColumnBudget, EntrywiseBox]
EnvironmentAmbiguitySet = Union[GelbrichBall, FrobeniusEmpirical]
