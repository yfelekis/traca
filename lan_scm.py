"""
Linear Additive-Noise SCM machinery.

Convention (row-vector form)
----------------------------
    X = X W + U          (structural)
    X = U A              (reduced),   A = (I - W)^{-1}

W is strictly upper-triangular: W[i,j] = coefficient on edge X_i → X_j, i < j.
Because W is nilpotent, A always exists: A = I + W + W² + ... + W^{d-1}.

Sample matrix:  X_samples = U_samples @ A       shape (n, d)
Gaussian mean:  μ_X = μ_U @ A
Gaussian cov:   Σ_X = A.T @ Σ_U @ A

Intervention do(X_J = v)
------------------------
1. Zero columns J of W  →  W_do
2. A_do = (I - W_do)^{-1}
3. Set fixed[J] = v; zero U[:,J] at sample time
4. X_do = U_do @ A_do + fixed @ A_do
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, NamedTuple

import argparse
import joblib
import sys
import yaml

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_rng(seed=None, rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng(seed)


def _matrix_norm(M: np.ndarray, norm: str = "spectral") -> float:
    if norm == "spectral":
        return float(np.linalg.norm(M, ord=2))
    if norm == "frobenius":
        return float(np.linalg.norm(M, ord="fro"))
    if norm == "inf":
        return float(np.linalg.norm(M, ord=np.inf))
    if norm == "1":
        return float(np.linalg.norm(M, ord=1))
    raise ValueError(f"Unknown norm '{norm}'. Choose: spectral, frobenius, inf, 1.")


def _strictly_upper(M: np.ndarray) -> np.ndarray:
    return np.triu(np.asarray(M, dtype=float), k=1)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class LANSCM:
    """
    Linear Additive-Noise SCM.

    Parameters
    ----------
    W : (d, d) array
        Structural coefficients. Only the strictly upper-triangular part is used.
    noise_mean : (d,) array, optional
        Mean of exogenous noise U. Defaults to zeros.
    noise_cov : (d, d) array, optional
        Covariance of U. Defaults to identity.
    var_names : list[str], optional
        Variable names in topological order. Defaults to ['X0', 'X1', ...].
    """

    def __init__(
        self,
        W: np.ndarray,
        noise_mean: Optional[np.ndarray] = None,
        noise_cov:  Optional[np.ndarray] = None,
        var_names:  Optional[Sequence[str]] = None,
        _J:     Optional[list[int]]  = None,
        _fixed: Optional[np.ndarray] = None,
    ):
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square, got shape {W.shape}.")
        self._W = _strictly_upper(W).copy()
        d = self._W.shape[0]

        self.noise_mean = np.zeros(d, dtype=float) if noise_mean is None \
            else np.asarray(noise_mean, dtype=float)
        if self.noise_mean.shape != (d,):
            raise ValueError(f"noise_mean must have shape ({d},), got {self.noise_mean.shape}.")

        self.noise_cov = np.eye(d, dtype=float) if noise_cov is None \
            else np.asarray(noise_cov, dtype=float)
        if self.noise_cov.shape != (d, d):
            raise ValueError(f"noise_cov must have shape ({d},{d}), got {self.noise_cov.shape}.")
        if not np.allclose(self.noise_cov, self.noise_cov.T):
            raise ValueError("noise_cov must be symmetric.")

        self.var_names: list[str] = [f"X{i}" for i in range(d)] if var_names is None \
            else list(var_names)
        if len(self.var_names) != d:
            raise ValueError(f"var_names must have length {d}, got {len(self.var_names)}.")

        # Intervention state (private)
        self._J:     list[int]   = [] if _J is None else list(_J)
        self._fixed: np.ndarray  = np.zeros(d, dtype=float) if _fixed is None \
            else np.asarray(_fixed, dtype=float)
        if self._fixed.shape != (d,):
            raise ValueError(f"_fixed must have shape ({d},), got {self._fixed.shape}.")

        self._A: Optional[np.ndarray] = None  # cached propagator

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def d(self) -> int:
        """Number of variables."""
        return self._W.shape[0]

    @property
    def W(self) -> np.ndarray:
        """Structural matrix (read-only copy)."""
        return self._W.copy()

    @property
    def A(self) -> np.ndarray:
        """Propagator A = (I - W)^{-1}, computed once and cached."""
        if self._A is None:
            I = np.eye(self.d, dtype=float)
            self._A = np.linalg.solve(I - self._W, I)
        return self._A

    # ------------------------------------------------------------------
    # Named-edge constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_edges(
        cls,
        var_names: Sequence[str],
        edges: Sequence[tuple[str, str, float]],
        noise_mean: Optional[np.ndarray] = None,
        noise_cov:  Optional[np.ndarray] = None,
        noise_std:  Optional[np.ndarray | float] = None,
    ) -> "LANSCM":
        """
        Construct a LANSCM from a named edge list.

        Parameters
        ----------
        var_names : list[str]
            In topological order (parents before children).
        edges : list of (parent, child, weight)
        noise_cov : (d, d) array, optional
            Takes priority over noise_std if both are given.
        noise_std : scalar or (d,) array, optional
            Standard deviations — squared internally: noise_cov = diag(noise_std²).
        """
        d = len(var_names)
        index = {name: i for i, name in enumerate(var_names)}
        W = np.zeros((d, d), dtype=float)
        for parent, child, weight in edges:
            if parent not in index:
                raise KeyError(f"Unknown variable '{parent}'.")
            if child not in index:
                raise KeyError(f"Unknown variable '{child}'.")
            i, j = index[parent], index[child]
            if i >= j:
                raise ValueError(
                    f"Edge {parent!r}→{child!r} violates topological order "
                    f"(index {i} ≥ {j}). List var_names with parents before children."
                )
            W[i, j] = float(weight)

        if noise_cov is None and noise_std is not None:
            std = np.broadcast_to(np.asarray(noise_std, dtype=float), (d,))
            noise_cov = np.diag(std ** 2)

        return cls(W, noise_mean=noise_mean, noise_cov=noise_cov, var_names=list(var_names))

    # ------------------------------------------------------------------
    # Edge mutation
    # ------------------------------------------------------------------

    def set_edge(self, i: int, j: int, value: float) -> "LANSCM":
        """Set W[i,j] = value (requires i < j). Invalidates cached A. Returns self."""
        if not (0 <= i < self.d and 0 <= j < self.d):
            raise IndexError(f"Indices ({i},{j}) out of range for d={self.d}.")
        if i >= j:
            raise ValueError(f"W is strictly upper-triangular: need i < j, got ({i},{j}).")
        self._W[i, j] = float(value)
        self._A = None
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int, seed=None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Draw n endogenous samples, shape (n, d).

        Under interventions, noise for intervened nodes is zeroed and
        fixed values are injected via:  X = U @ A + fixed @ A
        """
        X, _ = self.sample_with_noise(n, seed=seed, rng=rng)
        return X

    def sample_with_noise(
        self,
        n: int,
        seed=None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw n samples and return both endogenous and noise variables.

        Returns
        -------
        X : (n, d) endogenous samples
        U : (n, d) noise samples (columns of intervened nodes are zeroed)

        Useful for optimization: fix U, vary W, recompute X = U @ A_new.
        """
        if n <= 0:
            raise ValueError("n must be positive.")
        rng = _as_rng(seed=seed, rng=rng)
        U = rng.multivariate_normal(self.noise_mean, self.noise_cov, size=n)
        if self._J:
            U[:, self._J] = 0.0
        X = U @ self.A + self._fixed @ self.A
        return X, U

    def bundle(
        self,
        interventions: list[dict],
        n: int,
        seed=None,
    ) -> "SCMBundle":
        """
        Collect everything needed for optimization into a single bundle.

        For each intervention, stores the intervened SCM instance, the
        endogenous samples X, and the noise samples U. Noise samples are
        useful when the optimizer fixes U and recomputes X = U @ A_new
        under a modified propagator without resampling.

        Parameters
        ----------
        interventions : list[dict]
            Each dict passed to .intervene(). Use {} for observational.
        n : int
            Samples per intervention.
        seed : int, optional

        Returns
        -------
        SCMBundle
        """
        rng = np.random.default_rng(seed)
        child_seeds = [int(rng.integers(2**31 - 1)) for _ in interventions]

        intervened_scms: dict[int, LANSCM] = {}
        endogenous_samples: dict[int, np.ndarray] = {}
        noise_samples: dict[int, np.ndarray] = {}

        for i, iv in enumerate(interventions):
            scm_do = self.intervene(iv)
            X, U = scm_do.sample_with_noise(n, seed=child_seeds[i])
            intervened_scms[i] = scm_do
            endogenous_samples[i] = X
            noise_samples[i] = U

        return SCMBundle(
            scm=self,
            interventions=list(interventions),
            intervened_scms=intervened_scms,
            noise_mean=self.noise_mean.copy(),
            noise_cov=self.noise_cov.copy(),
            endogenous_samples=endogenous_samples,
            noise_samples=noise_samples,
            n=n,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Gaussian moments
    # ------------------------------------------------------------------

    def gaussian_joint(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (μ_X, Σ_X) for the full endogenous joint Gaussian.

        μ_X = (μ_U_eff + fixed) @ A
        Σ_X = A.T @ Σ_U_eff @ A

        Intervened nodes have their noise mean and covariance zeroed before
        propagation.
        """
        mu_U    = self.noise_mean.copy()
        Sigma_U = self.noise_cov.copy()
        if self._J:
            mu_U[self._J]    = 0.0
            Sigma_U[self._J, :] = 0.0
            Sigma_U[:, self._J] = 0.0
        mu_X    = (mu_U + self._fixed) @ self.A
        Sigma_X = self.A.T @ Sigma_U @ self.A
        return mu_X, Sigma_X

    def gaussian_marginal(self, vars: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Marginal Gaussian for the given variable indices."""
        idx = np.asarray(vars, dtype=int)
        mu, Sigma = self.gaussian_joint()
        return mu[idx], Sigma[np.ix_(idx, idx)]

    def gaussian_conditional(
        self,
        target: Sequence[int],
        given:  Sequence[int],
        given_vals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Conditional Gaussian X_target | X_given = given_vals.

        μ_{a|b} = μ_a + Σ_ab Σ_bb⁻¹ (given_vals - μ_b)
        Σ_{a|b} = Σ_aa - Σ_ab Σ_bb⁻¹ Σ_ba
        """
        a = np.asarray(target, dtype=int)
        b = np.asarray(given,  dtype=int)
        mu, Sigma = self.gaussian_joint()
        if b.size == 0:
            return mu[a], Sigma[np.ix_(a, a)]
        given_vals = np.asarray(given_vals, dtype=float)
        if given_vals.shape != (b.size,):
            raise ValueError(f"given_vals must have shape ({b.size},).")
        Sigma_aa = Sigma[np.ix_(a, a)]
        Sigma_ab = Sigma[np.ix_(a, b)]
        Sigma_bb = Sigma[np.ix_(b, b)]
        mu_cond    = mu[a] + Sigma_ab @ np.linalg.solve(Sigma_bb, given_vals - mu[b])
        Sigma_cond = Sigma_aa - Sigma_ab @ np.linalg.solve(Sigma_bb, Sigma_ab.T)
        return mu_cond, Sigma_cond

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------

    def _parse_assignment(self, assignment: dict) -> tuple[list[int], list[float]]:
        """Parse {name_or_index: value} → (J_idx, values)."""
        name_to_idx = {name: i for i, name in enumerate(self.var_names)}
        J_idx, values = [], []
        for key, val in assignment.items():
            if isinstance(key, str):
                if key not in name_to_idx:
                    raise KeyError(f"Unknown variable '{key}'.")
                J_idx.append(name_to_idx[key])
            else:
                idx = int(key)
                if not (0 <= idx < self.d):
                    raise IndexError(f"Variable index {idx} out of range for d={self.d}.")
                J_idx.append(idx)
            values.append(float(val))
        if len(set(J_idx)) != len(J_idx):
            raise ValueError("Duplicate nodes in intervention assignment.")
        return J_idx, values

    def intervene(self, assignment: dict) -> "LANSCM":
        """
        Return a new LANSCM representing do(X_J = values).

        Parameters
        ----------
        assignment : dict
            Maps node name (str) or index (int) to do-value (float).
            Empty dict → returns a fresh observational copy (no intervention).

        Returns
        -------
        LANSCM — always a new object, never self.
        """
        J_idx, values = self._parse_assignment(assignment)
        W_do = self._W.copy()
        if J_idx:
            W_do[:, J_idx] = 0.0
        fixed = np.zeros(self.d, dtype=float)
        for j, v in zip(J_idx, values):
            fixed[j] = v
        return LANSCM(
            W=W_do,
            noise_mean=self.noise_mean.copy(),
            noise_cov=self.noise_cov.copy(),
            var_names=list(self.var_names),
            _J=J_idx,
            _fixed=fixed,
        )

    # ------------------------------------------------------------------
    # Mechanism perturbations
    # ------------------------------------------------------------------

    def perturb(self, delta_W: np.ndarray, mask: Optional[np.ndarray] = None) -> "LANSCM":
        """Return a new LANSCM with W' = W + (mask ⊙ ΔW). Noise is inherited."""
        dW = _strictly_upper(np.asarray(delta_W, dtype=float))
        if dW.shape != self._W.shape:
            raise ValueError(f"delta_W shape {dW.shape} does not match W shape {self._W.shape}.")
        if mask is not None:
            dW = dW * _strictly_upper(np.asarray(mask, dtype=float))
        return LANSCM(
            W=self._W + dW,
            noise_mean=self.noise_mean.copy(),
            noise_cov=self.noise_cov.copy(),
            var_names=list(self.var_names),
        )

    def amplification(self, delta_W: np.ndarray, norm: str = "spectral") -> float:
        """‖A ΔW‖ — measures how ΔW is amplified through the graph."""
        dW = _strictly_upper(np.asarray(delta_W, dtype=float))
        return _matrix_norm(self.A @ dW, norm=norm)

    def propagator_deviation(self, delta_W: np.ndarray, norm: str = "spectral") -> float:
        """Exact ‖A' - A‖ after perturbing W by ΔW."""
        return _matrix_norm(self.perturb(delta_W).A - self.A, norm=norm)

    def finite_poly_bound(self, delta_W: np.ndarray, norm: str = "spectral") -> float:
        """Upper bound: ‖A‖ · Σ_{k=1}^{d-1} δ^k,  δ = ‖A ΔW‖."""
        delta = self.amplification(delta_W, norm=norm)
        return _matrix_norm(self.A, norm=norm) * sum(delta ** k for k in range(1, self.d))

    def neumann_bound(self, delta_W: np.ndarray, norm: str = "spectral") -> Optional[float]:
        """(δ/(1-δ)) · ‖A‖ when δ < 1, else None."""
        delta = self.amplification(delta_W, norm=norm)
        if delta >= 1.0:
            return None
        return (delta / (1.0 - delta)) * _matrix_norm(self.A, norm=norm)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        edges = ", ".join(
            f"{self.var_names[i]}→{self.var_names[j]}({self._W[i,j]:.3f})"
            for i in range(self.d) for j in range(i+1, self.d)
            if self._W[i, j] != 0
        ) or "none"
        s = f"LANSCM(d={self.d}, edges=[{edges}])"
        if self._J:
            do_str = ", ".join(f"{self.var_names[j]}={self._fixed[j]:.3f}" for j in self._J)
            s += f" | do({do_str})"
        return s


# ---------------------------------------------------------------------------
# Single-SCM data bundle (for optimization)
# ---------------------------------------------------------------------------

@dataclass
class SCMBundle:
    """
    Everything needed for optimization, captured from a single LANSCM
    at a fixed set of interventions.

    Fields
    ------
    scm : LANSCM
        Base (observational) SCM. W and A are accessible via scm.W / scm.A.
    interventions : list[dict]
        The canonical intervention list used to build this bundle.
    intervened_scms : dict[int, LANSCM]
        Instantiated intervened SCM for each intervention index.
        intervened_scms[i] = scm.intervene(interventions[i]).
    noise_mean : (d,) array
        Exogenous noise mean of the base SCM.
    noise_cov : (d, d) array
        Exogenous noise covariance of the base SCM.
    endogenous_samples : dict[int, np.ndarray]
        X samples for each intervention index, shape (n, d).
    noise_samples : dict[int, np.ndarray]
        U samples for each intervention index, shape (n, d).
        Columns of intervened nodes are zeroed. Useful for optimizers
        that fix U and recompute X = U @ A_new under a modified W.
    n : int
        Samples per intervention.
    seed : int or None
        Seed used to generate all samples.
    """
    scm:                LANSCM
    interventions:      list[dict]
    intervened_scms:    dict[int, "LANSCM"]
    noise_mean:         np.ndarray
    noise_cov:          np.ndarray
    endogenous_samples: dict[int, np.ndarray]
    noise_samples:      dict[int, np.ndarray]
    n:                  int
    seed:               Optional[int] = None
    districts:          Optional[list] = None   # [[0,1],[2]] — district index lists
    sel_nodes:          Optional[list] = None   # [1] — shifted node indices

    # Convenience accessors so callers don't need to go through .scm
    @property
    def W(self) -> np.ndarray:
        """Structural matrix of the base SCM (read-only copy)."""
        return self.scm.W

    @property
    def A(self) -> np.ndarray:
        """Propagator of the base SCM (cached)."""
        return self.scm.A

    @property
    def d(self) -> int:
        return self.scm.d

    def n_interventions(self) -> int:
        return len(self.interventions)

    def __repr__(self) -> str:
        return (
            f"SCMBundle(d={self.d}, n_interventions={self.n_interventions()}, "
            f"n={self.n}, scm={self.scm!r})"
        )


# ---------------------------------------------------------------------------
# Source–target pair
# ---------------------------------------------------------------------------

class LANPair:
    """
    A source/target pair of LANSCMs with the same graph structure.

    Parameters
    ----------
    source, target : LANSCM
    selection_mask : (d, d) binary array, optional
        Encodes which entries of ΔW are allowed to differ (selection diagram).
    """

    def __init__(
        self,
        source: LANSCM,
        target: LANSCM,
        selection_mask: Optional[np.ndarray] = None,
    ):
        if source.d != target.d:
            raise ValueError("source and target must have the same dimension.")
        self.source = source
        self.target = target
        if selection_mask is not None:
            selection_mask = _strictly_upper(np.asarray(selection_mask, dtype=float))
            if selection_mask.shape != (source.d, source.d):
                raise ValueError(f"selection_mask must have shape ({source.d},{source.d}).")
        self.selection_mask = selection_mask

    @property
    def delta_W(self) -> np.ndarray:
        """Target minus source structural matrix."""
        return self.target._W - self.source._W

    def check_mask(self) -> bool:
        """Verify ΔW respects the selection mask. Raises ValueError if not."""
        if self.selection_mask is None:
            return True
        dW = _strictly_upper(self.delta_W)
        if not np.allclose(dW, dW * self.selection_mask):
            raise ValueError("delta_W has non-zero entries outside the selection mask.")
        return True

    def propagator_deviation(self, norm: str = "spectral") -> float:
        """‖A_target - A_source‖."""
        return _matrix_norm(self.target.A - self.source.A, norm=norm)

    def amplification(self, norm: str = "spectral") -> float:
        """‖A_source ΔW‖."""
        return self.source.amplification(self.delta_W, norm=norm)

    def paired_dataset(
        self,
        interventions: list[dict],
        n: int,
        seed: Optional[int] = None,
    ) -> "LANDataset":
        """
        Sample source and target data for each intervention.

        Parameters
        ----------
        interventions : list[dict]
            Each dict is passed to .intervene(). Use {} for observational.
        n : int
            Samples per intervention per domain.
        seed : int, optional
        """
        rng     = np.random.default_rng(seed)
        src_seed = int(rng.integers(2**31 - 1))
        tgt_seed = int(rng.integers(2**31 - 1))
        rng_src  = np.random.default_rng(src_seed)
        rng_tgt  = np.random.default_rng(tgt_seed)
        src_child_seeds = [int(rng_src.integers(2**31 - 1)) for _ in interventions]
        tgt_child_seeds = [int(rng_tgt.integers(2**31 - 1)) for _ in interventions]

        src_samples, tgt_samples = {}, {}
        src_gaussian, tgt_gaussian = {}, {}

        for i, iv in enumerate(interventions):
            src_do = self.source.intervene(iv)
            tgt_do = self.target.intervene(iv)
            src_samples[i]  = src_do.sample(n, seed=src_child_seeds[i])
            tgt_samples[i]  = tgt_do.sample(n, seed=tgt_child_seeds[i])
            src_gaussian[i] = src_do.gaussian_joint()
            tgt_gaussian[i] = tgt_do.gaussian_joint()

        return LANDataset(
            source=self.source,
            target=self.target,
            interventions=list(interventions),
            source_samples=src_samples,
            target_samples=tgt_samples,
            source_gaussian=src_gaussian,
            target_gaussian=tgt_gaussian,
            delta_W=self.delta_W.copy(),
            selection_mask=None if self.selection_mask is None else self.selection_mask.copy(),
        )

    def __repr__(self) -> str:
        return (
            f"LANPair(d={self.source.d}, "
            f"‖ΔW‖_F={np.linalg.norm(self.delta_W, ord='fro'):.4f}, "
            f"‖A_t-A_s‖={self.propagator_deviation():.4f})"
        )


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class LANDataset:
    """Paired source/target samples and Gaussian moments across interventions."""
    source:   LANSCM
    target:   LANSCM
    interventions:   list[dict]
    source_samples:  dict[int, np.ndarray]   # {interv_idx: (n, d)}
    target_samples:  dict[int, np.ndarray]
    source_gaussian: dict[int, tuple]        # {interv_idx: (μ, Σ)}
    target_gaussian: dict[int, tuple]
    delta_W:         np.ndarray
    selection_mask:  Optional[np.ndarray] = None

    def n_interventions(self) -> int:
        return len(self.interventions)


# ---------------------------------------------------------------------------
# Benchmark container
# ---------------------------------------------------------------------------

class Benchmark(NamedTuple):
    """SCM paired with its canonical intervention list."""
    scm:           LANSCM
    interventions: list[dict]


# ---------------------------------------------------------------------------
# Generic factories
# ---------------------------------------------------------------------------

def make_random_scm(
    d: int,
    sparsity: float = 0.5,
    weight_range: tuple[float, float] = (0.1, 0.9),
    noise_std: float = 1.0,
    seed=None,
    var_names: Optional[Sequence[str]] = None,
) -> LANSCM:
    """Random acyclic LAN SCM with strictly upper-triangular W."""
    rng = np.random.default_rng(seed)
    W = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < sparsity:
                sign = float(rng.choice([-1.0, 1.0]))
                W[i, j] = sign * rng.uniform(*weight_range)
    return LANSCM(W=W, noise_cov=np.diag([noise_std ** 2] * d), var_names=var_names)


def make_perturbation(
    source: LANSCM,
    selection_mask: Optional[np.ndarray] = None,
    constraint: str = "frobenius",
    budget: float | np.ndarray = 0.5,
    seed=None,
) -> tuple[LANSCM, np.ndarray]:
    """
    Generate an admissible ΔW and return (target_scm, delta_W).

    Parameters
    ----------
    constraint : {'frobenius', 'row', 'elementwise'}
    budget : scalar or array
        'frobenius'   → ‖ΔW‖_F ≤ budget
        'row'         → per-row L1 budget (scalar broadcast or (d,) array)
        'elementwise' → per-entry absolute bound (scalar or (d,d) array)
    """
    rng = np.random.default_rng(seed)
    d   = source.d
    raw = _strictly_upper(rng.standard_normal((d, d)))
    if selection_mask is not None:
        raw *= _strictly_upper(np.asarray(selection_mask, dtype=float))

    if constraint == "frobenius":
        b  = float(np.asarray(budget))
        fn = np.linalg.norm(raw, ord="fro")
        delta_W = raw if fn <= 1e-12 else raw * (b / fn)

    elif constraint == "row":
        budgets = np.broadcast_to(np.asarray(budget, dtype=float), (d,)).copy()
        delta_W = raw.copy()
        for i in range(d):
            l1 = np.sum(np.abs(delta_W[i]))
            if l1 > 1e-12:
                delta_W[i] *= budgets[i] / l1

    elif constraint == "elementwise":
        b = np.asarray(budget, dtype=float)
        delta_W = _strictly_upper(np.clip(raw, -b, b))
        if selection_mask is not None:
            delta_W *= _strictly_upper(np.asarray(selection_mask, dtype=float))

    else:
        raise ValueError(f"Unknown constraint '{constraint}'. Choose: frobenius, row, elementwise.")

    return source.perturb(delta_W), delta_W


# ---------------------------------------------------------------------------
# Benchmark factories
# ---------------------------------------------------------------------------

def make_lilucas(
    w_smoking_lungcancer:  float = 0.9,
    w_genetics_lungcancer: float = 0.8,
    w_allergy_coughing:    float = 0.4,
    w_lungcancer_coughing: float = 0.6,
    w_lungcancer_fatigue:  float = 0.9,
    w_coughing_fatigue:    float = 0.5,
    noise_mean: Optional[Sequence[float]] = None,
    noise_std:  Optional[Sequence[float]] = None,
) -> Benchmark:
    """
    LiLuCaS — 6-variable lung-cancer DAG.

    Variable order (topological):
        Smoking(0), Genetics(1), Allergy(2), LungCancer(3), Coughing(4), Fatigue(5)

    Defaults from lilucas_config.yaml.
    noise_std values are standard deviations — squared internally into noise_cov.
    """
    _noise_mean = [0.0, 0.0, 0.1, 0.1, 0.3, 0.2] if noise_mean is None else list(noise_mean)
    _noise_std  = [0.5, 2.0, 1.5, 1.0, 0.8, 1.2] if noise_std  is None else list(noise_std)
    scm = LANSCM.from_edges(
        var_names=["Smoking", "Genetics", "Allergy", "LungCancer", "Coughing", "Fatigue"],
        edges=[
            ("Smoking",    "LungCancer", w_smoking_lungcancer),
            ("Genetics",   "LungCancer", w_genetics_lungcancer),
            ("Allergy",    "Coughing",   w_allergy_coughing),
            ("LungCancer", "Coughing",   w_lungcancer_coughing),
            ("LungCancer", "Fatigue",    w_lungcancer_fatigue),
            ("Coughing",   "Fatigue",    w_coughing_fatigue),
        ],
        noise_mean=np.array(_noise_mean, dtype=float),
        noise_std=np.array(_noise_std,  dtype=float),
    )
    interventions = [
        {},
        {"Smoking": 0}, {"Smoking": 1},
        {"LungCancer": 0}, {"LungCancer": 1},
        {"Smoking": 0, "LungCancer": 0}, {"Smoking": 1, "LungCancer": 1},
        {"Smoking": 0, "LungCancer": 1}, {"Smoking": 1, "LungCancer": 0},
        {"Genetics": 0}, {"Genetics": 1},
        {"Genetics": 0, "Smoking": 0}, {"Genetics": 1, "Smoking": 1},
        {"Genetics": 0, "Smoking": 1}, {"Genetics": 1, "Smoking": 0},
        {"Allergy": 0}, {"Allergy": 1},
        {"Coughing": 0},
        {"Coughing": 1, "Fatigue": 1}, {"Coughing": 1, "Fatigue": 0},
        {"Coughing": 0, "Fatigue": 1}
    ]
    return Benchmark(scm, interventions)


def make_lilucas_light(
    w_smoking_lungcancer:  float = 0.9,
    w_genetics_lungcancer: float = 0.8,
    w_allergy_coughing:    float = 0.4,
    w_lungcancer_coughing: float = 0.6,
    w_lungcancer_fatigue:  float = 0.9,
    w_coughing_fatigue:    float = 0.5,
    noise_mean: Optional[Sequence[float]] = None,
    noise_std:  Optional[Sequence[float]] = None,
    interventions: Optional[list] = None,
) -> Benchmark:
    """
    LiLuCaS Light — same SCM as make_lilucas(), but with a smaller
    intervention set for fast iteration.  This is the default LiLuCaS
    variant for all paper experiments.

    Default interventions (9, down from 21):
        observational + single-node do() on Smoking and Genetics
        + all four joint (Smoking, Genetics) combinations.

    Parameters
    ----------
    interventions : list of dicts, optional
        Override the default 9-intervention set with any subset of the
        full make_lilucas() interventions. If None, the default 9 are used.

    All other parameters are identical to make_lilucas().
    """
    bench = make_lilucas(
        w_smoking_lungcancer=w_smoking_lungcancer,
        w_genetics_lungcancer=w_genetics_lungcancer,
        w_allergy_coughing=w_allergy_coughing,
        w_lungcancer_coughing=w_lungcancer_coughing,
        w_lungcancer_fatigue=w_lungcancer_fatigue,
        w_coughing_fatigue=w_coughing_fatigue,
        noise_mean=noise_mean,
        noise_std=noise_std,
    )
    if interventions is None:
        interventions = [
        {},
        {"Smoking": 0}, {"Smoking": 1},
        {"Genetics": 0}, {"Genetics": 1},
        {"Genetics": 0, "Smoking": 0},
        {"Genetics": 1, "Smoking": 1},
        {"Genetics": 0, "Smoking": 1},
        {"Genetics": 1, "Smoking": 0}
    ]
    return Benchmark(bench.scm, interventions)


def make_atce(
    alpha: float = 0.3,
    beta:  float = 1.0,
    gamma: float = 0.5,
    mu_Z:  float = 0.5,
    mu_X:  float = 0.3,
    mu_Y:  float = 0.4,
    noise_std: float = 1.0,
) -> Benchmark:
    """
    ATCE — Age → Treatment → Causal Effect  (Z → X, Z → Y, X → Y).

    Structural equations (strictly linear additive-noise):
        Z = U_Z,        U_Z ~ N(mu_Z, noise_std²)
        X = alpha·Z + U_X,  U_X ~ N(mu_X, noise_std²)
        Y = beta·X + gamma·Z + U_Y,  U_Y ~ N(mu_Y, noise_std²)

    W[Z,X]=alpha, W[Z,Y]=gamma, W[X,Y]=beta.
    noise_mean = [mu_Z, mu_X, mu_Y];  noise_std is a standard deviation,
    squared internally: noise_cov = diag([noise_std**2] * 3).

    Inspired by the Pearl–Bareinboim LA/NYC transport example.
    LA and NYC correspond to different values of mu_Z; they are not encoded
    here — this is a single SCM.
    """
    scm = LANSCM.from_edges(
        var_names=["Z", "X", "Y"],
        edges=[
            ("Z", "X", alpha),
            ("Z", "Y", gamma),
            ("X", "Y", beta),
        ],
        noise_mean=np.array([mu_Z, mu_X, mu_Y], dtype=float),
        noise_std=noise_std,
    )
    interventions = [
        {},
        {"X": 0},
        {"X": 1},
    ]
    return Benchmark(scm, interventions)


def make_ate(
    beta:    float = 1.0,
    mu_X:    float = 0.5,
    mu_Y:    float = 0.3,
    sigma_X: float = 1.0,
    sigma_Y: float = 1.0,
    rho:     float = 0.6,
) -> Benchmark:
    """
    ATE — Average Treatment Effect benchmark.

    ADMG: X → Y, X ↔ Y, S → Y

    Structural equations:
        X = U_X,      U_X ~ N(mu_X, σ_X²)
        Y = beta·X + U_Y,  U_Y ~ N(mu_Y, σ_Y²)
    Noise covariance (non-diagonal — models the latent confounder X ↔ Y):
        Σ = [[σ_X², ρ·σ_X·σ_Y],
             [ρ·σ_X·σ_Y, σ_Y²]]

    Shift pattern: Y's mechanism (beta) differs between source and target.
    Districts: {X, Y} (one district — both in same c.c. of bidirected graph).
    """
    noise_cov = np.array([
        [sigma_X ** 2,          rho * sigma_X * sigma_Y],
        [rho * sigma_X * sigma_Y, sigma_Y ** 2         ],
    ], dtype=float)
    scm = LANSCM.from_edges(
        var_names=["X", "Y"],
        edges=[("X", "Y", beta)],
        noise_mean=np.array([mu_X, mu_Y], dtype=float),
        noise_cov=noise_cov,
    )
    interventions = [
        {},
        {"X": 0.0},
        {"X": 1.0},
    ]
    return Benchmark(scm, interventions)


# ---------------------------------------------------------------------------
# Config → bundle (minimal CLI)
# ---------------------------------------------------------------------------

def _scm_from_config(cfg: dict) -> LANSCM:
    scm_cfg = cfg["scm"]
    var_names = list(scm_cfg["var_names"])
    edges = [tuple(edge) for edge in scm_cfg["edges"]]
    noise_mean = scm_cfg.get("noise_mean")
    noise_cov = scm_cfg.get("noise_cov")
    noise_std = scm_cfg.get("noise_std")
    noise_corr = scm_cfg.get("noise_corr")

    # Build full covariance matrix from noise_std + noise_corr (off-diagonal correlations)
    if noise_corr is not None and noise_cov is None:
        d = len(var_names)
        std_arr = np.broadcast_to(np.asarray(noise_std, dtype=float), (d,)).copy()
        cov = np.diag(std_arr ** 2)
        for entry in noise_corr:
            i, j, rho = int(entry[0]), int(entry[1]), float(entry[2])
            cov[i, j] = rho * std_arr[i] * std_arr[j]
            cov[j, i] = rho * std_arr[i] * std_arr[j]
        noise_cov = cov
        noise_std = None  # already encoded in cov

    return LANSCM.from_edges(
        var_names=var_names,
        edges=edges,
        noise_mean=None if noise_mean is None else np.asarray(noise_mean, dtype=float),
        noise_cov=None if noise_cov is None else np.asarray(noise_cov, dtype=float),
        noise_std=noise_std,
    )


def _save_bundle_from_config(config_path: Path, out_root: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ensure pickles reference lan_scm classes even when run as a script.
    sys.modules["lan_scm"] = sys.modules[__name__]
    SCMBundle.__module__ = "lan_scm"
    LANSCM.__module__ = "lan_scm"

    scm = _scm_from_config(cfg)
    interventions = list(cfg["interventions"])
    n = int(cfg["n_samples"])
    seed = cfg.get("seed")
    bundle = scm.bundle(interventions=interventions, n=n, seed=seed)

    # Attach semi-Markovian metadata from the YAML if present
    scm_cfg = cfg["scm"]
    bundle.districts = scm_cfg.get("districts", None)
    bundle.sel_nodes  = scm_cfg.get("sel_nodes",  None)

    dataset = cfg["name"]
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bundle.pkl"
    joblib.dump(bundle, out_path)

    # Create and save CV folds alongside the bundle so they are fixed at data-creation time.
    try:
        from sklearn.model_selection import KFold
        obs_idx = bundle.interventions.index({})
        U_s = bundle.noise_samples[obs_idx]
        k_folds  = int(cfg.get("k_folds",  5))
        fold_seed = int(cfg.get("fold_seed", 42))
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=fold_seed)
        folds = [
            {"train": train_idx, "test": test_idx}
            for train_idx, test_idx in kf.split(U_s)
        ]
        folds_path = out_dir / "cv_folds.pkl"
        joblib.dump(folds, folds_path)
        print(f"Saved {k_folds}-fold split to {folds_path}")
    except Exception as exc:
        print(f"Warning: could not create CV folds: {exc}")

    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save an SCM bundle from a dataset config."
    )
    parser.add_argument("config", type=str, help="Path to a dataset YAML config.")
    parser.add_argument(
        "--out_root",
        type=str,
        default="data",
        help="Output root directory for bundles (default: data).",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out_path = _save_bundle_from_config(Path(args.config), Path(args.out_root))
    print(f"Saved bundle to {out_path}")
