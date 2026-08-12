"""
ConstructiveClass: specifies the structure of admissible transport maps τ.

Markovian (all districts = singletons):
    τ is diagonal. Invariant nodes have τ[i,i] = 1 (fixed). Shifted nodes
    have τ[i,i] free.

Semi-Markovian (non-singleton districts):
    τ is block-diagonal. Invariant districts have their block = I (fixed).
    Shifted districts (D_k ∩ shifted_nodes ≠ ∅) have their block free.
    Cross-district entries are always zero.

Row-vector convention: X_target = X_source @ τ.T  (τ acts on columns).
Actually: τ maps source samples row-wise, so the operation is
    X_t_pushed = X_s @ τ    when τ is d×d.

The project(tau) method enforces the structural constraint in-place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Districts helper (centralized — single source of truth)
# ---------------------------------------------------------------------------

def resolve_districts(
    d: int,
    districts: Iterable[Iterable[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    """Resolve a district specification to a canonical tuple-of-tuples.

    Parameters
    ----------
    d : int
        Ambient dimension.
    districts : iterable of iterables of int, or None
        If None, returns all-singleton partition: ((0,), (1,), ..., (d-1,)).

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Ordered partition of {0, ..., d-1}.
    """
    if districts is None:
        return tuple((i,) for i in range(d))
    canon = tuple(tuple(int(x) for x in D) for D in districts)
    # Validate: partition must cover {0,...,d-1} exactly
    covered = sorted(x for D in canon for x in D)
    if covered != list(range(d)):
        raise ValueError(
            f"districts must be a partition of {{0,...,{d-1}}}; got {canon}"
        )
    return canon


# ---------------------------------------------------------------------------
# ConstructiveClass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstructiveClass:
    """Specifies the constructive structure of admissible transport maps τ.

    Markovian case: each district is a singleton {i}, so τ is diagonal
    with τ[i,i] = 1 for invariant i and τ[i,i] free for shifted i.

    Semi-Markovian case: districts are non-singleton sets and τ is
    block-diagonal with one block per district.  Invariant districts
    (D_k ∩ shifted_nodes = ∅) have their block fixed to I; shifted
    districts have their block free.

    Parameters
    ----------
    d : int
        Ambient dimension.
    districts : tuple[tuple[int, ...], ...]
        Ordered partition of {0, ..., d-1}.
    shifted_nodes : tuple[int, ...]
        K, the set of shifted-node indices.
    """
    d: int
    districts: tuple[tuple[int, ...], ...]
    shifted_nodes: tuple[int, ...]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_markovian(self) -> bool:
        """True iff all districts are singletons."""
        return all(len(D) == 1 for D in self.districts)

    def shifted_districts(self) -> tuple[tuple[int, ...], ...]:
        """Return districts that contain at least one shifted node."""
        sn = set(self.shifted_nodes)
        return tuple(D for D in self.districts if set(D) & sn)

    def invariant_districts(self) -> tuple[tuple[int, ...], ...]:
        """Return districts with no shifted nodes."""
        sn = set(self.shifted_nodes)
        return tuple(D for D in self.districts if not (set(D) & sn))

    # ------------------------------------------------------------------
    # Block mask
    # ------------------------------------------------------------------

    def block_mask(self) -> np.ndarray:
        """(d, d) 0/1 mask of admissible non-zero entries of τ.

        Entry (i, j) is 1 iff i and j belong to the same district AND
        that district is a shifted district (or i == j and i is invariant,
        but invariant diagonal entries are fixed to 1, not optimized).

        In practice: 1 for any (i, j) in the same district. The project()
        method handles fixing invariant blocks to I separately.
        """
        mask = np.zeros((self.d, self.d), dtype=float)
        for D in self.districts:
            for i in D:
                for j in D:
                    mask[i, j] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, tau: np.ndarray) -> np.ndarray:
        """Project a (d, d) matrix onto the constructive class.

        Steps:
        1. Zero out all cross-district entries.
        2. For invariant districts (D_k ∩ shifted_nodes = ∅):
           replace the block with the identity of that size.
        3. Shifted-district blocks are left as-is.

        Parameters
        ----------
        tau : (d, d) array

        Returns
        -------
        (d, d) array in the constructive class
        """
        tau = np.asarray(tau, dtype=float).copy()
        sn = set(self.shifted_nodes)

        # Step 1: zero everything outside districts
        result = np.zeros_like(tau)
        for D in self.districts:
            for i in D:
                for j in D:
                    result[i, j] = tau[i, j]

        # Step 2: fix invariant-district blocks to identity
        for D in self.invariant_districts():
            for ii, i in enumerate(D):
                for jj, j in enumerate(D):
                    result[i, j] = 1.0 if i == j else 0.0

        return result

    def init_tau(self, mode: str = "identity", rng=None) -> np.ndarray:
        """Return an initialized transport map in the constructive class.

        Parameters
        ----------
        mode : "identity" | "zeros" | "random"
            Initialization mode for the free (shifted) entries.
            Invariant entries are always fixed to 1 by project().
            "identity" (default): all entries start at 1 (no-op transport).
            "zeros": shifted entries start at 0.
            "random": shifted entries start at standard-normal draws.
        rng : int, np.random.Generator, or None
            Seed or generator for "random" mode. Ignored for other modes.

        Returns
        -------
        (d, d) array in the constructive class.
        """
        if mode == "identity":
            return self.project(np.eye(self.d))
        elif mode == "zeros":
            return self.project(np.zeros((self.d, self.d)))
        elif mode == "random":
            rng = np.random.default_rng(rng)
            return self.project(rng.standard_normal((self.d, self.d)))
        else:
            raise ValueError(
                f"Unknown tau_init mode: {mode!r}. "
                "Expected 'identity', 'zeros', or 'random'."
            )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def markovian(
        cls,
        d: int,
        shifted: Iterable[int],
    ) -> "ConstructiveClass":
        """Create a Markovian constructive class (all singletons).

        Parameters
        ----------
        d : int
        shifted : iterable of int
            Indices of shifted nodes.
        """
        return cls(
            d=d,
            districts=tuple((i,) for i in range(d)),
            shifted_nodes=tuple(sorted(shifted)),
        )

    @classmethod
    def from_districts(
        cls,
        d: int,
        districts: Iterable[Iterable[int]],
        shifted: Iterable[int],
    ) -> "ConstructiveClass":
        """Create a semi-Markovian constructive class from explicit districts.

        Parameters
        ----------
        d : int
        districts : iterable of iterables of int
            Must be a partition of {0, ..., d-1}.
        shifted : iterable of int
            Indices of shifted nodes (must be subset of ∪ districts = {0,...,d-1}).
        """
        canon = resolve_districts(d, districts)
        shifted_t = tuple(sorted(int(x) for x in shifted))
        # Validate: every shifted node must be in some district (trivially true if
        # districts cover all of {0,...,d-1})
        all_nodes = set(x for D in canon for x in D)
        for s in shifted_t:
            if s not in all_nodes:
                raise ValueError(f"Shifted node {s} not found in any district.")
        return cls(d=d, districts=canon, shifted_nodes=shifted_t)
