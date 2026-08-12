"""
Parametric shift library for building pseudo-targets from a source SCM.

apply_shift(scm, shift_type, magnitude, ...) returns a new LANSCM with one
structural parameter perturbed.  The original SCM is never modified.

Shift types: mechanism_edge (perturb W[i,j]), noise_mean (shift exogenous
mean at node k), noise_std (scale std, variance ~ (1+d)^2), noise_cov
(scale variance directly).  All shifts are Gaussian-parametric.

The magnitude d is exogenous — the endogenous effect is d propagated
through the SCM's propagator A, which is generally smaller.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Avoid circular import at module level; LANSCM is only needed at runtime.
from lan_scm import LANSCM


# ---------------------------------------------------------------------------
# Shift registry
# ---------------------------------------------------------------------------

SHIFT_TYPES = ("mechanism_edge", "noise_mean", "noise_std", "noise_cov")


def apply_shift(
    scm: LANSCM,
    shift_type: str,
    magnitude: float,
    *,
    edge: tuple[int, int] | None = None,
    node: int | None = None,
) -> LANSCM:
    """Return a new LANSCM with one parameter shifted.

    Parameters
    ----------
    scm : source LANSCM (not modified)
    shift_type : one of "mechanism_edge", "noise_mean", "noise_std", "noise_cov"
    magnitude : δ — the shift amount (additive or multiplicative depending on type)
    edge : (i, j) for "mechanism_edge" shifts
    node : k for "noise_mean", "noise_std", "noise_cov" shifts

    Returns
    -------
    LANSCM — a new SCM with the shift applied

    Raises
    ------
    ValueError : invalid shift_type, missing/invalid element, structural violation
    """
    if shift_type not in SHIFT_TYPES:
        raise ValueError(
            f"Unknown shift_type {shift_type!r}. "
            f"Supported: {SHIFT_TYPES}"
        )

    if shift_type == "mechanism_edge":
        return _shift_mechanism_edge(scm, magnitude, edge)
    else:
        return _shift_noise(scm, shift_type, magnitude, node)


# ---------------------------------------------------------------------------
# Mechanism edge shift
# ---------------------------------------------------------------------------

def _shift_mechanism_edge(
    scm: LANSCM,
    delta: float,
    edge: tuple[int, int] | None,
) -> LANSCM:
    """W_t[i,j] = W_s[i,j] + δ.  Only existing edges (nonzero W[i,j]) allowed."""
    if edge is None:
        raise ValueError("mechanism_edge shift requires edge=(i, j)")

    i, j = edge
    d = scm.d
    if not (0 <= i < d and 0 <= j < d):
        raise ValueError(f"edge ({i}, {j}) out of range for d={d}")
    if i >= j:
        raise ValueError(
            f"edge ({i}, {j}) violates DAG order (need i < j for strict "
            f"upper-triangular W)"
        )

    W = scm.W  # read-only copy
    if W[i, j] == 0.0:
        raise ValueError(
            f"W[{i},{j}] = 0 in the source SCM — this is a structural zero "
            f"(no edge {scm.var_names[i]}→{scm.var_names[j]}). "
            f"Perturbing a structural zero would create a new causal edge. "
            f"Only existing edges can be shifted."
        )

    W_new = W.copy()
    W_new[i, j] += delta

    return LANSCM(
        W=W_new,
        noise_mean=scm.noise_mean.copy(),
        noise_cov=scm.noise_cov.copy(),
        var_names=list(scm.var_names),
    )


# ---------------------------------------------------------------------------
# Noise shifts
# ---------------------------------------------------------------------------

def _shift_noise(
    scm: LANSCM,
    shift_type: str,
    delta: float,
    node: int | None,
) -> LANSCM:
    """Shift a noise parameter at node k."""
    if node is None:
        raise ValueError(f"{shift_type} shift requires node=k")

    d = scm.d
    if not (0 <= node < d):
        raise ValueError(f"node {node} out of range for d={d}")

    W = scm.W
    noise_mean = scm.noise_mean.copy()
    noise_cov = scm.noise_cov.copy()

    if shift_type == "noise_mean":
        noise_mean[node] += delta

    elif shift_type == "noise_std":
        # Scale standard deviation: std_new = std_old * (1 + δ)
        # Variance: cov[k,k]_new = cov[k,k]_old * (1 + δ)²
        scale = 1.0 + delta
        if scale < 0:
            raise ValueError(
                f"noise_std shift with δ={delta} gives scale={scale} < 0 "
                f"(standard deviation cannot be negative)"
            )
        noise_cov[node, node] *= scale ** 2
        # Scale off-diagonal entries involving this node to preserve correlation structure
        for k in range(d):
            if k != node:
                noise_cov[node, k] *= scale
                noise_cov[k, node] *= scale

    elif shift_type == "noise_cov":
        # Scale variance directly: cov[k,k]_new = cov[k,k]_old * (1 + δ)
        scale = 1.0 + delta
        if scale < 0:
            raise ValueError(
                f"noise_cov shift with δ={delta} gives scale={scale} < 0 "
                f"(variance cannot be negative)"
            )
        noise_cov[node, node] *= scale
        # Scale off-diagonal to preserve correlation: corr stays, cov scales by sqrt
        sqrt_scale = np.sqrt(scale)
        for k in range(d):
            if k != node:
                noise_cov[node, k] *= sqrt_scale
                noise_cov[k, node] *= sqrt_scale

    return LANSCM(
        W=W,
        noise_mean=noise_mean,
        noise_cov=noise_cov,
        var_names=list(scm.var_names),
    )


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def list_edges(scm: LANSCM) -> list[tuple[int, int, float]]:
    """Return all nonzero edges as (i, j, weight) tuples."""
    W = scm.W
    d = scm.d
    edges = []
    for i in range(d):
        for j in range(i + 1, d):
            if W[i, j] != 0.0:
                edges.append((i, j, float(W[i, j])))
    return edges


def describe_shift(
    scm: LANSCM,
    shift_type: str,
    magnitude: float,
    *,
    edge: tuple[int, int] | None = None,
    node: int | None = None,
) -> str:
    """Human-readable description of a shift."""
    vn = scm.var_names
    if shift_type == "mechanism_edge":
        i, j = edge
        w_old = scm.W[i, j]
        return (f"mechanism_edge ({vn[i]}→{vn[j]}): "
                f"W[{i},{j}] = {w_old:.4f} → {w_old + magnitude:.4f} (δ={magnitude})")
    elif shift_type == "noise_mean":
        old = scm.noise_mean[node]
        return (f"noise_mean ({vn[node]}): "
                f"μ[{node}] = {old:.4f} → {old + magnitude:.4f} (δ={magnitude})")
    elif shift_type == "noise_std":
        old_var = scm.noise_cov[node, node]
        old_std = np.sqrt(old_var)
        new_std = old_std * (1 + magnitude)
        return (f"noise_std ({vn[node]}): "
                f"σ[{node}] = {old_std:.4f} → {new_std:.4f} (δ={magnitude})")
    elif shift_type == "noise_cov":
        old_var = scm.noise_cov[node, node]
        new_var = old_var * (1 + magnitude)
        return (f"noise_cov ({vn[node]}): "
                f"Σ[{node},{node}] = {old_var:.4f} → {new_var:.4f} (δ={magnitude})")
    return f"{shift_type}(δ={magnitude})"
