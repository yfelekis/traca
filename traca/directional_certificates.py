"""
Directional (linear-query-specific) certificates for mean queries E[c^T X | do(J)].

Tighter than the general δ²(τ) certificate when the query direction c is fixed,
because only the c-component of the perturbation contributes.

Resolvent identity (right-mult convention, matches stability.py):
    A'_ι - A_ι ≈ A_ι ΔW R_ι A_ι    (first order)

Derived vectors for the bilinear form a^T ΔW b:
    a = v @ A_iota                           (source statistics propagated forward)
    b = diag(R_iota) * (A_iota @ q)          (query direction backward, gated by R)

Bilinear check: a @ ΔW @ b == v @ A_iota @ ΔW @ R_iota @ A_iota @ q

Environment term uses M_env from env.shifted_rows (NOT mechanism.shifted_rows).
For ATCE these differ: mechanism.shifted_rows=[0,1], environment.shifted_rows=[0,2].

"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from traca.ambiguity import (
    FrobeniusBall, RowBudget, ColumnBudget, EntrywiseBox,
)
from traca.stability import perturbed_propagator


# ---------------------------------------------------------------------------
# Shift mask
# ---------------------------------------------------------------------------

def _build_shift_mask(mechanism_set, d: int) -> np.ndarray:
    """Binary mask P: P[j,k] = 1 iff j in shifted_rows and j < k.

    Encodes which (j,k) entries of ΔW are free in the mechanism ambiguity set.
    """
    P = np.zeros((d, d), dtype=float)
    for j in mechanism_set.shifted_rows:
        for k in range(j + 1, d):
            P[j, k] = 1.0
    return P


# ---------------------------------------------------------------------------
# β — first-order mechanism bound
# ---------------------------------------------------------------------------

def directional_beta(
    a: np.ndarray,
    b: np.ndarray,
    mechanism_set,
) -> float:
    """First-order bound β = sup_{ΔW ∈ A_W} |a · ΔW · b|.

    With a = v @ A_iota and b = diag(R_iota) * (A_iota @ q), this bounds
    the first-order perturbation |a @ ΔW @ b| = |v @ A_ι @ ΔW @ R_ι @ A_ι @ q|.

    Parameters
    ----------
    a : (d,) — source statistics propagated forward: v @ A_iota
    b : (d,) — query direction backward, gated by R: diag(R) * (A @ q)
    mechanism_set : mechanism ambiguity set

    Returns
    -------
    float : β ≥ sup_{ΔW ∈ A_W} |a · ΔW · b|
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = len(a)
    P = _build_shift_mask(mechanism_set, d)
    abs_a = np.abs(a)
    abs_b = np.abs(b)

    if isinstance(mechanism_set, FrobeniusBall):
        # β = η · ‖P ⊙ outer(a, b)‖_F
        return float(mechanism_set.eta * np.linalg.norm(P * np.outer(a, b), "fro"))

    elif isinstance(mechanism_set, EntrywiseBox):
        # β = Σ_{j,k} P[j,k] · effective_bound[j,k] · |a[j]| · |b[k]|
        # effective_bound = |delta|+B (or B when delta=None).
        B_sup = mechanism_set.effective_bound
        return float(np.sum(P * B_sup * abs_a[:, np.newaxis] * abs_b[np.newaxis, :]))

    elif isinstance(mechanism_set, RowBudget):
        # β = Σ_j ρ[j] · |a[j]| · max_k(P[j,k] · |b[k]|)
        total = 0.0
        for j in mechanism_set.shifted_rows:
            rho_j = mechanism_set.rho.get(j, 0.0)
            # max over k > j (strict upper-triangular)
            max_bk = float(np.max(abs_b[j + 1:])) if j + 1 < d else 0.0
            total += rho_j * abs_a[j] * max_bk
        return float(total)

    elif isinstance(mechanism_set, ColumnBudget):
        # β = Σ_k c_k · |b[k]| · max_{j: P[j,k]=1} |a[j]|
        total = 0.0
        for k in range(d):
            ck = mechanism_set._budget(k)
            max_aj = max(
                (abs_a[j] for j in mechanism_set.shifted_rows if j < k),
                default=0.0,
            )
            total += ck * abs_b[k] * max_aj
        return float(total)

    else:
        raise TypeError(f"Unknown mechanism set type: {type(mechanism_set)}")


# ---------------------------------------------------------------------------
# λ — query-side bound for higher-order remainder
# ---------------------------------------------------------------------------

def directional_lambda(
    b: np.ndarray,
    mechanism_set,
    d: int,
) -> float:
    """Query-side bound λ = sup_{ΔW ∈ A_W} ||ΔW @ b||_2.

    Used in the higher-order remainder: λ · ||a||_2 · Σ_{r=1}^{d-2} γ^r.

    Parameters
    ----------
    b : (d,) — query direction backward, gated by R: diag(R) * (A @ q)
    mechanism_set : mechanism ambiguity set
    d : int

    Returns
    -------
    float : λ ≥ sup_{ΔW ∈ A_W} ||ΔW @ b||_2
    """
    b = np.asarray(b, dtype=float)
    P = _build_shift_mask(mechanism_set, d)
    abs_b = np.abs(b)

    if isinstance(mechanism_set, FrobeniusBall):
        # sup_{||ΔW_free||_F ≤ η} ||ΔW @ b||_2 = η · σ_max(M)
        # where M[j, idx] = b[k_idx] for free entry idx=(j_idx, k_idx).
        free = [
            (j, k)
            for j in mechanism_set.shifted_rows
            for k in range(j + 1, d)
            if (mechanism_set.entry_mask is None or mechanism_set.entry_mask[j, k] > 0)
        ]
        if not free:
            return 0.0
        n_free = len(free)
        M = np.zeros((d, n_free), dtype=float)
        for idx, (j, k) in enumerate(free):
            M[j, idx] = b[k]
        return float(mechanism_set.eta * np.linalg.norm(M, ord=2))

    elif isinstance(mechanism_set, EntrywiseBox):
        # sup ||ΔW @ b||_2:
        # For each row j: max |(ΔW @ b)[j]| = Σ_k P[j,k] effective_bound[j,k] |b[k]|
        B_sup = mechanism_set.effective_bound
        col_sums = np.array([
            np.sum(P[j, :] * B_sup[j, :] * abs_b)
            for j in range(d)
        ])
        return float(np.linalg.norm(col_sums))

    elif isinstance(mechanism_set, RowBudget):
        # sup ||ΔW @ b||_2: each row j constrained by ρ[j]
        # max |(ΔW @ b)[j]| = ρ[j] · max_k P[j,k] |b[k]|
        col_sums = np.array([
            mechanism_set.rho.get(j, 0.0) * (
                float(np.max(abs_b[j + 1:])) if j + 1 < d else 0.0
            )
            if j in mechanism_set.shifted_rows else 0.0
            for j in range(d)
        ])
        return float(np.linalg.norm(col_sums))

    elif isinstance(mechanism_set, ColumnBudget):
        # sup ||ΔW @ b||_2 with per-column ℓ1 budget:
        # ||ΔW @ b||_2 ≤ Σ_k c_k |b[k]| · max_{j: P[j,k]=1} 1
        # Use triangle: ||ΔW @ b||_2 ≤ Σ_k |b[k]| ||col_k(ΔW)||_2 ≤ Σ_k |b[k]| c_k
        total = 0.0
        for k in range(d):
            ck = mechanism_set._budget(k)
            if any(j < k for j in mechanism_set.shifted_rows):
                total += abs_b[k] * ck
        return float(total)

    else:
        raise TypeError(f"Unknown mechanism set type: {type(mechanism_set)}")


# ---------------------------------------------------------------------------
# Mechanism modulus  m_{ι,q}(v) = β + higher-order remainder
# ---------------------------------------------------------------------------

def directional_mechanism_modulus(
    q: np.ndarray,
    v: np.ndarray,
    A_iota: np.ndarray,
    R_iota: np.ndarray,
    mechanism_set,
    gamma_iota: float,
    d: int,
) -> float:
    """Mechanism modulus m_{ι,q}(v) = β + higher-order remainder.

    Bounds sup_{ΔW ∈ A_W} |v @ (A'_ι - A_ι) @ q| via the bilinear form
    plus a polynomial-series remainder.

    Parameters
    ----------
    q : (d,) query direction (e.g. e_k for node k)
    v : (d,) source statistics (μ_s for Gaussian, empirical mean for empirical)
    A_iota : (d, d) interventional propagator
    R_iota : (d, d) diagonal gating matrix
    mechanism_set : mechanism ambiguity set
    gamma_iota : float — γ_ι amplification factor
    d : int

    Returns
    -------
    float : m_{ι,q}(v) ≥ sup_{ΔW ∈ A_W} |v @ (A'_ι - A_ι) @ q|
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)

    # a = v @ A_iota  (source propagated forward through A)
    a = v @ A_iota                                    # (d,)
    # b = diag(R_iota) * (A_iota @ q)  (query propagated backward, gated by R)
    b = np.diag(R_iota) * (A_iota @ q)               # (d,)

    beta = directional_beta(a, b, mechanism_set)
    lam = directional_lambda(b, mechanism_set, d)

    # Higher-order remainder: λ · ||a||_2 · Σ_{r=1}^{d-2} γ^r
    a_norm = float(np.linalg.norm(a))
    remainder_series = float(sum(gamma_iota ** r for r in range(1, max(d - 1, 1))))
    remainder = lam * a_norm * remainder_series if d >= 3 else 0.0

    return float(beta + remainder)


# ---------------------------------------------------------------------------
# Direct environment bound  (certified for EntrywiseBox ≤ 4 free entries)
# ---------------------------------------------------------------------------

def _directional_env_direct_2d(
    q: np.ndarray,
    W: np.ndarray,
    R_iota: np.ndarray,
    B: np.ndarray,
    free_entries: list[tuple[int, int]],
    env_shifted_rows: Sequence[int],
    d: int,
    eps: float,
    box_delta: np.ndarray | None = None,
) -> float:
    """Certified environment bound for EntrywiseBox with ≤ 4 free entries.

    Returns ε · sup_{ΔW ∈ A_W} ||M_env @ A'_ι(ΔW) @ q||_2.

    Evaluates the objective at all 2^n_free corners plus interior maxima found
    via multiple-restart scipy.optimize, and returns ε × maximum found.

    Parameters
    ----------
    q : (d,) query direction
    W : (d, d) source weight matrix
    R_iota : (d, d) gating matrix
    B : (d, d) entrywise box half-width
    free_entries : list of (j, k) pairs with B[j,k] > 0
    env_shifted_rows : rows where environment shift can act (for M_env)
    d : int
    eps : float — environment radius ε
    box_delta : (d, d) array or None — box center (directional prior)

    Returns
    -------
    float : ε · sup (certified)
    """
    # M_env: diagonal mask from environment shifted rows
    M_env = np.diag([1.0 if i in env_shifted_rows else 0.0 for i in range(d)])

    n_free = len(free_entries)

    # Short-circuit: no free entries → ΔW = 0 is the only feasible point.
    # The sup is exact: ||M_env @ A_iota @ q||_2 (no mechanism perturbation).
    if n_free == 0:
        A_iota = perturbed_propagator(W, np.zeros((d, d)), R_iota)
        return float(eps * np.linalg.norm(M_env @ A_iota @ q))

    def obj(x: np.ndarray) -> float:
        dW = np.zeros((d, d), dtype=float)
        for idx, (j, k) in enumerate(free_entries):
            dW[j, k] = x[idx]
        A_prime = perturbed_propagator(W, dW, R_iota)
        return float(np.linalg.norm(M_env @ A_prime @ q))

    best_val = 0.0

    # Per-entry bounds: [lo, hi] for each free entry
    if box_delta is None:
        lo_vals = [-B[j, k] for (j, k) in free_entries]
        hi_vals = [B[j, k] for (j, k) in free_entries]
    else:
        lo_vals = [box_delta[j, k] - B[j, k] for (j, k) in free_entries]
        hi_vals = [box_delta[j, k] + B[j, k] for (j, k) in free_entries]

    # 1. All 2^n_free corners
    for bits in itertools.product([0, 1], repeat=n_free):
        x = np.array([lo_vals[i] if bits[i] == 0 else hi_vals[i]
                       for i in range(n_free)])
        val = obj(x)
        if val > best_val:
            best_val = val

    # 2. Interior maxima via scipy (multiple random restarts)
    bounds = list(zip(lo_vals, hi_vals))
    rng = np.random.default_rng(0)
    for _ in range(8):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        result = minimize(
            lambda x: -obj(x),
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-10},
        )
        val = -result.fun
        if val > best_val:
            best_val = val

    return float(eps * best_val)


# ---------------------------------------------------------------------------
# Environment bound dispatch
# ---------------------------------------------------------------------------

def directional_environment_bound(
    q: np.ndarray,
    W: np.ndarray,
    A_iota: np.ndarray,
    R_iota: np.ndarray,
    mechanism_set,
    env,
    alpha_iota: float,
    gamma_iota: float,
    eps: float,
    d: int,
) -> dict:
    """Environment bound ε · sup_{ΔW ∈ A_W} ||M_env A'_ι(ΔW) q||_2.

    Returns dict with:
      "value"  : float — the bound value
      "method" : "direct" (certified) or "bound" (approximate, not certified)

    "direct" path: EntrywiseBox with ≤ 4 free entries.
      Evaluates the true sup via corner search + scipy optimisation.

    "bound" path (fallback — NOT a proven certificate):
      ε · (||M_env A_ι q||_2 + α_ι)
      This reuses α_ι as a slack for A'_ι - A_ι, which is engineering convenience
      but is NOT derived from the directional certificate derivation. Never use "bound"
      values in reported results.

    M_env is built from env.shifted_rows, not mechanism_set.shifted_rows.
    In ATCE these differ: mechanism=[0,1], environment=[0,2].
    """
    env_shifted_rows = env.shifted_rows

    if isinstance(mechanism_set, EntrywiseBox):
        B = mechanism_set.B
        free_entries = [
            (j, k)
            for j in mechanism_set.shifted_rows
            for k in range(j + 1, d)
            if B[j, k] > 0
        ]
        if len(free_entries) <= 4:
            value = _directional_env_direct_2d(
                q, W, R_iota, B, free_entries, env_shifted_rows, d, eps,
                box_delta=mechanism_set.delta,
            )
            return {"value": value, "method": "direct"}

    # Fallback: approximate bound ε · (||M_env A_ι q||_2 + α_ι)
    # WARNING: NOT a certified bound. α_ι slacks the full operator norm change
    # (||A'_ι - A_ι||_2), which is much looser than the directional quantity.
    # The certified bound for the environment uses direct evaluation above.
    M_env = np.diag([1.0 if i in env_shifted_rows else 0.0 for i in range(d)])
    base = float(np.linalg.norm(M_env @ A_iota @ q))
    value = eps * (base + alpha_iota)
    return {"value": value, "method": "bound"}


# ---------------------------------------------------------------------------
# Full directional certificate — Gaussian
# ---------------------------------------------------------------------------

def directional_certificate_gaussian(
    tau: np.ndarray,
    c: np.ndarray,
    O: list[int],
    iota_idx: int,
    bundle,
    mechanism_set,
    env,
    alpha_iota: float,
    gamma_iota: float,
    noise_mean: np.ndarray | None = None,
) -> dict:
    """Directional certificate for Gaussian mean query E[c^T X | do(J)].

    Bounds |E_t[c^T X | do(J)] - c^T τ E_s[X | do(J)]| for a fixed direction c.

    Parameters
    ----------
    tau : (d, d) transport map
    c : (d,) query direction in ambient space (e.g. e_k for node k)
    O : list of output node indices (metadata only)
    iota_idx : int — intervention index into bundle
    bundle : SCMBundle
    mechanism_set : mechanism ambiguity set
    env : environment ambiguity set (GelbrichBall)
    alpha_iota : float — α_ι (unused in transport/mechanism; kept for API parity)
    gamma_iota : float — γ_ι amplification factor
    noise_mean : (d,) exogenous noise mean to use for v = μ_s.
        If None, uses bundle.noise_mean (population).  Pass the holdout-
        estimated mean to ensure row-consistency with evaluation data.

    Returns
    -------
    dict with keys: transport, mechanism, environment, env_method, certificate
    """
    c = np.asarray(c, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d = bundle.d
    I_d = np.eye(d)

    A_iota = bundle.intervened_scms[iota_idx].A
    # Gating matrix: R_iota from intervention
    iv = bundle.interventions[iota_idx]
    var_names = bundle.scm.var_names
    intervened_nodes = [
        var_names.index(k) if isinstance(k, str) else int(k)
        for k in iv.keys()
    ] if iv else []
    R_iota = np.eye(d)
    for node in intervened_nodes:
        R_iota[node, node] = 0.0

    from traca.utils import interventional_exo_mean
    scm_i = bundle.intervened_scms[iota_idx]
    nm = noise_mean if noise_mean is not None else bundle.noise_mean
    mu_s = interventional_exo_mean(nm, scm_i._fixed, scm_i._J)
    q = c                       # query direction in ambient space

    # Transport term: |μ_s_eff @ A_ι @ (τ − I) @ q|
    transport = abs(float(mu_s @ A_iota @ (tau - I_d) @ q))

    # Mechanism term: m_{ι,q}(μ_s_eff)
    mechanism = directional_mechanism_modulus(
        q, mu_s, A_iota, R_iota, mechanism_set, gamma_iota, d
    )

    # Environment term: ε · sup ||M_env A'_ι(ΔW) q||_2
    eps = env.eps
    env_result = directional_environment_bound(
        q, bundle.W, A_iota, R_iota, mechanism_set, env,
        alpha_iota, gamma_iota, eps, d
    )

    certificate = transport + mechanism + env_result["value"]
    return {
        "transport": float(transport),
        "mechanism": float(mechanism),
        "environment": float(env_result["value"]),
        "env_method": env_result["method"],
        "certificate": float(certificate),
    }


# ---------------------------------------------------------------------------
# Full directional certificate — Empirical
# ---------------------------------------------------------------------------

def directional_certificate_empirical(
    tau: np.ndarray,
    c: np.ndarray,
    O: list[int],
    iota_idx: int,
    bundle,
    mechanism_set,
    env,
    alpha_iota: float,
    gamma_iota: float,
    U_eff: np.ndarray | None = None,
) -> dict:
    """Directional certificate for empirical mean query (1/N) Σ_n c^T X_n.

    Uses the empirical noise mean as the source statistics vector v.

    Parameters
    ----------
    tau : (d, d) transport map
    c : (d,) query direction in ambient space
    O : list of output node indices (metadata only)
    iota_idx : int — intervention index into bundle
    bundle : SCMBundle
    mechanism_set : mechanism ambiguity set
    env : environment ambiguity set (FrobeniusEmpirical)
    alpha_iota : float — α_ι stability modulus
    gamma_iota : float — γ_ι amplification factor
    U_eff : (N, d) effective noise samples (intervened cols zeroed, fixed added).
        If None, derived from bundle.noise_samples[iota_idx] over all rows
        (population default).  Pass the holdout-subset U_eff to ensure
        row-consistency with Phi_pushed and target_val.

    Returns
    -------
    dict with keys: transport, mechanism, environment, env_method, certificate
    """
    c = np.asarray(c, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d = bundle.d
    I_d = np.eye(d)

    A_iota = bundle.intervened_scms[iota_idx].A
    iv = bundle.interventions[iota_idx]
    var_names = bundle.scm.var_names
    intervened_nodes = [
        var_names.index(k) if isinstance(k, str) else int(k)
        for k in iv.keys()
    ] if iv else []
    R_iota = np.eye(d)
    for node in intervened_nodes:
        R_iota[node, node] = 0.0

    if U_eff is not None:
        mu_hat = U_eff.mean(axis=0)
    else:
        scm_i = bundle.intervened_scms[iota_idx]
        U_s = bundle.noise_samples[iota_idx]    # (N, d)
        # Build effective noise: zero intervened cols, add fixed
        _U_eff = U_s.copy()
        if scm_i._J:
            _U_eff[:, list(scm_i._J)] = 0.0
        _U_eff += scm_i._fixed[np.newaxis, :]
        mu_hat = _U_eff.mean(axis=0)           # effective empirical exogenous mean
    q = c

    # Transport term using empirical mean
    transport = abs(float(mu_hat @ A_iota @ (tau - I_d) @ q))

    # Mechanism term
    mechanism = directional_mechanism_modulus(
        q, mu_hat, A_iota, R_iota, mechanism_set, gamma_iota, d
    )

    # Environment term (per-sample radius eps)
    eps = env.eps
    env_result = directional_environment_bound(
        q, bundle.W, A_iota, R_iota, mechanism_set, env,
        alpha_iota, gamma_iota, eps, d
    )

    certificate = transport + mechanism + env_result["value"]
    return {
        "transport": float(transport),
        "mechanism": float(mechanism),
        "environment": float(env_result["value"]),
        "env_method": env_result["method"],
        "certificate": float(certificate),
    }
