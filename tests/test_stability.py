"""
Tests for traca.stability.
"""
import numpy as np
import pytest

from traca.stability import gamma, alpha_polynomial, alpha_neumann, perturbed_propagator, gating_matrix
from traca.ambiguity import FrobeniusBall, RowBudget, ColumnBudget, EntrywiseBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chain3_propagators():
    """Simple 3-node chain: X0 -> X1 -> X2."""
    W = np.array([[0, 0.5, 0.0],
                  [0, 0.0, 0.4],
                  [0, 0.0, 0.0]], dtype=float)
    d = 3
    A = np.linalg.inv(np.eye(d) - W)
    R = np.eye(d)  # observational (no intervention)
    return W, A, R, d


# ---------------------------------------------------------------------------
# gating_matrix
# ---------------------------------------------------------------------------

def test_gating_matrix_observational():
    R = gating_matrix(3, [])
    assert np.allclose(R, np.eye(3))


def test_gating_matrix_intervention():
    R = gating_matrix(3, [1])
    assert R[1, 1] == 0.0
    assert R[0, 0] == 1.0
    assert R[2, 2] == 1.0


# ---------------------------------------------------------------------------
# gamma (amplification factor)
# ---------------------------------------------------------------------------

def test_gamma_zero_for_zero_eta():
    W, A, R, d = make_chain3_propagators()
    ball = FrobeniusBall(eta=0.0, shifted_rows=(1,), d=d)
    assert np.isclose(gamma(A, R, ball), 0.0)


def test_gamma_frobenius_mc():
    """Monte Carlo: empirical sup <= closed-form gamma."""
    W, A, R, d = make_chain3_propagators()
    eta = 0.3
    ball = FrobeniusBall(eta=eta, shifted_rows=(1,), d=d)
    gamma_val = gamma(A, R, ball)

    rng = np.random.default_rng(42)
    max_val = 0.0
    for _ in range(2000):
        dW = rng.standard_normal((d, d))
        # Only shifted rows non-zero
        dW_masked = np.zeros_like(dW)
        dW_masked[1] = dW[1]
        norm = np.linalg.norm(dW_masked, "fro")
        if norm > 0:
            dW_masked *= eta / norm
        # Right-mult: A @ (ΔW @ R), matching resolvent A'−A = A(ΔW R)A'
        val = np.linalg.norm(A @ dW_masked @ R, ord=2)
        max_val = max(max_val, val)
    assert max_val <= gamma_val + 1e-6, f"MC max {max_val:.6f} > gamma {gamma_val:.6f}"


def test_gamma_row_budget_mc():
    """Monte Carlo: empirical sup <= closed-form gamma (row budget)."""
    W, A, R, d = make_chain3_propagators()
    rb = RowBudget(rho={1: 0.2}, shifted_rows=(1,), d=d)
    gamma_val = gamma(A, R, rb)

    rng = np.random.default_rng(7)
    max_val = 0.0
    for _ in range(2000):
        dW = np.zeros((d, d))
        # Only upper-triangular entries of row 1 (column 2 for d=3)
        v = rng.standard_normal(d)
        v = np.triu(np.outer(np.ones(d), v), k=1)[1]  # upper-tri mask for row 1
        l1 = np.sum(np.abs(v))
        if l1 > 0:
            v *= 0.2 / l1
        dW[1] = v
        # Right-mult: A @ (ΔW @ R), matching resolvent A'−A = A(ΔW R)A'
        val = np.linalg.norm(A @ dW @ R, ord=2)
        max_val = max(max_val, val)
    assert max_val <= gamma_val + 1e-6


# ---------------------------------------------------------------------------
# alpha_polynomial
# ---------------------------------------------------------------------------

def test_alpha_polynomial_zero_gamma():
    W, A, R, d = make_chain3_propagators()
    alpha = alpha_polynomial(A, 0.0, d)
    assert np.isclose(alpha, 0.0)


def test_alpha_polynomial_holds_empirically():
    """For random ΔW, ||A'_ι - A_ι||_2 <= alpha_polynomial."""
    W, A, R, d = make_chain3_propagators()
    eta = 0.2
    ball = FrobeniusBall(eta=eta, shifted_rows=(1,), d=d)
    gamma_val = gamma(A, R, ball)
    alpha = alpha_polynomial(A, gamma_val, d)

    rng = np.random.default_rng(99)
    for _ in range(500):
        dW = ball.project(rng.standard_normal((d, d)))
        A_prime = perturbed_propagator(W, dW, R)
        diff = np.linalg.norm(A_prime - A, ord=2)
        assert diff <= alpha + 1e-8, f"diff {diff:.8f} > alpha {alpha:.8f}"


# ---------------------------------------------------------------------------
# alpha_neumann
# ---------------------------------------------------------------------------

def test_alpha_neumann_requires_gamma_lt_1():
    W, A, R, d = make_chain3_propagators()
    with pytest.raises(ValueError):
        alpha_neumann(A, 1.5, 0.1)


def test_alpha_neumann_holds_empirically():
    """For small ΔW (gamma < 1), ||A' - A||_2 <= alpha_neumann."""
    W, A, R, d = make_chain3_propagators()
    eta = 0.05
    ball = FrobeniusBall(eta=eta, shifted_rows=(1,), d=d)
    gamma_val = gamma(A, R, ball)
    if gamma_val >= 1.0:
        pytest.skip("gamma >= 1 for this config, can't test Neumann")
    sup_RdW = eta * np.linalg.norm(R @ np.eye(d), ord=2)  # conservative bound
    alpha = alpha_neumann(A, gamma_val, sup_RdW)

    rng = np.random.default_rng(55)
    for _ in range(500):
        dW = ball.project(rng.standard_normal((d, d)))
        A_prime = perturbed_propagator(W, dW, R)
        diff = np.linalg.norm(A_prime - A, ord=2)
        assert diff <= alpha + 1e-6


# ---------------------------------------------------------------------------
# Regression: gamma R-placement at LiLuCaS ι=6
# ---------------------------------------------------------------------------

def test_gamma_nonzero_when_shifted_rows_subset_of_J():
    """Regression for gamma R-placement bug (2026-07-22).

    At LiLuCaS ι=6 (do(Smoking=1, Genetics=1)), shifted_rows=(0,1) are
    a subset of J=(0,1).  The old code computed A@R then bounded
    ||A(RΔW)|| — since R zeros rows 0,1 of ΔW, gamma was 0, violating the
    bound by a concrete ΔW that produces nonzero propagator change via the
    _fixed do-values.

    The fix gates the *columns* of the budget by R (right-mult convention:
    resolvent is A'−A = A(ΔW R)A').  Gamma must be nonzero here because
    ΔW has nonzero entries in column 3 (LungCancer's mechanism), and
    R[3,3]=1 (column 3 is NOT intervened on).
    """
    # LiLuCaS light DAG: 6 nodes
    d = 6
    W = np.zeros((d, d))
    W[0, 3] = 0.7; W[1, 3] = 0.5
    W[0, 4] = 0.4; W[3, 4] = 0.6
    W[2, 5] = 0.3; W[3, 5] = 0.5; W[4, 5] = 0.4

    # iota=6: do(Smoking=1, Genetics=1) — intervene on nodes 0 and 1
    J = [0, 1]
    R = gating_matrix(d, J)
    A_iota = np.linalg.inv(np.eye(d) - W @ R)

    # EntrywiseBox: B[0,3]=0.5, B[1,3]=0.5 — perturb LungCancer's mechanism
    B = np.zeros((d, d))
    B[0, 3] = 0.5; B[1, 3] = 0.5
    box = EntrywiseBox(B=B, shifted_rows=(0, 1), d=d)

    # gamma must be nonzero — the perturbation acts on column 3 which is
    # NOT zeroed by R (R[3,3]=1)
    g = gamma(A_iota, R, box)
    assert g > 0.1, f"gamma should be nonzero; got {g}"

    # The concrete ΔW from the investigation
    dW = np.zeros((d, d))
    dW[0, 3] = -0.429; dW[1, 3] = -0.393

    # Actual propagator change
    A_prime = perturbed_propagator(W, dW, R)
    diff = A_prime - A_iota

    # Build U_eff for iota=6: zero intervened columns, add _fixed
    rng = np.random.default_rng(42)
    N = 200
    noise_mean = np.array([0.1, 0.2, 0.15, 0.1, 0.3, 0.2])
    U_obs = rng.normal(loc=noise_mean, scale=1.0, size=(N, d))
    U_eff = U_obs.copy()
    U_eff[:, J] = 0.0
    fixed = np.zeros(d); fixed[0] = 1.0; fixed[1] = 1.0
    U_eff += fixed[np.newaxis, :]

    # Actual query effect
    actual = np.linalg.norm(U_eff @ diff, 'fro') / np.sqrt(N)
    assert actual > 1.0, f"actual query effect should be ~1.135; got {actual}"

    # alpha bound must exceed actual
    alpha = alpha_polynomial(A_iota, g, d)
    bound = alpha * np.linalg.norm(U_eff, 'fro') / np.sqrt(N)
    assert bound > actual, (
        f"alpha bound {bound:.6f} must exceed actual {actual:.6f} "
        f"(gamma={g:.6f}, alpha={alpha:.6f})"
    )

    # Corner-search worst case must also be bounded
    I = np.eye(d)
    worst = 0.0
    for s0 in [-1, 1]:
        for s1 in [-1, 1]:
            dW_c = np.zeros((d, d))
            dW_c[0, 3] = s0 * 0.5; dW_c[1, 3] = s1 * 0.5
            A_p = np.linalg.inv(I - (W + dW_c) @ R)
            val = np.linalg.norm(U_eff @ (A_p - A_iota), 'fro') / np.sqrt(N)
            worst = max(worst, val)
    assert bound > worst, (
        f"alpha bound {bound:.6f} must exceed worst-case {worst:.6f}"
    )
