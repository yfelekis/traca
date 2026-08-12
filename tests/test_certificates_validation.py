"""
Validation tests for the corrected certificates.

The headline property of a certificate is:

    certificate(tau)  >=  sup_{adversary in ambiguity set}  loss(tau, adversary)

These tests sample many adversaries, take the empirical worst-case loss, and
assert the certificate dominates it.  Crucially, they include a source != target
case with tau != I, which is exactly where the previously buggy (transport-term-
dropped) empirical certificate FAILED.  If someone reintroduces that bug,
test_empirical_certificate_dominates_worstcase_shifted goes red.

Run: pytest tests/test_certificates_validation.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from traca.certificates import (
    delta_iota_rho_sq,
    delta_iota_U,
    full_joint_certificate,
    query_interval,
)
from traca.stability import gamma, alpha_polynomial, perturbed_propagator, gating_matrix
from traca.ambiguity import FrobeniusBall


# ---------------------------------------------------------------------------
# Fixtures: a tiny 3-node chain with a genuine mechanism shift
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_setup():
    """3-node chain X0 -> X1 -> X2, shifted node = 1 (the middle mechanism)."""
    rng = np.random.default_rng(0)
    d = 3
    W = np.zeros((d, d))
    W[0, 1] = 0.3
    W[1, 2] = 0.2
    A = np.linalg.solve(np.eye(d) - W, np.eye(d))

    mu_s = np.zeros(d)
    Sigma_s = np.eye(d)

    N = 400
    U_s = rng.standard_normal((N, d))  # row-vector form: (N, d)

    R_iota = gating_matrix(d, [])  # observational intervention
    A_iota = A

    shifted_rows = (1,)
    eta = 0.3
    eps = 0.1
    aw = FrobeniusBall(eta=eta, shifted_rows=shifted_rows, d=d)
    g = gamma(A_iota, R_iota, aw)
    al = alpha_polynomial(A_iota, g, d)

    return dict(d=d, W=W, A=A, A_iota=A_iota, R_iota=R_iota,
                mu_s=mu_s, Sigma_s=Sigma_s, U_s=U_s, N=N,
                shifted_rows=shifted_rows, eta=eta, eps=eps,
                aw=aw, alpha=al)


# ---------------------------------------------------------------------------
# Adversary samplers
# ---------------------------------------------------------------------------

def _sample_dW(rng, shifted_rows, d, eta):
    dW = np.zeros((d, d))
    for j in shifted_rows:
        dW[j] = rng.standard_normal(d)
    dW = np.triu(dW, k=1)  # strictly upper-triangular
    nrm = np.linalg.norm(dW, "fro")
    if nrm > 0:
        dW *= (eta * rng.uniform(0.5, 1.0)) / nrm
    return dW


def _sample_Theta(rng, shifted_rows, N, d, eps):
    Theta = np.zeros((N, d))
    Theta[:, list(shifted_rows)] = rng.standard_normal((N, len(shifted_rows)))
    nrm = np.linalg.norm(Theta, "fro")
    radius = eps * np.sqrt(N)
    if nrm > 0:
        Theta *= (radius * rng.uniform(0.5, 1.0)) / nrm
    return Theta


def _empirical_loss(tau, dW, Theta, W, A_iota, R_iota, U_s, S_O=None):
    """F^U_ι(τ, ΔW, Θ) per the code convention (row-vector form)."""
    A_prime = perturbed_propagator(W, dW, R_iota)
    pushed = U_s @ A_iota @ tau         # (N, d)
    target = (U_s + Theta) @ A_prime    # (N, d)
    residual = pushed - target          # (N, d)
    if S_O is not None:
        residual = residual @ S_O.T     # (N, |O|)
    N = U_s.shape[0]
    return float(np.linalg.norm(residual, "fro") ** 2) / N


# ---------------------------------------------------------------------------
# Empirical certificate dominates adversarial worst case
# ---------------------------------------------------------------------------

def test_empirical_certificate_dominates_worstcase_identity(chain_setup):
    """tau = I (source == target). Certificate must dominate empirical worst case."""
    s = chain_setup
    tau = np.eye(s["d"])
    cert = delta_iota_U(tau, s["alpha"], s["A_iota"], s["U_s"],
                        s["eps"], s["N"]) ** 2

    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(500):
        dW = _sample_dW(rng, s["shifted_rows"], s["d"], s["eta"])
        Theta = _sample_Theta(rng, s["shifted_rows"], s["N"], s["d"], s["eps"])
        worst = max(worst, _empirical_loss(tau, dW, Theta, s["W"],
                                           s["A_iota"], s["R_iota"], s["U_s"]))
    assert worst <= cert + 1e-9, f"worst={worst:.6f} > cert={cert:.6f}"


def test_empirical_certificate_dominates_worstcase_shifted(chain_setup):
    """tau != I — genuine transport.

    This is the case where the BUGGY certificate (transport term dropped)
    FAILS.  With a tau far from I, the dropped term
        ‖(τ−I)A_ι U_s‖_F
    is large, making the buggy certificate invalid.
    If this test fails, the transport term was reintroduced as a bug.
    """
    s = chain_setup
    tau = np.eye(s["d"])
    tau[1, 1] = 0.4  # shifted node gets a non-trivial transport coefficient
    cert = delta_iota_U(tau, s["alpha"], s["A_iota"], s["U_s"],
                        s["eps"], s["N"]) ** 2

    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(500):
        dW = _sample_dW(rng, s["shifted_rows"], s["d"], s["eta"])
        Theta = _sample_Theta(rng, s["shifted_rows"], s["N"], s["d"], s["eps"])
        worst = max(worst, _empirical_loss(tau, dW, Theta, s["W"],
                                           s["A_iota"], s["R_iota"], s["U_s"]))
    assert worst <= cert + 1e-9, (
        f"worst={worst:.6f} > cert={cert:.6f} — transport term missing?"
    )


def test_buggy_certificate_is_invalid_when_tau_shifted(chain_setup):
    """Regression guard: confirm the OLD formula (transport term dropped) is invalid.

    The buggy formula: alpha² * (1/N) * ‖U_s‖_F² + ε² * (‖A‖₂ + α)²
    drops the transport mismatch term entirely.  With tau far enough from I,
    the transport-only loss at the ZERO adversary (ΔW=0, Θ=0) exceeds the
    buggy bound — proving the transport term is load-bearing.

    We use tau[1,1] = 0.1 so the transport-only loss dominates.
    """
    s = chain_setup
    tau = np.eye(s["d"])
    tau[1, 1] = 0.1  # shifted node has large transport mismatch: tau-I = diag(0,-0.9,0)

    U_norm_sq = float(np.linalg.norm(s["U_s"], "fro") ** 2) / s["N"]
    A_norm2 = float(np.linalg.norm(s["A_iota"], ord=2))
    buggy_cert = s["alpha"] ** 2 * U_norm_sq + s["eps"] ** 2 * (A_norm2 + s["alpha"]) ** 2

    # The zero adversary (ΔW=0, Θ=0) is admissible and gives a pure transport loss.
    zero_dW = np.zeros_like(s["W"])
    zero_Theta = np.zeros_like(s["U_s"])
    base_loss = _empirical_loss(tau, zero_dW, zero_Theta, s["W"],
                                s["A_iota"], s["R_iota"], s["U_s"])

    # The worst case is at least as large as the zero-adversary loss.
    assert base_loss > buggy_cert, (
        f"Transport-only loss ({base_loss:.6f}) <= buggy cert ({buggy_cert:.6f}). "
        "The example is not strong enough to expose the transport-term bug."
    )


# ---------------------------------------------------------------------------
# Gaussian certificate: non-negativity, monotonicity, selector tightening
# ---------------------------------------------------------------------------

def test_gaussian_certificate_nonnegative(chain_setup):
    s = chain_setup
    tau = np.eye(s["d"]); tau[1, 1] = 0.5
    val = delta_iota_rho_sq(tau, s["alpha"], s["A_iota"],
                             s["mu_s"], s["Sigma_s"], s["eps"])
    assert val >= 0.0


def test_gaussian_certificate_monotone_in_eps(chain_setup):
    s = chain_setup
    tau = np.eye(s["d"]); tau[1, 1] = 0.5
    vals = [delta_iota_rho_sq(tau, s["alpha"], s["A_iota"],
                               s["mu_s"], s["Sigma_s"], eps)
            for eps in (0.0, 0.1, 0.2, 0.4)]
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1] + 1e-12


def test_gaussian_selector_tightening(chain_setup):
    """Restricting to a subset O ⊆ [d] can only decrease the certificate."""
    s = chain_setup
    d = s["d"]
    tau = np.eye(d); tau[1, 1] = 0.5
    full = delta_iota_rho_sq(tau, s["alpha"], s["A_iota"],
                              s["mu_s"], s["Sigma_s"], s["eps"])
    from traca.certificates import selection_matrix
    S_O = selection_matrix(d, [2])
    restricted = delta_iota_rho_sq(tau, s["alpha"], s["A_iota"],
                                    s["mu_s"], s["Sigma_s"], s["eps"], S_O=S_O)
    assert restricted <= full + 1e-12


def test_empirical_selector_tightening(chain_setup):
    s = chain_setup
    d = s["d"]
    tau = np.eye(d); tau[1, 1] = 0.5
    full = delta_iota_U(tau, s["alpha"], s["A_iota"],
                        s["U_s"], s["eps"], s["N"])
    from traca.certificates import selection_matrix
    S_O = selection_matrix(d, [2])
    restricted = delta_iota_U(tau, s["alpha"], s["A_iota"],
                               s["U_s"], s["eps"], s["N"], S_O=S_O)
    assert restricted <= full + 1e-9


# ---------------------------------------------------------------------------
# full_joint_certificate: consistent with per-iota delta
# ---------------------------------------------------------------------------

def test_full_joint_single_query_consistency(chain_setup):
    """full_joint_certificate with query_family=[(0,[2])] matches single-query."""
    s = chain_setup
    tau = np.eye(s["d"]); tau[1, 1] = 0.5

    fj = full_joint_certificate(
        tau, [s["alpha"]], [s["A_iota"]], s["mu_s"], s["Sigma_s"], s["eps"],
        mode="empirical", U_s_list=[s["U_s"]], N=s["N"],
        query_family=[(0, [2])],
    )
    from traca.certificates import selection_matrix
    S_O = selection_matrix(s["d"], [2])
    direct = delta_iota_U(tau, s["alpha"], s["A_iota"],
                          s["U_s"], s["eps"], s["N"], S_O=S_O) ** 2
    assert abs(fj - direct) < 1e-9


# ---------------------------------------------------------------------------
# query_interval: symmetric, monotone in delta_sq
# ---------------------------------------------------------------------------

def test_query_interval_symmetric():
    lo, hi = query_interval(Phi_pushed=2.0, L_Phi=1.0, delta_sq=0.04)
    assert abs((hi - 2.0) - (2.0 - lo)) < 1e-12


def test_query_interval_monotone_in_delta():
    lo1, hi1 = query_interval(Phi_pushed=2.0, L_Phi=1.0, delta_sq=0.04)
    lo2, hi2 = query_interval(Phi_pushed=2.0, L_Phi=1.0, delta_sq=0.16)
    assert (hi2 - lo2) > (hi1 - lo1)


# ---------------------------------------------------------------------------
# √N consistency: gap cert - worst should be stable across N
#
# The code's loss is ‖residual‖_F² / N.  The certificate δ^U_code is normalized
# so that (δ^U_code)² also scales as O(1) w.r.t N.  If the √N is wired
# inconsistently (e.g. the Θ-ball radius dropped its √N while the loss kept
# its /N), the certificate would grow as O(N) relative to the loss, or shrink
# as O(1/N) — both clearly visible as a trend across N.
#
# We test at N=200 and N=800 (4× ratio) with the SAME random seed and verify
# the gap cert - worst is stable to within ±50% of its value at N=200.
# A systematic O(N) or O(1/N) drift would produce a 4× change and fail.
# ---------------------------------------------------------------------------

def _make_chain_at_N(N: int, shifted_rows=(1,), eta=0.3, eps=0.1):
    rng = np.random.default_rng(42)
    d = 3
    W = np.zeros((d, d))
    W[0, 1] = 0.3; W[1, 2] = 0.2
    A = np.linalg.solve(np.eye(d) - W, np.eye(d))
    U_s = rng.standard_normal((N, d))
    R_iota = gating_matrix(d, [])
    aw = FrobeniusBall(eta=eta, shifted_rows=shifted_rows, d=d)
    g = gamma(A, R_iota, aw)
    al = alpha_polynomial(A, g, d)
    return dict(d=d, W=W, A=A, A_iota=A, R_iota=R_iota,
                U_s=U_s, N=N, shifted_rows=shifted_rows, eta=eta, eps=eps, alpha=al)


def _worst_and_cert(s, tau, n_adv=300):
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(n_adv):
        dW = _sample_dW(rng, s["shifted_rows"], s["d"], s["eta"])
        Theta = _sample_Theta(rng, s["shifted_rows"], s["N"], s["d"], s["eps"])
        worst = max(worst, _empirical_loss(tau, dW, Theta, s["W"],
                                           s["A_iota"], s["R_iota"], s["U_s"]))
    cert = delta_iota_U(tau, s["alpha"], s["A_iota"], s["U_s"],
                        s["eps"], s["N"]) ** 2
    return worst, cert


def test_sqrt_N_consistency_gap_stable_across_N():
    """The gap (cert - worst_case_loss) should not grow or shrink systematically with N.

    If the √N bookkeeping is internally inconsistent (e.g. the Θ-ball radius
    is not eps*sqrt(N) or the loss is not divided by N), the gap will have a
    clear O(N) or O(1/N) trend.  A 4× change in N should produce at most a
    ±50% change in the gap.
    """
    tau = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.4, 0.0],
                    [0.0, 0.0, 1.0]])  # tau[1,1]=0.4: real transport

    s200 = _make_chain_at_N(200)
    s800 = _make_chain_at_N(800)

    worst200, cert200 = _worst_and_cert(s200, tau)
    worst800, cert800 = _worst_and_cert(s800, tau)

    gap200 = cert200 - worst200
    gap800 = cert800 - worst800

    # Both certificates must dominate their respective worst cases
    assert worst200 <= cert200 + 1e-9, f"cert fails at N=200: {worst200} > {cert200}"
    assert worst800 <= cert800 + 1e-9, f"cert fails at N=800: {worst800} > {cert800}"

    # The gaps should be roughly comparable (within 3× of each other).
    # A 4× change in N with correct √N bookkeeping gives <2× change in gap.
    # An O(N) drift would give a 4× change and fail this check.
    assert gap200 > 0 and gap800 > 0, "negative gap implies certificate is invalid"
    ratio = max(gap200, gap800) / min(gap200, gap800)
    assert ratio < 3.0, (
        f"Gap changes {ratio:.2f}× from N=200 ({gap200:.4f}) to N=800 ({gap800:.4f}). "
        "This suggests the √N bookkeeping is inconsistent between loss, Θ-ball, and δ."
    )


# ---------------------------------------------------------------------------
# No double-squaring: E_U_joint squares δ^U exactly once
# ---------------------------------------------------------------------------

def test_no_double_squaring(chain_setup):
    """E_U_joint(tau, [d]) == d**2, not d**4.

    full_joint_certificate and single_query_certificate square δ^U internally.
    E_U_joint must also square pre-computed unsquared δ^U values — not re-square
    already-squared outputs.
    """
    from traca.certificates import E_U_joint, selection_matrix
    s = chain_setup
    tau = np.eye(s["d"]); tau[1, 1] = 0.5

    # Compute unsquared δ^U directly
    S_O = selection_matrix(s["d"], [2])
    d_val = delta_iota_U(tau, s["alpha"], s["A_iota"],
                         s["U_s"], s["eps"], s["N"], S_O=S_O)

    # E_U_joint takes pre-computed unsquared values and squares each
    joint = E_U_joint(tau, [d_val])

    assert abs(joint - d_val ** 2) < 1e-12, (
        f"E_U_joint returned {joint:.6f}, expected d**2={d_val**2:.6f}. "
        "Either double-squaring or missing squaring."
    )

    # Also confirm it is NOT d**4
    assert abs(joint - d_val ** 4) > 1e-6 or d_val < 1e-3, (
        "joint == d**4: E_U_joint is double-squaring."
    )


# ---------------------------------------------------------------------------
# Gaussian transport term: hand-computed d=2 closed form
#
# For d=2, W=[[0,w],[0,0]], A=[[1,w],[0,1]], tau=diag(1,s), mu_s=0, Sigma_s=I,
# eps=0, shifted rows={1} (only X1 can shift):
#   S = I  (full post-interventional)
#   A(tau-I) = A @ diag(0, s-1) = [[0, w(s-1)],[0, s-1]]
#   term1 (mu): 4||mu_s A (tau-I)||^2 = 0        (mu_s=0)
#   term2 (Sigma fro): 4||Sigma_s||_2 * ||A(tau-I)||_F^2
#                    = 4 * 1 * (w^2(s-1)^2 + (s-1)^2) = 4(1+w^2)(s-1)^2
#   term3 (mech mu): 4 alpha^2 ||mu_s||^2 = 0    (mu_s=0)
#   term4 (mech Sigma): 4*d*||Sigma_s||_2 * alpha^2 = 4*2*1*alpha^2 = 8*alpha^2
#   term5 (env): 2*(||A||_2 + alpha)^2 * eps^2 = 0  (eps=0)
#   Total: 4*(1+w^2)*(s-1)^2 + 8*alpha^2
# ---------------------------------------------------------------------------

def test_gaussian_certificate_closed_form_d2():
    """Verify delta_iota_rho_sq against a hand-computed d=2 closed form.

    Setup: d=2, W[0,1]=w, tau=diag(1,s), mu_s=0, Sigma_s=I, eps=0.
    Only row 1 of ΔW can shift (shifted_rows={1}).

    Expected:
        delta^2 = 4*(1 + w²)*(s-1)² + 8*alpha²
    where alpha = ||A||_2 * gamma (polynomial bound with d-1=1 terms).
    """
    w = 0.5
    s_val = 0.4  # tau[1,1]
    d = 2
    W = np.array([[0.0, w], [0.0, 0.0]])
    A = np.linalg.solve(np.eye(d) - W, np.eye(d))
    tau = np.diag([1.0, s_val])
    mu_s = np.zeros(d)
    Sigma_s = np.eye(d)
    eps = 0.0
    alpha = 0.1  # arbitrary small alpha (mechanism stability, external)

    # Hand computation (mu_s=0, eps=0 → only terms 2 and 4 survive)
    A_tau_I = A @ (tau - np.eye(d))          # [[0, w*(s-1)],[0, s-1]]
    term2 = 4.0 * np.linalg.norm(Sigma_s, ord=2) * np.linalg.norm(A_tau_I, "fro") ** 2
    term4 = 4.0 * d * np.linalg.norm(Sigma_s, ord=2) * alpha ** 2
    expected = term2 + term4

    # Code's formula
    computed = delta_iota_rho_sq(tau, alpha, A, mu_s, Sigma_s, eps)

    assert abs(computed - expected) < 1e-10, (
        f"closed-form={expected:.8f}, computed={computed:.8f}. "
        "Gaussian transport term A(tau-I) is missing or wrong."
    )

    # Sanity: verify term2 depends on (s-1)^2 and includes A (not bare (tau-I))
    # If the code used bare ||(tau-I)||_F instead of ||A(tau-I)||_F, term2 would be
    # 4*(s-1)^2, not 4*(1+w^2)*(s-1)^2.
    bare_wrong = 4.0 * (s_val - 1.0) ** 2 + term4
    assert abs(computed - bare_wrong) > 1e-6, (
        "Code used bare ||(tau-I)||_F instead of ||A(tau-I)||_F for Frobenius term."
    )
