"""
Tests for traca.losses.

Part 1 (Phase 4): value tests.
Part 2 (Phase 5.5): gradient tests (added later).
"""
import numpy as np
import pytest

from traca.losses import GaussianLoss, EmpiricalLoss, numerical_gradient
from traca.stability import gating_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_2node():
    """2-node SCM: X0 -> X1, no intervention."""
    d = 2
    W = np.array([[0, 0.5], [0, 0]], dtype=float)
    A = np.linalg.inv(np.eye(d) - W)
    R = np.eye(d)
    mu_s = np.array([0.0, 0.0])
    Sigma_s = np.eye(d)
    return d, W, A, R, mu_s, Sigma_s


# ---------------------------------------------------------------------------
# GaussianLoss
# ---------------------------------------------------------------------------

class TestGaussianLoss:
    def setup_method(self):
        self.loss = GaussianLoss()
        self.d, self.W, self.A, self.R, self.mu_s, self.Sigma_s = make_2node()

    def test_zero_at_identity_no_shift(self):
        """F = 0 when tau=I, dW=0, source=target."""
        tau = np.eye(self.d)
        dW = np.zeros((self.d, self.d))
        val = self.loss.value(tau, dW, self.W, self.A, self.R, self.mu_s, self.Sigma_s)
        assert np.isclose(val, 0.0, atol=1e-10), f"Expected 0, got {val}"

    def test_surrogate_leq_exact(self):
        """Surrogate <= exact for random inputs."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            tau = rng.standard_normal((self.d, self.d))
            dW = np.zeros((self.d, self.d))  # no mechanism shift
            dW[1, 0] = rng.uniform(-0.1, 0.1)
            exact = self.loss.value(tau, dW, self.W, self.A, self.R,
                                    self.mu_s, self.Sigma_s)
            surr = self.loss.surrogate(tau, dW, self.W, self.A, self.R,
                                       self.mu_s, self.Sigma_s)
            assert surr <= exact + 1e-9, f"Surrogate {surr} > exact {exact}"

    def test_nonnegative(self):
        rng = np.random.default_rng(5)
        tau = rng.standard_normal((self.d, self.d))
        dW = np.zeros((self.d, self.d))
        val = self.loss.value(tau, dW, self.W, self.A, self.R, self.mu_s, self.Sigma_s)
        assert val >= -1e-10

    def test_grad_tau_numerical(self):
        """grad_tau should agree with numerical gradient."""
        rng = np.random.default_rng(10)
        tau = np.eye(self.d) + 0.1 * rng.standard_normal((self.d, self.d))
        dW = np.zeros((self.d, self.d))
        mu_t = self.mu_s @ self.A
        Sigma_t = self.A.T @ self.Sigma_s @ self.A

        def f(t):
            return self.loss.surrogate(t, dW, self.W, self.A, self.R,
                                       self.mu_s, self.Sigma_s, mu_t, Sigma_t)

        analytic = self.loss.grad_tau(tau, dW, self.W, self.A, self.R,
                                       self.mu_s, self.Sigma_s, mu_t, Sigma_t)
        numeric = numerical_gradient(f, tau)
        assert np.allclose(analytic, numeric, atol=1e-5), \
            f"Max diff: {np.max(np.abs(analytic - numeric)):.2e}"

    def test_grad_dW_gaussian(self):
        """grad_dW must match numerical gradient with adversary fixed in EXOGENOUS space.

        Adversary params (mu_t_exo, Sigma_t_exo) are held fixed in exogenous space.
        For each perturbed dW inside f(dw), mu_t_obs and Sigma_t_obs are recomputed
        from the perturbed A' — this is the exact chain-rule path that grad_dW implements.
        Freezing the observed params instead would bypass the chain-rule and give a
        structurally invalid test.
        """
        rng = np.random.default_rng(30)
        tau = np.eye(self.d) + 0.1 * rng.standard_normal((self.d, self.d))
        dW = np.zeros((self.d, self.d))
        dW[0, 1] = 0.1

        # Adversary fixed in exogenous space
        mu_t_exo = np.array([0.1, -0.05])
        Sigma_t_exo = np.eye(self.d) * 1.1

        def f(dw):
            Ap = np.linalg.inv(np.eye(self.d) - self.W - dw)
            mo = mu_t_exo @ Ap              # recomputed from perturbed A'
            So = Ap.T @ Sigma_t_exo @ Ap   # recomputed from perturbed A'
            return self.loss.surrogate(tau, dw, self.W, self.A, self.R,
                                       self.mu_s, self.Sigma_s, mo, So)

        # Nominal observed params for analytic call
        A_prime_nom = np.linalg.inv(np.eye(self.d) - self.W - dW)
        mu_t_obs = mu_t_exo @ A_prime_nom
        Sigma_t_obs = A_prime_nom.T @ Sigma_t_exo @ A_prime_nom

        analytic = self.loss.grad_dW(tau, dW, self.W, self.A, self.R,
                                      self.mu_s, self.Sigma_s, mu_t_obs, Sigma_t_obs)
        numeric = numerical_gradient(f, dW)
        assert np.allclose(analytic, numeric, atol=1e-4), \
            f"grad_dW max diff: {np.max(np.abs(analytic - numeric)):.2e}"

    def test_grad_dW_nonidentity_R(self):
        """grad_dW with non-identity R distinguishes left-mult from right-mult.

        Uses d=3, R=diag(1.0, 0.7, 0.3) — unequal diagonal entries make
        left-mult and right-mult resolvent identities produce different results.
        Only the right-mult formula (our convention) matches finite differences.
        """
        d = 3
        W = np.zeros((d, d))
        W[0, 1] = 0.4
        W[0, 2] = 0.3
        W[1, 2] = 0.6
        A = np.linalg.inv(np.eye(d) - W)
        R = np.diag([1.0, 0.7, 0.3])

        rng = np.random.default_rng(77)
        mu_s = np.array([0.3, -0.2, 0.1])
        Sigma_s = np.array([[1.2, 0.2, 0.1],
                            [0.2, 0.9, 0.15],
                            [0.1, 0.15, 1.1]])
        tau = np.eye(d) + 0.12 * rng.standard_normal((d, d))
        dW = np.zeros((d, d))
        dW[0, 1] = 0.08
        dW[0, 2] = -0.05

        # Adversary fixed in exogenous space
        mu_t_exo = np.array([0.1, -0.1, 0.05])
        Sigma_t_exo = np.diag([1.1, 0.95, 1.05])

        # A' uses right-mult: (I - (W+dW) @ R)^{-1}
        A_prime = np.linalg.inv(np.eye(d) - (W + dW) @ R)
        mu_t_obs = mu_t_exo @ A_prime
        Sigma_t_obs = A_prime.T @ Sigma_t_exo @ A_prime

        loss = GaussianLoss()

        def f(dw):
            Ap = np.linalg.inv(np.eye(d) - (W + dw) @ R)
            mo = mu_t_exo @ Ap
            So = Ap.T @ Sigma_t_exo @ Ap
            return loss.surrogate(tau, dw, W, A, R, mu_s, Sigma_s, mo, So)

        analytic = loss.grad_dW(tau, dW, W, A, R, mu_s, Sigma_s, mu_t_obs, Sigma_t_obs)
        numeric = numerical_gradient(f, dW)
        max_diff = np.max(np.abs(analytic - numeric))
        assert max_diff < 1e-4, \
            f"grad_dW non-identity R: max diff = {max_diff:.2e} (fails → wrong R convention)"

    def test_grad_Sigma_t_closed_form(self):
        """grad_Sigma_t closed-form matches finite differences."""
        rng = np.random.default_rng(55)
        tau = np.eye(self.d) + 0.1 * rng.standard_normal((self.d, self.d))
        dW = np.zeros((self.d, self.d))
        mu_t = np.array([0.1, -0.05])
        Sigma_t = np.eye(self.d) * 1.2

        analytic = self.loss.grad_Sigma_t(
            tau, dW, self.W, self.A, self.R,
            self.mu_s, self.Sigma_s, mu_t, Sigma_t,
        )

        # Numerical gradient
        eps = 1e-6
        d = self.d
        numeric = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                e = np.zeros((d, d))
                e[i, j] = eps
                if i != j:
                    e[j, i] = eps
                f_plus = self.loss.surrogate(
                    tau, dW, self.W, self.A, self.R,
                    self.mu_s, self.Sigma_s, mu_t, Sigma_t + e,
                )
                f_minus = self.loss.surrogate(
                    tau, dW, self.W, self.A, self.R,
                    self.mu_s, self.Sigma_s, mu_t, Sigma_t - e,
                )
                g = (f_plus - f_minus) / (2 * eps)
                numeric[i, j] = g
                numeric[j, i] = g

        max_diff = np.max(np.abs(analytic - numeric))
        assert max_diff < 1e-5, \
            f"grad_Sigma_t closed-form vs FD: max diff = {max_diff:.2e}"

        # Structure check: must be scalar * I
        off_diag = analytic - np.diag(np.diag(analytic))
        assert np.allclose(off_diag, 0.0, atol=1e-15), "grad_Sigma_t must be diagonal"
        assert np.allclose(analytic[0, 0], analytic[1, 1], atol=1e-15), \
            "grad_Sigma_t must be scalar * I"

    def test_grad_tau_offdiagonal_covariance(self):
        """grad_tau with off-diagonal Sigma_s (latent confounder between X and Y).

        Chain X->Z->Y with hidden X<->Y confounder, Cov(U_X, U_Y)=0.5.
        Sigma_s is dense-symmetric with genuine off-diagonals. A is upper-triangular
        (non-symmetric), so this test catches both sandwich-orientation errors
        (A.T Sigma A vs A Sigma A.T) and off-diagonal Sigma handling.

        Variables: X=0, Z=1, Y=2. Edges: X->Z (0.5), Z->Y (0.8).
        """
        # Chain X->Z->Y with hidden X<->Y confounder
        d = 3
        # W[src, dst]: W[X,Z]=0.5, W[Z,Y]=0.8 (row-vector X=UA convention)
        W = np.zeros((d, d))
        W[0, 1] = 0.5   # X -> Z
        W[1, 2] = 0.8   # Z -> Y
        A = np.linalg.inv(np.eye(d) - W)
        R = np.eye(d)  # no intervention (observational)

        mu_s = np.zeros(d)
        # Sigma_s: noise_std=[1,1,1], Cov(U_X, U_Y)=0.5
        Sigma_s = np.eye(d)
        Sigma_s[0, 2] = 0.5
        Sigma_s[2, 0] = 0.5  # symmetric, off-diagonal confounder entry

        mu_t = mu_s @ A
        Sigma_t = A.T @ Sigma_s @ A

        rng = np.random.default_rng(42)
        tau = np.eye(d) + 0.1 * rng.standard_normal((d, d))

        def f(t):
            return self.loss.surrogate(t, np.zeros((d, d)), W, A, R,
                                       mu_s, Sigma_s, mu_t, Sigma_t)

        analytic = self.loss.grad_tau(tau, np.zeros((d, d)), W, A, R,
                                      mu_s, Sigma_s, mu_t, Sigma_t)
        numeric = numerical_gradient(f, tau)
        assert np.allclose(analytic, numeric, atol=1e-5), \
            f"off-diagonal covariance grad_tau: max diff = {np.max(np.abs(analytic - numeric)):.2e}"


# ---------------------------------------------------------------------------
# EmpiricalLoss
# ---------------------------------------------------------------------------

class TestEmpiricalLoss:
    def setup_method(self):
        self.loss = EmpiricalLoss()
        self.d, self.W, self.A, self.R, self.mu_s, self.Sigma_s = make_2node()
        rng = np.random.default_rng(0)
        self.N = 50
        self.U_s = rng.multivariate_normal(self.mu_s, self.Sigma_s, self.N)

    def test_zero_at_identity_no_shift(self):
        """F = 0 when tau=I, dW=0, Theta=0."""
        tau = np.eye(self.d)
        dW = np.zeros((self.d, self.d))
        Theta = np.zeros((self.N, self.d))
        val = self.loss.value(tau, dW, Theta, self.W, self.A, self.R, self.U_s)
        assert np.isclose(val, 0.0, atol=1e-10), f"Expected 0, got {val}"

    def test_nonnegative(self):
        rng = np.random.default_rng(11)
        tau = rng.standard_normal((self.d, self.d))
        dW = np.zeros((self.d, self.d))
        Theta = np.zeros((self.N, self.d))
        val = self.loss.value(tau, dW, Theta, self.W, self.A, self.R, self.U_s)
        assert val >= -1e-10

    def test_hand_computed_2node(self):
        """Hand-computed check: tau=2*I, dW=0, Theta=0.
        F = (1/N) ||U_s A (2I - I)||_F^2 = (1/N) ||U_s A||_F^2
        """
        tau = 2.0 * np.eye(self.d)
        dW = np.zeros((self.d, self.d))
        Theta = np.zeros((self.N, self.d))
        val = self.loss.value(tau, dW, Theta, self.W, self.A, self.R, self.U_s)
        # pushed = U_s A (2I) = 2 U_s A; target = U_s A I = U_s A
        # residual = U_s A (2I - I) = U_s A
        expected = float(np.linalg.norm(self.U_s @ self.A, "fro") ** 2) / self.N
        assert np.isclose(val, expected, atol=1e-8)

    def test_grad_tau_analytic(self):
        """grad_tau analytic should agree with numerical gradient."""
        rng = np.random.default_rng(20)
        tau = np.eye(self.d) + 0.1 * rng.standard_normal((self.d, self.d))
        dW = np.zeros((self.d, self.d))
        Theta = 0.01 * rng.standard_normal((self.N, self.d))

        def f(t):
            return self.loss.value(t, dW, Theta, self.W, self.A, self.R, self.U_s)

        analytic = self.loss.grad_tau(tau, dW, Theta, self.W, self.A, self.R, self.U_s)
        numeric = numerical_gradient(f, tau)
        assert np.allclose(analytic, numeric, atol=1e-5), \
            f"Max diff: {np.max(np.abs(analytic - numeric)):.2e}"

    def test_grad_Theta_analytic(self):
        """grad_Theta analytic should agree with numerical gradient."""
        rng = np.random.default_rng(21)
        tau = np.eye(self.d)
        dW = np.zeros((self.d, self.d))
        Theta = 0.01 * rng.standard_normal((self.N, self.d))

        def f(th):
            return self.loss.value(tau, dW, th, self.W, self.A, self.R, self.U_s)

        analytic = self.loss.grad_Theta(tau, dW, Theta, self.W, self.A, self.R, self.U_s)
        numeric = numerical_gradient(f, Theta)
        assert np.allclose(analytic, numeric, atol=1e-5), \
            f"Max diff: {np.max(np.abs(analytic - numeric)):.2e}"

    def test_grad_dW_analytic(self):
        """grad_dW analytic should agree with numerical gradient."""
        rng = np.random.default_rng(22)
        tau = np.eye(self.d)
        dW = 0.05 * rng.standard_normal((self.d, self.d))
        Theta = np.zeros((self.N, self.d))

        def f(dw):
            return self.loss.value(tau, dw, Theta, self.W, self.A, self.R, self.U_s)

        analytic = self.loss.grad_dW(tau, dW, Theta, self.W, self.A, self.R, self.U_s)
        numeric = numerical_gradient(f, dW)
        assert np.allclose(analytic, numeric, atol=1e-5), \
            f"Max diff: {np.max(np.abs(analytic - numeric)):.2e}"
