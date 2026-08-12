"""
Tests for traca.query.
"""
import numpy as np
import pytest

from traca.query import S_O, F_iota_O_rho, F_iota_O_U
from traca.losses import GaussianLoss, EmpiricalLoss
from traca.stability import gating_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_2node():
    d = 2
    W = np.array([[0, 0.5], [0, 0]], dtype=float)
    A = np.linalg.inv(np.eye(d) - W)
    R = gating_matrix(d, [])
    mu_s = np.array([0.0, 0.0])
    Sigma_s = np.eye(d)
    return d, W, A, R, mu_s, Sigma_s


# ---------------------------------------------------------------------------
# S_O
# ---------------------------------------------------------------------------

def test_S_O_shape():
    S = S_O(4, [1, 3])
    assert S.shape == (2, 4)
    assert S[0, 1] == 1.0
    assert S[1, 3] == 1.0
    assert S[0, 0] == 0.0


# ---------------------------------------------------------------------------
# F_iota_O_rho
# ---------------------------------------------------------------------------

class TestFiotaOrho:
    def setup_method(self):
        self.d, self.W, self.A, self.R, self.mu_s, self.Sigma_s = make_2node()

    def test_full_O_recovers_full_joint(self):
        """F_iota_O_rho with O=[0,1] should equal GaussianLoss.value."""
        tau = np.eye(self.d)
        dW = np.zeros((self.d, self.d))
        full_loss = GaussianLoss().value(
            tau, dW, self.W, self.A, self.R, self.mu_s, self.Sigma_s
        )
        restricted = F_iota_O_rho(
            tau, dW, self.W, self.A, self.R, self.mu_s, self.Sigma_s,
            None, None, [0, 1]
        )
        assert np.isclose(full_loss, restricted, atol=1e-8), \
            f"full={full_loss:.8f}, restricted={restricted:.8f}"

    def test_singleton_O_gives_scalar_loss(self):
        """F_iota_O_rho with O=[0] gives a scalar (1D Gaussian loss)."""
        tau = np.eye(self.d)
        dW = np.zeros((self.d, self.d))
        val = F_iota_O_rho(
            tau, dW, self.W, self.A, self.R, self.mu_s, self.Sigma_s,
            None, None, [0]
        )
        assert isinstance(val, float)
        assert val >= 0.0

    def test_nonnegative(self):
        rng = np.random.default_rng(5)
        tau = rng.standard_normal((self.d, self.d))
        dW = np.zeros((self.d, self.d))
        val = F_iota_O_rho(
            tau, dW, self.W, self.A, self.R, self.mu_s, self.Sigma_s,
            None, None, [1]
        )
        assert val >= -1e-10


# ---------------------------------------------------------------------------
# F_iota_O_U
# ---------------------------------------------------------------------------

class TestFiotaOU:
    def setup_method(self):
        self.d, self.W, self.A, self.R, self.mu_s, self.Sigma_s = make_2node()
        self.N = 50
        rng = np.random.default_rng(0)
        self.U_s = rng.multivariate_normal(self.mu_s, self.Sigma_s, self.N)

    def test_full_O_recovers_full_joint(self):
        """F_iota_O_U with O=[0,1] should equal EmpiricalLoss.value."""
        tau = np.eye(self.d)
        dW = np.zeros((self.d, self.d))
        Theta = np.zeros((self.N, self.d))
        full_loss = EmpiricalLoss().value(tau, dW, Theta, self.W, self.A, self.R, self.U_s)
        restricted = F_iota_O_U(tau, dW, Theta, self.W, self.A, self.R, self.U_s, [0, 1])
        assert np.isclose(full_loss, restricted, atol=1e-8)

    def test_singleton_O_gives_scalar(self):
        tau = np.eye(self.d)
        dW = np.zeros((self.d, self.d))
        Theta = np.zeros((self.N, self.d))
        val = F_iota_O_U(tau, dW, Theta, self.W, self.A, self.R, self.U_s, [0])
        assert isinstance(val, float)
        assert val >= 0.0

    def test_nonnegative(self):
        rng = np.random.default_rng(3)
        tau = rng.standard_normal((self.d, self.d))
        dW = np.zeros((self.d, self.d))
        Theta = np.zeros((self.N, self.d))
        val = F_iota_O_U(tau, dW, Theta, self.W, self.A, self.R, self.U_s, [1])
        assert val >= -1e-10
