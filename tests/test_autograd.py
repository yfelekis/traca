"""
Gradient verification: analytic TraCA gradients vs PyTorch autograd.

Four tests:
    1. grad_tau (analytic, surrogate) vs torch autograd through surrogate
    2. Surrogate vs exact τ gradient gap — diagnostic, no assertion
    3. grad_dW (analytic) vs torch autograd through surrogate
    4. grad_mu_t (analytic) vs torch autograd through exact mean term

All tests use a non-trivial adversary (ΔW≠0, μ_t≠μ_s@A, Σ_t≠A.T@Σ_s@A)
to exercise the full chain-rule paths in the corrected G_mu / G_Sigma formulas.

The entire module is skipped cleanly if PyTorch is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping autograd tests")

from traca.losses import GaussianLoss
from traca.losses_torch import (
    _to_tensor,
    surrogate_torch,
    exact_loss_torch,
    perturbed_propagator_torch,
)


# ---------------------------------------------------------------------------
# Shared fixture: 2-node chain SCM with non-trivial adversary
# ---------------------------------------------------------------------------

def _make_nontrivial() -> dict:
    """2-node SCM X0→X1 with a non-zero adversary.

    ΔW[0,1]=0.1 (mechanism shift), μ_t ≠ μ_s@A, Σ_t ≠ A.T@Σ_s@A.
    Adversary params are fixed in EXOGENOUS space (as in optim._adv_update).
    """
    d = 2
    W = np.array([[0.0, 0.5], [0.0, 0.0]])
    A = np.linalg.inv(np.eye(d) - W)
    R = np.eye(d)
    mu_s = np.array([0.2, -0.1])
    Sigma_s = np.array([[1.2, 0.3], [0.3, 0.9]])

    rng = np.random.default_rng(99)
    tau = np.eye(d) + 0.15 * rng.standard_normal((d, d))
    dW = np.zeros((d, d))
    dW[0, 1] = 0.1

    # Adversary fixed in exogenous space
    mu_t_exo = np.array([0.15, -0.08])
    Sigma_t_exo = np.array([[1.1, 0.2], [0.2, 0.85]])

    # Convert to observed space at nominal dW (for analytic calls)
    A_prime = np.linalg.inv(np.eye(d) - W - dW)
    mu_t_obs = mu_t_exo @ A_prime
    Sigma_t_obs = A_prime.T @ Sigma_t_exo @ A_prime

    return dict(
        d=d, W=W, A=A, R=R,
        mu_s=mu_s, Sigma_s=Sigma_s,
        tau=tau, dW=dW,
        mu_t_exo=mu_t_exo, Sigma_t_exo=Sigma_t_exo,
        mu_t_obs=mu_t_obs, Sigma_t_obs=Sigma_t_obs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAutograd:

    def setup_method(self):
        self.loss = GaussianLoss()
        self.p = _make_nontrivial()

    def _t(self, x, requires_grad: bool = False) -> "torch.Tensor":
        t = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64)
        if requires_grad:
            t.requires_grad_(True)
        return t

    # ------------------------------------------------------------------
    # Test 1: grad_tau analytic (surrogate) vs torch autograd (surrogate)
    # ------------------------------------------------------------------
    def test_grad_tau_analytic_vs_surrogate_autograd(self):
        """grad_tau (analytic) matches torch autograd through surrogate, atol=1e-4."""
        p = self.p

        tau_t = self._t(p["tau"], requires_grad=True)
        W_t = self._t(p["W"])
        A_t = self._t(p["A"])
        R_t = self._t(p["R"])
        mu_s_t = self._t(p["mu_s"])
        Sigma_s_t = self._t(p["Sigma_s"])
        mu_t_t = self._t(p["mu_t_obs"])
        Sigma_t_t = self._t(p["Sigma_t_obs"])
        dW_t = self._t(p["dW"])

        loss_val = surrogate_torch(
            tau_t, dW_t, W_t, A_t, R_t, mu_s_t, Sigma_s_t, mu_t_t, Sigma_t_t
        )
        loss_val.backward()
        g_tau_autograd = tau_t.grad.detach().numpy()

        g_tau_analytic = self.loss.grad_tau(
            p["tau"], p["dW"], p["W"], p["A"], p["R"],
            p["mu_s"], p["Sigma_s"], p["mu_t_obs"], p["Sigma_t_obs"],
        )

        max_diff = float(np.max(np.abs(g_tau_analytic - g_tau_autograd)))
        assert np.allclose(g_tau_analytic, g_tau_autograd, atol=1e-4), (
            f"grad_tau analytic vs autograd (surrogate): max diff = {max_diff:.2e}"
        )

    # ------------------------------------------------------------------
    # Test 2: surrogate vs exact τ gradient gap — diagnostic, no assertion
    # ------------------------------------------------------------------
    def test_grad_tau_surrogate_vs_exact_gap(self):
        """Report ‖∇_τF̃ − ∇_τF‖_F.  No assertion — purely diagnostic."""
        p = self.p

        W_t = self._t(p["W"])
        A_t = self._t(p["A"])
        R_t = self._t(p["R"])
        mu_s_t = self._t(p["mu_s"])
        Sigma_s_t = self._t(p["Sigma_s"])
        mu_t_t = self._t(p["mu_t_obs"])
        Sigma_t_t = self._t(p["Sigma_t_obs"])
        dW_t = self._t(p["dW"])

        # Surrogate gradient
        tau_s = self._t(p["tau"], requires_grad=True)
        surrogate_torch(
            tau_s, dW_t, W_t, A_t, R_t, mu_s_t, Sigma_s_t, mu_t_t, Sigma_t_t
        ).backward()
        g_surr = tau_s.grad.detach().numpy().copy()

        # Exact gradient
        tau_e = self._t(p["tau"], requires_grad=True)
        exact_loss_torch(
            tau_e, dW_t, W_t, A_t, R_t, mu_s_t, Sigma_s_t, mu_t_t, Sigma_t_t
        ).backward()
        g_exact = tau_e.grad.detach().numpy().copy()

        gap_abs = float(np.linalg.norm(g_surr - g_exact, "fro"))
        gap_rel = gap_abs / (float(np.linalg.norm(g_exact, "fro")) + 1e-12)
        print(
            f"\n[diagnostic] ‖∇_τF̃ − ∇_τF‖_F = {gap_abs:.4e}  "
            f"(relative: {gap_rel:.4e})"
        )
        # No assertion

    # ------------------------------------------------------------------
    # Test 3: grad_dW analytic vs torch autograd (surrogate)
    # ------------------------------------------------------------------
    def test_grad_dW_analytic_vs_autograd(self):
        """grad_dW (analytic) matches torch autograd through surrogate, atol=1e-4.

        The adversary is fixed in EXOGENOUS space.  Observed params (mu_t_obs,
        Sigma_t_obs) are recomputed from the perturbed A' inside the torch
        graph, which is the same chain-rule path that grad_dW implements.
        """
        p = self.p

        W_t = self._t(p["W"])
        A_t = self._t(p["A"])
        R_t = self._t(p["R"])
        mu_s_t = self._t(p["mu_s"])
        Sigma_s_t = self._t(p["Sigma_s"])
        tau_t = self._t(p["tau"])
        mu_t_exo_t = self._t(p["mu_t_exo"])
        Sigma_t_exo_t = self._t(p["Sigma_t_exo"])

        dW_t = self._t(p["dW"], requires_grad=True)

        # Recompute observed params from A'(dW) inside the graph
        A_prime_t = perturbed_propagator_torch(W_t, dW_t, R_t)
        mu_t_obs_t = mu_t_exo_t @ A_prime_t
        Sigma_t_obs_t = A_prime_t.T @ Sigma_t_exo_t @ A_prime_t

        loss_val = surrogate_torch(
            tau_t, dW_t, W_t, A_t, R_t, mu_s_t, Sigma_s_t, mu_t_obs_t, Sigma_t_obs_t
        )
        loss_val.backward()
        g_dW_autograd = dW_t.grad.detach().numpy()

        g_dW_analytic = self.loss.grad_dW(
            p["tau"], p["dW"], p["W"], p["A"], p["R"],
            p["mu_s"], p["Sigma_s"], p["mu_t_obs"], p["Sigma_t_obs"],
        )

        max_diff = float(np.max(np.abs(g_dW_analytic - g_dW_autograd)))
        assert np.allclose(g_dW_analytic, g_dW_autograd, atol=1e-4), (
            f"grad_dW analytic vs autograd (surrogate): max diff = {max_diff:.2e}"
        )

    # ------------------------------------------------------------------
    # Test 3b: grad_dW with non-identity R (distinguishes conventions)
    # ------------------------------------------------------------------
    def test_grad_dW_nonidentity_R_autograd(self):
        """grad_dW with R=diag(1.0, 0.7, 0.3) matches autograd.

        Non-identity R makes left-mult and right-mult resolvent identities
        differ. Only the right-mult formula (our convention) passes.
        """
        d = 3
        W = np.zeros((d, d))
        W[0, 1] = 0.4
        W[0, 2] = 0.3
        W[1, 2] = 0.6
        A = np.linalg.inv(np.eye(d) - W)
        R = np.diag([1.0, 0.7, 0.3])

        mu_s = np.array([0.3, -0.2, 0.1])
        Sigma_s = np.array([[1.2, 0.2, 0.1],
                            [0.2, 0.9, 0.15],
                            [0.1, 0.15, 1.1]])

        rng = np.random.default_rng(77)
        tau = np.eye(d) + 0.12 * rng.standard_normal((d, d))
        dW = np.zeros((d, d))
        dW[0, 1] = 0.08
        dW[0, 2] = -0.05

        mu_t_exo = np.array([0.1, -0.1, 0.05])
        Sigma_t_exo = np.diag([1.1, 0.95, 1.05])

        # Observed params at nominal dW
        A_prime = np.linalg.inv(np.eye(d) - (W + dW) @ R)
        mu_t_obs = mu_t_exo @ A_prime
        Sigma_t_obs = A_prime.T @ Sigma_t_exo @ A_prime

        # Autograd: differentiate through A'(dW)
        W_t = self._t(W)
        A_t = self._t(A)
        R_t = self._t(R)
        mu_s_t = self._t(mu_s)
        Sigma_s_t = self._t(Sigma_s)
        tau_t = self._t(tau)
        mu_t_exo_t = self._t(mu_t_exo)
        Sigma_t_exo_t = self._t(Sigma_t_exo)

        dW_t = self._t(dW, requires_grad=True)

        A_prime_t = perturbed_propagator_torch(W_t, dW_t, R_t)
        mu_t_obs_t = mu_t_exo_t @ A_prime_t
        Sigma_t_obs_t = A_prime_t.T @ Sigma_t_exo_t @ A_prime_t

        loss_val = surrogate_torch(
            tau_t, dW_t, W_t, A_t, R_t, mu_s_t, Sigma_s_t, mu_t_obs_t, Sigma_t_obs_t
        )
        loss_val.backward()
        g_dW_autograd = dW_t.grad.detach().numpy()

        # Analytic
        loss_fn = GaussianLoss()
        g_dW_analytic = loss_fn.grad_dW(
            tau, dW, W, A, R, mu_s, Sigma_s, mu_t_obs, Sigma_t_obs,
        )

        max_diff = float(np.max(np.abs(g_dW_analytic - g_dW_autograd)))
        assert max_diff < 1e-4, (
            f"grad_dW non-identity R analytic vs autograd: max diff = {max_diff:.2e}"
        )

    # ------------------------------------------------------------------
    # Test 3c: grad_Sigma_t closed-form vs autograd
    # ------------------------------------------------------------------
    def test_grad_Sigma_t_vs_autograd(self):
        """grad_Sigma_t closed-form matches autograd through surrogate."""
        p = self.p

        W_t = self._t(p["W"])
        A_t = self._t(p["A"])
        R_t = self._t(p["R"])
        mu_s_t = self._t(p["mu_s"])
        Sigma_s_t = self._t(p["Sigma_s"])
        mu_t_t = self._t(p["mu_t_obs"])
        tau_t = self._t(p["tau"])
        dW_t = self._t(p["dW"])

        Sigma_t_t = self._t(p["Sigma_t_obs"], requires_grad=True)

        loss_val = surrogate_torch(
            tau_t, dW_t, W_t, A_t, R_t, mu_s_t, Sigma_s_t, mu_t_t, Sigma_t_t
        )
        loss_val.backward()
        g_Sigma_autograd = Sigma_t_t.grad.detach().numpy()

        loss_fn = GaussianLoss()
        g_Sigma_analytic = loss_fn.grad_Sigma_t(
            p["tau"], p["dW"], p["W"], p["A"], p["R"],
            p["mu_s"], p["Sigma_s"], p["mu_t_obs"], p["Sigma_t_obs"],
        )

        max_diff = float(np.max(np.abs(g_Sigma_analytic - g_Sigma_autograd)))
        assert max_diff < 1e-5, (
            f"grad_Sigma_t analytic vs autograd: max diff = {max_diff:.2e}"
        )

    # ------------------------------------------------------------------
    # Test 4: grad_mu_t analytic vs torch autograd
    # ------------------------------------------------------------------
    def test_grad_mu_t_analytic_vs_autograd(self):
        """grad_mu_t (analytic) matches torch autograd, atol=1e-6.

        The mean term is exact in both surrogate and exact loss, so the
        gradient is identical — tight tolerance expected.
        """
        p = self.p

        W_t = self._t(p["W"])
        A_t = self._t(p["A"])
        R_t = self._t(p["R"])
        mu_s_t = self._t(p["mu_s"])
        Sigma_s_t = self._t(p["Sigma_s"])
        Sigma_t_t = self._t(p["Sigma_t_obs"])
        tau_t = self._t(p["tau"])
        dW_t = self._t(p["dW"])

        mu_t_t = self._t(p["mu_t_obs"], requires_grad=True)

        loss_val = surrogate_torch(
            tau_t, dW_t, W_t, A_t, R_t, mu_s_t, Sigma_s_t, mu_t_t, Sigma_t_t
        )
        loss_val.backward()
        g_mu_autograd = mu_t_t.grad.detach().numpy()

        g_mu_analytic = self.loss.grad_mu_t(
            p["tau"], p["dW"], p["W"], p["A"], p["R"],
            p["mu_s"], p["Sigma_s"], p["mu_t_obs"], p["Sigma_t_obs"],
        )

        max_diff = float(np.max(np.abs(g_mu_analytic - g_mu_autograd)))
        assert np.allclose(g_mu_analytic, g_mu_autograd, atol=1e-6), (
            f"grad_mu_t analytic vs autograd: max diff = {max_diff:.2e}"
        )

    # ------------------------------------------------------------------
    # Test 5: Default backend τ-step differentiates EXACT, not surrogate
    # ------------------------------------------------------------------
    def test_default_backend_tau_uses_exact(self):
        """Lock: default grad_backend's τ-gradient matches FD of exact W₂²,
        and does NOT match FD of the surrogate.

        Uses an ATE-like setup with off-diagonal pushed/target covariance
        so exact ≠ surrogate (the cross-term differs).
        """
        from traca.optim import OptimConfig
        assert OptimConfig().grad_backend == "autograd", (
            "Default grad_backend is not 'autograd' — someone flipped it back"
        )

        p = self.p

        W_t = self._t(p["W"])
        A_t = self._t(p["A"])
        R_t = self._t(p["R"])
        mu_s_t = self._t(p["mu_s"])
        Sigma_s_t = self._t(p["Sigma_s"])
        mu_t_t = self._t(p["mu_t_obs"])
        Sigma_t_t = self._t(p["Sigma_t_obs"])
        dW_t = self._t(p["dW"])

        # Autograd τ-gradient through EXACT W₂²
        tau_e = self._t(p["tau"], requires_grad=True)
        exact_loss_torch(
            tau_e, dW_t, W_t, A_t, R_t,
            mu_s_t, Sigma_s_t, mu_t_t, Sigma_t_t,
        ).backward()
        g_exact_autograd = tau_e.grad.detach().numpy().copy()

        # FD of exact W₂²
        eps = 1e-5
        loss_fn = GaussianLoss()
        tau_np = p["tau"].copy()
        g_fd_exact = np.zeros_like(tau_np)
        for i in range(tau_np.shape[0]):
            for j in range(tau_np.shape[1]):
                tau_np[i, j] += eps
                f_plus = loss_fn.value(
                    tau_np, p["dW"], p["W"], p["A"], p["R"],
                    p["mu_s"], p["Sigma_s"], p["mu_t_obs"], p["Sigma_t_obs"],
                )
                tau_np[i, j] -= 2 * eps
                f_minus = loss_fn.value(
                    tau_np, p["dW"], p["W"], p["A"], p["R"],
                    p["mu_s"], p["Sigma_s"], p["mu_t_obs"], p["Sigma_t_obs"],
                )
                g_fd_exact[i, j] = (f_plus - f_minus) / (2 * eps)
                tau_np[i, j] += eps

        # FD of surrogate
        g_fd_surr = np.zeros_like(tau_np)
        for i in range(tau_np.shape[0]):
            for j in range(tau_np.shape[1]):
                tau_np[i, j] += eps
                f_plus = loss_fn.surrogate(
                    tau_np, p["dW"], p["W"], p["A"], p["R"],
                    p["mu_s"], p["Sigma_s"], p["mu_t_obs"], p["Sigma_t_obs"],
                )
                tau_np[i, j] -= 2 * eps
                f_minus = loss_fn.surrogate(
                    tau_np, p["dW"], p["W"], p["A"], p["R"],
                    p["mu_s"], p["Sigma_s"], p["mu_t_obs"], p["Sigma_t_obs"],
                )
                g_fd_surr[i, j] = (f_plus - f_minus) / (2 * eps)
                tau_np[i, j] += eps

        diff_exact = float(np.max(np.abs(g_exact_autograd - g_fd_exact)))
        diff_surr = float(np.max(np.abs(g_exact_autograd - g_fd_surr)))

        # Must match FD-of-exact
        assert diff_exact < 1e-4, (
            f"Default τ-grad vs FD-of-exact: max diff = {diff_exact:.2e} — "
            f"τ-step is not differentiating the exact W₂²"
        )
        # Must NOT match FD-of-surrogate (exact ≠ surrogate for off-diagonal Σ)
        assert diff_surr > 1e-3, (
            f"Default τ-grad vs FD-of-surrogate: max diff = {diff_surr:.2e} — "
            f"exact and surrogate gradients should differ with off-diagonal Σ"
        )


# ---------------------------------------------------------------------------
# Empirical gradient parity: closed-form vs autograd oracle
# ---------------------------------------------------------------------------

def _empirical_loss_torch(tau, dW, Theta, W, A_iota, R_iota, U_s):
    """Differentiable PyTorch twin of EmpiricalLoss.value.

    (1/N) || U_s @ A_iota @ tau - (U_s + Theta) @ A'_iota ||_F^2
    """
    d = W.shape[0]
    I = torch.eye(d, dtype=W.dtype)
    F = I - (W + dW) @ R_iota
    A_prime = torch.linalg.solve(F, I)
    pushed = U_s @ A_iota @ tau
    target = (U_s + Theta) @ A_prime
    residual = pushed - target
    N = U_s.shape[0]
    return (torch.linalg.norm(residual, ord="fro") ** 2) / N


def _make_empirical_case(d, seed=42):
    """Build a random empirical setup with dW≠0, Θ≠0."""
    rng = np.random.default_rng(seed)
    W = np.zeros((d, d))
    # Fill strictly upper triangular
    for i in range(d):
        for j in range(i + 1, d):
            W[i, j] = rng.uniform(-0.5, 0.5)
    A = np.linalg.inv(np.eye(d) - W)
    R = np.eye(d)
    N = 50
    U_s = rng.standard_normal((N, d))
    tau = np.eye(d) + 0.1 * rng.standard_normal((d, d))
    dW = np.zeros((d, d))
    dW[0, 1] = 0.15
    if d > 2:
        dW[0, 2] = -0.1
    Theta = 0.3 * rng.standard_normal((N, d))
    return dict(d=d, W=W, A=A, R=R, U_s=U_s, tau=tau, dW=dW, Theta=Theta, N=N)


class TestEmpiricalGradientParity:
    """Closed-form empirical gradients vs PyTorch autograd oracle.

    The empirical training path is and must remain torch-free.
    This test uses autograd as an independent oracle to verify the
    closed-form formulas are exact.
    """

    @pytest.mark.parametrize("d,seed", [(3, 42), (6, 99)], ids=["d3_atce", "d6_lilucas"])
    def test_empirical_grad_tau_parity(self, d, seed):
        """Closed-form grad_tau matches autograd, proving the formula is exact."""
        from traca.losses import EmpiricalLoss
        p = _make_empirical_case(d, seed)
        loss_fn = EmpiricalLoss()

        # Closed-form
        g_cf = loss_fn.grad_tau(
            p["tau"], p["dW"], p["Theta"], p["W"], p["A"], p["R"], p["U_s"],
        )

        # Autograd
        tau_t = torch.tensor(p["tau"], dtype=torch.float64, requires_grad=True)
        W_t = torch.tensor(p["W"], dtype=torch.float64)
        A_t = torch.tensor(p["A"], dtype=torch.float64)
        R_t = torch.tensor(p["R"], dtype=torch.float64)
        U_t = torch.tensor(p["U_s"], dtype=torch.float64)
        dW_t = torch.tensor(p["dW"], dtype=torch.float64)
        Th_t = torch.tensor(p["Theta"], dtype=torch.float64)

        loss = _empirical_loss_torch(tau_t, dW_t, Th_t, W_t, A_t, R_t, U_t)
        loss.backward()
        g_ag = tau_t.grad.detach().numpy()

        diff = float(np.max(np.abs(g_cf - g_ag)))
        assert diff < 1e-8, (
            f"Empirical grad_tau d={d}: closed-form vs autograd max diff = {diff:.2e}"
        )

    @pytest.mark.parametrize("d,seed", [(3, 42), (6, 99)], ids=["d3_atce", "d6_lilucas"])
    def test_empirical_grad_dW_parity(self, d, seed):
        """Closed-form grad_dW matches autograd."""
        from traca.losses import EmpiricalLoss
        p = _make_empirical_case(d, seed)
        loss_fn = EmpiricalLoss()

        g_cf = loss_fn.grad_dW(
            p["tau"], p["dW"], p["Theta"], p["W"], p["A"], p["R"], p["U_s"],
        )

        dW_t = torch.tensor(p["dW"], dtype=torch.float64, requires_grad=True)
        tau_t = torch.tensor(p["tau"], dtype=torch.float64)
        W_t = torch.tensor(p["W"], dtype=torch.float64)
        A_t = torch.tensor(p["A"], dtype=torch.float64)
        R_t = torch.tensor(p["R"], dtype=torch.float64)
        U_t = torch.tensor(p["U_s"], dtype=torch.float64)
        Th_t = torch.tensor(p["Theta"], dtype=torch.float64)

        loss = _empirical_loss_torch(tau_t, dW_t, Th_t, W_t, A_t, R_t, U_t)
        loss.backward()
        g_ag = dW_t.grad.detach().numpy()

        diff = float(np.max(np.abs(g_cf - g_ag)))
        assert diff < 1e-8, (
            f"Empirical grad_dW d={d}: closed-form vs autograd max diff = {diff:.2e}"
        )

    @pytest.mark.parametrize("d,seed", [(3, 42), (6, 99)], ids=["d3_atce", "d6_lilucas"])
    def test_empirical_grad_Theta_parity(self, d, seed):
        """Closed-form grad_Theta matches autograd."""
        from traca.losses import EmpiricalLoss
        p = _make_empirical_case(d, seed)
        loss_fn = EmpiricalLoss()

        g_cf = loss_fn.grad_Theta(
            p["tau"], p["dW"], p["Theta"], p["W"], p["A"], p["R"], p["U_s"],
        )

        Th_t = torch.tensor(p["Theta"], dtype=torch.float64, requires_grad=True)
        tau_t = torch.tensor(p["tau"], dtype=torch.float64)
        W_t = torch.tensor(p["W"], dtype=torch.float64)
        A_t = torch.tensor(p["A"], dtype=torch.float64)
        R_t = torch.tensor(p["R"], dtype=torch.float64)
        U_t = torch.tensor(p["U_s"], dtype=torch.float64)
        dW_t = torch.tensor(p["dW"], dtype=torch.float64)

        loss = _empirical_loss_torch(tau_t, dW_t, Th_t, W_t, A_t, R_t, U_t)
        loss.backward()
        g_ag = Th_t.grad.detach().numpy()

        diff = float(np.max(np.abs(g_cf - g_ag)))
        assert diff < 1e-8, (
            f"Empirical grad_Theta d={d}: closed-form vs autograd max diff = {diff:.2e}"
        )
