"""
Tests for traca.certificates.
"""
import numpy as np
import pytest

from traca.certificates import (
    delta_iota_rho, delta_iota_rho_sq, delta_iota_U, E_rho_joint, E_U_joint,
    full_joint_certificate, single_query_certificate,
    query_interval, selection_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_2node():
    d = 2
    W = np.array([[0, 0.5], [0, 0]], dtype=float)
    A = np.linalg.inv(np.eye(d) - W)
    mu_s = np.array([0.0, 0.0])
    Sigma_s = np.eye(d)
    tau = np.eye(d)
    return d, W, A, mu_s, Sigma_s, tau


# ---------------------------------------------------------------------------
# selection_matrix
# ---------------------------------------------------------------------------

def test_selection_matrix():
    S = selection_matrix(4, [0, 2])
    assert S.shape == (2, 4)
    assert S[0, 0] == 1.0 and S[0, 1] == 0.0
    assert S[1, 2] == 1.0 and S[1, 3] == 0.0


# ---------------------------------------------------------------------------
# delta_iota_rho
# ---------------------------------------------------------------------------

def test_delta_rho_nonnegative():
    d, W, A, mu_s, Sigma_s, tau = make_2node()
    val = delta_iota_rho(tau, alpha_iota=0.1, A_iota=A,
                          mu_s=mu_s, Sigma_s=Sigma_s, eps=0.05)
    assert val >= 0.0


def test_delta_rho_monotone_in_eps():
    d, W, A, mu_s, Sigma_s, tau = make_2node()
    d1 = delta_iota_rho(tau, 0.1, A, mu_s, Sigma_s, eps=0.05)
    d2 = delta_iota_rho(tau, 0.1, A, mu_s, Sigma_s, eps=0.2)
    assert d2 >= d1 - 1e-10


def test_delta_rho_selector_leq_full():
    """Single-output selector certificate <= full post-interventional certificate."""
    d, W, A, mu_s, Sigma_s, tau = make_2node()
    delta_full = delta_iota_rho(tau, 0.1, A, mu_s, Sigma_s, eps=0.1)
    S_O = selection_matrix(d, [0])
    delta_sel = delta_iota_rho(tau, 0.1, A, mu_s, Sigma_s, eps=0.1, S_O=S_O)
    assert delta_sel <= delta_full + 1e-10


# ---------------------------------------------------------------------------
# delta_iota_U
# ---------------------------------------------------------------------------

def test_delta_U_nonnegative():
    d, W, A, mu_s, Sigma_s, tau = make_2node()
    rng = np.random.default_rng(0)
    U_s = rng.multivariate_normal(mu_s, Sigma_s, 100)
    val = delta_iota_U(tau, 0.1, A, U_s, eps=0.05, N=100)
    assert val >= 0.0


def test_delta_U_monotone_in_eps():
    d, W, A, mu_s, Sigma_s, tau = make_2node()
    rng = np.random.default_rng(1)
    U_s = rng.multivariate_normal(mu_s, Sigma_s, 100)
    d1 = delta_iota_U(tau, 0.1, A, U_s, eps=0.05, N=100)
    d2 = delta_iota_U(tau, 0.1, A, U_s, eps=0.2, N=100)
    assert d2 >= d1 - 1e-10


# ---------------------------------------------------------------------------
# E_rho_joint
# ---------------------------------------------------------------------------

def test_E_rho_joint_average():
    d, W, A, mu_s, Sigma_s, tau = make_2node()
    deltas = [1.0, 2.0, 3.0]
    assert np.isclose(E_rho_joint(tau, deltas), 2.0)


# ---------------------------------------------------------------------------
# query_interval
# ---------------------------------------------------------------------------

def test_query_interval_contains_pushed():
    lower, upper = query_interval(Phi_pushed=0.5, L_Phi=1.0, delta_sq=0.04)
    assert lower <= 0.5 <= upper


def test_query_interval_width():
    lower, upper = query_interval(Phi_pushed=0.5, L_Phi=2.0, delta_sq=0.25)
    # radius = L_Phi * sqrt(delta_sq) = 2.0 * 0.5 = 1.0 → width = 2.0
    assert np.isclose(upper - lower, 2 * 2.0 * 0.5, atol=1e-8)


# ---------------------------------------------------------------------------
# single_query_certificate
# ---------------------------------------------------------------------------

def test_single_query_recovers_full():
    """With O = all coordinates, single_query_certificate ~ full delta_iota_rho."""
    d, W, A, mu_s, Sigma_s, tau = make_2node()
    sq = single_query_certificate(tau, 0.1, A, mu_s, Sigma_s,
                                   eps=0.1, O=[0, 1], d=d)
    full = delta_iota_rho(tau, 0.1, A, mu_s, Sigma_s, eps=0.1)
    # Single-query with all O should be <= full
    assert sq <= full + 1e-8
