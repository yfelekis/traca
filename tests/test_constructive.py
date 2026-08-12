"""
Tests for traca.constructive.ConstructiveClass.
"""
import numpy as np
import pytest

from traca.constructive import ConstructiveClass, resolve_districts


# ---------------------------------------------------------------------------
# resolve_districts
# ---------------------------------------------------------------------------

def test_resolve_districts_none():
    d = resolve_districts(3, None)
    assert d == ((0,), (1,), (2,))


def test_resolve_districts_explicit():
    d = resolve_districts(4, [[0, 1], [2], [3]])
    assert d == ((0, 1), (2,), (3,))


def test_resolve_districts_invalid():
    with pytest.raises(ValueError):
        resolve_districts(3, [[0, 1], [1, 2]])  # overlapping


# ---------------------------------------------------------------------------
# Markovian factory
# ---------------------------------------------------------------------------

def test_markovian_factory():
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    assert cc.is_markovian
    assert cc.shifted_nodes == (1,)
    assert cc.districts == ((0,), (1,), (2,))


def test_markovian_block_mask():
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    mask = cc.block_mask()
    # Diagonal only (singletons → no off-diagonal within district)
    assert np.allclose(mask, np.eye(3))


def test_markovian_project_identity():
    """project(I) should return I (invariant nodes fixed, shifted node free)."""
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    tau = np.eye(3)
    proj = cc.project(tau)
    assert np.allclose(proj, np.eye(3))


def test_markovian_project_zeroes_offdiag():
    """project should zero cross-district entries."""
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    tau = np.ones((3, 3))
    proj = cc.project(tau)
    # Only diagonal entries survive
    assert np.allclose(proj, np.diag(np.diag(proj)))


def test_markovian_project_invariant_fixed():
    """Invariant nodes' diagonal entries must be 1."""
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    tau = 5.0 * np.eye(3)
    proj = cc.project(tau)
    assert np.isclose(proj[0, 0], 1.0)  # invariant
    assert np.isclose(proj[2, 2], 1.0)  # invariant
    assert np.isclose(proj[1, 1], 5.0)  # shifted — free


def test_markovian_project_idempotent():
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    rng = np.random.default_rng(0)
    tau = rng.standard_normal((3, 3))
    p1 = cc.project(tau)
    p2 = cc.project(p1)
    assert np.allclose(p1, p2, atol=1e-12)


# ---------------------------------------------------------------------------
# Semi-Markovian factory
# ---------------------------------------------------------------------------

def test_semi_markovian_factory():
    cc = ConstructiveClass.from_districts(d=4, districts=[[0], [1, 2], [3]],
                                          shifted=[1, 2])
    assert not cc.is_markovian
    assert cc.shifted_nodes == (1, 2)


def test_semi_markovian_block_mask():
    cc = ConstructiveClass.from_districts(d=4, districts=[[0], [1, 2], [3]],
                                          shifted=[1, 2])
    mask = cc.block_mask()
    # District {1,2} should have a 2x2 block
    assert mask[1, 2] == 1.0
    assert mask[2, 1] == 1.0
    # Cross-district entries should be 0
    assert mask[0, 1] == 0.0
    assert mask[1, 3] == 0.0


def test_semi_markovian_project_invariant_block():
    """Invariant district {0} and {3} blocks are fixed to I."""
    cc = ConstructiveClass.from_districts(d=4, districts=[[0], [1, 2], [3]],
                                          shifted=[1, 2])
    tau = 3.0 * np.eye(4)
    proj = cc.project(tau)
    assert np.isclose(proj[0, 0], 1.0)  # invariant
    assert np.isclose(proj[3, 3], 1.0)  # invariant
    assert np.isclose(proj[1, 1], 3.0)  # shifted free
    assert np.isclose(proj[2, 2], 3.0)  # shifted free


def test_semi_markovian_project_zeroes_cross():
    """Cross-district entries are zero."""
    cc = ConstructiveClass.from_districts(d=4, districts=[[0], [1, 2], [3]],
                                          shifted=[1, 2])
    tau = np.ones((4, 4))
    proj = cc.project(tau)
    assert np.isclose(proj[0, 1], 0.0)
    assert np.isclose(proj[1, 3], 0.0)
    assert np.isclose(proj[0, 3], 0.0)


def test_semi_markovian_project_idempotent():
    cc = ConstructiveClass.from_districts(d=4, districts=[[0], [1, 2], [3]],
                                          shifted=[1, 2])
    rng = np.random.default_rng(1)
    tau = rng.standard_normal((4, 4))
    p1 = cc.project(tau)
    p2 = cc.project(p1)
    assert np.allclose(p1, p2, atol=1e-12)


def test_init_tau_is_identity():
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    tau = cc.init_tau()
    assert np.allclose(tau, np.eye(3))


def test_init_tau_identity_explicit():
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    tau = cc.init_tau(mode="identity")
    assert np.allclose(tau, np.eye(3))


def test_init_tau_zeros_invariant_entries_are_one():
    """zeros mode: invariant entries = 1 (enforced by project), shifted entry = 0."""
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    tau = cc.init_tau(mode="zeros")
    assert np.isclose(tau[0, 0], 1.0)   # invariant
    assert np.isclose(tau[2, 2], 1.0)   # invariant
    assert np.isclose(tau[1, 1], 0.0)   # shifted — starts at 0


def test_init_tau_random_invariant_entries_are_one():
    """random mode: invariant entries = 1, shifted entry ≠ 1 (almost surely)."""
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    tau = cc.init_tau(mode="random", rng=42)
    assert np.isclose(tau[0, 0], 1.0)   # invariant
    assert np.isclose(tau[2, 2], 1.0)   # invariant
    assert not np.isclose(tau[1, 1], 1.0)  # shifted — random, not 1


def test_init_tau_bad_mode_raises():
    cc = ConstructiveClass.markovian(d=3, shifted=[1])
    with pytest.raises(ValueError, match="Unknown tau_init mode"):
        cc.init_tau(mode="bogus")
