"""
Tests for traca.shifts (apply_shift, list_edges, describe_shift, SHIFT_TYPES).
"""
import copy

import numpy as np
import pytest
import yaml

from lan_scm import LANSCM, _scm_from_config
from traca.shifts import apply_shift, list_edges, describe_shift, SHIFT_TYPES
from experiments.run import _load_bundle


def _has_mechanism_shift(source_bundle, target_bundle, atol=1e-8):
    """Return True if the target has a different W matrix than the source."""
    return not np.allclose(source_bundle.W, target_bundle.W, atol=atol)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def atce_scm():
    """ATCE source SCM (d=3, Gaussian benchmark)."""
    with open("data_configs/atce.yaml") as f:
        cfg = yaml.safe_load(f)
    return _scm_from_config(cfg)


@pytest.fixture
def atce_bundle():
    """ATCE source bundle (small n for speed)."""
    with open("data_configs/atce.yaml") as f:
        cfg = yaml.safe_load(f)
    scm = _scm_from_config(cfg)
    return scm.bundle([{}, {"X": 0}, {"X": 1}], n=200, seed=0)


@pytest.fixture
def lilucas_scm():
    """LiLuCaS source SCM (d=6, empirical benchmark)."""
    with open("data_configs/lilucas.yaml") as f:
        cfg = yaml.safe_load(f)
    return _scm_from_config(cfg)


@pytest.fixture
def lilucas_bundle():
    """LiLuCaS source bundle (small n for speed)."""
    with open("data_configs/lilucas.yaml") as f:
        cfg = yaml.safe_load(f)
    scm = _scm_from_config(cfg)
    interventions = [{}, {"Smoking": 0}, {"LungCancer": 0}]
    return scm.bundle(interventions, n=200, seed=0)


# ---------------------------------------------------------------------------
# apply_shift correctness — mechanism_edge
# ---------------------------------------------------------------------------

class TestMechanismEdge:
    """mechanism_edge shift produces correct W perturbation."""

    def test_W_perturbed(self, atce_scm):
        """Shifted SCM has W[i,j] = W_old + δ."""
        shifted = apply_shift(atce_scm, "mechanism_edge", 0.2, edge=(0, 1))
        assert abs(shifted.W[0, 1] - (atce_scm.W[0, 1] + 0.2)) < 1e-12

    def test_other_edges_unchanged(self, atce_scm):
        """Only the specified edge changes."""
        shifted = apply_shift(atce_scm, "mechanism_edge", 0.2, edge=(0, 1))
        W_orig = atce_scm.W
        W_new = shifted.W
        # Zero out the perturbed entry and compare
        W_new[0, 1] = W_orig[0, 1]
        assert np.allclose(W_new, W_orig)

    def test_noise_unchanged(self, atce_scm):
        """Noise params are not modified by mechanism_edge shift."""
        shifted = apply_shift(atce_scm, "mechanism_edge", 0.2, edge=(0, 1))
        assert np.allclose(shifted.noise_mean, atce_scm.noise_mean)
        assert np.allclose(shifted.noise_cov, atce_scm.noise_cov)

    def test_original_unchanged(self, atce_scm):
        """Original SCM is not mutated."""
        W_before = atce_scm.W.copy()
        apply_shift(atce_scm, "mechanism_edge", 0.5, edge=(0, 1))
        assert np.allclose(atce_scm.W, W_before)

    def test_has_mechanism_shift_fires(self, atce_scm):
        """_has_mechanism_shift detects the W difference."""
        shifted = apply_shift(atce_scm, "mechanism_edge", 0.3, edge=(0, 1))
        b_src = atce_scm.bundle([{}], n=10, seed=0)
        b_tgt = shifted.bundle([{}], n=10, seed=0)
        assert _has_mechanism_shift(b_src, b_tgt)

    def test_structural_zero_rejected(self, atce_scm):
        """Perturbing a structural zero raises ValueError."""
        # ATCE: W[1,0] = 0 (no X→Z edge, and i>j so DAG-invalid anyway)
        # W[0,0] is diagonal — also zero. Use (1,2) which IS an edge.
        # Find a structural zero: W[1,0] doesn't exist (i>=j),
        # but (0,0) has i>=j. Try (1,0):
        with pytest.raises(ValueError, match="DAG order"):
            apply_shift(atce_scm, "mechanism_edge", 0.1, edge=(1, 0))

    def test_nonedge_structural_zero(self, lilucas_scm):
        """Perturbing a zero in the upper triangle (non-edge) raises ValueError."""
        # LiLuCaS: Smoking(0)→Allergy(2) has W[0,2]=0 (no edge)
        with pytest.raises(ValueError, match="structural zero"):
            apply_shift(lilucas_scm, "mechanism_edge", 0.1, edge=(0, 2))

    def test_missing_edge_raises(self, atce_scm):
        """Missing edge= argument raises ValueError."""
        with pytest.raises(ValueError, match="requires edge"):
            apply_shift(atce_scm, "mechanism_edge", 0.1)

    def test_out_of_range_raises(self, atce_scm):
        """Edge indices out of range raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            apply_shift(atce_scm, "mechanism_edge", 0.1, edge=(0, 10))

    def test_propagates_to_downstream(self, atce_scm):
        """Mechanism shift changes downstream interventional means."""
        shifted = apply_shift(atce_scm, "mechanism_edge", 0.5, edge=(0, 1))
        # Observational mean of X: μ_X = μ_U @ A
        # Z→X weight changed → A changes → μ_X changes
        mu_src = atce_scm.noise_mean @ atce_scm.A
        mu_tgt = shifted.noise_mean @ shifted.A
        # With zero noise mean, both are zero. Use non-zero mean.
        shifted2 = LANSCM(
            W=atce_scm.W,
            noise_mean=np.array([1.0, 0.0, 0.0]),
            noise_cov=atce_scm.noise_cov,
            var_names=atce_scm.var_names,
        )
        shifted3 = apply_shift(shifted2, "mechanism_edge", 0.5, edge=(0, 1))
        mu_src2 = shifted2.noise_mean @ shifted2.A
        mu_tgt2 = shifted3.noise_mean @ shifted3.A
        # X (node 1) mean should differ
        assert abs(mu_tgt2[1] - mu_src2[1]) > 0.1


# ---------------------------------------------------------------------------
# apply_shift correctness — noise_mean
# ---------------------------------------------------------------------------

class TestNoiseMean:
    """noise_mean shift operates correctly on Gaussian parameters."""

    def test_mean_shifted(self, atce_scm):
        """noise_mean[k] += δ."""
        shifted = apply_shift(atce_scm, "noise_mean", 0.5, node=0)
        assert abs(shifted.noise_mean[0] - (atce_scm.noise_mean[0] + 0.5)) < 1e-12

    def test_other_nodes_unchanged(self, atce_scm):
        shifted = apply_shift(atce_scm, "noise_mean", 0.5, node=0)
        assert shifted.noise_mean[1] == atce_scm.noise_mean[1]
        assert shifted.noise_mean[2] == atce_scm.noise_mean[2]

    def test_cov_unchanged(self, atce_scm):
        shifted = apply_shift(atce_scm, "noise_mean", 0.5, node=0)
        assert np.allclose(shifted.noise_cov, atce_scm.noise_cov)

    def test_W_unchanged(self, atce_scm):
        shifted = apply_shift(atce_scm, "noise_mean", 0.5, node=0)
        assert np.allclose(shifted.W, atce_scm.W)

    def test_no_mechanism_shift(self, atce_scm):
        """noise_mean shift does NOT trigger _has_mechanism_shift."""
        shifted = apply_shift(atce_scm, "noise_mean", 0.5, node=0)
        b_src = atce_scm.bundle([{}], n=10, seed=0)
        b_tgt = shifted.bundle([{}], n=10, seed=0)
        assert not _has_mechanism_shift(b_src, b_tgt)

    def test_original_unchanged(self, atce_scm):
        mean_before = atce_scm.noise_mean.copy()
        apply_shift(atce_scm, "noise_mean", 0.5, node=0)
        assert np.allclose(atce_scm.noise_mean, mean_before)

    def test_propagates_to_downstream(self, atce_scm):
        """Mean shift on root Z propagates to downstream X and Y means."""
        shifted = apply_shift(atce_scm, "noise_mean", 1.0, node=0)
        mu_src = atce_scm.noise_mean @ atce_scm.A
        mu_tgt = shifted.noise_mean @ shifted.A
        # Z mean shifted by 1.0 → X mean changes by W[0,1]*1.0 via A
        assert abs(mu_tgt[1] - mu_src[1]) > 0.1  # X
        assert abs(mu_tgt[2] - mu_src[2]) > 0.1  # Y

    def test_empirical_samples_differ(self, lilucas_scm):
        """Empirical path: shifted SCM produces different samples."""
        shifted = apply_shift(lilucas_scm, "noise_mean", 1.0, node=3)
        b_src = lilucas_scm.bundle([{}], n=500, seed=42)
        b_tgt = shifted.bundle([{}], n=500, seed=42)
        # Mean of node 3 (LungCancer) should differ
        mean_src = np.mean(b_src.endogenous_samples[0][:, 3])
        mean_tgt = np.mean(b_tgt.endogenous_samples[0][:, 3])
        assert abs(mean_tgt - mean_src) > 0.5

    def test_missing_node_raises(self, atce_scm):
        with pytest.raises(ValueError, match="requires node"):
            apply_shift(atce_scm, "noise_mean", 0.5)


# ---------------------------------------------------------------------------
# apply_shift correctness — noise_std
# ---------------------------------------------------------------------------

class TestNoiseStd:
    """noise_std shift scales standard deviation."""

    def test_variance_scaled_quadratically(self, atce_scm):
        """cov[k,k] *= (1+δ)²."""
        shifted = apply_shift(atce_scm, "noise_std", 0.5, node=0)
        expected = atce_scm.noise_cov[0, 0] * (1.5 ** 2)
        assert abs(shifted.noise_cov[0, 0] - expected) < 1e-12

    def test_other_diag_unchanged(self, atce_scm):
        shifted = apply_shift(atce_scm, "noise_std", 0.5, node=0)
        assert abs(shifted.noise_cov[1, 1] - atce_scm.noise_cov[1, 1]) < 1e-12
        assert abs(shifted.noise_cov[2, 2] - atce_scm.noise_cov[2, 2]) < 1e-12

    def test_W_unchanged(self, atce_scm):
        shifted = apply_shift(atce_scm, "noise_std", 0.5, node=0)
        assert np.allclose(shifted.W, atce_scm.W)

    def test_no_mechanism_shift(self, atce_scm):
        shifted = apply_shift(atce_scm, "noise_std", 0.5, node=0)
        b_src = atce_scm.bundle([{}], n=10, seed=0)
        b_tgt = shifted.bundle([{}], n=10, seed=0)
        assert not _has_mechanism_shift(b_src, b_tgt)

    def test_negative_scale_rejected(self, atce_scm):
        """δ < -1 gives negative std → error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            apply_shift(atce_scm, "noise_std", -1.5, node=0)

    def test_empirical_dispersion_increases(self, lilucas_scm):
        """Empirical path: larger std → more spread in samples."""
        shifted = apply_shift(lilucas_scm, "noise_std", 1.0, node=3)
        b_src = lilucas_scm.bundle([{}], n=2000, seed=42)
        b_tgt = shifted.bundle([{}], n=2000, seed=42)
        std_src = np.std(b_src.endogenous_samples[0][:, 3])
        std_tgt = np.std(b_tgt.endogenous_samples[0][:, 3])
        # δ=1.0 doubles exogenous std; endogenous std increases less due
        # to upstream propagation through A mixing other (unchanged) noise.
        assert std_tgt > std_src * 1.2

    def test_off_diagonal_correlation_preserved(self):
        """With off-diagonal cov, noise_std preserves correlation structure."""
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        scm = LANSCM(W=np.array([[0, 0.3], [0, 0]]),
                      noise_mean=np.zeros(2), noise_cov=cov,
                      var_names=["A", "B"])
        shifted = apply_shift(scm, "noise_std", 1.0, node=0)
        # Correlation should be preserved: corr = cov[0,1] / sqrt(cov[0,0]*cov[1,1])
        corr_src = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        corr_tgt = shifted.noise_cov[0, 1] / np.sqrt(
            shifted.noise_cov[0, 0] * shifted.noise_cov[1, 1])
        assert abs(corr_src - corr_tgt) < 1e-10


# ---------------------------------------------------------------------------
# apply_shift correctness — noise_cov
# ---------------------------------------------------------------------------

class TestNoiseCov:
    """noise_cov shift scales variance directly."""

    def test_variance_scaled_linearly(self, atce_scm):
        """cov[k,k] *= (1+δ)."""
        shifted = apply_shift(atce_scm, "noise_cov", 0.5, node=0)
        expected = atce_scm.noise_cov[0, 0] * 1.5
        assert abs(shifted.noise_cov[0, 0] - expected) < 1e-12

    def test_differs_from_noise_std(self, atce_scm):
        """noise_cov and noise_std produce different results for same δ."""
        shifted_std = apply_shift(atce_scm, "noise_std", 0.5, node=0)
        shifted_cov = apply_shift(atce_scm, "noise_cov", 0.5, node=0)
        # noise_std: cov *= (1.5)² = 2.25; noise_cov: cov *= 1.5
        assert abs(shifted_std.noise_cov[0, 0] - 2.25) < 1e-12
        assert abs(shifted_cov.noise_cov[0, 0] - 1.5) < 1e-12

    def test_no_mechanism_shift(self, atce_scm):
        shifted = apply_shift(atce_scm, "noise_cov", 0.5, node=0)
        b_src = atce_scm.bundle([{}], n=10, seed=0)
        b_tgt = shifted.bundle([{}], n=10, seed=0)
        assert not _has_mechanism_shift(b_src, b_tgt)

    def test_negative_scale_rejected(self, atce_scm):
        with pytest.raises(ValueError, match="cannot be negative"):
            apply_shift(atce_scm, "noise_cov", -1.5, node=0)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

class TestInvalidInputs:
    """Bad shift_type, missing element, etc. raise clear errors."""

    def test_unknown_shift_type(self, atce_scm):
        with pytest.raises(ValueError, match="Unknown shift_type"):
            apply_shift(atce_scm, "bogus", 0.1, node=0)

    def test_mechanism_edge_with_node(self, atce_scm):
        """mechanism_edge with node= instead of edge= raises."""
        with pytest.raises(ValueError, match="requires edge"):
            apply_shift(atce_scm, "mechanism_edge", 0.1, node=0)

    def test_noise_mean_with_edge(self, atce_scm):
        """noise_mean with edge= instead of node= raises."""
        with pytest.raises(ValueError, match="requires node"):
            apply_shift(atce_scm, "noise_mean", 0.1, edge=(0, 1))

    def test_node_out_of_range(self, atce_scm):
        with pytest.raises(ValueError, match="out of range"):
            apply_shift(atce_scm, "noise_mean", 0.1, node=10)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_list_edges(self, atce_scm):
        edges = list_edges(atce_scm)
        assert len(edges) == 3
        assert (0, 1, 0.3) in edges
        assert (0, 2, 0.5) in edges
        assert (1, 2, 1.0) in edges

    def test_describe_shift(self, atce_scm):
        desc = describe_shift(atce_scm, "mechanism_edge", 0.2, edge=(0, 1))
        assert "Z→X" in desc
        assert "0.2" in desc


# ---------------------------------------------------------------------------
# Support matrix: all shift_types × bundle types
# ---------------------------------------------------------------------------

class TestSupportMatrix:
    """Confirm each shift_type works on both Gaussian and empirical SCMs."""

    @pytest.mark.parametrize("shift_type,kwargs", [
        ("mechanism_edge", {"edge": (0, 1)}),       # Z→X
        ("noise_mean",     {"node": 0}),
        ("noise_std",      {"node": 0}),
        ("noise_cov",      {"node": 0}),
    ])
    def test_gaussian_scm(self, atce_scm, shift_type, kwargs):
        """All shift types produce a valid LANSCM from Gaussian source."""
        shifted = apply_shift(atce_scm, shift_type, 0.2, **kwargs)
        assert isinstance(shifted, LANSCM)
        assert shifted.d == atce_scm.d
        # Can build a bundle
        b = shifted.bundle([{}], n=10, seed=0)
        assert b.n == 10

    @pytest.mark.parametrize("shift_type,kwargs", [
        ("mechanism_edge", {"edge": (0, 3)}),       # Smoking→LungCancer
        ("noise_mean",     {"node": 3}),
        ("noise_std",      {"node": 3}),
        ("noise_cov",      {"node": 3}),
    ])
    def test_empirical_scm(self, lilucas_scm, shift_type, kwargs):
        """All shift types produce a valid LANSCM from empirical source."""
        shifted = apply_shift(lilucas_scm, shift_type, 0.2, **kwargs)
        assert isinstance(shifted, LANSCM)
        assert shifted.d == lilucas_scm.d
        b = shifted.bundle([{}], n=10, seed=0)
        assert b.n == 10

    @pytest.mark.parametrize("shift_type,kwargs", [
        ("mechanism_edge", {"edge": (0, 1)}),
        ("noise_mean",     {"node": 0}),
        ("noise_std",      {"node": 0}),
        ("noise_cov",      {"node": 0}),
    ])
    def test_measurable_difference(self, atce_scm, shift_type, kwargs):
        """Each shift at δ=0.5 produces a measurably different target."""
        shifted = apply_shift(atce_scm, shift_type, 0.5, **kwargs)
        b_src = atce_scm.bundle([{}], n=1000, seed=42)
        b_tgt = shifted.bundle([{}], n=1000, seed=42)
        # At least one coordinate's sample mean or std should differ
        X_src = b_src.endogenous_samples[0]
        X_tgt = b_tgt.endogenous_samples[0]
        mean_diff = np.max(np.abs(X_src.mean(axis=0) - X_tgt.mean(axis=0)))
        std_diff = np.max(np.abs(X_src.std(axis=0) - X_tgt.std(axis=0)))
        assert mean_diff > 0.05 or std_diff > 0.05, \
            f"Shift {shift_type} at δ=0.5 produced no measurable difference"
