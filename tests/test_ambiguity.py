"""
Tests for traca.ambiguity.

Each projection is tested for:
  - Mask invariance: proj(M_K @ X) == M_K @ proj(M_K @ X)
  - Idempotence: proj(proj(X)) == proj(X)
  - Feasibility: proj(X) lies in the constraint set
  - Boundary: proj(scale * feasible) == feasible for scale <= 1
"""
import numpy as np
import pytest

from traca.ambiguity import (
    FrobeniusBall, RowBudget, ColumnBudget, EntrywiseBox,
    GelbrichBall, FrobeniusEmpirical,
    _apply_shift_mask,
)
from traca.utils import gelbrich_distance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 3
SHIFTED = (1,)  # only node 1 is shifted


def make_dW(seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dW = rng.standard_normal((D, D))
    np.fill_diagonal(dW, 0)
    return dW


# ---------------------------------------------------------------------------
# FrobeniusBall
# ---------------------------------------------------------------------------

class TestFrobeniusBall:
    def setup_method(self):
        self.ball = FrobeniusBall(eta=0.5, shifted_rows=SHIFTED, d=D)

    def test_feasibility(self):
        dW = make_dW()
        proj = self.ball.project(dW)
        # Only shifted rows should be non-zero
        for j in range(D):
            if j not in SHIFTED:
                assert np.allclose(proj[j], 0), f"Row {j} should be zero"
        # Frobenius norm within radius
        assert np.linalg.norm(proj, "fro") <= self.ball.eta + 1e-9

    def test_idempotence(self):
        dW = make_dW()
        p1 = self.ball.project(dW)
        p2 = self.ball.project(p1)
        assert np.allclose(p1, p2, atol=1e-10)

    def test_boundary_inside(self):
        """A small dW already inside the ball should be unchanged."""
        dW = np.zeros((D, D))
        dW[1, 2] = 0.1  # upper-triangular entry, norm = 0.1 < 0.5
        proj = self.ball.project(dW)
        assert np.allclose(proj, dW, atol=1e-10)

    def test_mask_invariance(self):
        """proj(M_K @ X) should equal M_K @ proj(M_K @ X)."""
        dW = make_dW()
        masked = _apply_shift_mask(dW, SHIFTED)
        p = self.ball.project(masked)
        p_masked = _apply_shift_mask(p, SHIFTED)
        assert np.allclose(p, p_masked, atol=1e-10)

    def test_scale(self):
        ball2 = self.ball.scale(2.0)
        assert np.isclose(ball2.eta, 1.0)


# ---------------------------------------------------------------------------
# RowBudget
# ---------------------------------------------------------------------------

class TestRowBudget:
    def setup_method(self):
        self.rb = RowBudget(rho={1: 0.3}, shifted_rows=SHIFTED, d=D)

    def test_feasibility(self):
        dW = make_dW()
        proj = self.rb.project(dW)
        for j in SHIFTED:
            assert np.sum(np.abs(proj[j])) <= self.rb.rho[j] + 1e-9
        for j in range(D):
            if j not in SHIFTED:
                assert np.allclose(proj[j], 0)

    def test_idempotence(self):
        dW = make_dW()
        p1 = self.rb.project(dW)
        p2 = self.rb.project(p1)
        assert np.allclose(p1, p2, atol=1e-10)

    def test_boundary_inside(self):
        dW = np.zeros((D, D))
        dW[1, 2] = 0.1  # upper-triangular, L1 norm 0.1 < 0.3
        proj = self.rb.project(dW)
        assert np.allclose(proj[1], dW[1], atol=1e-10)


# ---------------------------------------------------------------------------
# ColumnBudget
# ---------------------------------------------------------------------------

class TestColumnBudget:
    def setup_method(self):
        self.cb = ColumnBudget(c=0.4, shifted_rows=SHIFTED, d=D)

    def test_feasibility(self):
        dW = make_dW()
        proj = self.cb.project(dW)
        for k in range(D):
            col_l1 = sum(abs(proj[j, k]) for j in SHIFTED)
            assert col_l1 <= 0.4 + 1e-9
        for j in range(D):
            if j not in SHIFTED:
                assert np.allclose(proj[j], 0)

    def test_idempotence(self):
        dW = make_dW()
        p1 = self.cb.project(dW)
        p2 = self.cb.project(p1)
        assert np.allclose(p1, p2, atol=1e-10)


# ---------------------------------------------------------------------------
# EntrywiseBox
# ---------------------------------------------------------------------------

class TestEntrywiseBox:
    def setup_method(self):
        B = np.full((D, D), 0.5)
        self.box = EntrywiseBox(B=B, shifted_rows=SHIFTED, d=D)

    def test_feasibility(self):
        dW = make_dW()
        proj = self.box.project(dW)
        for j in SHIFTED:
            assert np.all(np.abs(proj[j]) <= 0.5 + 1e-9)
        for j in range(D):
            if j not in SHIFTED:
                assert np.allclose(proj[j], 0)

    def test_idempotence(self):
        dW = make_dW()
        p1 = self.box.project(dW)
        p2 = self.box.project(p1)
        assert np.allclose(p1, p2, atol=1e-10)

    def test_boundary_inside(self):
        dW = np.zeros((D, D))
        dW[1, 2] = 0.2  # upper-triangular entry, within box
        proj = self.box.project(dW)
        assert np.allclose(proj[1, 2], 0.2)


# ---------------------------------------------------------------------------
# FrobeniusEmpirical
# ---------------------------------------------------------------------------

class TestFrobeniusEmpirical:
    def setup_method(self):
        self.fe = FrobeniusEmpirical(eps=0.1, N=100, shifted_rows=SHIFTED)

    def test_feasibility(self):
        rng = np.random.default_rng(7)
        Theta = rng.standard_normal((100, D))
        proj = self.fe.project(Theta)
        # Only shifted columns non-zero
        for j in range(D):
            if j not in SHIFTED:
                assert np.allclose(proj[:, j], 0)
        # Frobenius norm within radius
        assert np.linalg.norm(proj, "fro") <= self.fe.radius() + 1e-9

    def test_idempotence(self):
        rng = np.random.default_rng(8)
        Theta = rng.standard_normal((100, D))
        p1 = self.fe.project(Theta)
        p2 = self.fe.project(p1)
        assert np.allclose(p1, p2, atol=1e-10)


# ---------------------------------------------------------------------------
# GelbrichBall (projection-only tests)
# ---------------------------------------------------------------------------

class TestGelbrichBall:
    def setup_method(self):
        self.mu_s = np.zeros(D)
        self.Sigma_s = np.eye(D)
        self.ball = GelbrichBall(mu_s=self.mu_s, Sigma_s=self.Sigma_s,
                                  eps=0.3, shifted_rows=SHIFTED)

    def test_invariance_pinning(self):
        """Non-shifted coords must be pinned to source."""
        mu_raw = np.array([5.0, 5.0, 5.0])
        Sigma_raw = np.eye(D) * 3.0
        mu_p, Sigma_p = self.ball.project(mu_raw, Sigma_raw)
        for i in range(D):
            if i not in SHIFTED:
                assert np.isclose(mu_p[i], self.mu_s[i], atol=1e-10)

    def test_inside_unchanged(self):
        """A point inside the ball should be returned unchanged after pinning."""
        mu_t = np.zeros(D)
        mu_t[1] = 0.01  # tiny shift in shifted coord
        Sigma_t = np.eye(D)
        mu_p, Sigma_p = self.ball.project(mu_t, Sigma_t)
        assert np.allclose(mu_p, mu_t, atol=1e-8)


# ===========================================================================
# Sampler tests
# ===========================================================================

class TestFrobeniusBallSample:
    """Sampler for FrobeniusBall: uniform-on-sphere at radius eta."""

    def test_admissibility(self):
        fb = FrobeniusBall(eta=0.5, shifted_rows=(0, 1), d=D)
        rng = np.random.default_rng(10)
        dW = fb.sample(rng)
        dW_proj = fb.project(dW)
        np.testing.assert_allclose(dW, dW_proj, atol=1e-10,
                                   err_msg="sample must be admissible")

    def test_magnitude_on_sphere(self):
        fb = FrobeniusBall(eta=0.7, shifted_rows=(0, 1), d=D)
        rng = np.random.default_rng(11)
        dW = fb.sample(rng)
        np.testing.assert_allclose(np.linalg.norm(dW, "fro"), 0.7, atol=1e-10,
                                   err_msg="sample must lie on sphere")

    def test_magnitude_with_entry_mask(self):
        mask = np.zeros((D, D))
        mask[0, 2] = 1; mask[1, 2] = 1
        fb = FrobeniusBall(eta=0.3, shifted_rows=(0, 1), d=D, entry_mask=mask)
        rng = np.random.default_rng(12)
        dW = fb.sample(rng)
        np.testing.assert_allclose(np.linalg.norm(dW, "fro"), 0.3, atol=1e-10)
        # Non-free entries must be zero
        for j in range(D):
            for k in range(D):
                if mask[j, k] == 0:
                    assert dW[j, k] == 0.0, f"Non-free entry ({j},{k}) must be zero"

    def test_mask_respected(self):
        fb = FrobeniusBall(eta=0.5, shifted_rows=(1,), d=D)
        rng = np.random.default_rng(13)
        dW = fb.sample(rng)
        # Row 0 and row 2 must be zero
        assert np.allclose(dW[0], 0)
        assert np.allclose(dW[2], 0)
        # Lower triangle must be zero
        assert np.allclose(np.tril(dW), 0)

    def test_determinism(self):
        fb = FrobeniusBall(eta=0.5, shifted_rows=(0, 1), d=D)
        dW1 = fb.sample(np.random.default_rng(77))
        dW2 = fb.sample(np.random.default_rng(77))
        assert np.array_equal(dW1, dW2)


class TestEntrywiseBoxSample:
    """Sampler for EntrywiseBox: uniform-in-box."""

    def test_admissibility(self):
        B = np.array([[0.0, 0.5], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2)
        rng = np.random.default_rng(20)
        dW = box.sample(rng)
        dW_proj = box.project(dW)
        np.testing.assert_allclose(dW, dW_proj, atol=1e-10,
                                   err_msg="sample must be admissible")

    def test_within_box_bounds(self):
        B = np.array([[0.0, 0.0, 0.3],
                       [0.0, 0.0, 0.3],
                       [0.0, 0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0, 1), d=3)
        rng = np.random.default_rng(21)
        for _ in range(50):
            dW = box.sample(rng)
            assert np.all(np.abs(dW) <= B + 1e-12), "Entry exceeds box bound"

    def test_structural_zeros(self):
        B = np.array([[0.0, 0.5], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2)
        rng = np.random.default_rng(22)
        dW = box.sample(rng)
        assert dW[0, 0] == 0.0, "Diagonal must be zero (strict upper-triangular)"
        assert dW[1, 0] == 0.0, "Non-shifted row must be zero"
        assert dW[1, 1] == 0.0, "Non-shifted row must be zero"

    def test_mask_respected(self):
        B = np.array([[0.0, 0.0, 0.3],
                       [0.0, 0.0, 0.3],
                       [0.0, 0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0, 1), d=3)
        rng = np.random.default_rng(23)
        dW = box.sample(rng)
        # Row 2 must be zero (not shifted)
        assert np.allclose(dW[2], 0)
        # Columns 0, 1 must be zero (B is zero there)
        assert np.allclose(dW[:, 0], 0)
        assert np.allclose(dW[:, 1], 0)

    def test_determinism(self):
        B = np.array([[0.0, 0.5], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2)
        dW1 = box.sample(np.random.default_rng(88))
        dW2 = box.sample(np.random.default_rng(88))
        assert np.array_equal(dW1, dW2)


class TestGelbrichBallSample:
    """Sampler for GelbrichBall: projected onto ball boundary."""

    def test_admissibility(self):
        gb = GelbrichBall(mu_s=np.zeros(D), Sigma_s=np.eye(D),
                          eps=0.3, shifted_rows=SHIFTED)
        rng = np.random.default_rng(30)
        mu_t, Sigma_t = gb.sample(rng)
        mu_p, Sigma_p = gb.project(mu_t, Sigma_t)
        np.testing.assert_allclose(mu_t, mu_p, atol=1e-8,
                                   err_msg="sampled mu must be admissible")
        np.testing.assert_allclose(Sigma_t, Sigma_p, atol=1e-8,
                                   err_msg="sampled Sigma must be admissible")

    def test_invariance_pinning_on_sample(self):
        mu_s = np.array([1.0, 2.0, 3.0])
        Sigma_s = np.diag([1.0, 2.0, 3.0])
        gb = GelbrichBall(mu_s=mu_s, Sigma_s=Sigma_s,
                          eps=0.5, shifted_rows=(1,))
        rng = np.random.default_rng(31)
        mu_t, Sigma_t = gb.sample(rng)
        # Non-shifted coords must be pinned to source
        for i in [0, 2]:
            assert np.isclose(mu_t[i], mu_s[i], atol=1e-10)

    def test_within_ball(self):
        gb = GelbrichBall(mu_s=np.zeros(D), Sigma_s=np.eye(D),
                          eps=0.3, shifted_rows=SHIFTED)
        rng = np.random.default_rng(32)
        mu_t, Sigma_t = gb.sample(rng)
        dist2 = gelbrich_distance(mu_t, Sigma_t, gb.mu_s, gb.Sigma_s)
        assert dist2 <= gb.eps ** 2 + 1e-8, \
            f"Sample outside ball: dist²={dist2}, eps²={gb.eps**2}"

    def test_determinism(self):
        gb = GelbrichBall(mu_s=np.zeros(D), Sigma_s=np.eye(D),
                          eps=0.3, shifted_rows=SHIFTED)
        mu1, S1 = gb.sample(np.random.default_rng(55))
        mu2, S2 = gb.sample(np.random.default_rng(55))
        assert np.array_equal(mu1, mu2) and np.array_equal(S1, S2)

    def test_source_arrays_not_mutated(self):
        """Shared mu_s/Sigma_s must not be written through by sample()."""
        mu_s = np.array([1.0, 2.0, 3.0])
        Sigma_s = np.diag([1.0, 2.0, 3.0])
        mu_s_copy = mu_s.copy()
        Sigma_s_copy = Sigma_s.copy()
        gb = GelbrichBall(mu_s=mu_s, Sigma_s=Sigma_s,
                          eps=0.5, shifted_rows=(1,))
        rng = np.random.default_rng(33)
        gb.sample(rng)
        np.testing.assert_array_equal(mu_s, mu_s_copy)
        np.testing.assert_array_equal(Sigma_s, Sigma_s_copy)


class TestFrobeniusEmpiricalSample:
    """Sampler for FrobeniusEmpirical: uniform-on-sphere at eps*sqrt(N)."""

    def test_admissibility(self):
        fe = FrobeniusEmpirical(eps=0.2, N=100, shifted_rows=SHIFTED)
        rng = np.random.default_rng(40)
        Theta = fe.sample(rng, d=D)
        Theta_proj = fe.project(Theta)
        np.testing.assert_allclose(Theta, Theta_proj, atol=1e-10,
                                   err_msg="sample must be admissible")

    def test_magnitude_on_sphere(self):
        fe = FrobeniusEmpirical(eps=0.2, N=100, shifted_rows=SHIFTED)
        rng = np.random.default_rng(41)
        Theta = fe.sample(rng, d=D)
        expected_radius = 0.2 * np.sqrt(100)  # = 2.0
        np.testing.assert_allclose(np.linalg.norm(Theta, "fro"),
                                   expected_radius, atol=1e-10)

    def test_mask_respected(self):
        fe = FrobeniusEmpirical(eps=0.2, N=50, shifted_rows=(1,))
        rng = np.random.default_rng(42)
        Theta = fe.sample(rng, d=D)
        assert np.allclose(Theta[:, 0], 0), "Non-shifted col 0 must be zero"
        assert np.allclose(Theta[:, 2], 0), "Non-shifted col 2 must be zero"

    def test_N_override(self):
        fe = FrobeniusEmpirical(eps=0.2, N=100, shifted_rows=SHIFTED)
        rng = np.random.default_rng(43)
        Theta = fe.sample(rng, N=50, d=D)
        assert Theta.shape == (50, D)
        # Radius uses the caller-specified N (50), not self.N (100)
        expected_radius = 0.2 * np.sqrt(50)
        np.testing.assert_allclose(np.linalg.norm(Theta, "fro"),
                                   expected_radius, atol=1e-10)

    def test_determinism(self):
        fe = FrobeniusEmpirical(eps=0.2, N=100, shifted_rows=SHIFTED)
        T1 = fe.sample(np.random.default_rng(66), d=D)
        T2 = fe.sample(np.random.default_rng(66), d=D)
        assert np.array_equal(T1, T2)


# ---------------------------------------------------------------------------
# Scale non-mutation tests
# ---------------------------------------------------------------------------

class TestScaleNonMutation:
    """scale(a) and scale(b) from the same original give independent results."""

    def test_frobenius_ball(self):
        mask = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        orig = FrobeniusBall(eta=1.0, shifted_rows=(0, 1), d=3, entry_mask=mask)
        eta_before = orig.eta
        mask_id_before = id(orig.entry_mask)
        sa = orig.scale(2.0)
        sb = orig.scale(3.0)
        assert orig.eta == eta_before, "original mutated"
        assert sa.eta == 2.0 and sb.eta == 3.0
        # entry_mask shared but read-only — confirm identity preserved
        assert id(orig.entry_mask) == mask_id_before

    def test_entrywise_box(self):
        B = np.array([[0, 0.5], [0, 0]], dtype=float)
        orig = EntrywiseBox(B=B, shifted_rows=(0,), d=2)
        B_copy = orig.B.copy()
        sa = orig.scale(2.0)
        sb = orig.scale(3.0)
        np.testing.assert_array_equal(orig.B, B_copy, err_msg="original B mutated")
        np.testing.assert_allclose(sa.B, B * 2.0)
        np.testing.assert_allclose(sb.B, B * 3.0)

    def test_gelbrich_ball(self):
        mu_s = np.zeros(2)
        Sigma_s = np.eye(2)
        orig = GelbrichBall(mu_s=mu_s, Sigma_s=Sigma_s,
                            eps=0.5, shifted_rows=(1,))
        eps_before = orig.eps
        sa = orig.scale(2.0)
        sb = orig.scale(0.5)
        assert orig.eps == eps_before
        assert sa.eps == 1.0 and sb.eps == 0.25

    def test_frobenius_empirical(self):
        orig = FrobeniusEmpirical(eps=0.2, N=100, shifted_rows=(1,))
        eps_before = orig.eps
        sa = orig.scale(2.0)
        sb = orig.scale(3.0)
        assert orig.eps == eps_before
        assert np.isclose(sa.eps, 0.4) and np.isclose(sb.eps, 0.6)


# ---------------------------------------------------------------------------
# EntrywiseBox shifted box (directional prior) tests
# ---------------------------------------------------------------------------

class TestEntrywiseBoxDeltaNoneBitIdentity:
    """delta=None must produce bit-identical results to pre-change code."""

    def test_project_bit_identical(self):
        B = np.array([[0.0, 0.3], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=None)
        dW = np.array([[0.5, 0.8], [-0.1, 0.2]])
        result = box.project(dW)
        # Pre-change code: _apply_shift_mask then np.clip(dW, -B, B)
        expected = np.clip(_apply_shift_mask(dW, (0,)), -B, B)
        np.testing.assert_array_equal(result, expected)

    def test_sample_bit_identical(self):
        B = np.array([[0.0, 0.3], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=None)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = box.sample(rng1)
        # Pre-change code: rng.uniform(-1,1,...) * B then _apply_shift_mask
        s2_raw = rng2.uniform(-1.0, 1.0, size=(2, 2)) * B
        s2 = _apply_shift_mask(s2_raw, (0,))
        np.testing.assert_array_equal(s1, s2)

    def test_scale_preserves_none(self):
        B = np.array([[0.0, 0.3], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=None)
        box_s = box.scale(2.0)
        assert box_s.delta is None, "scale(r) must preserve None, not convert to zeros"

    def test_effective_bound_returns_B(self):
        B = np.array([[0.0, 0.3], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=None)
        assert box.effective_bound is box.B, "effective_bound must return same B object"


class TestEntrywiseBoxShiftedBox:
    """Tests for the shifted box [delta-B, delta+B] with delta != None."""

    def test_project_clips_to_shifted_range(self):
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 1.0], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        # Box is [0, 2] at entry (0,1)
        dW = np.array([[-1.0, 5.0], [0.0, 0.0]])
        result = box.project(dW)
        assert result[0, 1] == 2.0, "Should clip to delta+B=2.0"
        assert result[0, 0] == 0.0, "Diagonal must be zero"

        dW2 = np.array([[0.0, -1.0], [0.0, 0.0]])
        result2 = box.project(dW2)
        assert result2[0, 1] == 0.0, "Should clip to delta-B=0.0"

    def test_scale_preserves_shifted_range(self):
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 1.0], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        box_s = box.scale(0.5)
        # Scaled box: delta=0.5, B=0.5 → [0, 1.0]
        np.testing.assert_allclose(box_s.delta, delta * 0.5)
        np.testing.assert_allclose(box_s.B, B * 0.5)
        result = box_s.project(np.array([[0.0, 3.0], [0.0, 0.0]]))
        assert result[0, 1] == 1.0, "Scaled box should clip to 1.0"

    def test_sample_within_shifted_range(self):
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 1.0], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        rng = np.random.default_rng(99)
        for _ in range(100):
            dW = box.sample(rng)
            # Must be in [delta-B, delta+B] = [0, 2] at (0,1)
            assert dW[0, 1] >= -1e-12, f"Sample below lower bound: {dW[0, 1]}"
            assert dW[0, 1] <= 2.0 + 1e-12, f"Sample above upper bound: {dW[0, 1]}"

    def test_effective_bound_shifted(self):
        B = np.array([[0.0, 0.3], [0.0, 0.0]])
        delta = np.array([[0.0, 0.1], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        expected = np.abs(delta) + B  # [[0, 0.4], [0, 0]]
        np.testing.assert_allclose(box.effective_bound, expected)

    def test_idempotence_shifted(self):
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 0.5], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        dW = np.array([[0.0, 3.0], [0.0, 0.0]])
        p1 = box.project(dW)
        p2 = box.project(p1)
        np.testing.assert_allclose(p1, p2, atol=1e-15)

    def test_scale_nonmutation_with_delta(self):
        B = np.array([[0.0, 0.5], [0.0, 0.0]])
        delta = np.array([[0.0, 0.3], [0.0, 0.0]])
        orig = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        B_copy = orig.B.copy()
        delta_copy = orig.delta.copy()
        sa = orig.scale(2.0)
        sb = orig.scale(3.0)
        np.testing.assert_array_equal(orig.B, B_copy, err_msg="original B mutated")
        np.testing.assert_array_equal(orig.delta, delta_copy, err_msg="original delta mutated")
        np.testing.assert_allclose(sa.B, B * 2.0)
        np.testing.assert_allclose(sa.delta, delta * 2.0)
        np.testing.assert_allclose(sb.B, B * 3.0)
        np.testing.assert_allclose(sb.delta, delta * 3.0)
