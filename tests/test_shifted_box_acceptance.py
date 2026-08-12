"""
Acceptance tests for the shifted box (directional prior).

Tier 1: delta=None bit-identity (in test_ambiguity.py — fast, pure unit tests)
Tier 2: τ gradient verification on shifted corners (this file)
Tier 3: Shifted box beats symmetric at equal effective_bound (this file)

Both tiers require training (fit_gaussian) so they are slower (~120s total).
"""
import numpy as np
import pytest
import joblib
import yaml

from traca.ambiguity import EntrywiseBox, GelbrichBall, _apply_shift_mask
from traca.constructive import ConstructiveClass
from traca.optim import fit_gaussian, OptimConfig, _enumerate_box_corners
from traca.stability import perturbed_propagator, gating_matrix, alpha_polynomial, gamma
from traca.query import F_iota_O_rho
from traca.certificates import delta_iota_rho_sq, selection_matrix
from traca.utils import bundle_exo_means


# ---------------------------------------------------------------------------
# Fixture: ATE bundle
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ate_setup():
    """Load ATE bundle and build shared test objects."""
    import os
    base = 'results_production/ate/ate_gaussian_entrywise_subfamily'
    if not os.path.isdir(base):
        pytest.skip(f"Production artifacts not found at {base} — run production first")
    bundle = joblib.load(f'{base}/bundle.pkl')
    with open(f'{base}/config.yaml') as f:
        config = yaml.safe_load(f)

    d = bundle.d
    n_iota = bundle.n_interventions()
    var_names = bundle.scm.var_names

    pairs = []
    for qf in config['training']['query_family']:
        pairs.append((int(qf['intervention_idx']), qf['O']))

    A_iotas = [bundle.intervened_scms[i].A for i in range(n_iota)]
    R_iotas = []
    for i, iv in enumerate(bundle.interventions):
        if iv:
            nodes = [var_names.index(k) if isinstance(k, str) else int(k)
                     for k in iv.keys()]
        else:
            nodes = []
        R_iotas.append(gating_matrix(d, nodes))

    mu_s_effs = bundle_exo_means(bundle)
    Sigma_s_raw = bundle.noise_cov
    _fixeds = [bundle.intervened_scms[i]._fixed for i in range(n_iota)]
    _Js = [bundle.intervened_scms[i]._J for i in range(n_iota)]
    Sigma_s_effs = []
    for i in range(n_iota):
        S = Sigma_s_raw.copy()
        if _Js[i]:
            jj = list(_Js[i])
            S[jj, :] = 0.0
            S[:, jj] = 0.0
        Sigma_s_effs.append(S)

    env = GelbrichBall(
        mu_s=bundle.noise_mean.copy(),
        Sigma_s=bundle.noise_cov.copy(),
        eps=2.0,
        shifted_rows=(1,),
    )
    constructive = ConstructiveClass.from_districts(d, [[0, 1]], [1])

    return {
        'bundle': bundle, 'config': config, 'd': d, 'n_iota': n_iota,
        'pairs': pairs, 'A_iotas': A_iotas, 'R_iotas': R_iotas,
        'mu_s_effs': mu_s_effs, 'Sigma_s_raw': Sigma_s_raw,
        'Sigma_s_effs': Sigma_s_effs,
        '_fixeds': _fixeds, '_Js': _Js,
        'env': env, 'constructive': constructive,
    }


def _compute_loss(setup, tau_, dW_, mu_t_exo, Sigma_t_exo):
    """Compute full-joint loss for ATE."""
    losses = []
    for iota_idx, O in setup['pairs']:
        A_prime = perturbed_propagator(setup['bundle'].W, dW_,
                                        setup['R_iotas'][iota_idx])
        mu = mu_t_exo.copy()
        Sigma = Sigma_t_exo.copy()
        J = setup['_Js'][iota_idx]
        fixed = setup['_fixeds'][iota_idx]
        if J:
            jj = list(J)
            mu[jj] = 0.0
            Sigma[jj, :] = 0.0
            Sigma[:, jj] = 0.0
        mu = mu + np.asarray(fixed, dtype=float)
        mu_t_obs = mu @ A_prime
        Sigma_t_obs = A_prime.T @ Sigma @ A_prime
        val = F_iota_O_rho(tau_, dW_, setup['bundle'].W,
                           setup['A_iotas'][iota_idx],
                           setup['R_iotas'][iota_idx],
                           setup['mu_s_effs'][iota_idx],
                           setup['Sigma_s_effs'][iota_idx],
                           mu_t_obs, Sigma_t_obs, O)
        losses.append(val)
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Tier 2: τ gradient verification on shifted corners
# ---------------------------------------------------------------------------

class TestTauMovesWithShiftedCorners:
    """At δ≠0, τ must move MORE from identity than at δ=0 (same effective_bound)."""

    def test_directional_moves_more_than_symmetric(self, ate_setup):
        """Shifted box makes τ move further from I than symmetric box with
        the same effective_bound.

        Symmetric B=1.0 (eff_bound=1.0) vs shifted δ=0.5, B=0.5
        (eff_bound=1.0). Equal adversary budget, but the shifted box has a
        net directional signal that allows τ to follow the shift direction.
        """
        B_half = np.array([[0.0, 0.5], [0.0, 0.0]])
        B_full = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 0.5], [0.0, 0.0]])
        cfg = OptimConfig(
            eta_tau=0.001, eta_adv=0.005, k_tau=1, k_adv=2,
            n_iters=500, tol=1e-5, conv_window=30, seed=42,
            grad_backend='autograd',
        )
        # Symmetric with effective_bound = 1.0
        mech_sym = EntrywiseBox(B=B_full, shifted_rows=(0,), d=2, delta=None)
        opt_sym = fit_gaussian(ate_setup['bundle'], mech_sym, ate_setup['env'],
                               ate_setup['constructive'], cfg, ate_setup['pairs'])
        drift_sym = np.linalg.norm(opt_sym.tau - np.eye(2), 'fro')

        # Shifted with effective_bound = |0.5| + 0.5 = 1.0
        mech_dir = EntrywiseBox(B=B_half, shifted_rows=(0,), d=2, delta=delta)
        opt_dir = fit_gaussian(ate_setup['bundle'], mech_dir, ate_setup['env'],
                               ate_setup['constructive'], cfg, ate_setup['pairs'])
        drift_dir = np.linalg.norm(opt_dir.tau - np.eye(2), 'fro')

        assert drift_dir > drift_sym, (
            f"Directional ||τ-I||={drift_dir:.4f} should exceed "
            f"symmetric ||τ-I||={drift_sym:.4f} at equal effective_bound"
        )

    def test_directional_tau_moves(self, ate_setup):
        """Shifted box [0, 2]: τ must move away from identity."""
        delta = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        mech_dir = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        cfg = OptimConfig(
            eta_tau=0.001, eta_adv=0.005, k_tau=1, k_adv=2,
            n_iters=500, tol=1e-5, conv_window=30, seed=42,
            grad_backend='autograd',
        )
        opt = fit_gaussian(ate_setup['bundle'], mech_dir, ate_setup['env'],
                           ate_setup['constructive'], cfg, ate_setup['pairs'])
        drift = np.linalg.norm(opt.tau - np.eye(2), 'fro')
        assert drift > 0.1, (
            f"Directional: τ must move from I but ||τ-I||={drift:.4f}"
        )

    def test_directional_lower_loss(self, ate_setup):
        """Directional box achieves lower training loss than symmetric at
        equal effective_bound.

        This is the FAIR comparison: both boxes have effective_bound = 1.0
        (symmetric B=1.0 vs shifted δ=0.5, B=0.5). The shifted box
        constrains the adversary to a directional subregion, allowing τ
        to exploit the shift direction for a lower minimax loss.
        """
        B_full = np.array([[0.0, 1.0], [0.0, 0.0]])
        B_half = np.array([[0.0, 0.5], [0.0, 0.0]])
        delta = np.array([[0.0, 0.5], [0.0, 0.0]])
        cfg = OptimConfig(
            eta_tau=0.001, eta_adv=0.005, k_tau=1, k_adv=2,
            n_iters=500, tol=1e-5, conv_window=30, seed=42,
            grad_backend='autograd',
        )
        # Symmetric: eff_bound = 1.0
        mech_sym = EntrywiseBox(B=B_full, shifted_rows=(0,), d=2, delta=None)
        opt_sym = fit_gaussian(ate_setup['bundle'], mech_sym, ate_setup['env'],
                               ate_setup['constructive'], cfg, ate_setup['pairs'])

        # Shifted: eff_bound = |0.5| + 0.5 = 1.0
        mech_dir = EntrywiseBox(B=B_half, shifted_rows=(0,), d=2, delta=delta)
        opt_dir = fit_gaussian(ate_setup['bundle'], mech_dir, ate_setup['env'],
                               ate_setup['constructive'], cfg, ate_setup['pairs'])

        assert opt_dir.final_loss < opt_sym.final_loss, (
            f"Directional loss ({opt_dir.final_loss:.4f}) should be < "
            f"symmetric loss ({opt_sym.final_loss:.4f}) at equal effective_bound"
        )


# ---------------------------------------------------------------------------
# Tier 2b: Corner enumeration correctness
# ---------------------------------------------------------------------------

class TestCornerEnumerationShifted:
    """Corner enumeration with delta produces correct endpoints."""

    def test_corners_symmetric(self):
        """delta=None: corners are ±B as before."""
        B = np.array([[0.0, 0.5], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=None)
        corners = _enumerate_box_corners(box)
        assert len(corners) == 2  # 2^1 = 2
        vals = sorted([c[0, 1] for c in corners])
        np.testing.assert_allclose(vals, [-0.5, 0.5])

    def test_corners_shifted(self):
        """delta≠None: corners are {delta-B, delta+B}."""
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 1.0], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        corners = _enumerate_box_corners(box)
        assert len(corners) == 2  # 2^1 = 2
        vals = sorted([c[0, 1] for c in corners])
        # delta-B = 0, delta+B = 2
        np.testing.assert_allclose(vals, [0.0, 2.0])

    def test_corners_two_entries_shifted(self):
        """Two free entries with delta: 2^2 = 4 corners."""
        B = np.array([[0.0, 0.0, 0.3],
                       [0.0, 0.0, 0.3],
                       [0.0, 0.0, 0.0]])
        delta = np.array([[0.0, 0.0, 0.1],
                           [0.0, 0.0, -0.05],
                           [0.0, 0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0, 1), d=3, delta=delta)
        corners = _enumerate_box_corners(box)
        assert len(corners) == 4
        # Entry (0,2): {0.1-0.3, 0.1+0.3} = {-0.2, 0.4}
        # Entry (1,2): {-0.05-0.3, -0.05+0.3} = {-0.35, 0.25}
        vals_02 = sorted(set(c[0, 2] for c in corners))
        vals_12 = sorted(set(c[1, 2] for c in corners))
        np.testing.assert_allclose(vals_02, [-0.2, 0.4])
        np.testing.assert_allclose(vals_12, [-0.35, 0.25])


# ---------------------------------------------------------------------------
# Tier 3: Shifted box at equal effective_bound beats symmetric
# ---------------------------------------------------------------------------

class TestShiftedBoxBehavior:
    """Verify shifted box geometry, certificate, and dominance over symmetric."""

    def test_project_matches_onesided(self):
        """Shifted box [0, 2.0] must clip correctly."""
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 1.0], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)

        # Clips to [0, 2] at entry (0,1)
        dW_hi = np.array([[0.0, 5.0], [0.0, 0.0]])
        np.testing.assert_array_equal(box.project(dW_hi),
                                       np.array([[0.0, 2.0], [0.0, 0.0]]))
        dW_lo = np.array([[0.0, -1.0], [0.0, 0.0]])
        np.testing.assert_array_equal(box.project(dW_lo),
                                       np.array([[0.0, 0.0], [0.0, 0.0]]))
        dW_mid = np.array([[0.0, 1.3], [0.0, 0.0]])
        np.testing.assert_array_equal(box.project(dW_mid),
                                       np.array([[0.0, 1.3], [0.0, 0.0]]))

    def test_corners_match_onesided(self):
        """Corners must be {0, 2.0} at (0,1)."""
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 1.0], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        corners = _enumerate_box_corners(box)
        vals = sorted([c[0, 1] for c in corners])
        np.testing.assert_allclose(vals, [0.0, 2.0])

    def test_sample_within_onesided_range(self):
        """All samples must be in [0, 2.0] at (0,1)."""
        B = np.array([[0.0, 1.0], [0.0, 0.0]])
        delta = np.array([[0.0, 1.0], [0.0, 0.0]])
        box = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        rng = np.random.default_rng(2026)
        for _ in range(500):
            dW = box.sample(rng)
            assert dW[0, 1] >= -1e-12, f"Sample below 0: {dW[0, 1]}"
            assert dW[0, 1] <= 2.0 + 1e-12, f"Sample above 2: {dW[0, 1]}"

    def test_shifted_beats_symmetric_at_equal_effective_bound(self, ate_setup):
        """At equal effective_bound, shifted box achieves lower minimax loss.

        This is the central claim of the directional prior: constraining the
        adversary to a directional subregion [δ-B, δ+B] instead of the
        symmetric [-eff, +eff] allows τ to exploit the shift direction.

        Uses eps=1.0 to keep radii moderate. Symmetric B=0.5 (eff=0.5) vs
        shifted δ=0.25, B=0.25 (eff=|0.25|+0.25=0.5).
        """
        B_sym = np.array([[0.0, 0.5], [0.0, 0.0]])
        B_dir = np.array([[0.0, 0.25], [0.0, 0.0]])
        delta = np.array([[0.0, 0.25], [0.0, 0.0]])
        env = GelbrichBall(
            mu_s=ate_setup['bundle'].noise_mean.copy(),
            Sigma_s=ate_setup['bundle'].noise_cov.copy(),
            eps=1.0, shifted_rows=(1,),
        )
        cfg = OptimConfig(
            eta_tau=0.001, eta_adv=0.005, k_tau=1, k_adv=2,
            n_iters=500, tol=1e-5, conv_window=30, seed=42,
            grad_backend='autograd',
        )

        mech_sym = EntrywiseBox(B=B_sym, shifted_rows=(0,), d=2, delta=None)
        opt_sym = fit_gaussian(ate_setup['bundle'], mech_sym, env,
                               ate_setup['constructive'], cfg, ate_setup['pairs'])

        mech_dir = EntrywiseBox(B=B_dir, shifted_rows=(0,), d=2, delta=delta)
        opt_dir = fit_gaussian(ate_setup['bundle'], mech_dir, env,
                               ate_setup['constructive'], cfg, ate_setup['pairs'])

        # Shifted τ should move more
        drift_sym = np.linalg.norm(opt_sym.tau - np.eye(2), 'fro')
        drift_dir = np.linalg.norm(opt_dir.tau - np.eye(2), 'fro')
        assert drift_dir > drift_sym, (
            f"Shifted ||τ-I||={drift_dir:.4f} should exceed "
            f"symmetric ||τ-I||={drift_sym:.4f}"
        )

        # Shifted should achieve lower loss
        assert opt_dir.final_loss < opt_sym.final_loss, (
            f"Shifted loss ({opt_dir.final_loss:.4f}) should be < "
            f"symmetric loss ({opt_sym.final_loss:.4f})"
        )

    def test_mc_coverage_shifted(self, ate_setup):
        """MC coverage: general cert must hold for all in-ball samples."""
        B = np.array([[0.0, 0.5], [0.0, 0.0]])
        delta = np.array([[0.0, 0.25], [0.0, 0.0]])
        mech = EntrywiseBox(B=B, shifted_rows=(0,), d=2, delta=delta)
        env = GelbrichBall(
            mu_s=ate_setup['bundle'].noise_mean.copy(),
            Sigma_s=ate_setup['bundle'].noise_cov.copy(),
            eps=1.0, shifted_rows=(1,),
        )
        cfg = OptimConfig(
            eta_tau=0.001, eta_adv=0.005, k_tau=1, k_adv=2,
            n_iters=500, tol=1e-5, conv_window=30, seed=42,
            grad_backend='autograd',
        )
        opt = fit_gaussian(ate_setup['bundle'], mech, env,
                           ate_setup['constructive'], cfg, ate_setup['pairs'])

        # Compute general certificate using effective_bound (gamma picks it up)
        cert_total = 0.0
        for iota_idx, O in ate_setup['pairs']:
            A_i = ate_setup['A_iotas'][iota_idx]
            R_i = ate_setup['R_iotas'][iota_idx]
            S_O = selection_matrix(2, O)
            g = gamma(A_i, R_i, mech)
            a = alpha_polynomial(A_i, g, 2)
            c = delta_iota_rho_sq(
                opt.tau, a, A_i,
                ate_setup['bundle'].noise_mean,
                ate_setup['bundle'].noise_cov,
                env.eps, S_O=S_O,
            )
            cert_total += c
        cert = cert_total / len(ate_setup['pairs'])

        # MC coverage
        K = 500
        rng_mc = np.random.default_rng(2026)
        violations = 0
        for _ in range(K):
            dW_k = mech.sample(rng_mc)
            mu_t_k, Sigma_t_k = env.sample(rng_mc)
            l = _compute_loss(ate_setup, opt.tau, dW_k, mu_t_k, Sigma_t_k)
            if l > cert:
                violations += 1

        assert violations == 0, (
            f"Certificate violated: {violations}/{K} violations"
        )
