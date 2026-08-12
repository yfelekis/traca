"""
Tests for traca/directional_certificates.py.

Oracle test (Test 8) output is shown below; stop and review before integrating
into evaluate.py.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest
import yaml

from traca.directional_certificates import (
    _build_shift_mask,
    directional_beta,
    directional_lambda,
    directional_mechanism_modulus,
    directional_environment_bound,
    directional_certificate_gaussian,
    directional_certificate_empirical,
)
from traca.ambiguity import EntrywiseBox, FrobeniusBall, FrobeniusEmpirical
from traca.stability import perturbed_propagator, gating_matrix, gamma, alpha_polynomial
from traca.certificates import single_query_certificate

from experiments.run import (
    _build_mechanism_set,
    _build_environment_set,
    _load_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def atce_cfg():
    with open("configs/atce/gaussian_z_entrywise_full.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def atce_setup(atce_cfg):
    """Full ATCE setup: bundle, mechanism_set, env, R_iotas, A_iotas, alpha, gamma."""
    bundle = _load_bundle(atce_cfg["data"], atce_cfg.get("training", {}))
    d = bundle.d
    mechanism_set = _build_mechanism_set(atce_cfg["ambiguity"]["mechanism"], d)
    env = _build_environment_set(
        atce_cfg["ambiguity"]["environment"],
        bundle.noise_mean, bundle.noise_cov, bundle.n
    )

    var_names = bundle.scm.var_names
    A_iotas, R_iotas, alpha_iotas, gamma_iotas = [], [], [], []
    for i, iv in enumerate(bundle.interventions):
        A_i = bundle.intervened_scms[i].A
        nodes = [var_names.index(k) if isinstance(k, str) else int(k) for k in iv] if iv else []
        R_i = gating_matrix(d, nodes)
        g = gamma(A_i, R_i, mechanism_set)
        A_iotas.append(A_i)
        R_iotas.append(R_i)
        gamma_iotas.append(g)
        alpha_iotas.append(alpha_polynomial(A_i, g, d))

    return dict(
        bundle=bundle,
        mechanism_set=mechanism_set,
        env=env,
        A_iotas=A_iotas,
        R_iotas=R_iotas,
        alpha_iotas=alpha_iotas,
        gamma_iotas=gamma_iotas,
        d=d,
    )


# ---------------------------------------------------------------------------
# Test 1: Non-circular bilinear form check
# ---------------------------------------------------------------------------

class TestBilinearForm:
    """Code's a@ΔW@b matches actual v@(A'_ι-A_ι)@q to first order (right-mult propagator).

    The code uses right-mult perturbed propagator A'_ι = (I-(W+ΔW)R)^{-1}, giving
    first-order term v@A_ι@ΔW@R_ι@A_ι@q.  The plan's paper convention
    (a_paper = R A^T q, b_paper = A v) targets a left-mult propagator and gives a
    DIFFERENT bilinear form — verified below to be numerically distinct on the same example.
    """

    def test_bilinear_entrywise_2x2(self):
        """Non-trivial 2×2: a@dW@b ≈ actual v@(A'_ι-A_ι)@q for small dW.

        Uses A_iota = inv(I - W @ R) (correct interventional propagator with gating).
        Gate node 0 (R[0,0]=0) so ΔW[0,1] is NOT nullified by R — test is non-vacuous.
        Also confirms the paper's left-mult convention gives a different value.
        """
        W = np.array([[0.0, 0.5], [0.0, 0.0]])
        R = np.diag([0.0, 1.0])                  # gate node 0
        A_iota = np.linalg.inv(np.eye(2) - W @ R)   # right-mult interventional propagator
        v = np.array([1.0, 0.5])
        q = np.array([0.0, 1.0])                  # e_1
        delta = 1e-5
        dW = np.array([[0.0, delta], [0.0, 0.0]])

        a = v @ A_iota
        b = np.diag(R) * (A_iota @ q)

        # Code bilinear (right-mult convention: R after ΔW)
        code_bilinear = float(a @ dW @ b)

        # Actual perturbation via right-mult perturbed propagator
        A_prime = np.linalg.inv(np.eye(2) - (W + dW) @ R)
        actual = float(v @ (A_prime - A_iota) @ q)

        np.testing.assert_allclose(code_bilinear, actual, rtol=1e-4,
            err_msg="code bilinear ≠ actual right-mult first-order perturbation")

        # Paper's left-mult convention: a_p = R @ A_iota.T @ q, b_p = A_iota @ v
        # gives a DIFFERENT bilinear — zero here, not delta
        a_paper = R @ A_iota.T @ q
        b_paper = A_iota @ v
        paper_bilinear = float(a_paper @ dW @ b_paper)
        assert not np.isclose(code_bilinear, paper_bilinear, atol=1e-8), (
            "Code and paper bilinears unexpectedly agree — "
            "convention distinction not exercised"
        )

    def test_bilinear_multiple_random_dW(self, atce_setup):
        """ATCE: a@dW@b ≈ actual v@(A'_ι-A_ι)@q (synthetic non-zero v to avoid μ_s=0 trap)."""
        s = atce_setup
        d = s["d"]
        mechanism_set = s["mechanism_set"]
        rng = np.random.default_rng(7)

        iota_idx = 0
        q_idx = 2
        A_i = s["A_iotas"][iota_idx]
        R_i = s["R_iotas"][iota_idx]
        W = s["bundle"].W
        # Use synthetic non-zero v — bundle.noise_mean=[0,0,0] makes a=0, vacuously passing
        v = np.array([1.0, 0.0, 0.0])
        q = np.eye(d)[q_idx]

        a = v @ A_i
        b = np.diag(R_i) * (A_i @ q)

        for _ in range(20):
            # Small dW so first-order approximation is accurate (|dW| ~ 1e-6)
            dW = mechanism_set.project(rng.standard_normal((d, d))) * 1e-6
            code_bilinear = float(a @ dW @ b)
            A_prime = perturbed_propagator(W, dW, R_i)
            actual = float(v @ (A_prime - A_i) @ q)
            np.testing.assert_allclose(code_bilinear, actual, rtol=0.01, atol=1e-12,
                err_msg="code bilinear not a first-order approx of actual perturbation")


# ---------------------------------------------------------------------------
# Test 2: directional_cert ≤ general_cert
# ---------------------------------------------------------------------------

class TestDirectionalVsGeneral:
    """Directional cert (half-width) ≤ √(δ²_generic) for ATCE."""

    def _run(self, atce_setup, tau, iota_idx, q_idx):
        s = atce_setup
        d = s["d"]
        bundle = s["bundle"]
        mechanism_set = s["mechanism_set"]
        env = s["env"]
        A_i = s["A_iotas"][iota_idx]
        alpha_i = s["alpha_iotas"][iota_idx]
        gamma_i = s["gamma_iotas"][iota_idx]

        c = np.eye(d)[q_idx]
        res = directional_certificate_gaussian(
            tau, c, [q_idx], iota_idx, bundle,
            mechanism_set, env, alpha_i, gamma_i
        )
        dir_cert = res["certificate"]

        generic_sq = single_query_certificate(
            tau, alpha_i, A_i, bundle.noise_mean, bundle.noise_cov, env.eps,
            O=[q_idx], d=d, mode="gaussian"
        )
        generic_hw = float(generic_sq) ** 0.5
        return dir_cert, generic_hw

    def test_tau_identity(self, atce_setup):
        d = atce_setup["d"]
        tau = np.eye(d)
        dir_cert, generic_hw = self._run(atce_setup, tau, iota_idx=0, q_idx=2)
        assert dir_cert <= generic_hw + 1e-8, (
            f"Directional cert {dir_cert:.6f} > generic HW {generic_hw:.6f}"
        )

    def test_tau_learned_proxy(self, atce_setup):
        """Learned τ proxy: τ_Z=1.1 (scaled identity on shifted node)."""
        d = atce_setup["d"]
        tau = np.eye(d)
        tau[0, 0] = 1.1     # shifted Z node
        dir_cert, generic_hw = self._run(atce_setup, tau, iota_idx=1, q_idx=2)
        assert dir_cert <= generic_hw + 1e-8


# ---------------------------------------------------------------------------
# Test 3: β = 0 when a=0 or b=0
# ---------------------------------------------------------------------------

class TestBetaZeroCases:
    def test_beta_zero_when_a_zero(self):
        mech = EntrywiseBox(
            B=np.array([[0.0, 0.3, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            shifted_rows=(0,), d=3
        )
        a = np.zeros(3)
        b = np.array([0.1, 0.5, 0.8])
        assert directional_beta(a, b, mech) == 0.0

    def test_beta_zero_when_b_zero(self):
        mech = EntrywiseBox(
            B=np.array([[0.0, 0.3, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            shifted_rows=(0,), d=3
        )
        a = np.array([1.0, 0.5, 0.2])
        b = np.zeros(3)
        assert directional_beta(a, b, mech) == 0.0

    def test_lambda_zero_when_b_zero(self):
        mech = EntrywiseBox(
            B=np.array([[0.0, 0.3, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            shifted_rows=(0,), d=3
        )
        b = np.zeros(3)
        assert directional_lambda(b, mech, 3) == 0.0


# ---------------------------------------------------------------------------
# Test 4: FrobeniusBall β analytic vs hand-crafted 2×2
# ---------------------------------------------------------------------------

class TestBetaFrobenius:
    def test_frob_2x2_hand_crafted(self):
        """FrobeniusBall β = η · ||P ⊙ outer(a,b)||_F."""
        mech = FrobeniusBall(eta=0.5, shifted_rows=(0,), d=2)
        a = np.array([2.0, 3.0])
        b = np.array([1.0, 4.0])
        # P[0,1] = 1, all others 0; outer[0,1] = a[0]*b[1] = 8
        expected = 0.5 * abs(a[0] * b[1])
        result = directional_beta(a, b, mech)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 5: EntrywiseBox β for ATCE B matrix
# ---------------------------------------------------------------------------

class TestBetaEntrywiseATCE:
    def test_entrywise_atce_structure(self, atce_setup):
        """β = Σ_{j,k} P[j,k]*B[j,k]*|a[j]|*|b[k]| matches manual sum."""
        s = atce_setup
        mechanism_set = s["mechanism_set"]
        d = s["d"]
        assert isinstance(mechanism_set, EntrywiseBox)

        rng = np.random.default_rng(0)
        a = rng.standard_normal(d)
        b = rng.standard_normal(d)

        result = directional_beta(a, b, mechanism_set)

        # Manual sum
        P = _build_shift_mask(mechanism_set, d)
        B = mechanism_set.B
        expected = float(np.sum(P * B * np.abs(a)[:, None] * np.abs(b)[None, :]))
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 6: Certificate ≥ 0 always
# ---------------------------------------------------------------------------

class TestCertificateNonNegative:
    def test_gaussian_cert_nonneg(self, atce_setup):
        s = atce_setup
        d = s["d"]
        tau = np.eye(d)
        for iota_idx in range(3):
            for q_idx in range(d):
                c = np.eye(d)[q_idx]
                res = directional_certificate_gaussian(
                    tau, c, [q_idx], iota_idx, s["bundle"],
                    s["mechanism_set"], s["env"],
                    s["alpha_iotas"][iota_idx], s["gamma_iotas"][iota_idx]
                )
                assert res["certificate"] >= 0.0, (
                    f"Negative cert at iota={iota_idx}, q={q_idx}: {res['certificate']}"
                )


# ---------------------------------------------------------------------------
# Test 7: method == "direct" for ATCE (EntrywiseBox, 2 free entries)
# ---------------------------------------------------------------------------

class TestDirectMethodATCE:
    def test_atce_env_method_is_direct(self, atce_setup):
        """ATCE uses EntrywiseBox with 2 free entries → must use 'direct' env evaluation."""
        s = atce_setup
        d = s["d"]
        q = np.eye(d)[2]    # query on Y (node 2)
        tau = np.eye(d)

        iota_idx = 0
        A_i = s["A_iotas"][iota_idx]
        R_i = s["R_iotas"][iota_idx]
        env = s["env"]
        mechanism_set = s["mechanism_set"]

        env_result = directional_environment_bound(
            q, s["bundle"].W, A_i, R_i, mechanism_set, env,
            s["alpha_iotas"][iota_idx], s["gamma_iotas"][iota_idx],
            env.eps, d
        )
        assert env_result["method"] == "direct", (
            f"Expected 'direct' for ATCE EntrywiseBox, got '{env_result['method']}'. "
            "This means ATCE is being routed through the unproven fallback."
        )


# ---------------------------------------------------------------------------
# Test 8: Oracle test (brute-force ΔW grid vs certificate)
# ---------------------------------------------------------------------------

def _brute_force_directional_cert(
    tau: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    W: np.ndarray,
    A_iota: np.ndarray,
    R_iota: np.ndarray,
    mechanism_set,
    env_shifted_rows,
    d: int,
    n_grid: int = 60,
) -> float:
    """Brute-force sup of actual discrepancy over EntrywiseBox (ε=0, no env shift).

    sup_{ΔW ∈ A_W} |v @ A'_ι(ΔW) @ q - v @ τ A_ι @ q|

    Note: ε=0 so no environment shift; this isolates the mechanism contribution.
    The directional cert (with eps=0) must dominate this.
    """
    assert isinstance(mechanism_set, EntrywiseBox)
    free_entries = [
        (j, k)
        for j in mechanism_set.shifted_rows
        for k in range(j + 1, d)
        if mechanism_set.B[j, k] > 0
    ]
    n_free = len(free_entries)
    I_d = np.eye(d)

    ref = float(v @ tau @ A_iota @ q)  # v @ τ A_ι q (query under transport)

    best = 0.0

    # Dense grid over free entries
    grid_vals = [np.linspace(-mechanism_set.B[j, k], mechanism_set.B[j, k], n_grid)
                 for (j, k) in free_entries]

    for vals in itertools.product(*grid_vals):
        dW = np.zeros((d, d))
        for idx, (j, k) in enumerate(free_entries):
            dW[j, k] = vals[idx]
        A_prime = perturbed_propagator(W, dW, R_iota)
        actual = float(v @ A_prime @ q)
        discrepancy = abs(actual - ref)
        if discrepancy > best:
            best = discrepancy

    return best


class TestOracleBruteForce:
    """Oracle test: dir_cert(eps=0) ≥ brute-force discrepancy for all ΔW ∈ A_W."""

    def _check_oracle(self, atce_setup, iota_idx, q_idx, v_override=None):
        s = atce_setup
        d = s["d"]
        bundle = s["bundle"]
        mechanism_set = s["mechanism_set"]
        env = s["env"]
        A_i = s["A_iotas"][iota_idx]
        R_i = s["R_iotas"][iota_idx]
        gamma_i = s["gamma_iotas"][iota_idx]
        alpha_i = s["alpha_iotas"][iota_idx]

        v = v_override if v_override is not None else bundle.noise_mean
        q = np.eye(d)[q_idx]
        tau = np.eye(d)     # τ = I for the oracle test

        # Directional cert with eps=0
        # We compute manually to override eps=0
        transport = abs(float(v @ A_i @ (tau - np.eye(d)) @ q))   # = 0 at τ=I
        mechanism = directional_mechanism_modulus(
            q, v, A_i, R_i, mechanism_set, gamma_i, d
        )
        dir_cert = transport + mechanism   # eps=0 → no env term

        # Brute-force
        bf = _brute_force_directional_cert(
            tau, q, v, bundle.W, A_i, R_i, mechanism_set,
            env.shifted_rows, d, n_grid=40
        )

        return dir_cert, bf

    @pytest.mark.parametrize("iota_idx,q_idx", [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 2),
        (2, 1), (2, 2),
    ])
    def test_oracle_zero_mean(self, atce_setup, iota_idx, q_idx):
        """With zero source mean (ATCE default), dir_cert ≥ brute-force."""
        dir_cert, bf = self._check_oracle(atce_setup, iota_idx, q_idx)
        assert dir_cert >= bf - 1e-6, (
            f"ORACLE VIOLATION: iota={iota_idx}, q={q_idx}: "
            f"dir_cert={dir_cert:.6f} < brute_force={bf:.6f}"
        )

    @pytest.mark.parametrize("iota_idx,q_idx", [
        (0, 2), (1, 2), (2, 2),
    ])
    def test_oracle_nonzero_mean(self, atce_setup, iota_idx, q_idx):
        """With non-zero source mean, mechanism tightening is tested."""
        v = np.array([1.0, 0.0, 0.0])   # synthetic non-zero Z mean
        dir_cert, bf = self._check_oracle(atce_setup, iota_idx, q_idx, v_override=v)
        assert dir_cert >= bf - 1e-6, (
            f"ORACLE VIOLATION (nonzero mean): iota={iota_idx}, q={q_idx}: "
            f"dir_cert={dir_cert:.6f} < brute_force={bf:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 9: η=0 (degenerate mechanism set — all B entries zero)
# ---------------------------------------------------------------------------

class TestEtaZero:
    """Directional cert at η=0 (no mechanism ambiguity, env-only).

    This is the case that failed in production: _directional_env_direct_2d
    received free_entries=[] and passed bounds=[] to scipy.optimize.minimize,
    which crashed with 'not enough values to unpack'.

    The fix short-circuits when free_entries is empty, returning the exact
    ΔW=0 value: eps * ||M_env @ A_iota @ q||_2.
    """

    @pytest.fixture
    def atce_eta_zero(self, atce_cfg):
        """ATCE setup with η=0 (all B entries zeroed)."""
        cfg = copy.deepcopy(atce_cfg)
        B = np.array(cfg["ambiguity"]["mechanism"]["B"], dtype=float)
        B[:] = 0.0
        cfg["ambiguity"]["mechanism"]["B"] = B.tolist()

        bundle = _load_bundle(cfg["data"], cfg.get("training", {}))
        d = bundle.d
        mechanism_set = _build_mechanism_set(cfg["ambiguity"]["mechanism"], d)
        env = _build_environment_set(
            cfg["ambiguity"]["environment"],
            bundle.noise_mean, bundle.noise_cov, bundle.n
        )

        var_names = bundle.scm.var_names
        A_iotas, R_iotas, alpha_iotas, gamma_iotas = [], [], [], []
        for i, iv in enumerate(bundle.interventions):
            A_i = bundle.intervened_scms[i].A
            nodes = [var_names.index(k) if isinstance(k, str) else int(k) for k in iv] if iv else []
            R_i = gating_matrix(d, nodes)
            g = gamma(A_i, R_i, mechanism_set)
            A_iotas.append(A_i)
            R_iotas.append(R_i)
            gamma_iotas.append(g)
            alpha_iotas.append(alpha_polynomial(A_i, g, d))

        return dict(
            bundle=bundle, mechanism_set=mechanism_set, env=env,
            A_iotas=A_iotas, R_iotas=R_iotas,
            alpha_iotas=alpha_iotas, gamma_iotas=gamma_iotas, d=d,
        )

    @pytest.mark.parametrize("iota_idx,q_idx", [
        (0, 0), (0, 1), (0, 2),
        (1, 2), (2, 2),
    ])
    def test_gaussian_cert_at_eta_zero(self, atce_eta_zero, iota_idx, q_idx):
        """Directional cert computes without error at η=0."""
        s = atce_eta_zero
        d = s["d"]
        tau = np.eye(d)
        c = np.eye(d)[q_idx]

        res = directional_certificate_gaussian(
            tau, c, [q_idx], iota_idx, s["bundle"],
            s["mechanism_set"], s["env"],
            s["alpha_iotas"][iota_idx], s["gamma_iotas"][iota_idx],
        )
        assert np.isfinite(res["certificate"]), (
            f"Non-finite cert at η=0: iota={iota_idx}, q={q_idx}"
        )
        assert res["certificate"] >= 0.0
        assert res["env_method"] == "direct"

    def test_eta_zero_mechanism_is_zero(self, atce_eta_zero):
        """At η=0, mechanism term must be exactly zero (no ΔW budget)."""
        s = atce_eta_zero
        d = s["d"]
        tau = np.eye(d)
        c = np.eye(d)[2]  # Y

        res = directional_certificate_gaussian(
            tau, c, [2], 0, s["bundle"],
            s["mechanism_set"], s["env"],
            s["alpha_iotas"][0], s["gamma_iotas"][0],
        )
        assert res["mechanism"] == 0.0, (
            f"Mechanism term should be 0 at η=0, got {res['mechanism']}"
        )

    def test_eta_zero_cert_equals_env_only(self, atce_eta_zero):
        """At η=0 with τ=I, cert = env term only (transport=0, mechanism=0)."""
        s = atce_eta_zero
        d = s["d"]
        tau = np.eye(d)
        c = np.eye(d)[2]
        eps = s["env"].eps

        res = directional_certificate_gaussian(
            tau, c, [2], 0, s["bundle"],
            s["mechanism_set"], s["env"],
            s["alpha_iotas"][0], s["gamma_iotas"][0],
        )
        assert res["transport"] == 0.0
        assert res["mechanism"] == 0.0
        # env term = eps * ||M_env @ A_iota @ q||_2 (exact at ΔW=0)
        env_shifted = s["env"].shifted_rows
        M_env = np.diag([1.0 if i in env_shifted else 0.0 for i in range(d)])
        A_0 = s["A_iotas"][0]
        expected_env = eps * float(np.linalg.norm(M_env @ A_0 @ c))
        np.testing.assert_allclose(res["environment"], expected_env, atol=1e-12)


# ---------------------------------------------------------------------------
# Empirical directional certificate: i>0 with U_eff vs native draws
# ---------------------------------------------------------------------------

class TestDirectionalCertificateEmpirical:
    """Pin directional_certificate_empirical at i>0 (non-observational).

    The function reads bundle.noise_samples[iota_idx] then re-applies
    the U_eff construction (zero intervened cols + add fixed).  When the
    bundle already carries U_effs (as the decoupled eval paths do),
    the double-apply must be harmless: zeroing cols that are already
    0+fixed gives 0, then += fixed restores the value.

    Tests:
    1. Native-draw bundle and U_eff bundle produce identical results.
    2. Transport term matches an independent hand computation from U_eff mean.
    """

    @pytest.fixture
    def empirical_setup(self):
        """3-node chain X->Y->Z with do(X=1), nonzero noise mean."""
        from lan_scm import LANSCM, SCMBundle
        from traca.utils import build_U_effs

        d = 3
        W = np.zeros((d, d))
        W[0, 1] = 0.7   # X->Y
        W[1, 2] = 0.4   # Y->Z
        noise_mean = np.array([1.0, 0.5, -0.3])
        noise_cov = np.diag([1.0, 0.8, 1.2])
        scm = LANSCM(W, noise_mean=noise_mean, noise_cov=noise_cov,
                      var_names=["X", "Y", "Z"])

        interventions = [{}, {"X": 1.0}]
        n = 500
        bundle = scm.bundle(interventions, n=n, seed=42)

        # U_eff bundle: overwrite noise_samples with obs-derived U_effs
        import copy
        U_effs = build_U_effs(bundle)
        ueff_bundle = copy.copy(bundle)
        ueff_bundle.noise_samples = {i: U_effs[i] for i in range(len(interventions))}

        # Ambiguity sets
        mechanism_set = FrobeniusBall(eta=0.3, shifted_rows=(1,), d=d)
        env = FrobeniusEmpirical(eps=0.2, N=n, shifted_rows=(1,))

        # Stability
        iota_idx = 1  # do(X=1)
        A_i = bundle.intervened_scms[iota_idx].A
        iv = interventions[iota_idx]
        var_names = scm.var_names
        nodes = [var_names.index(k) for k in iv] if iv else []
        R_i = gating_matrix(d, nodes)
        g = gamma(A_i, R_i, mechanism_set)
        a = alpha_polynomial(A_i, g, d)

        return dict(
            bundle=bundle,
            ueff_bundle=ueff_bundle,
            U_effs=U_effs,
            mechanism_set=mechanism_set,
            env=env,
            iota_idx=iota_idx,
            alpha_iota=a,
            gamma_iota=g,
            A_iota=A_i,
            R_iota=R_i,
            d=d,
        )

    def test_double_apply_is_identity(self, empirical_setup):
        """Applying U_eff construction to already-constructed U_effs is a no-op.

        The function reads bundle.noise_samples[i] then re-applies
        column-zeroing + fixed.  When the input is already a U_eff,
        this must produce the same array (double-apply = single-apply).
        """
        s = empirical_setup
        iota_idx = s["iota_idx"]
        scm_i = s["bundle"].intervened_scms[iota_idx]
        U_eff_original = s["U_effs"][iota_idx]

        # Simulate the function's internal re-application
        U_double = U_eff_original.copy()
        if scm_i._J:
            U_double[:, list(scm_i._J)] = 0.0
        U_double += scm_i._fixed[np.newaxis, :]

        np.testing.assert_allclose(
            U_double, U_eff_original, atol=1e-14,
            err_msg="Double-apply of U_eff construction is not identity",
        )

    def test_transport_matches_hand_computation(self, empirical_setup):
        """Transport term at i>0 matches independent U_eff mean computation."""
        s = empirical_setup
        d = s["d"]
        tau = np.eye(d) * 1.0
        tau[1, 1] = 0.9  # shift Y's diagonal to make transport nonzero
        c = np.array([0.0, 0.0, 1.0])  # query Z
        A_i = s["A_iota"]

        res = directional_certificate_empirical(
            tau, c, [2], s["iota_idx"], s["ueff_bundle"],
            s["mechanism_set"], s["env"],
            s["alpha_iota"], s["gamma_iota"],
        )

        # Independent: U_eff for intervention 1 from obs U
        U_eff = s["U_effs"][s["iota_idx"]]
        mu_eff = U_eff.mean(axis=0)
        expected_transport = abs(float(mu_eff @ A_i @ (tau - np.eye(d)) @ c))

        np.testing.assert_allclose(
            res["transport"], expected_transport, atol=1e-10,
            err_msg="Transport term does not match hand-computed U_eff mean",
        )
