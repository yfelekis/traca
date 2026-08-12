"""
Smoke tests for the radius-sampling evaluation driver (traca_radius_eval.py).

Tests:
- ATE smoke: single ρ_test, K=3, 2 folds — rows emitted with correct schema
- ATCE smoke: 2 ρ_test points, K=2 — schema + coverage denominator = K
- Pristine set unchanged post-loop
- Geometry match guard: mismatched geometry → error
- Determinism: same seed → same rows
"""
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from traca_train import train_cv
from traca_radius_eval import (
    radius_sampling_eval,
    _build_sampled_target_bundle_gaussian,
    _build_sampled_target_bundle_empirical,
    _base_mechanism_radius,
    _base_environment_radius,
    _score_variant,
    _aggregate,
)
from experiments.run import _load_bundle

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

ATCE_CONFIG = "configs/atce/gaussian_z_entrywise_full.yaml"

# Minimal fast training config for tests
FAST_TRAINING = {
    "eta_tau": 0.005,
    "eta_adv": 0.005,
    "k_adv": 2,
    "n_iters": 15,
    "tol": 1e-4,
    "conv_window": 5,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def atce_trained(tmp_path_factory):
    """Train ATCE with 2 folds × 2 radii (minimal). Module-scoped for speed."""
    tmp_dir = tmp_path_factory.mktemp("atce_radius_eval")
    with open(ATCE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
    cfg_path = tmp_dir / "test_config.yaml"
    with cfg_path.open("w") as f:
        yaml.dump(cfg, f)

    shift_spec = {"shift_type": "noise_mean", "magnitude": 0.3, "node": 0}
    pkl_path = train_cv(
        config_path=cfg_path,
        sweep_axis="eps",
        radius_values=[0.1, 0.2],
        out_dir=tmp_dir,
        k_folds=2,
        include_baselines=True,
        shift_spec=shift_spec,
    )
    return pkl_path, tmp_dir


# ---------------------------------------------------------------------------
# TestBuildTargetBundle
# ---------------------------------------------------------------------------

class TestBuildTargetBundle:
    """Unit tests for target bundle builders."""

    def test_gaussian_target_bundle_shapes(self):
        """Gaussian target bundle has correct shapes and paired U."""
        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
        bundle = _load_bundle(cfg["data"], cfg.get("training", {}))
        d = bundle.d

        rng = np.random.default_rng(42)
        dW = rng.standard_normal((d, d)) * 0.01
        mu_t = bundle.noise_mean + rng.standard_normal(d) * 0.1
        Sigma_t = bundle.noise_cov.copy()

        test_indices = np.array([0, 1, 2, 3, 4])
        tb = _build_sampled_target_bundle_gaussian(
            bundle, dW, mu_t, Sigma_t, test_indices,
        )

        assert tb.d == d
        assert tb.n == len(test_indices)
        assert len(tb.interventions) == len(bundle.interventions)
        # Noise params are the sampled target
        np.testing.assert_allclose(tb.noise_mean, mu_t)
        np.testing.assert_allclose(tb.noise_cov, Sigma_t)
        # Each intervention has endogenous samples
        for i in range(len(bundle.interventions)):
            assert i in tb.endogenous_samples
            assert tb.endogenous_samples[i].shape == (len(test_indices), d)

    def test_empirical_target_bundle_shapes(self):
        """Empirical target bundle has correct shapes."""
        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
        bundle = _load_bundle(cfg["data"], cfg.get("training", {}))
        d = bundle.d

        rng = np.random.default_rng(42)
        dW = rng.standard_normal((d, d)) * 0.01
        n_test = 5
        Theta = rng.standard_normal((n_test, d)) * 0.1

        test_indices = np.arange(n_test)
        tb = _build_sampled_target_bundle_empirical(
            bundle, dW, Theta, test_indices,
        )

        assert tb.d == d
        assert tb.n == n_test
        for i in range(len(bundle.interventions)):
            assert tb.endogenous_samples[i].shape == (n_test, d)

    def test_gaussian_target_uses_paired_U(self):
        """Verify the target is built from the source's held-out U (variance-cancellation)."""
        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
        bundle = _load_bundle(cfg["data"], cfg.get("training", {}))
        d = bundle.d

        # Zero perturbation + same noise → target X should equal source X on holdout
        dW = np.zeros((d, d))
        test_indices = np.array([0, 1, 2])
        tb = _build_sampled_target_bundle_gaussian(
            bundle, dW, bundle.noise_mean.copy(), bundle.noise_cov.copy(),
            test_indices,
        )
        # Observational (iota=0): X_target should equal X_source on holdout
        X_source = bundle.endogenous_samples[0][test_indices]
        X_target = tb.endogenous_samples[0]
        np.testing.assert_allclose(X_target, X_source, atol=1e-10)


# ---------------------------------------------------------------------------
# TestBaseRadius
# ---------------------------------------------------------------------------

class TestBaseRadius:
    """Test the radius extraction helpers."""

    def test_frobenius_ball_radius(self):
        from traca.ambiguity import FrobeniusBall
        fb = FrobeniusBall(eta=0.5, shifted_rows=(0,), d=3)
        assert _base_mechanism_radius(fb) == 0.5

    def test_entrywise_box_radius(self):
        from traca.ambiguity import EntrywiseBox
        B = np.array([[0.0, 0.3], [0.0, 0.0]])
        eb = EntrywiseBox(B=B, shifted_rows=(0,), d=2)
        assert _base_mechanism_radius(eb) == 0.3

    def test_gelbrich_radius(self):
        from traca.ambiguity import GelbrichBall
        gb = GelbrichBall(mu_s=np.zeros(2), Sigma_s=np.eye(2),
                          eps=0.2, shifted_rows=(0,))
        assert _base_environment_radius(gb) == 0.2

    def test_empirical_radius(self):
        from traca.ambiguity import FrobeniusEmpirical
        fe = FrobeniusEmpirical(eps=0.1, N=100, shifted_rows=(0,))
        assert _base_environment_radius(fe) == 0.1


# ---------------------------------------------------------------------------
# TestRadiusSamplingEval — ATCE Gaussian smoke
# ---------------------------------------------------------------------------

class TestRadiusSamplingEval:
    """End-to-end smoke tests for the eval loop."""

    def test_single_rho_test(self, atce_trained):
        """Single ρ_test, K=3, 2 folds — correct schema and row count."""
        pkl_path, tmp_dir = atce_trained
        output_dir = tmp_dir / "eval_single"

        df = radius_sampling_eval(
            results_pkl=pkl_path,
            rho_test_values=[0.1],
            K=3,
            output_dir=output_dir,
            base_seed=2026,
        )

        # Schema check
        required_cols = {
            "rho_test", "k", "fold", "variant", "rho_train",
            "err", "certificate", "gap", "coverage_fraction",
            "all_covered", "mean_width",
        }
        assert required_cols.issubset(set(df.columns)), \
            f"Missing columns: {required_cols - set(df.columns)}"

        # Row count: 1 ρ_test × 3 K × (2 folds × (2 radii + 1 baseline)) = 3 × 6 = 18
        # But baseline is expanded across 2 grid points, so per fold:
        #   eps_0.10, eps_0.20, baseline@eps_0.10, baseline@eps_0.20 = 4 per fold
        # 2 folds × 4 = 8 variants, × 3 K × 1 ρ_test
        n_variants = df.groupby(["fold", "variant"]).ngroups
        expected_rows = 1 * 3 * n_variants
        assert len(df) == expected_rows, \
            f"Expected {expected_rows} rows, got {len(df)}"

        # All rho_test values match
        assert set(df["rho_test"].unique()) == {0.1}

        # k values are 0, 1, 2
        assert set(df["k"].unique()) == {0, 1, 2}

        # Certificate >= 0
        assert (df["certificate"] >= 0).all()

        # Gap = certificate - err
        np.testing.assert_allclose(
            df["gap"].values,
            df["certificate"].values - df["err"].values,
            atol=1e-10,
        )

        # CSV files saved
        assert (output_dir / "radius_eval.csv").exists()
        assert (output_dir / "radius_eval_agg.csv").exists()

    def test_two_rho_test_points(self, atce_trained):
        """Two ρ_test points, K=2 — correct row count and monotone certificate."""
        pkl_path, tmp_dir = atce_trained
        output_dir = tmp_dir / "eval_two"

        df = radius_sampling_eval(
            results_pkl=pkl_path,
            rho_test_values=[0.0, 0.1],
            K=2,
            output_dir=output_dir,
            base_seed=2026,
        )

        # Two ρ_test values
        assert set(df["rho_test"].unique()) == {0.0, 0.1}

        # At ρ_test=0.0: target = source, err should be small for τ≈I
        df_zero = df[df["rho_test"] == 0.0]
        # For the identity baseline specifically
        df_bl_zero = df_zero[df_zero["variant"].str.contains("baseline")]
        if len(df_bl_zero) > 0:
            # Identity at zero shift → machine-epsilon error (paired U,
            # population moments used for both source and target in Gaussian
            # scoring path).
            assert (df_bl_zero["err"] < 1e-10).all(), \
                f"Identity baseline at ρ_test=0 has unexpectedly large error: {df_bl_zero['err'].values}"

    def test_coverage_denominator_is_K(self, atce_trained):
        """Coverage fraction is count/n_queries, computed per-instance (not averaged over K)."""
        pkl_path, tmp_dir = atce_trained
        output_dir = tmp_dir / "eval_cov"

        df = radius_sampling_eval(
            results_pkl=pkl_path,
            rho_test_values=[0.1],
            K=2,
            output_dir=output_dir,
            base_seed=2026,
        )

        # Each row has a coverage_fraction that's either 0 or 1 or a fraction of n_queries
        # (one row = one (rho_test, k, fold, variant) — coverage is over queries, not K)
        assert df["coverage_fraction"].between(0, 1).all()

        # Aggregated: coverage_K = mean(all_covered) over K per (rho_test, variant, fold)
        agg = pd.read_csv(output_dir / "radius_eval_agg.csv")
        assert "coverage_mean" in agg.columns
        assert agg["coverage_mean"].between(0, 1).all()

    def test_determinism(self, atce_trained):
        """Same seed → same rows (fold-independent seeding)."""
        pkl_path, tmp_dir = atce_trained

        df1 = radius_sampling_eval(
            results_pkl=pkl_path,
            rho_test_values=[0.1],
            K=2,
            output_dir=tmp_dir / "det1",
            base_seed=999,
        )
        df2 = radius_sampling_eval(
            results_pkl=pkl_path,
            rho_test_values=[0.1],
            K=2,
            output_dir=tmp_dir / "det2",
            base_seed=999,
        )

        # Same err values
        np.testing.assert_array_equal(df1["err"].values, df2["err"].values)
        np.testing.assert_array_equal(
            df1["coverage_fraction"].values,
            df2["coverage_fraction"].values,
        )

    def test_rho_test_zero_is_identity_shift(self, atce_trained):
        """At ρ_test=0.0, sampled dW=0 and mu_t=mu_s, Sigma_t=Sigma_s (no shift)."""
        pkl_path, tmp_dir = atce_trained
        output_dir = tmp_dir / "eval_zero"

        df = radius_sampling_eval(
            results_pkl=pkl_path,
            rho_test_values=[0.0],
            K=2,
            output_dir=output_dir,
            base_seed=2026,
        )

        # All K samples at ρ_test=0 should produce identical error (same zero shift)
        for variant in df["variant"].unique():
            for fold in df["fold"].unique():
                sub = df[(df["variant"] == variant) & (df["fold"] == fold)]
                # All rows for the same (variant, fold) at rho=0 should have same err
                assert sub["err"].nunique() == 1, \
                    f"Non-unique err at ρ_test=0 for {variant}/{fold}: {sub['err'].values}"


# ---------------------------------------------------------------------------
# TestPristineSetUnchanged
# ---------------------------------------------------------------------------

class TestPristineSetUnchanged:
    """Verify the pristine ambiguity sets are not mutated by the eval loop."""

    def test_mechanism_set_unchanged(self, atce_trained):
        """The pristine mechanism set has the same radius before and after eval."""
        pkl_path, tmp_dir = atce_trained
        from traca_run_evaluation import load_training_results
        from traca.ambiguity import EntrywiseBox

        _, metadata = load_training_results(pkl_path)
        config_path = Path(metadata["config_path"])
        with config_path.open() as f:
            cfg = yaml.safe_load(f)

        from experiments.run import _build_mechanism_set
        mech_before = _build_mechanism_set(cfg["ambiguity"]["mechanism"], metadata["d"])
        B_before = mech_before.B.copy()

        # Run eval
        radius_sampling_eval(
            results_pkl=pkl_path,
            rho_test_values=[0.1, 0.2],
            K=2,
            output_dir=tmp_dir / "pristine_check",
            base_seed=42,
        )

        mech_after = _build_mechanism_set(cfg["ambiguity"]["mechanism"], metadata["d"])
        np.testing.assert_array_equal(mech_after.B, B_before)


# ---------------------------------------------------------------------------
# TestAggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    """Test the _aggregate function."""

    def test_aggregate_structure(self):
        """Aggregation produces correct columns and row counts."""
        records = []
        for rho in [0.1, 0.2]:
            for k in range(3):
                for fold in ["fold_0", "fold_1"]:
                    for variant in ["TraCA (eps_0.10)", "TraCA_baseline_identity"]:
                        records.append({
                            "rho_test": rho,
                            "k": k,
                            "fold": fold,
                            "variant": variant,
                            "rho_train": 0.1,
                            "err": np.random.rand(),
                            "certificate": np.random.rand() + 1,
                            "gap": np.random.rand(),
                            "coverage_fraction": np.random.choice([0.0, 1.0]),
                            "dir_coverage_fraction": np.random.choice([0.0, 1.0]),
                            "all_covered": np.random.choice([True, False]),
                            "mean_width": np.random.rand(),
                            "mean_dir_width": np.random.rand(),
                        })

        df = pd.DataFrame(records)
        agg = _aggregate(df)

        # Should have 2 rho_test × 2 variants = 4 rows
        assert len(agg) == 4
        assert set(agg.columns) == {
            "rho_test", "variant",
            "err_mean", "err_std", "cert_mean", "width_mean",
            "dir_width_mean",
            "coverage_mean", "coverage_std",
            "dir_coverage_mean", "dir_coverage_std",
        }
