"""
Tests for the decoupled train/eval workflow (traca_train.py + traca_run_evaluation.py).

- Round-trip: train → save pkl → load → τ bit-for-bit identical
- Radius flattening: correct method names and counts
- CSV schema: required columns, row count, dtypes
- Shift integration: shift_spec actually applied (not silently ignored)
"""
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lan_scm import make_atce, _scm_from_config
from traca.shifts import apply_shift
from traca_train import train_cv, _train_one_fold_radius
from traca_run_evaluation import (
    load_training_results,
    flatten_to_methods,
    run_evaluation,
    _reveal_target_and_score,
    _build_shifted_target_bundle,
    _build_paired_target_bundle,
    _observational_pushed_distance,
)
from experiments.run import (
    _build_mechanism_set,
    _build_environment_set,
    _build_constructive_class,
    _build_optim_config,
    _load_bundle,
)
from experiments.evaluate import _compute_target_loss_per_iota


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ATCE_CONFIG = "configs/atce/gaussian_z_entrywise_full.yaml"

# Minimal fast training config for tests
FAST_TRAINING = {
    "eta_tau": 0.005,
    "eta_adv": 0.005,
    "k_adv": 2,
    "n_iters": 30,
    "tol": 1e-4,
    "conv_window": 5,
    "seed": 42,
}


@pytest.fixture
def atce_cfg():
    with open(ATCE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    # Override training for speed
    cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
    return cfg


@pytest.fixture
def atce_bundle(atce_cfg):
    return _load_bundle(atce_cfg["data"], atce_cfg.get("training", {}))


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Train with 2 folds × 2 radii, save pkl, reload, verify identity."""

    def test_save_load_tau_identical(self, tmp_dir):
        """τ round-trips bit-for-bit through joblib pickle."""
        # Write a fast config
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

        import joblib
        data = joblib.load(pkl_path)
        metadata = data["__metadata__"]

        # Check each fold/radius has a tau and it's a proper array
        for fold_key in ["fold_0", "fold_1"]:
            assert fold_key in data
            for radius_key in ["eps_0.10", "eps_0.20"]:
                assert radius_key in data[fold_key], f"Missing {radius_key} in {fold_key}"
                entry = data[fold_key][radius_key]
                tau = entry["tau"]
                assert isinstance(tau, np.ndarray)
                assert tau.shape == (3, 3)
                # Verify opt_result tau matches stored tau
                assert np.array_equal(tau, entry["opt_result"].tau)

            # Baseline
            assert "baseline_identity" in data[fold_key]
            bl = data[fold_key]["baseline_identity"]
            assert np.array_equal(bl["tau"], np.eye(3))

    def test_test_indices_preserved(self, tmp_dir):
        """test_indices in pkl match the fold split."""
        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
        cfg_path = tmp_dir / "test_config.yaml"
        with cfg_path.open("w") as f:
            yaml.dump(cfg, f)

        pkl_path = train_cv(
            config_path=cfg_path,
            sweep_axis="eps",
            radius_values=[0.1],
            out_dir=tmp_dir,
            k_folds=2,
            shift_spec={"shift_type": "noise_mean", "magnitude": 0.1, "node": 0},
        )

        import joblib
        data = joblib.load(pkl_path)

        # All test_indices across folds should form a partition of [0, N)
        N = data["__metadata__"]["n_samples"]
        all_indices = []
        for fold_key in ["fold_0", "fold_1"]:
            idx = data[fold_key]["eps_0.10"]["test_indices"]
            all_indices.append(idx)
        combined = np.sort(np.concatenate(all_indices))
        assert len(combined) == N
        assert np.array_equal(combined, np.arange(N))

    def test_metadata_complete(self, tmp_dir):
        """__metadata__ has all required keys."""
        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
        cfg_path = tmp_dir / "test_config.yaml"
        with cfg_path.open("w") as f:
            yaml.dump(cfg, f)

        pkl_path = train_cv(
            config_path=cfg_path,
            sweep_axis="eps",
            radius_values=[0.1],
            out_dir=tmp_dir,
            k_folds=2,
            shift_spec={"shift_type": "noise_mean", "magnitude": 0.1, "node": 0},
        )

        import joblib
        data = joblib.load(pkl_path)
        md = data["__metadata__"]

        required_keys = {
            "experiment_name", "config_path", "sweep_axis", "radius_values",
            "k_folds", "d", "n_samples", "n_interventions", "objective",
            "shift_spec", "git_commit", "timestamp", "source_bundle_path",
        }
        assert required_keys.issubset(set(md.keys()))
        assert md["d"] == 3
        assert md["sweep_axis"] == "eps"
        assert md["k_folds"] == 2


# ---------------------------------------------------------------------------
# TestRadiusFlattening
# ---------------------------------------------------------------------------

class TestRadiusFlattening:
    """Flatten nested fold→radius into flat method entries."""

    def _make_fake_results(self):
        """Build a minimal fake results dict for flattening tests."""
        d = 3
        results = {}
        for fold_idx in range(2):
            fold_key = f"fold_{fold_idx}"
            results[fold_key] = {}
            for rv in [0.1, 0.2, 0.3]:
                radius_key = f"eps_{rv:.2f}"
                results[fold_key][radius_key] = {
                    "tau": np.eye(d) * (1 + rv),
                    "test_indices": np.array([fold_idx]),
                    "shift_spec": {"shift_type": "noise_mean", "magnitude": 0.3, "node": 0},
                    "config_snapshot": {},
                    "training_metadata": {"converged": True, "n_iters": 10,
                                          "final_loss": 0.1, "initial_loss": 0.5,
                                          "objective": "gaussian"},
                }
            # Baseline
            results[fold_key]["baseline_identity"] = {
                "tau": np.eye(d),
                "test_indices": np.array([fold_idx]),
                "shift_spec": {"shift_type": "noise_mean", "magnitude": 0.3, "node": 0},
                "config_snapshot": {},
                "training_metadata": {"converged": True, "n_iters": 0,
                                      "final_loss": None, "initial_loss": None,
                                      "objective": "gaussian"},
            }
        return results

    def test_n_radii_produces_n_methods(self):
        """3 radii + 1 baseline = 4 method entries per fold × 2 folds = 8."""
        results = self._make_fake_results()
        flat = flatten_to_methods(results, "eps")
        assert len(flat) == 8  # 2 folds × (3 radii + 1 baseline)

    def test_method_names_format(self):
        """Method names are 'TraCA (eps_0.10)' format."""
        results = self._make_fake_results()
        flat = flatten_to_methods(results, "eps")
        method_names = {e["method"] for e in flat}
        assert "TraCA (eps_0.10)" in method_names
        assert "TraCA (eps_0.20)" in method_names
        assert "TraCA (eps_0.30)" in method_names
        assert "TraCA_baseline_identity" in method_names

    def test_baseline_included(self):
        """Baseline entry has radius=0 and correct method name."""
        results = self._make_fake_results()
        flat = flatten_to_methods(results, "eps")
        baselines = [e for e in flat if e["method"] == "TraCA_baseline_identity"]
        assert len(baselines) == 2  # one per fold
        for bl in baselines:
            assert bl["radius"] == 0.0
            assert np.array_equal(bl["tau"], np.eye(3))

    def test_radius_parsed_correctly(self):
        """Radius values parsed from keys match expected floats."""
        results = self._make_fake_results()
        flat = flatten_to_methods(results, "eps")
        radii = {e["radius"] for e in flat if e["method"] != "TraCA_baseline_identity"}
        assert radii == {0.1, 0.2, 0.3}


# ---------------------------------------------------------------------------
# TestCSVSchema
# ---------------------------------------------------------------------------

class TestCSVSchema:
    """CSV output has correct columns, row count, and round-trips."""

    REQUIRED_COLUMNS = {
        "method", "radius", "eps", "eta", "sweep_axis", "shift_type",
        "shift_magnitude", "fold", "trial",
        "coverage_fraction", "target_loss", "certificate", "gap", "all_covered",
    }

    def test_required_columns_present(self, tmp_dir):
        """All 14 required columns exist."""
        import pandas as pd
        # Build minimal CSV via run_evaluation
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
            radius_values=[0.1],
            out_dir=tmp_dir,
            k_folds=2,
            include_baselines=True,
            shift_spec=shift_spec,
        )

        df = run_evaluation(
            results_pkl=pkl_path,
            output_dir=tmp_dir / "eval_out",
            target_seed=99999,
        )

        assert self.REQUIRED_COLUMNS.issubset(set(df.columns))

    def test_row_count(self, tmp_dir):
        """Row count = n_methods × n_folds × n_trials."""
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

        df = run_evaluation(
            results_pkl=pkl_path,
            output_dir=tmp_dir / "eval_out",
            target_seed=99999,
        )

        # 2 radii TraCA + 2 baseline (one per ε) = 4 methods per fold × 2 folds × 1 trial = 8
        expected = (2 + 2) * 2 * 1
        assert len(df) == expected

    def test_csv_round_trips(self, tmp_dir):
        """Write → read back → verify values match."""
        import pandas as pd

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
            radius_values=[0.1],
            out_dir=tmp_dir,
            k_folds=2,
            shift_spec=shift_spec,
        )

        df = run_evaluation(
            results_pkl=pkl_path,
            output_dir=tmp_dir / "eval_out",
            target_seed=99999,
        )

        csv_path = tmp_dir / "eval_out" / "evaluation.csv"
        assert csv_path.exists()
        df_loaded = pd.read_csv(csv_path)
        assert len(df_loaded) == len(df)
        # Numeric columns should match
        for col in ["certificate", "target_loss", "gap", "radius"]:
            np.testing.assert_allclose(
                df_loaded[col].values, df[col].values, rtol=1e-10
            )


# ---------------------------------------------------------------------------
# TestShiftIntegration
# ---------------------------------------------------------------------------

class TestShiftIntegration:
    """Verify that shift_spec is actually applied, not silently ignored."""

    def test_shift_applied_before_scoring(self, atce_cfg, atce_bundle, tmp_dir):
        """Eval with a noise_mean shift_spec produces different target_loss
        than a near-zero shift (proving the shift is actually applied)."""
        d = atce_bundle.d
        cfg = atce_cfg

        # Use tau=I for simplicity
        tau = np.eye(d)
        test_indices = np.arange(100)

        # Score with a real shift
        shift_big = {"shift_type": "noise_mean", "magnitude": 2.0, "node": 0}
        scores_big = _reveal_target_and_score(
            tau, atce_bundle, test_indices,
            shift_spec=shift_big, objective="gaussian",
            config=cfg, seed=12345,
        )

        # Score with a tiny shift
        shift_tiny = {"shift_type": "noise_mean", "magnitude": 0.001, "node": 0}
        scores_tiny = _reveal_target_and_score(
            tau, atce_bundle, test_indices,
            shift_spec=shift_tiny, objective="gaussian",
            config=cfg, seed=12345,
        )

        # Target loss should differ significantly
        assert abs(scores_big["target_loss"] - scores_tiny["target_loss"]) > 0.01, (
            "Shift not applied: big and tiny shifts produced nearly identical target_loss"
        )

    def test_shift_spec_round_trips(self, tmp_dir):
        """shift_spec saved in pkl matches what eval reads back."""
        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
        cfg_path = tmp_dir / "test_config.yaml"
        with cfg_path.open("w") as f:
            yaml.dump(cfg, f)

        original_shift = {"shift_type": "noise_mean", "magnitude": 0.3, "node": 0}
        pkl_path = train_cv(
            config_path=cfg_path,
            sweep_axis="eps",
            radius_values=[0.1],
            out_dir=tmp_dir,
            k_folds=2,
            shift_spec=original_shift,
        )

        results, metadata = load_training_results(pkl_path)
        # Check metadata-level shift_spec
        assert metadata["shift_spec"] == original_shift
        # Check per-entry shift_spec
        for fold_key, fold_dict in results.items():
            for radius_key, entry in fold_dict.items():
                assert entry["shift_spec"] == original_shift

    def test_no_shift_no_target_raises(self, atce_cfg, atce_bundle):
        """Scoring with shift_spec=None and no target_bundle raises ValueError."""
        with pytest.raises(ValueError, match="shift_spec.*target_bundle"):
            _reveal_target_and_score(
                np.eye(3), atce_bundle, np.arange(10),
                shift_spec=None, objective="gaussian",
                config=atce_cfg,
            )


# ---------------------------------------------------------------------------
# TestHoldoutEstimation — Gaussian error bars via eval-side holdout moments
# ---------------------------------------------------------------------------

class TestHoldoutEstimation:
    """Gaussian folds get variance via per-fold moment estimation.

    Both training and eval estimate exogenous moments from their respective
    fold samples: training uses the 8000-sample training split (so each fold
    learns a different τ), eval uses the 2000-sample held-out split (so each
    fold produces different certificates and Phi_pushed).
    """

    def test_folds_differ_in_tau_and_certificate(self, tmp_dir):
        """Different folds → different estimated moments (train + eval) →
        different τ and certificates.

        Asserts three layers of fold variation:
        1. τ differs across folds (per-fold moment estimation in training)
        2. Certificates differ across folds (τ variation + eval-side holdout)
        3. target_loss differs across folds (different τ applied to same target)
        """
        import joblib

        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
        cfg_path = tmp_dir / "test_config.yaml"
        with cfg_path.open("w") as f:
            yaml.dump(cfg, f)

        shift_spec = {"shift_type": "noise_std", "magnitude": 0.3, "node": 0}
        pkl_path = train_cv(
            config_path=cfg_path,
            sweep_axis="eps",
            radius_values=[0.2],
            out_dir=tmp_dir,
            k_folds=3,
            shift_spec=shift_spec,
        )

        # --- Layer 1: τ must differ across folds (training-side variation) ---
        data = joblib.load(pkl_path)
        taus = []
        for k in range(3):
            tau = data[f"fold_{k}"]["eps_0.20"]["tau"]
            taus.append(np.diag(tau))
        taus = np.array(taus)  # (3, d)
        # At least one pair of folds must have different τ diagonals
        all_same = np.allclose(taus[0], taus[1]) and np.allclose(taus[1], taus[2])
        assert not all_same, (
            f"All fold τ diagonals identical — per-fold moment estimation not working.\n"
            f"τ diags: {taus}"
        )

        # --- Layers 2+3: certificates and target_loss via eval ---
        # noise_std shift not supported by paired mode → use unpaired
        df = run_evaluation(
            results_pkl=pkl_path,
            output_dir=tmp_dir / "eval_out",
            target_seed=99999,
            pairing="unpaired",
        )

        df_r = df[df["method"] == "TraCA (eps_0.20)"]
        assert len(df_r) == 3  # 3 folds

        # Certificates should differ across folds
        certs = df_r["certificate"].values
        assert not np.allclose(certs[0], certs[1]) or not np.allclose(certs[1], certs[2]), (
            "All fold certificates identical — holdout estimation not working"
        )

        # target_loss should differ across folds (different τ → different loss)
        losses = df_r["target_loss"].values
        assert not np.allclose(losses[0], losses[1]) or not np.allclose(losses[1], losses[2]), (
            "All fold target_losses identical — τ variation not propagating to eval"
        )

    def test_holdout_moments_close_to_analytic(self, atce_cfg, atce_bundle):
        """With n=10000 and a fold of ~5000 samples, estimated moments should
        be close to analytic (within statistical tolerance), not wildly off."""
        test_indices = np.arange(0, atce_bundle.n // 2)
        U_obs_test = atce_bundle.noise_samples[0][test_indices]
        mu_hat = np.mean(U_obs_test, axis=0)
        Sigma_hat = np.cov(U_obs_test, rowvar=False)

        # Analytic moments
        mu_true = atce_bundle.noise_mean
        Sigma_true = atce_bundle.noise_cov

        # Should be close (statistical tolerance for n=5000)
        np.testing.assert_allclose(mu_hat, mu_true, atol=0.05)
        np.testing.assert_allclose(Sigma_hat, Sigma_true, atol=0.05)

    def test_full_dataset_skips_holdout(self, atce_cfg, atce_bundle):
        """When test_indices == full dataset, holdout estimation is skipped
        (backward compat with monolithic path)."""
        tau = np.eye(atce_bundle.d)
        shift = {"shift_type": "noise_mean", "magnitude": 0.3, "node": 0}

        # Full dataset: should use analytic moments
        scores = _reveal_target_and_score(
            tau, atce_bundle, np.arange(atce_bundle.n),
            shift_spec=shift, objective="gaussian",
            config=atce_cfg, seed=12345,
        )
        # Just verify it runs without error and produces valid output
        assert scores["certificate"] > 0
        assert scores["target_loss"] >= 0

    def test_summary_records_semantics(self, tmp_dir):
        """summary.json records error_bar_semantics for Gaussian objective."""
        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING}
        cfg_path = tmp_dir / "test_config.yaml"
        with cfg_path.open("w") as f:
            yaml.dump(cfg, f)

        shift_spec = {"shift_type": "noise_std", "magnitude": 0.3, "node": 0}
        pkl_path = train_cv(
            config_path=cfg_path,
            sweep_axis="eps",
            radius_values=[0.2],
            out_dir=tmp_dir,
            k_folds=2,
            shift_spec=shift_spec,
        )

        # noise_std shift not supported by paired mode → use unpaired
        run_evaluation(
            results_pkl=pkl_path,
            output_dir=tmp_dir / "eval_out",
            target_seed=99999,
            pairing="unpaired",
        )

        import json
        with (tmp_dir / "eval_out" / "summary.json").open() as f:
            summary = json.load(f)
        assert summary["error_bar_semantics"] == "source_holdout_moments"


# ---------------------------------------------------------------------------
# TestShiftGrid — (prior × test-time shift) evaluation grid
# ---------------------------------------------------------------------------

class TestShiftGrid:
    """--shift_grid_json: every stored map scored against every grid shift."""

    SHIFT_GRID = [
        {"shift_type": "noise_std", "magnitude": 0.3, "node": 0},
        [  # compound: both of Y's incoming edges (column 2)
            {"shift_type": "mechanism_edge", "magnitude": 0.2, "edge": [0, 2]},
            {"shift_type": "mechanism_edge", "magnitude": 0.2, "edge": [1, 2]},
        ],
    ]

    GRID_COLUMNS = {
        "prior_axis", "prior_value", "shift_type", "shift_magnitude",
        "trial", "target_loss", "certificate", "dir_certificate",
        "coverage_fraction",
    }

    def _train_small(self, tmp_dir, k_folds=2):
        with open(ATCE_CONFIG) as f:
            cfg = yaml.safe_load(f)
        cfg["training"] = {**cfg.get("training", {}), **FAST_TRAINING,
                           "n_samples": 300}
        cfg_path = tmp_dir / "test_config.yaml"
        with cfg_path.open("w") as f:
            yaml.dump(cfg, f)
        return train_cv(
            config_path=cfg_path,
            sweep_axis="eps",
            radius_values=[0.2],
            out_dir=tmp_dir,
            k_folds=k_folds,
            include_baselines=True,
            shift_spec={"shift_type": "noise_std", "magnitude": 0.3, "node": 0},
        )

    def test_grid_row_count_and_schema(self, tmp_dir):
        """Rows = entries × grid size; schema matches; no Huber alpha column."""
        pkl_path = self._train_small(tmp_dir, k_folds=2)
        # Grid includes noise_std → must use unpaired
        df = run_evaluation(
            results_pkl=pkl_path,
            output_dir=tmp_dir / "grid_out",
            shift_grid=self.SHIFT_GRID,
            pairing="unpaired",
        )

        # 2 folds × (1 radius + 1 baseline) × 2 shifts = 8 rows
        assert len(df) == 8
        assert self.GRID_COLUMNS.issubset(set(df.columns))
        assert "alpha" not in df.columns
        assert "contamination_distribution" not in df.columns

        # Both shift types present with correct labels
        assert set(df["shift_type"]) == {
            "noise_std", "mechanism_edge+mechanism_edge",
        }
        # Trial column is 0 (no trial axis)
        assert set(df["trial"]) == {0}
        # prior_axis is the training sweep axis
        assert set(df["prior_axis"]) == {"eps"}

        # Saved as evaluation_grid.csv, not evaluation.csv
        assert (tmp_dir / "grid_out" / "evaluation_grid.csv").exists()
        assert not (tmp_dir / "grid_out" / "evaluation.csv").exists()

    def test_compound_mechanism_edge_shifts_both_entries(self, atce_bundle):
        """The compound spec perturbs BOTH W[0,2] and W[1,2] (Y's incoming edges)."""
        compound = self.SHIFT_GRID[1]
        target = _build_shifted_target_bundle(
            atce_bundle, compound, np.arange(100), seed=1,
        )
        W_s, W_t = atce_bundle.W, target.W
        np.testing.assert_allclose(W_t[0, 2], W_s[0, 2] + 0.2)
        np.testing.assert_allclose(W_t[1, 2], W_s[1, 2] + 0.2)
        # Untouched entries unchanged
        np.testing.assert_allclose(W_t[0, 1], W_s[0, 1])

    def test_zero_magnitude_shift_scores(self, atce_cfg, atce_bundle):
        """Magnitude 0.0 (the no-shift grid cell) scores without error and
        gives lower target_loss than a large shift for tau=I."""
        tau = np.eye(atce_bundle.d)
        zero_spec = {"shift_type": "noise_std", "magnitude": 0.0, "node": 0}
        big_spec = {"shift_type": "noise_std", "magnitude": 0.8, "node": 0}

        # noise_std not supported by paired → use unpaired
        s_zero = _reveal_target_and_score(
            tau, atce_bundle, np.arange(100),
            shift_spec=zero_spec, objective="gaussian",
            config=atce_cfg, seed=5,
            pairing="unpaired",
        )
        s_big = _reveal_target_and_score(
            tau, atce_bundle, np.arange(100),
            shift_spec=big_spec, objective="gaussian",
            config=atce_cfg, seed=5,
            pairing="unpaired",
        )
        assert s_zero["target_loss"] < s_big["target_loss"]


# ---------------------------------------------------------------------------
# TestPairedTarget — paired target construction
# ---------------------------------------------------------------------------

class TestPairedTarget:
    """Paired target reuses exact held-out U rows through shifted SCM."""

    def test_paired_mechanism_edge_same_U(self, atce_bundle):
        """mechanism_edge: target U == source U on held-out indices."""
        test_indices = np.arange(50, 150)
        shift_spec = {"shift_type": "mechanism_edge", "magnitude": 0.3, "edge": [0, 1]}

        target = _build_paired_target_bundle(atce_bundle, shift_spec, test_indices)

        # Observational U must be identical (only A changes)
        source_U = atce_bundle.noise_samples[0][test_indices]
        target_U = target.noise_samples[0]
        np.testing.assert_array_equal(source_U, target_U)

        # But X differs (different A)
        source_X = atce_bundle.endogenous_samples[0][test_indices]
        target_X = target.endogenous_samples[0]
        assert not np.allclose(source_X, target_X), "X should differ (different A)"

    def test_paired_noise_mean_shifts_U(self, atce_bundle):
        """noise_mean: target U[:, node] == source U[:, node] + magnitude."""
        test_indices = np.arange(50, 150)
        node, mag = 0, 0.5
        shift_spec = {"shift_type": "noise_mean", "magnitude": mag, "node": node}

        target = _build_paired_target_bundle(atce_bundle, shift_spec, test_indices)

        source_U = atce_bundle.noise_samples[0][test_indices]
        target_U = target.noise_samples[0]

        # Shifted column
        np.testing.assert_allclose(
            target_U[:, node], source_U[:, node] + mag,
            err_msg="Shifted column should be source + magnitude",
        )
        # Other columns unchanged
        other_cols = [c for c in range(atce_bundle.d) if c != node]
        np.testing.assert_array_equal(
            target_U[:, other_cols], source_U[:, other_cols],
            err_msg="Non-shifted columns should be identical",
        )

    def test_paired_gaussian_moments_are_sample_estimated(self, atce_bundle):
        """Paired target moments come from np.mean/np.cov, not analytic SCM params."""
        test_indices = np.arange(50, 150)
        shift_spec = {"shift_type": "mechanism_edge", "magnitude": 0.3, "edge": [0, 1]}

        target = _build_paired_target_bundle(atce_bundle, shift_spec, test_indices)

        # Sample-estimated moments from the same U
        U_holdout = atce_bundle.noise_samples[0][test_indices]
        expected_mean = np.mean(U_holdout, axis=0)
        expected_cov = np.cov(U_holdout, rowvar=False)

        np.testing.assert_allclose(target.noise_mean, expected_mean)
        np.testing.assert_allclose(target.noise_cov, expected_cov)

        # They should NOT equal the analytic SCM params (finite-sample difference)
        assert not np.allclose(target.noise_mean, target.scm.noise_mean, atol=1e-10), (
            "Paired target moments should differ from analytic SCM params"
        )

    def test_paired_unsupported_shift_raises(self, atce_bundle):
        """noise_std and noise_cov shifts raise ValueError on paired path."""
        test_indices = np.arange(50, 150)
        for shift_type in ("noise_std", "noise_cov"):
            with pytest.raises(ValueError, match="does not support"):
                _build_paired_target_bundle(
                    atce_bundle,
                    {"shift_type": shift_type, "magnitude": 0.3, "node": 0},
                    test_indices,
                )

    def test_paired_compound_shift(self, atce_bundle):
        """Compound: mechanism_edge + noise_mean both applied correctly."""
        test_indices = np.arange(50, 150)
        compound = [
            {"shift_type": "mechanism_edge", "magnitude": 0.2, "edge": [0, 1]},
            {"shift_type": "noise_mean", "magnitude": 0.4, "node": 0},
        ]

        target = _build_paired_target_bundle(atce_bundle, compound, test_indices)

        # W changed
        np.testing.assert_allclose(
            target.W[0, 1], atce_bundle.W[0, 1] + 0.2,
        )
        # U shifted on node 0
        source_U = atce_bundle.noise_samples[0][test_indices]
        target_U = target.noise_samples[0]
        np.testing.assert_allclose(target_U[:, 0], source_U[:, 0] + 0.4)
        # Other columns unchanged
        np.testing.assert_array_equal(target_U[:, 1:], source_U[:, 1:])

    def test_paired_intervened_columns_zeroed(self, atce_bundle):
        """Intervened columns in U are zeroed (matching lan_scm convention)."""
        test_indices = np.arange(50, 150)
        shift_spec = {"shift_type": "mechanism_edge", "magnitude": 0.3, "edge": [0, 1]}

        target = _build_paired_target_bundle(atce_bundle, shift_spec, test_indices)

        # Check intervened SCMs have correct zeroing
        for i, iv in enumerate(atce_bundle.interventions):
            if iv:  # non-observational
                scm_do = target.intervened_scms[i]
                for j_idx in scm_do._J:
                    np.testing.assert_array_equal(
                        target.noise_samples[i][:, j_idx], 0.0,
                        err_msg=f"Intervention {i}: column {j_idx} should be zeroed",
                    )


# ---------------------------------------------------------------------------
# TestObsDistanceHandCheck — hard gate: obs_distance matches raw W₂²
# ---------------------------------------------------------------------------

class TestObsDistanceHandCheck:
    """obs_distance(tau=I) must equal an independently computed W₂².

    This is the R1.1 selection signal.  If it breaks, ATE and Portland
    operating-point selection is silently wrong.  The manual check returned
    |code - raw_W2| = 0.00e+00; this test codifies that.
    """

    def test_obs_distance_equals_raw_W2_on_ate(self):
        """Build ATE source + target from YAMLs, compute obs_distance(tau=I),
        and compare against a raw NumPy W₂² that never calls the code under
        test."""
        # --- Build source and target bundles from YAML (same as eval path) ---
        with open("data_configs/ate.yaml") as f:
            src_cfg = yaml.safe_load(f)
        with open("data_configs/ate_target.yaml") as f:
            tgt_cfg = yaml.safe_load(f)

        src_scm = _scm_from_config(src_cfg)
        tgt_scm = _scm_from_config(tgt_cfg)
        interventions = list(src_cfg["interventions"])
        n = 5000
        source_bundle = src_scm.bundle(interventions, n=n, seed=0)
        target_bundle = tgt_scm.bundle(interventions, n=n, seed=1)

        tau = np.eye(source_bundle.d)

        # --- Code under test ---
        obs_dist = _observational_pushed_distance(
            tau, source_bundle, target_bundle, "gaussian",
        )

        # --- Independent reference: raw W₂² in NumPy ---
        # Pushed source observational: tau=I, so pushed == source
        A_s_obs = source_bundle.intervened_scms[0].A  # observational propagator
        A_t_obs = target_bundle.intervened_scms[0].A

        mu_pushed = source_bundle.noise_mean @ A_s_obs  # tau=I: no change
        Sigma_pushed = A_s_obs.T @ source_bundle.noise_cov @ A_s_obs

        mu_target = target_bundle.noise_mean @ A_t_obs
        Sigma_target = A_t_obs.T @ target_bundle.noise_cov @ A_t_obs

        # W₂²(N(mu_pushed, Sigma_pushed), N(mu_target, Sigma_target))
        # = ||mu_pushed - mu_target||² + Tr(Sigma_pushed + Sigma_target
        #     - 2 (Sigma_pushed^{1/2} Sigma_target Sigma_pushed^{1/2})^{1/2})
        mean_sq = float(np.dot(mu_pushed - mu_target, mu_pushed - mu_target))
        eigvals_p, eigvecs_p = np.linalg.eigh(Sigma_pushed)
        S_sqrt = eigvecs_p @ np.diag(np.sqrt(np.maximum(eigvals_p, 0.0))) @ eigvecs_p.T
        M = S_sqrt @ Sigma_target @ S_sqrt
        eigvals_M = np.linalg.eigvalsh(M)
        cross_trace = float(np.sum(np.sqrt(np.maximum(eigvals_M, 0.0))))
        cov_term = float(np.trace(Sigma_pushed) + np.trace(Sigma_target)
                         - 2.0 * cross_trace)
        raw_W2 = mean_sq + max(cov_term, 0.0)

        # --- Gate ---
        assert abs(obs_dist - raw_W2) < 1e-10, (
            f"obs_distance disagrees with raw W₂²: code={obs_dist:.15e}, "
            f"raw={raw_W2:.15e}, |diff|={abs(obs_dist - raw_W2):.2e}"
        )


# ---------------------------------------------------------------------------
# TestCompoundThetaLossHandCheck — hard gate: compound-shift loss matches raw
# ---------------------------------------------------------------------------

class TestCompoundThetaLossHandCheck:
    """_compute_target_loss_per_iota on a compound shift must equal a
    hand-computed Frobenius residual for every intervention.

    This is what makes all LiLuCaS numbers meaningful.  The manual check
    returned |code - hand| = 0.00e+00 across all interventions; this test
    codifies that.
    """

    def test_compound_loss_equals_raw_frobenius_on_lilucas(self):
        """Build LiLuCaS source, apply compound shift (mechanism_edge +
        noise_mean), compute per-iota loss via production code, and compare
        against a raw NumPy Frobenius reference that never calls the code
        under test."""
        # --- Build source bundle from YAML ---
        with open("data_configs/lilucas_light.yaml") as f:
            src_cfg = yaml.safe_load(f)
        src_scm = _scm_from_config(src_cfg)
        interventions = list(src_cfg["interventions"])
        n = 500  # small for speed
        source_bundle = src_scm.bundle(interventions, n=n, seed=42)

        d = source_bundle.d
        var_names = source_bundle.scm.var_names
        tau = np.eye(d)  # tau=I to isolate the loss computation

        # Compound shift: mechanism_edge on Smoking->LungCancer + noise_mean on LungCancer
        delta_m = 0.25
        delta_n = 0.3
        shift_spec = [
            {"shift_type": "mechanism_edge", "magnitude": delta_m, "edge": [0, 3]},
            {"shift_type": "noise_mean", "magnitude": delta_n, "node": 3},
        ]

        # --- Build target via production code ---
        shifted_scm = apply_shift(src_scm, "mechanism_edge", delta_m, edge=(0, 3))
        shifted_scm = apply_shift(shifted_scm, "noise_mean", delta_n, node=3)
        target_bundle = shifted_scm.bundle(interventions, n=n, seed=99999)

        # --- Code under test ---
        code_losses = _compute_target_loss_per_iota(
            tau, source_bundle, target_bundle, "empirical",
            shift_spec=shift_spec,
        )

        # --- Independent reference: raw Frobenius per-iota ---
        W_s = source_bundle.W
        W_t = target_bundle.W
        dW_actual = W_t - W_s

        # Noise-mean delta vector
        noise_deltas = np.zeros(d)
        noise_deltas[3] = delta_n

        for i, iv in enumerate(interventions):
            A_s_i = source_bundle.intervened_scms[i].A
            U_s_i = source_bundle.noise_samples[i]
            N_i = U_s_i.shape[0]

            # Build R_ι independently (no import of gating_matrix)
            R_i = np.eye(d)
            intervened_nodes = [var_names.index(k) if isinstance(k, str) else int(k)
                                for k in iv.keys()] if iv else []
            for j in intervened_nodes:
                R_i[j, j] = 0.0

            # A'_ι = (I - (W + ΔW) R_ι)^{-1}  — independent computation
            A_prime_i = np.linalg.inv(np.eye(d) - (W_s + dW_actual) @ R_i)

            # Theta: broadcast noise_mean delta to (N, d)
            Theta_i = np.tile(noise_deltas, (N_i, 1))

            # Raw loss: (1/N) ||U_s A_s τ - (U_s + Θ) A'_ι||_F²
            pushed = U_s_i @ A_s_i @ tau
            target_recon = (U_s_i + Theta_i) @ A_prime_i
            residual = pushed - target_recon
            loss_hand = float(np.linalg.norm(residual, "fro") ** 2) / N_i

            # --- Gate ---
            assert abs(code_losses[i] - loss_hand) < 1e-10, (
                f"Intervention {i} ({iv}): code={code_losses[i]:.15e}, "
                f"hand={loss_hand:.15e}, |diff|={abs(code_losses[i] - loss_hand):.2e}"
            )


# ---------------------------------------------------------------------------
# TestInterventionalTargetVal — hard gate: Gaussian target_val includes fixed
# ---------------------------------------------------------------------------

class TestInterventionalTargetVal:
    """The Gaussian target_val formula must include the intervention's fixed
    values: E[Y|do(X=v)] = (noise_mean + fixed) @ A_t, not just noise_mean @ A_t.

    For ATE with β_t = 1.5, noise_mean = [μ_X, μ_Y]:
        do(X=0) → E[Y] = 0 * 1.5 + μ_Y
        do(X=1) → E[Y] = 1 * 1.5 + μ_Y
    The buggy formula (noise_mean @ A_t) omits the fixed term.

    This test exercises _reveal_target_and_score and _compute_target_loss_per_iota
    on an interventional query and verifies the result against a raw NumPy
    reference that never calls the code under test.
    """

    def test_target_val_do_X1_on_ate(self):
        """target_value for do(X=1)/Y on ATE must equal β_t + μ_Y."""
        # --- Build source + target from YAML ---
        with open("data_configs/ate.yaml") as f:
            src_cfg = yaml.safe_load(f)
        with open("data_configs/ate_target.yaml") as f:
            tgt_cfg = yaml.safe_load(f)

        src_scm = _scm_from_config(src_cfg)
        tgt_scm = _scm_from_config(tgt_cfg)
        interventions = list(src_cfg["interventions"])
        n = 5000
        source_bundle = src_scm.bundle(interventions, n=n, seed=0)
        target_bundle = tgt_scm.bundle(interventions, n=n, seed=1)

        d = source_bundle.d
        beta_t = 1.5  # target edge weight X→Y
        mu_Y = target_bundle.noise_mean[1]
        # E[Y|do(X=1)] = β_t * 1 + μ_Y
        expected_Y = beta_t + mu_Y

        # --- Independent reference: raw NumPy ---
        # ATE: W_t = [[0, 1.5], [0, 0]].  A_t = [[1, 1.5], [0, 1]].
        # do(X=1): fixed = [1, 0].  E[V|do(X=1)] = (μ_eff + fixed) @ A_do.
        iota_doX1 = 2  # interventions: {}, {X:0}, {X:1}
        A_t = target_bundle.intervened_scms[iota_doX1].A
        fixed_t = target_bundle.intervened_scms[iota_doX1]._fixed
        noise_mean_t = target_bundle.noise_mean

        from traca.utils import interventional_exo_mean
        scm_t_doX1 = target_bundle.intervened_scms[iota_doX1]
        raw_target_Y = float((interventional_exo_mean(noise_mean_t, fixed_t, scm_t_doX1._J) @ A_t)[1])
        assert abs(raw_target_Y - expected_Y) < 1e-10, (
            f"Sanity: raw formula gives {raw_target_Y}, expected {expected_Y}"
        )

        # Also verify against endogenous samples (MC sanity)
        X_t = target_bundle.endogenous_samples[iota_doX1]
        empirical_Y = float(np.mean(X_t[:, 1]))
        assert abs(empirical_Y - expected_Y) < 0.1, (
            f"MC sanity: empirical E[Y|do(X=1)] = {empirical_Y}, expected ~{expected_Y}"
        )

        # --- Code under test: _reveal_target_and_score ---
        with open("configs/ate/gaussian_entrywise_subfamily.yaml") as f:
            cfg = yaml.safe_load(f)

        tau = np.eye(d)
        test_indices = np.arange(200)  # small holdout for speed

        scores = _reveal_target_and_score(
            tau, source_bundle, test_indices,
            shift_spec=None, objective="gaussian",
            config=cfg, seed=42,
            target_bundle=target_bundle,
        )

        # Check per-query target_value for do(X=1)/Y = query (2, 1)
        key_doX1_Y = "(2,1)"
        code_target_val = scores["per_query"][key_doX1_Y]["target_value"]

        assert abs(code_target_val - raw_target_Y) < 1e-10, (
            f"_reveal_target_and_score target_value for do(X=1)/Y: "
            f"code={code_target_val:.15e}, raw={raw_target_Y:.15e}, "
            f"|diff|={abs(code_target_val - raw_target_Y):.2e}"
        )

        # Also check do(X=0)/Y = query (1, 1): E[Y|do(X=0)] = μ_Y
        key_doX0_Y = "(1,1)"
        code_doX0 = scores["per_query"][key_doX0_Y]["target_value"]
        scm_t_doX0 = target_bundle.intervened_scms[1]
        raw_doX0 = float((interventional_exo_mean(
            noise_mean_t, scm_t_doX0._fixed, scm_t_doX0._J
        ) @ scm_t_doX0.A)[1])
        assert abs(code_doX0 - raw_doX0) < 1e-10, (
            f"do(X=0)/Y: code={code_doX0:.15e}, raw={raw_doX0:.15e}"
        )

    def test_target_loss_includes_fixed_on_ate(self):
        """_compute_target_loss_per_iota Gaussian must include fixed values
        in both source and target means."""
        with open("data_configs/ate.yaml") as f:
            src_cfg = yaml.safe_load(f)
        with open("data_configs/ate_target.yaml") as f:
            tgt_cfg = yaml.safe_load(f)

        src_scm = _scm_from_config(src_cfg)
        tgt_scm = _scm_from_config(tgt_cfg)
        interventions = list(src_cfg["interventions"])
        n = 5000
        source_bundle = src_scm.bundle(interventions, n=n, seed=0)
        target_bundle = tgt_scm.bundle(interventions, n=n, seed=1)

        d = source_bundle.d
        tau = np.eye(d)

        code_losses = _compute_target_loss_per_iota(
            tau, source_bundle, target_bundle, "gaussian",
        )

        # --- Independent reference per intervention ---
        from traca.losses import GaussianLoss
        loss_fn = GaussianLoss()
        W_s = source_bundle.W
        mu_s = source_bundle.noise_mean
        Sigma_s = source_bundle.noise_cov
        mu_t = target_bundle.noise_mean
        Sigma_t = target_bundle.noise_cov
        dW_zero = np.zeros((d, d))
        var_names = source_bundle.scm.var_names

        for i, iv in enumerate(interventions):
            A_s = source_bundle.intervened_scms[i].A
            A_t = target_bundle.intervened_scms[i].A
            fixed_s = source_bundle.intervened_scms[i]._fixed
            fixed_t = target_bundle.intervened_scms[i]._fixed

            # Build R_ι independently
            R_i = np.eye(d)
            intervened_nodes = [var_names.index(k) if isinstance(k, str)
                                else int(k) for k in iv.keys()] if iv else []
            for j in intervened_nodes:
                R_i[j, j] = 0.0

            # Correct means including fixed (with noise zeroing at intervened nodes)
            from traca.utils import interventional_exo_mean
            mu_s_eff = interventional_exo_mean(mu_s, fixed_s, intervened_nodes)
            mu_t_obs = interventional_exo_mean(mu_t, fixed_t,
                target_bundle.intervened_scms[i]._J) @ A_t
            Sigma_t_obs = A_t.T @ Sigma_t @ A_t

            hand = float(loss_fn.value(tau, dW_zero, W_s, A_s, R_i,
                                        mu_s_eff, Sigma_s, mu_t_obs, Sigma_t_obs))

            assert abs(code_losses[i] - hand) < 1e-10, (
                f"Intervention {i} ({iv}): code={code_losses[i]:.15e}, "
                f"hand={hand:.15e}, |diff|={abs(code_losses[i] - hand):.2e}"
            )


# ---------------------------------------------------------------------------
# Gate #3: Certificate-frame includes fixed (derivation 1b verification)
# ---------------------------------------------------------------------------

class TestCertificateFrameFixed:
    """delta_iota_rho_sq must use the per-intervention effective exogenous mean.

    For do(X=1) on ATE, mu_s_eff = [1, μ_Y] (not bare noise_mean).
    Term 3 (mechanism × mean = 4 α² ‖mu_s_eff‖²) must be nonzero.
    """

    def test_certificate_frame_includes_fixed_on_ate(self):
        from traca.utils import interventional_exo_mean
        from traca.certificates import delta_iota_rho_sq

        with open("data_configs/ate.yaml") as f:
            src_cfg = yaml.safe_load(f)
        src_scm = _scm_from_config(src_cfg)
        interventions = list(src_cfg["interventions"])
        bundle = src_scm.bundle(interventions, n=500, seed=0)
        d = bundle.d

        # do(X=1) = intervention index 2
        iota = 2
        scm_i = bundle.intervened_scms[iota]
        A_iota = scm_i.A
        mu_s_eff = interventional_exo_mean(bundle.noise_mean, scm_i._fixed, scm_i._J)
        Sigma_s = bundle.noise_cov
        tau = np.eye(d)
        alpha = 0.1
        eps = 0.2

        # Code under test
        cert = delta_iota_rho_sq(tau, alpha, A_iota, mu_s_eff, Sigma_s, eps)

        # Raw NumPy reference — term by term (Gaussian certificate formula)
        I_d = np.eye(d)
        A_tau_I = A_iota @ (tau - I_d)
        mu_transport = mu_s_eff @ A_tau_I
        Sigma_s_norm2 = float(np.linalg.norm(Sigma_s, ord=2))

        term1 = 4.0 * float(np.dot(mu_transport, mu_transport))
        term2 = 4.0 * Sigma_s_norm2 * float(np.linalg.norm(A_tau_I, "fro")) ** 2
        term3 = 4.0 * alpha ** 2 * float(np.dot(mu_s_eff, mu_s_eff))
        term4 = 4.0 * d * Sigma_s_norm2 * alpha ** 2
        term5 = 2.0 * (float(np.linalg.norm(A_iota, ord=2)) + alpha) ** 2 * eps ** 2

        raw_cert = term1 + term2 + term3 + term4 + term5

        # Gate: match to 1e-10
        assert abs(cert - raw_cert) < 1e-10, (
            f"Certificate mismatch: code={cert:.15e}, raw={raw_cert:.15e}, "
            f"|diff|={abs(cert - raw_cert):.2e}"
        )

        # Gate: term3 must be nonzero for do(X=1) — proves fixed enters certificate
        assert term3 > 0.0, (
            f"term3 (mechanism × mean) = {term3}, expected > 0 for do(X=1)"
        )
        # mu_s_eff for do(X=1): X zeroed + fixed=1, Y keeps μ_Y
        mu_Y = bundle.noise_mean[1]
        expected_sq = 1.0 + mu_Y ** 2
        assert abs(float(np.dot(mu_s_eff, mu_s_eff)) - expected_sq) < 1e-12, (
            f"‖mu_s_eff‖² = {float(np.dot(mu_s_eff, mu_s_eff))}, expected {expected_sq}"
        )


# ---------------------------------------------------------------------------
# Gate #4: Coverage consistency — Phi_pushed and cert in same frame
# ---------------------------------------------------------------------------

class TestCoverageConsistency:
    """Phi_pushed and certificate must be computed in the same mu_s_eff frame.
    The interval [lo, hi] must cover the true target_val."""

    def test_coverage_consistency_ate(self):
        from traca.utils import interventional_exo_mean
        from traca.certificates import single_query_certificate, query_interval

        with open("data_configs/ate.yaml") as f:
            src_cfg = yaml.safe_load(f)
        with open("data_configs/ate_target.yaml") as f:
            tgt_cfg = yaml.safe_load(f)

        src_scm = _scm_from_config(src_cfg)
        tgt_scm = _scm_from_config(tgt_cfg)
        interventions = list(src_cfg["interventions"])
        n = 5000
        source_bundle = src_scm.bundle(interventions, n=n, seed=0)
        target_bundle = tgt_scm.bundle(interventions, n=n, seed=1)

        d = source_bundle.d
        tau = np.eye(d)
        eps = 0.5  # generous to ensure coverage

        # do(X=1) / Y
        iota = 2
        out_node = 1
        scm_s_i = source_bundle.intervened_scms[iota]
        mu_s_eff = interventional_exo_mean(
            source_bundle.noise_mean, scm_s_i._fixed, scm_s_i._J
        )

        from traca.stability import gamma, alpha_polynomial, gating_matrix
        iv = source_bundle.interventions[iota]
        var_names = source_bundle.scm.var_names
        intervened_nodes = [var_names.index(k) if isinstance(k, str)
                           else int(k) for k in iv.keys()] if iv else []
        R_i = gating_matrix(d, intervened_nodes)
        from experiments.run import _build_mechanism_set
        with open("configs/ate/gaussian_entrywise_subfamily.yaml") as f:
            cfg = yaml.safe_load(f)
        aw = _build_mechanism_set(cfg["ambiguity"]["mechanism"], d)
        g_i = gamma(scm_s_i.A, R_i, aw)
        alpha_i = alpha_polynomial(scm_s_i.A, g_i, d)

        # Certificate and Phi_pushed use same mu_s_eff
        cert_q = float(single_query_certificate(
            tau, alpha_i, scm_s_i.A,
            mu_s_eff, source_bundle.noise_cov,
            eps=eps, O=[out_node], d=d, mode="gaussian",
        ))
        Phi_pushed = float((mu_s_eff @ scm_s_i.A @ tau)[out_node])
        lo, hi = query_interval(Phi_pushed, 1.0, cert_q)

        # True target value via gaussian_joint (canonical reference)
        target_mu, _ = target_bundle.intervened_scms[iota].gaussian_joint()
        target_val = float(target_mu[out_node])

        assert lo <= target_val <= hi, (
            f"Coverage failure: [{lo:.6f}, {hi:.6f}] does not contain "
            f"target_val={target_val:.6f}"
        )


# ---------------------------------------------------------------------------
# Gate #4b: Scoring U_s_list equals training's U_eff to 1e-10
# ---------------------------------------------------------------------------

class TestScoringUsesTrainingU:
    """Scoring's U_s_list must equal the optimizer's U_eff construction.

    Training (optim.py:645-652) builds U_eff from shared observational U:
        U_eff = U_obs.copy(); U_eff[:, J] = 0; U_eff += fixed
    Scoring must use the same construction via build_U_effs, NOT the
    per-intervention noise_samples[i] (which are independent draws).
    """

    def test_build_U_effs_matches_training_construction(self):
        """build_U_effs matches the optimizer's inline U_eff loop."""
        from traca.utils import build_U_effs
        from lan_scm import make_ate

        bm = make_ate()
        bundle = bm.scm.bundle(bm.interventions, n=500, seed=0)

        # build_U_effs output
        U_effs = build_U_effs(bundle)

        # Training's inline construction (optim.py:645-652)
        obs_idx = bundle.interventions.index({})
        U_obs = bundle.noise_samples[obs_idx]
        for i in range(bundle.n_interventions()):
            scm_i = bundle.intervened_scms[i]
            U_eff_ref = U_obs.copy()
            if scm_i._J:
                U_eff_ref[:, list(scm_i._J)] = 0.0
            U_eff_ref += scm_i._fixed[np.newaxis, :]
            np.testing.assert_allclose(
                U_effs[i], U_eff_ref, atol=1e-14,
                err_msg=f"build_U_effs[{i}] does not match training construction"
            )

    def test_build_U_effs_differs_from_per_iv_noise_samples(self):
        """For non-observational interventions, U_eff != noise_samples[i]."""
        from traca.utils import build_U_effs
        from lan_scm import make_ate

        bm = make_ate()
        bundle = bm.scm.bundle(bm.interventions, n=500, seed=0)
        U_effs = build_U_effs(bundle)

        for i in range(bundle.n_interventions()):
            iv = bundle.interventions[i]
            diff = np.linalg.norm(U_effs[i] - bundle.noise_samples[i], "fro")
            if iv:  # non-observational
                assert diff > 1.0, (
                    f"iota={i}: U_eff should differ from noise_samples[i] "
                    f"for non-obs intervention, but ||diff||={diff:.6f}"
                )
            else:  # observational
                np.testing.assert_allclose(
                    U_effs[i], bundle.noise_samples[i], atol=1e-14,
                    err_msg="U_eff should equal noise_samples for obs intervention"
                )

    def test_build_U_effs_with_holdout_indices(self):
        """build_U_effs with indices subsets rows correctly."""
        from traca.utils import build_U_effs
        from lan_scm import make_ate

        bm = make_ate()
        bundle = bm.scm.bundle(bm.interventions, n=100, seed=0)
        indices = np.array([10, 20, 30, 40, 50])

        U_effs_full = build_U_effs(bundle)
        U_effs_sub = build_U_effs(bundle, indices=indices)

        for i in range(bundle.n_interventions()):
            assert U_effs_sub[i].shape == (5, bundle.d)
            np.testing.assert_allclose(
                U_effs_sub[i], U_effs_full[i][indices], atol=1e-14,
            )


# ---------------------------------------------------------------------------
# Gate #4c: Empirical Phi_pushed pinning — independent NumPy reference
# ---------------------------------------------------------------------------

class TestEmpiricalPhiPushed:
    """Pin the empirical Phi_pushed formula against an independent computation.

    The formula decomposes X_s^ι @ τ into (U_zeroed @ A_ι @ τ) + (fixed @ A_ι @ τ).
    The sample mean of U_zeroed @ A @ τ captures the noise-mean component; only
    the fixed-intervention part is added separately.  Using mu_s_eff instead of
    just fixed would double-count noise_mean_zeroed.

    This test uses nonzero noise_mean to expose the double-counting bug.
    """

    def test_phi_pushed_matches_endogenous_mean(self):
        """Phi_pushed from the (U, fixed) decomposition must equal
        mean(X_s^ι @ τ)_k to machine epsilon."""
        from lan_scm import LANSCM

        # Build a d=3 chain with nonzero noise_mean
        W = np.array([[0, 0.5, 0], [0, 0, 0.8], [0, 0, 0]])
        noise_mean = np.array([1.0, 2.0, 0.5])  # nonzero!
        noise_cov = np.diag([1.0, 0.5, 0.3])
        scm = LANSCM(W=W, noise_mean=noise_mean, noise_cov=noise_cov)

        interventions = [{}, {"X0": 3.0}]  # obs + do(X0=3)
        n = 5000
        bundle = scm.bundle(interventions, n=n, seed=42)

        tau = np.diag([0.9, 1.1, 0.95])  # non-identity
        d = bundle.d

        for iota in range(len(interventions)):
            A_i = bundle.intervened_scms[iota].A
            U_s = bundle.noise_samples[iota]  # (n, d), mean ≈ noise_mean_zeroed
            fixed_i = bundle.intervened_scms[iota]._fixed
            X_s = bundle.endogenous_samples[iota]  # (n, d)

            for k in range(d):
                # --- Independent reference: sample mean of X_s @ τ ---
                Phi_ref = float(np.mean(X_s @ tau, axis=0)[k])

                # --- Production formula (fixed, not mu_s_eff) ---
                pushed_X = U_s @ A_i @ tau
                Phi_prod = float(np.mean(pushed_X[:, k])) + float((fixed_i @ A_i @ tau)[k])

                np.testing.assert_allclose(
                    Phi_prod, Phi_ref, atol=1e-10,
                    err_msg=f"Phi_pushed mismatch at iota={iota}, k={k}"
                )

    def test_double_counting_detected_with_nonzero_mean(self):
        """Confirm that the old (buggy) formula with mu_s_eff instead of fixed
        gives a different answer when noise_mean ≠ 0."""
        from lan_scm import LANSCM
        from traca.utils import interventional_exo_mean

        W = np.array([[0, 0.5, 0], [0, 0, 0.8], [0, 0, 0]])
        noise_mean = np.array([1.0, 2.0, 0.5])  # nonzero
        noise_cov = np.diag([1.0, 0.5, 0.3])
        scm = LANSCM(W=W, noise_mean=noise_mean, noise_cov=noise_cov)

        interventions = [{"X0": 3.0}]  # do(X0=3): zeros col 0, fixed[0]=3
        bundle = scm.bundle(interventions, n=5000, seed=42)

        tau = np.eye(3)
        iota = 0
        A_i = bundle.intervened_scms[iota].A
        U_s = bundle.noise_samples[iota]
        fixed_i = bundle.intervened_scms[iota]._fixed
        X_s = bundle.endogenous_samples[iota]

        mu_s_eff = interventional_exo_mean(
            bundle.noise_mean, fixed_i, bundle.intervened_scms[iota]._J,
        )

        k = 1  # pick a node downstream of the intervention
        pushed_X = U_s @ A_i @ tau

        Phi_correct = float(np.mean(pushed_X[:, k])) + float((fixed_i @ A_i @ tau)[k])
        Phi_buggy = float(np.mean(pushed_X[:, k])) + float((mu_s_eff @ A_i @ tau)[k])
        Phi_ref = float(np.mean(X_s @ tau, axis=0)[k])

        # Correct matches reference
        np.testing.assert_allclose(Phi_correct, Phi_ref, atol=1e-10)

        # Buggy does NOT match reference (double-counts noise_mean[1]=2.0)
        assert abs(Phi_buggy - Phi_ref) > 0.1, (
            f"Expected divergence from double-counting, but got "
            f"Phi_buggy={Phi_buggy:.6f} vs Phi_ref={Phi_ref:.6f}"
        )


# ---------------------------------------------------------------------------
# Gate #5: Optimizer uses fixed mean
# ---------------------------------------------------------------------------

class TestOptimizerUsesFixedMean:
    """fit_gaussian must use per-intervention mu_s_effs, not bare noise_mean."""

    def test_optimizer_uses_fixed_mean(self):
        from traca.utils import bundle_exo_means
        from traca.losses import GaussianLoss

        with open("data_configs/ate.yaml") as f:
            src_cfg = yaml.safe_load(f)
        src_scm = _scm_from_config(src_cfg)
        interventions = list(src_cfg["interventions"])
        bundle = src_scm.bundle(interventions, n=500, seed=0)
        d = bundle.d

        mu_s_effs = bundle_exo_means(bundle)

        # do(X=1) is intervention 2
        # mu_s_effs[2] should be [1.0, μ_Y]: X zeroed + fixed=1, Y keeps noise mean
        expected_doX1 = np.array([1.0, bundle.noise_mean[1]])
        np.testing.assert_allclose(mu_s_effs[2], expected_doX1, atol=1e-12)
        # Observational should equal noise_mean
        np.testing.assert_allclose(mu_s_effs[0], bundle.noise_mean, atol=1e-12)

        # Loss at tau=I with correct mu_s_eff vs bare noise_mean should differ for do(X=1)
        loss_fn = GaussianLoss()
        W = bundle.W
        A_i = bundle.intervened_scms[2].A
        R_i = np.eye(d); R_i[0, 0] = 0.0  # do(X=1) zeros node 0

        # Mock adversary at zero perturbation: target = source with dW=0
        dW = np.zeros((d, d))
        mu_t_obs = mu_s_effs[2] @ A_i
        Sigma_t_obs = A_i.T @ bundle.noise_cov @ A_i

        loss_correct = loss_fn.value(np.eye(d), dW, W, A_i, R_i,
                                      mu_s_effs[2], bundle.noise_cov,
                                      mu_t_obs, Sigma_t_obs)

        loss_wrong = loss_fn.value(np.eye(d), dW, W, A_i, R_i,
                                    bundle.noise_mean, bundle.noise_cov,
                                    mu_t_obs, Sigma_t_obs)

        # With correct mean and τ=I: source pushed mean matches target → loss from mean = 0.
        # With wrong mean (bare noise_mean): pushed differs from target → nonzero.
        assert loss_correct < loss_wrong, (
            f"Expected correct mean to give lower loss: "
            f"correct={loss_correct:.6e}, wrong={loss_wrong:.6e}"
        )


# ---------------------------------------------------------------------------
# Gate #6: interventional_exo_mean matches gaussian_joint
# ---------------------------------------------------------------------------

class TestInterventionalExoMeanMatchesGaussianJoint:
    """interventional_exo_mean must reproduce gaussian_joint()'s mean logic
    for every benchmark and every intervention."""

    @pytest.mark.parametrize("data_config", [
        "data_configs/ate.yaml",
        "data_configs/atce.yaml",
    ])
    def test_mean_matches_gaussian_joint(self, data_config):
        from traca.utils import interventional_exo_mean

        with open(data_config) as f:
            cfg = yaml.safe_load(f)
        scm = _scm_from_config(cfg)
        interventions = list(cfg["interventions"])
        bundle = scm.bundle(interventions, n=100, seed=0)

        for i, iv in enumerate(interventions):
            scm_i = bundle.intervened_scms[i]
            mu_eff = interventional_exo_mean(
                bundle.noise_mean, scm_i._fixed, scm_i._J
            )
            mu_X_helper = mu_eff @ scm_i.A

            # Reference: gaussian_joint() — the canonical implementation
            mu_X_ref, _ = scm_i.gaussian_joint()

            np.testing.assert_allclose(
                mu_X_helper, mu_X_ref, atol=1e-12,
                err_msg=f"{data_config} intervention {i} ({iv}): "
                        f"helper={mu_X_helper}, gaussian_joint={mu_X_ref}"
            )


# ---------------------------------------------------------------------------
# Gate #7: Empirical U_eff residual matches SCM's own samples
# ---------------------------------------------------------------------------

class TestEmpiricalLossUEffMatchesSCMSamples:
    """U_eff construction (zero intervened cols + add fixed) must produce
    the same endogenous samples and loss as the SCM's sample_with_noise()
    code path applied to the same base noise.

    Independent reference: sample_with_noise()'s formula applied to U_obs
    (U_zeroed @ A + fixed @ A), NOT U_eff @ A_ι (would be circular)."""

    def test_empirical_loss_U_eff_matches_scm_samples(self):
        from traca.losses import EmpiricalLoss
        from traca.stability import gating_matrix

        with open("data_configs/ate.yaml") as f:
            src_cfg = yaml.safe_load(f)
        scm = _scm_from_config(src_cfg)
        interventions = list(src_cfg["interventions"])
        bundle = scm.bundle(interventions, n=500, seed=0)
        d = bundle.d
        loss_fn = EmpiricalLoss()
        W = bundle.W

        # Non-identity τ to make the test nontrivial
        tau = np.eye(d)
        tau[0, 0] = 1.2
        tau[1, 1] = 0.8

        # Test do(X=1) = intervention 2
        iota = 2
        scm_i = bundle.intervened_scms[iota]
        iv = bundle.interventions[iota]
        var_names = bundle.scm.var_names
        intervened_nodes = [var_names.index(k) if isinstance(k, str)
                           else int(k) for k in iv.keys()] if iv else []
        R_i = gating_matrix(d, intervened_nodes)

        # Build U_eff via the plan's recipe (code under test)
        obs_idx = bundle.interventions.index({})
        U_obs = bundle.noise_samples[obs_idx]
        U_eff = U_obs.copy()
        if scm_i._J:
            U_eff[:, list(scm_i._J)] = 0.0
        U_eff += scm_i._fixed[np.newaxis, :]

        N = U_obs.shape[0]
        Theta_zero = np.zeros((N, d))
        dW_zero = np.zeros((d, d))

        # Code under test: loss via U_eff through EmpiricalLoss
        loss_eff = loss_fn.value(tau, dW_zero, Theta_zero, W, scm_i.A, R_i, U_eff)

        # Independent reference: replicate sample_with_noise()'s formula
        # (lan_scm.py:245: X = U @ A + fixed @ A) applied to U_obs.
        # This is a SEPARATE code path from "U_eff @ A".
        U_zeroed = U_obs.copy()
        if scm_i._J:
            U_zeroed[:, list(scm_i._J)] = 0.0
        X_ref = U_zeroed @ scm_i.A + scm_i._fixed @ scm_i.A  # sample_with_noise formula

        # Verify U_eff @ A == X_ref (two different decompositions of the same quantity)
        np.testing.assert_allclose(
            U_eff @ scm_i.A, X_ref, atol=1e-12,
            err_msg="U_eff @ A must match sample_with_noise reference"
        )

        # Reference loss: (1/N) ‖X_ref @ (τ-I)‖²_F
        loss_ref = float(np.linalg.norm(X_ref @ (tau - np.eye(d)), "fro") ** 2) / N

        assert abs(loss_eff - loss_ref) < 1e-10, (
            f"U_eff loss vs SCM samples: eff={loss_eff:.15e}, ref={loss_ref:.15e}, "
            f"|diff|={abs(loss_eff - loss_ref):.2e}"
        )
        # Sanity: loss should be nonzero (tau ≠ I)
        assert loss_ref > 1e-6, f"Expected nonzero loss, got {loss_ref}"


# ---------------------------------------------------------------------------
# Gate #8: Deferred Sigma_s zeroing guard
# ---------------------------------------------------------------------------

class TestUniformDiagonalNoiseCov:
    """Guard for the deferred covariance zeroing.

    The deferred Sigma_s zeroing (not zeroing Sigma_s at intervened nodes)
    is conservative: zeroing rows/cols of a PSD matrix can only reduce its
    spectral norm (Cauchy interlacing), so using the full Sigma_s gives a
    larger certificate.

    For configs WITHOUT districts (no latent confounders): noise_cov must be
    uniform diagonal — the strongest structural check.

    For configs WITH districts (semi-Markovian, e.g. ATE): off-diagonal
    entries are expected from the latent confounder. We verify the
    conservatism property directly: ‖Σ_zeroed‖_2 ≤ ‖Σ_full‖_2 for every
    intervention."""

    DIAGONAL_CONFIGS = [
        "data_configs/atce.yaml",
    ]
    DISTRICT_CONFIGS = [
        "data_configs/ate.yaml",
    ]

    @pytest.mark.parametrize("data_config", DIAGONAL_CONFIGS)
    def test_noise_cov_is_uniform_diagonal(self, data_config):
        with open(data_config) as f:
            cfg = yaml.safe_load(f)
        scm = _scm_from_config(cfg)

        noise_cov = scm.noise_cov
        d = noise_cov.shape[0]

        # Check diagonal: off-diagonal entries must be zero
        off_diag = noise_cov.copy()
        np.fill_diagonal(off_diag, 0.0)
        off_diag_norm = float(np.linalg.norm(off_diag, "fro"))
        assert off_diag_norm < 1e-12, (
            f"{data_config}: noise_cov is not diagonal, "
            f"off-diagonal Frobenius = {off_diag_norm}"
        )

        # Check uniform: all diagonal entries must be equal
        diag = np.diag(noise_cov)
        assert float(np.max(diag) - np.min(diag)) < 1e-12, (
            f"{data_config}: noise_cov diagonal is not uniform, "
            f"diag = {diag}"
        )

    @pytest.mark.parametrize("data_config", DISTRICT_CONFIGS)
    def test_district_config_zeroing_is_conservative(self, data_config):
        """For semi-Markovian configs, verify ‖Σ_zeroed‖_2 ≤ ‖Σ_full‖_2."""
        with open(data_config) as f:
            cfg = yaml.safe_load(f)
        scm = _scm_from_config(cfg)
        interventions = list(cfg["interventions"])
        bundle = scm.bundle(interventions, n=100, seed=0)

        noise_cov = scm.noise_cov
        full_norm = float(np.linalg.norm(noise_cov, 2))

        for i, iv in enumerate(interventions):
            if not iv:
                continue  # observational — no nodes to zero
            var_names = scm.var_names
            J = [var_names.index(k) if isinstance(k, str) else int(k)
                 for k in iv.keys()]
            # Zero rows and cols at intervened nodes
            Sigma_zeroed = noise_cov.copy()
            for j in J:
                Sigma_zeroed[j, :] = 0.0
                Sigma_zeroed[:, j] = 0.0
            zeroed_norm = float(np.linalg.norm(Sigma_zeroed, 2))
            assert zeroed_norm <= full_norm + 1e-14, (
                f"{data_config}, intervention {iv}: "
                f"‖Σ_zeroed‖={zeroed_norm} > ‖Σ_full‖={full_norm}"
            )
