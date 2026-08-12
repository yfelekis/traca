"""
Portland-specific evaluation: triptych (source-only, target-only, transported)
with real-CSV ground truth and coverage checks.

Usage
-----
    python -m experiments.portland_eval results/portland_backdoor_gaussian_qrestricted

Reads
-----
    <exp_dir>/result.pkl, bundle.pkl, config.yaml, eval.json
    data/portland/portland_backdoor.csv          — real observations
    data/portland/portland_metadata.json         — standardisation params

Writes
------
    <exp_dir>/portland_triptych.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lan_scm  # noqa: E402

from traca.stability import gamma, alpha_polynomial, gating_matrix
from traca.certificates import single_query_certificate, query_interval
from experiments.run import _build_mechanism_set


# ---------------------------------------------------------------------------
# Ground-truth: back-door estimate from real CSV data
# ---------------------------------------------------------------------------

def _real_csv_ground_truth(
    data_dir: Path,
    tcc_values: list[float],
) -> dict:
    """Compute E[Y|do(X=x)] for Fanno Creek using real observations.

    Three estimates per TCC level:
      source_only  — back-door formula with source Z distribution
      target_only  — back-door formula with Fanno Creek Z distribution (ground truth)
      target_mean  — raw mean of Fanno Creek Y (unstandardized, for reference)

    Both source_only and target_only use the regression E[Y|X,Z,S] fitted on SOURCE
    data, then marginalize over the respective Z distribution (back-door adjustment).
    This mirrors the paper's model (ii) vs model (i).
    """
    meta = json.load(open(data_dir / "portland_metadata.json"))
    sp = meta["standardisation"]

    df = pd.read_csv(data_dir / "portland_backdoor.csv")
    src = df[df["watershed"] != "Fanno Creek"].copy()
    tgt = df[df["watershed"] == "Fanno Creek"].copy()

    # Standardize using source stats
    Z_raw_cols = sp["Z_raw_cols"]
    mu_Z_raw = np.array(sp["Z_raw_mu"])
    std_Z_raw = np.array(sp["Z_raw_std"])
    pc1 = np.array(sp["pca_pc1_loadings"])
    mu_Z_src = sp["Z_pc_mu_src"]
    std_Z_src = sp["Z_pc_std_src"]
    mu_S_src = sp["S_mu_src"]
    std_S_src = sp["S_std_src"]
    mu_X_src = sp["X_mu_src"]
    std_X_src = sp["X_std_src"]
    mu_Y_src = sp["Y_mu_src"]
    std_Y_src = sp["Y_std_src"]

    def std_Z(raw_df):
        z = (raw_df[Z_raw_cols].values - mu_Z_raw) / std_Z_raw
        return (z @ pc1 - mu_Z_src) / std_Z_src

    Z_src = std_Z(src)
    Z_tgt = std_Z(tgt)
    S_src = (src["season_num"].astype(float).values - mu_S_src) / std_S_src
    S_tgt = (tgt["season_num"].astype(float).values - mu_S_src) / std_S_src
    X_src = (src["X"].values - mu_X_src) / std_X_src
    Y_src = (src["Y"].values - mu_Y_src) / std_Y_src

    # Fit Y ~ Z + S + X on source (no intercept — zero-mean in source)
    A_Y = np.column_stack([Z_src, S_src, X_src])
    coef_Y, _, _, _ = np.linalg.lstsq(A_Y, Y_src, rcond=None)
    w_ZY, w_SY, w_XY = float(coef_Y[0]), float(coef_Y[1]), float(coef_Y[2])

    results = {"tcc_values": tcc_values, "estimates": []}

    for tcc in tcc_values:
        x_std = (tcc - mu_X_src) / std_X_src

        # Source-only: E[Y|do(X=x)] = (1/N_src) Σ_i [w_ZY*Z_i + w_SY*S_i + w_XY*x]
        ey_src_std = float(np.mean(w_ZY * Z_src + w_SY * S_src + w_XY * x_std))
        ey_src_mgl = ey_src_std * std_Y_src + mu_Y_src

        # Target-only: same formula but marginalize over Fanno Creek (Z, S)
        ey_tgt_std = float(np.mean(w_ZY * Z_tgt + w_SY * S_tgt + w_XY * x_std))
        ey_tgt_mgl = ey_tgt_std * std_Y_src + mu_Y_src

        # Raw target mean (just for reference, not a causal estimate)
        raw_tgt_mean = float(tgt["Y"].mean())

        results["estimates"].append({
            "tcc_pct": tcc,
            "x_standardized": x_std,
            "source_only_mgl": ey_src_mgl,
            "target_only_mgl": ey_tgt_mgl,
            "source_only_std": ey_src_std,
            "target_only_std": ey_tgt_std,
            "raw_target_mean_mgl": raw_tgt_mean,
        })

    results["regression_coefficients"] = {
        "w_ZY": w_ZY, "w_SY": w_SY, "w_XY": w_XY,
    }
    return results


# ---------------------------------------------------------------------------
# Triptych assembly
# ---------------------------------------------------------------------------

def _iv_to_nodes(iv: dict, var_names: list[str]) -> list[int]:
    if not iv:
        return []
    return [var_names.index(k) if isinstance(k, str) else int(k) for k in iv.keys()]


def portland_triptych(
    exp_dir: Path,
    data_dir: Path = Path("data/portland"),
) -> dict:
    """Full Portland triptych: source-only, target-only (real CSV), transported + cert.

    Returns dict with per-TCC results and coverage summary.
    """
    # Load experiment artifacts
    result = joblib.load(exp_dir / "result.pkl")
    bundle = joblib.load(exp_dir / "bundle.pkl")
    with (exp_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)

    tau = result.tau
    d = bundle.d
    N = bundle.n
    var_names = bundle.scm.var_names
    objective = cfg.get("objective", "empirical")
    mode = "gaussian" if objective == "gaussian" else "empirical"
    Y_idx = var_names.index("Y")

    meta = json.load(open(data_dir / "portland_metadata.json"))
    sp = meta["standardisation"]
    tcc_values = sp["interventions_raw_tcc"]
    mu_Y_src = sp["Y_mu_src"]
    std_Y_src = sp["Y_std_src"]

    # Stability moduli
    eps = float(cfg["ambiguity"]["environment"]["eps"])
    mechanism_set = _build_mechanism_set(cfg["ambiguity"]["mechanism"], d)
    A_iotas, alpha_iotas = [], []
    for i, iv in enumerate(bundle.interventions):
        A_i = bundle.intervened_scms[i].A
        R_i = gating_matrix(d, _iv_to_nodes(iv, var_names))
        g = gamma(A_i, R_i, mechanism_set)
        alpha_iotas.append(alpha_polynomial(A_i, g, d))
        A_iotas.append(A_i)

    ns = bundle.noise_samples
    U_s_list = [ns[i] for i in range(bundle.n_interventions())] if isinstance(ns, dict) else list(ns)

    # Ground truth from real CSV
    gt = _real_csv_ground_truth(data_dir, tcc_values)

    # Build triptych per canopy level
    triptych = []
    all_covered = True

    # Endogenous samples include the fixed intervention contribution
    es = bundle.endogenous_samples
    X_s_list = [es[i] for i in range(bundle.n_interventions())] if isinstance(es, dict) else list(es)

    for tcc_idx, tcc in enumerate(tcc_values):
        iota_idx = tcc_idx + 1  # intervention 0 is observational
        gt_est = gt["estimates"][tcc_idx]

        # Fixed intervention contribution (needed for Gaussian mode)
        iv = bundle.interventions[iota_idx]
        fixed = np.zeros(d)
        for var_name, val in iv.items():
            idx = var_names.index(var_name) if isinstance(var_name, str) else int(var_name)
            fixed[idx] = val

        # Source-only (τ=I): E[Y | do(X=v), source] using endogenous samples
        source_only_std = float(np.mean(X_s_list[iota_idx], axis=0)[Y_idx])
        source_only_mgl = source_only_std * std_Y_src + mu_Y_src

        # Transported (τ): Phi_pushed = E[Y | do(X=v), τ-pushed]
        # endogenous_samples @ τ = (U_s @ A_ι + fixed @ A_ι) @ τ
        pushed_std = float(np.mean(X_s_list[iota_idx] @ tau, axis=0)[Y_idx])
        transported_mgl = pushed_std * std_Y_src + mu_Y_src

        # Certificate for query (iota_idx, Y)
        if mode == "empirical":
            cert_q = float(single_query_certificate(
                tau, alpha_iotas[iota_idx], A_iotas[iota_idx],
                bundle.noise_mean, bundle.noise_cov,
                eps=eps, O=[Y_idx], d=d, mode="empirical",
                U_s=U_s_list[iota_idx], N=N,
            ))
        else:
            cert_q = float(single_query_certificate(
                tau, alpha_iotas[iota_idx], A_iotas[iota_idx],
                bundle.noise_mean, bundle.noise_cov,
                eps=eps, O=[Y_idx], d=d, mode="gaussian",
            ))

        L_Phi = 1.0
        lo_std, hi_std = query_interval(pushed_std, L_Phi, cert_q)
        lo_mgl = lo_std * std_Y_src + mu_Y_src
        hi_mgl = hi_std * std_Y_src + mu_Y_src

        target_only_mgl = gt_est["target_only_mgl"]
        covered = lo_mgl <= target_only_mgl <= hi_mgl
        if not covered:
            all_covered = False

        triptych.append({
            "tcc_pct": tcc,
            "source_only_mgl": source_only_mgl,
            "target_only_mgl": target_only_mgl,
            "transported_mgl": transported_mgl,
            "cert_interval_mgl": [lo_mgl, hi_mgl],
            "cert_width_mgl": hi_mgl - lo_mgl,
            "delta_sq": cert_q,
            "covered": covered,
            "source_only_std": source_only_std,
            "transported_std": pushed_std,
            "target_only_std": gt_est["target_only_std"],
        })

    # Load eval.json if it exists for adversary_sup check
    eval_path = exp_dir / "eval.json"
    eval_info = {}
    if eval_path.exists():
        eval_data = json.load(open(eval_path))
        eval_info = {
            "certificate": eval_data.get("certificate"),
            "adversary_sup_estimate": eval_data.get("adversary_sup_estimate"),
            "certificate_valid": eval_data.get("certificate_valid"),
        }

    out = {
        "experiment": str(exp_dir),
        "objective": objective,
        "eps_config": eps,
        "ground_truth": gt,
        "triptych": triptych,
        "all_covered": all_covered,
        "eval_summary": eval_info,
        "tau_diagonal": [float(x) for x in np.diag(tau)],
    }

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Portland Triptych — {objective} mode, ε={eps}")
    print(f"{'='*80}")
    print(f"{'TCC%':>5}  {'Source':>9}  {'Target':>9}  {'Transport':>9}  "
          f"{'Interval':>20}  {'Width':>6}  {'Cov':>4}")
    print(f"{'-'*80}")
    for t in triptych:
        lo, hi = t["cert_interval_mgl"]
        cov_str = "YES" if t["covered"] else " NO"
        print(f"{t['tcc_pct']:>5}  {t['source_only_mgl']:>9.4f}  {t['target_only_mgl']:>9.4f}  "
              f"{t['transported_mgl']:>9.4f}  [{lo:>8.4f}, {hi:>8.4f}]  "
              f"{t['cert_width_mgl']:>6.3f}  {cov_str}")
    print(f"{'-'*80}")
    print(f"All covered: {all_covered}")
    if eval_info:
        print(f"Certificate valid: {eval_info.get('certificate_valid')}")
    print(f"{'='*80}\n")

    # Save
    out_path = exp_dir / "portland_triptych.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {out_path}")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Portland triptych evaluation.")
    p.add_argument("exp_dir", help="Experiment output directory.")
    p.add_argument("--data_dir", default="data/portland", help="Portland data directory.")
    args = p.parse_args()
    portland_triptych(Path(args.exp_dir), Path(args.data_dir))
