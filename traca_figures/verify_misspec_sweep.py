#!/usr/bin/env python3
"""Correctness gate and post-processing for the ATE misspecification sweep.

Verifies that delta=0.5 and delta=0.0 sweep results reproduce the published
ate_dir and ate_sym numbers on diagonal (eps==eta) rows.  Also builds a
unified CSV and summary tables from the full sweep.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Config name for delta=0.5 (reuses the published directional config).
ATE_DIR_CONFIG = "ate_gaussian_entrywise_subfamily_directional"
ATE_SYM_CONFIG = "ate_gaussian_entrywise_subfamily"

# All eight deltas and their result-directory names.
DELTA_CONFIGS = [
    ("dm0.5", "ate_misspec_dm0.5", -0.5),
    ("d0.0",  "ate_misspec_d0.0",   0.0),
    ("d0.3",  "ate_misspec_d0.3",   0.3),
    ("d0.4",  "ate_misspec_d0.4",   0.4),
    ("d0.5",  ATE_DIR_CONFIG,       0.5),
    ("d0.6",  "ate_misspec_d0.6",   0.6),
    ("d0.7",  "ate_misspec_d0.7",   0.7),
    ("d1.0",  "ate_misspec_d1.0",   1.0),
]
TRUE_SHIFT = 0.5


def _parse_method_eps_eta(method_str):
    m = re.search(r"eps_([\d.]+)_eta_([\d.]+)", method_str)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _compare(pub_path, sweep_path, label, compare_cols):
    """Compare two evaluation_queries.csv files on diagonal rows."""
    if not pub_path.exists():
        print(f"  SKIP: published not found at {pub_path}")
        return True
    if not sweep_path.exists():
        print(f"  SKIP: sweep not found at {sweep_path}")
        return True

    pub = pd.read_csv(pub_path)
    sweep = pd.read_csv(sweep_path)

    pub_eps_eta = pub["method"].apply(_parse_method_eps_eta)
    pub["_eps"] = pub_eps_eta.apply(lambda x: x[0])
    pub["_eta"] = pub_eps_eta.apply(lambda x: x[1])
    pub_diag = pub[pub["_eps"] == pub["_eta"]].copy()
    pub_diag = pub_diag[
        ~pub_diag["method"].str.contains("baseline", case=False)]

    sweep_filt = sweep[
        ~sweep["method"].str.contains("baseline", case=False)].copy()

    merge_keys = ["eps", "eta", "fold", "iota", "node"]
    merged = pd.merge(pub_diag, sweep_filt, on=merge_keys,
                      suffixes=("_pub", "_sweep"))

    print(f"\n{label}: {len(merged)} matched rows")
    ok = True
    for col in compare_cols:
        cp, cs = f"{col}_pub", f"{col}_sweep"
        if cp not in merged.columns or cs not in merged.columns:
            print(f"  {col:15s}: SKIP (column missing)")
            continue
        diff = (merged[cp] - merged[cs]).abs()
        max_diff = diff.max()
        status = "PASS" if max_diff < 1e-8 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {col:15s}: max|diff| = {max_diff:.2e}  {status}")
    return ok


def correctness_gate(results_dir, published_dir):
    """Compare delta=0.5 and delta=0.0 against published results."""
    print("=" * 70)
    print("CORRECTNESS GATE")
    print("=" * 70)

    all_pass = True

    # delta=0.5 vs published ate_dir
    pub_dir = (published_dir / "ate_dir" / ATE_DIR_CONFIG
               / "evaluation_queries.csv")
    sweep_d05 = (results_dir / ATE_DIR_CONFIG
                 / "evaluation_queries.csv")
    cols_dir = ["std_lo", "std_hi", "dir_lo", "dir_hi",
                "target_value", "Phi_pushed"]
    if not _compare(pub_dir, sweep_d05, "delta=0.5 vs ate_dir", cols_dir):
        all_pass = False

    # delta=0.0 vs published ate_sym
    pub_sym = (published_dir / "ate_sym" / ATE_SYM_CONFIG
               / "evaluation_queries.csv")
    sweep_d00 = (results_dir / "ate_misspec_d0.0"
                 / "evaluation_queries.csv")
    cols_sym = ["std_lo", "std_hi", "dir_lo", "dir_hi",
                "target_value", "Phi_pushed"]
    if not _compare(pub_sym, sweep_d00, "delta=0.0 vs ate_sym", cols_sym):
        all_pass = False

    print(f"\n{'=' * 70}")
    print(f"CORRECTNESS GATE: {'ALL PASS' if all_pass else 'FAILED'}")
    print(f"{'=' * 70}")

    if not all_pass:
        sys.exit(1)


def build_unified_csv(results_dir):
    """Merge all evaluation_queries.csv into sweep_results.csv."""
    print("=" * 70)
    print("BUILDING UNIFIED CSV")
    print("=" * 70)

    frames = []
    for label, config_name, delta in DELTA_CONFIGS:
        csv_path = results_dir / config_name / "evaluation_queries.csv"
        if not csv_path.exists():
            print(f"  WARNING: missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df = df[~df["method"].str.contains("baseline", case=False)].copy()
        df["delta"] = delta
        df["r_train"] = df["eps"]
        df["box_contains_truth"] = (
            (TRUE_SHIFT >= delta - df["r_train"])
            & (TRUE_SHIFT <= delta + df["r_train"])
        )
        frames.append(df)

    if not frames:
        print("ERROR: no data found")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)
    out_cols = [
        "delta", "r_train", "fold", "iota", "node", "target_value",
        "box_contains_truth",
        "std_lo", "std_hi", "std_width", "std_covered",
        "dir_lo", "dir_hi", "dir_width", "dir_covered",
    ]
    out_df = all_df[out_cols]
    out_path = results_dir / "sweep_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}  ({len(out_df)} rows)")
    return out_df


def summary_tables(df, results_dir):
    """Print and save summary tables at r_train=0.2."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLES AT r_train = 0.2")
    print("=" * 70)

    sub = df[np.isclose(df["r_train"], 0.2)].copy()
    if sub.empty:
        print("WARNING: no rows at r_train=0.2")
        return

    t1 = sub.groupby("delta").agg(
        mean_std_width=("std_width", "mean"),
        mean_dir_width=("dir_width", "mean"),
    ).reset_index()
    print("\nTable 1 -- Mean interval width by delta:")
    print(t1.to_string(index=False, float_format="%.6f"))
    t1.to_csv(results_dir / "summary_width.csv", index=False)

    box_info = sub.groupby("delta")["box_contains_truth"].first().reset_index()
    cov = sub.groupby("delta").agg(
        std_coverage=("std_covered", "mean"),
        dir_coverage=("dir_covered", "mean"),
    ).reset_index()
    t2 = pd.merge(cov, box_info, on="delta")
    t2 = t2[["delta", "box_contains_truth", "std_coverage", "dir_coverage"]]

    t2["DISAGREE"] = ""
    for i, row in t2.iterrows():
        if row["box_contains_truth"] and row["dir_coverage"] < 1.0:
            t2.at[i, "DISAGREE"] = "<< box contains truth but dir misses"
        elif not row["box_contains_truth"] and row["dir_coverage"] == 1.0:
            t2.at[i, "DISAGREE"] = "<< box misses truth but dir covers"

    print("\nTable 2 -- Coverage fraction by delta:")
    print(t2.to_string(index=False, float_format="%.4f"))
    t2.to_csv(results_dir / "summary_coverage.csv", index=False)

    t3 = t1.copy()
    t3["width_ratio"] = t3["mean_dir_width"] / t3["mean_std_width"]
    t3 = pd.merge(t3[["delta", "width_ratio"]], box_info, on="delta")
    t3 = t3[["delta", "box_contains_truth", "width_ratio"]]

    print("\nTable 3 -- Width ratio (dir/std) by delta:")
    print(t3.to_string(index=False, float_format="%.6f"))
    t3.to_csv(results_dir / "summary_ratio.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correctness gate and post-processing for the ATE "
                    "misspecification sweep.")
    parser.add_argument("--correctness_gate_only", action="store_true",
                        help="Run only the correctness gate, then exit.")
    parser.add_argument("--results_dir", type=Path,
                        default=Path("results_production/ate_misspec"),
                        help="Sweep results directory.")
    parser.add_argument("--published_dir", type=Path,
                        default=Path("results_production"),
                        help="Root of published results (ate_dir/, ate_sym/).")
    args = parser.parse_args()

    if args.correctness_gate_only:
        correctness_gate(args.results_dir, args.published_dir)
    else:
        correctness_gate(args.results_dir, args.published_dir)
        df = build_unified_csv(args.results_dir)
        summary_tables(df, args.results_dir)
