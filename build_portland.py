"""
Build the Portland real-world benchmark from EDI edi.1978.1 (seasonal
water-quality field data, 100 sites) and USGS ScienceBase landscape
covariates.  Outputs bundles and CSVs to data/portland/.

Source: all watersheds except Fanno Creek (N~247).
Target: Fanno Creek (N=100), root-node distribution only — no target Y.
DAG: Z -> X -> Y, Z -> Y, S -> Y  (d=4, simple back-door).
All variables standardised to source mean=0, std=1 before fitting W.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data source URLs
# ---------------------------------------------------------------------------

EDI_URL = (
    "https://pasta.lternet.edu/package/data/eml/edi/1978/1/"
    "240a45e219f14fa914f8209db0dafef6"
)
USGS_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    "65ccd274d34ef4b119cb3b50"
    "?f=__disk__c9%2Fc1%2F97%2Fc9c197fb4fbbe8c75b5135508c14a9f37f34ef16"
)

SEASON_MAP = {1: "Summer", 2: "Fall", 3: "Winter", 4: "Spring"}

# Paper's fixed TCC values for interventions (raw %)
TCC_INTERVENTION_VALUES = [45, 50, 55, 60]

# Min sites per watershed to include in rotated splits
ROTATED_MIN_SITES = 5


# ---------------------------------------------------------------------------
# Download helpers  (raw CSVs cached to data/portland/raw/ after first download)
# ---------------------------------------------------------------------------

RAW_EDI_PATH  = Path("data/portland/raw/edi_raw.csv")
RAW_USGS_PATH = Path("data/portland/raw/usgs_raw.csv")


def _fetch_raw(url: str, cache_path: Path, label: str) -> str:
    """Return raw CSV text, using a local cache if available; download + cache on first run."""
    if cache_path.exists():
        print(f"  {label}: using cached file ({cache_path})", flush=True)
        return cache_path.read_text(encoding="utf-8")
    print(f"  {label}: downloading from URL …", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "TraCA_Portland_Benchmark"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8", errors="replace")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(raw, encoding="utf-8")
    print(f"  {label}: saved to {cache_path}", flush=True)
    return raw


def _download_edi() -> pd.DataFrame:
    """Load EDI water-quality data (downloads once, then reads from cache)."""
    from io import StringIO
    raw = _fetch_raw(EDI_URL, RAW_EDI_PATH, "EDI water-quality data")
    df = pd.read_csv(StringIO(raw))

    # "4a" marks a second spring sampling round (April 29 vs May 6-10); treat as season 4.
    df["synoptic_event"] = df["synoptic_event"].replace({"4a": "4"})

    # Coerce numeric columns (EDI encodes missing values as "NA" strings)
    for col in ["curbid", "synoptic_event"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["canopy_site_dens_pct", "do_mgl", "temp_c", "ph",
                "do_pctsat", "cond_uS_cm", "ORP_mv", "turb_ntu"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  EDI rows loaded: {len(df)}", flush=True)
    return df


def _download_usgs() -> pd.DataFrame:
    """Load USGS landscape covariates (downloads once, then reads from cache). Returns PDX rows only."""
    from io import StringIO
    raw = _fetch_raw(USGS_URL, RAW_USGS_PATH, "USGS landscape data")
    df = pd.read_csv(StringIO(raw))
    pdx = df[df["City"] == "PDX"].copy()
    pdx["curbid"] = pd.to_numeric(pdx["curbid"], errors="coerce").astype("Int64")

    # Mean annual temperature = average of max and min
    pdx["Z_mean_temp"] = (pdx["prism_atemp_max_C"] + pdx["prism_atemp_min_C"]) / 2.0

    # canopy_site_dens_pct in the EDI data is all NA; use NLCD 2021 overwater TCC
    # (matches analysis.R: ggplot and boot_backdoor both use the overwater column)
    pdx = pdx.rename(columns={"nlcd_2021_ttc_overwater_%": "X_canopy_pct"})

    keep = ["curbid", "Latitude", "Longitude",
            "X_canopy_pct", "prism_elev_m", "prism_ppt_mm", "Z_mean_temp"]
    pdx = pdx[keep].rename(columns={
        "prism_elev_m": "Z_elevation",
        "prism_ppt_mm": "Z_precip",
    })
    print(f"  USGS Portland rows: {len(pdx)}", flush=True)
    return pdx


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge(edi: pd.DataFrame, usgs: pd.DataFrame) -> pd.DataFrame:
    """Inner-join EDI and USGS on curbid."""
    merged = edi.merge(usgs, on="curbid", how="inner")
    print(f"  After inner join: {len(merged)} rows", flush=True)
    return merged


# ---------------------------------------------------------------------------
# Cleaning  (reproduces paper's logic exactly)
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Apply the five cleaning steps from the paper.

    Returns
    -------
    df_clean : cleaned DataFrame
    report   : list of dicts describing each step
    """
    report = []
    n0 = len(df)

    # --- Step 0: drop rows missing key variables ---
    before = len(df)
    df = df.dropna(subset=["do_mgl", "temp_c", "X_canopy_pct"])
    removed_na = before - len(df)
    report.append({
        "step": "drop_missing_key_vars",
        "description": "Remove rows where do_mgl, temp_c, or canopy_site_dens_pct is NaN",
        "removed_n": removed_na,
        "removed_ids": [],   # NaN rows have no meaningful ID to report
    })

    # --- Step 1: remove Columbia Slough ---
    before = len(df)
    mask = df["watershed"] == "Columbia Slough"
    removed_ids = df.loc[mask, "curbid"].unique().tolist()
    df = df[~mask].copy()
    report.append({
        "step": "remove_columbia_slough",
        "description": "Remove all observations from Columbia Slough watershed",
        "removed_n": before - len(df),
        "removed_ids": [int(x) for x in removed_ids],
    })

    # --- Step 2: remove zero-canopy site ---
    before = len(df)
    mask = df["X_canopy_pct"] == 0
    removed_ids = df.loc[mask, "curbid"].unique().tolist()
    df = df[~mask].copy()
    report.append({
        "step": "remove_zero_canopy",
        "description": "Remove any site with zero tree canopy cover",
        "removed_n": before - len(df),
        "removed_ids": [int(x) for x in removed_ids],
    })

    # --- Step 3: remove duplicate site-season pairs (keep first) ---
    before = len(df)
    dup_mask = df.duplicated(subset=["curbid", "synoptic_event"], keep="first")
    removed_ids = df.loc[dup_mask, "curbid"].unique().tolist()
    df = df[~dup_mask].copy()
    report.append({
        "step": "remove_duplicate_site_season",
        "description": "Remove duplicate observations from the same site and season (keep first)",
        "removed_n": before - len(df),
        "removed_ids": [int(x) for x in removed_ids],
    })

    # --- Step 4: remove outlier oxygen saturation (do_pctsat >= 180) ---
    # Matches analysis.R: filter(do_pctsat < 180)
    before = len(df)
    mask = df["do_pctsat"] >= 180
    removed_ids = df.loc[mask, "curbid"].unique().tolist()
    removed_pctsat = df.loc[mask, "do_pctsat"].tolist()
    df = df[~mask].copy()
    report.append({
        "step": "remove_outlier_oxygen_saturation",
        "description": f"Remove observations with do_pctsat >= 180 "
                       f"(matches analysis.R filter)",
        "removed_n": before - len(df),
        "removed_ids": [int(x) for x in removed_ids],
        "outlier_do_pctsat": [float(x) for x in removed_pctsat],
    })

    n_final = len(df)
    total_removed = n0 - n_final

    print(f"\nCleaning report:")
    print(f"  Starting rows  : {n0}")
    for step in report:
        print(f"  {step['step']:<35s}: -{step['removed_n']:3d} rows", end="")
        if step["removed_ids"]:
            print(f"  (site IDs: {step['removed_ids'][:5]}{'…' if len(step['removed_ids'])>5 else ''})", end="")
        print()
    print(f"  {'FINAL':<35s}: {n_final} rows")

    if n_final != 347:
        print(f"\n  WARNING: expected 347 rows after cleaning, got {n_final}.")
        print(f"  (total removed: {total_removed})")
    else:
        print(f"\n  OK: 347 rows confirmed.")

    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Build CSV outputs
# ---------------------------------------------------------------------------

def _build_csvs(df: pd.DataFrame, out_root: Path) -> None:
    """Save all five benchmark CSV files."""

    # Rename for benchmark consistency
    df = df.copy()
    df["site_id"]      = df["curbid"].astype("int64")
    df["season_num"]   = df["synoptic_event"].astype("int64")
    df["season_name"]  = df["season_num"].map(SEASON_MAP)
    df["date"]         = df["collection_date"]
    df["X_canopy"]     = df["X_canopy_pct"]  # NLCD 2021 riparian TCC from USGS
    df["Y_do"]         = df["do_mgl"]
    df["W_temp"]       = df["temp_c"]

    # --- portland_obs_long.csv ---
    obs_long = df[[
        "watershed", "site_id", "season_num", "season_name", "date",
        "X_canopy", "Y_do", "W_temp",
        "Z_elevation", "Z_precip", "Z_mean_temp",
    ]].copy()
    obs_long.to_csv(out_root / "portland_obs_long.csv", index=False)
    print(f"  Saved portland_obs_long.csv   ({len(obs_long)} rows)")

    # --- portland_site_level.csv ---
    # canopy is site-level (constant across seasons); take first occurrence per site
    site_level = (
        df.groupby("site_id", as_index=False)
        .agg(
            watershed    = ("watershed", "first"),
            X_canopy     = ("X_canopy", "first"),
            Z_elevation  = ("Z_elevation", "first"),
            Z_precip     = ("Z_precip", "first"),
            Z_mean_temp  = ("Z_mean_temp", "first"),
            Latitude     = ("Latitude", "first"),
            Longitude    = ("Longitude", "first"),
        )
        .sort_values("site_id")
    )
    site_level.to_csv(out_root / "portland_site_level.csv", index=False)
    print(f"  Saved portland_site_level.csv ({len(site_level)} rows / sites)")

    # --- portland_backdoor.csv ---
    backdoor = df[[
        "watershed", "site_id", "season_num",
        "X_canopy", "Y_do",
        "Z_elevation", "Z_precip", "Z_mean_temp",
    ]].rename(columns={"X_canopy": "X", "Y_do": "Y"}).copy()
    backdoor.to_csv(out_root / "portland_backdoor.csv", index=False)
    print(f"  Saved portland_backdoor.csv   ({len(backdoor)} rows)")

    # --- portland_mediation.csv ---
    mediation = df[[
        "watershed", "site_id", "season_num",
        "X_canopy", "W_temp", "Y_do",
        "Z_elevation", "Z_precip", "Z_mean_temp",
    ]].rename(columns={"X_canopy": "X", "W_temp": "W", "Y_do": "Y"}).copy()
    mediation.to_csv(out_root / "portland_mediation.csv", index=False)
    print(f"  Saved portland_mediation.csv  ({len(mediation)} rows)")

    # --- portland_sim_base.csv ---
    # Real covariates that serve as fixed backbone for semi-synthetic generation (Appendix B)
    sim_base = df[[
        "watershed", "site_id", "season_num",
        "Z_elevation", "Z_precip", "Z_mean_temp",
    ]].copy()
    sim_base.to_csv(out_root / "portland_sim_base.csv", index=False)
    print(f"  Saved portland_sim_base.csv   ({len(sim_base)} rows)")


# ---------------------------------------------------------------------------
# Build splits JSON
# ---------------------------------------------------------------------------

def _build_splits(df: pd.DataFrame) -> dict:
    """
    Define main and rotated source/target splits.

    Main split:  target = Fanno Creek, source = all other watersheds.
    Rotated:     each watershed with ≥ ROTATED_MIN_SITES unique sites as target,
                 while keeping Fanno Creek in the source pool.
    """
    df = df.copy()
    df["site_id"] = df["curbid"].astype("int64")

    all_watersheds = sorted(df["watershed"].unique().tolist())
    fanno = "Fanno Creek"

    # count unique sites per watershed
    site_counts = (
        df.groupby("watershed")["site_id"]
        .nunique()
        .to_dict()
    )
    obs_counts = df["watershed"].value_counts().to_dict()

    fanno_sites  = sorted(df.loc[df["watershed"] == fanno, "site_id"].unique().tolist())
    source_ws    = [w for w in all_watersheds if w != fanno]

    main_split = {
        "target": fanno,
        "source": source_ws,
        "target_n_obs": int(obs_counts.get(fanno, 0)),
        "source_n_obs": int(sum(obs_counts.get(w, 0) for w in source_ws)),
        "target_n_sites": int(site_counts.get(fanno, 0)),
        "source_n_sites": int(sum(site_counts.get(w, 0) for w in source_ws)),
        "target_site_ids": fanno_sites,
    }

    # rotated splits: watersheds with ≥ ROTATED_MIN_SITES unique sites
    rotated = []
    for tgt in all_watersheds:
        if tgt == fanno:
            continue
        if site_counts.get(tgt, 0) < ROTATED_MIN_SITES:
            continue
        src_ws = [w for w in all_watersheds if w != tgt]
        tgt_sites = sorted(df.loc[df["watershed"] == tgt, "site_id"].unique().tolist())
        rotated.append({
            "target": tgt,
            "source": src_ws,
            "target_n_obs": int(obs_counts.get(tgt, 0)),
            "source_n_obs": int(sum(obs_counts.get(w, 0) for w in src_ws)),
            "target_n_sites": int(site_counts.get(tgt, 0)),
            "target_site_ids": tgt_sites,
        })

    return {
        "main": main_split,
        "rotated": rotated,
        "all_watersheds": all_watersheds,
        "obs_by_watershed": {k: int(v) for k, v in obs_counts.items()},
        "sites_by_watershed": {k: int(v) for k, v in site_counts.items()},
    }


# ---------------------------------------------------------------------------
# Fit LAN SCM (simple back-door DAG)
# ---------------------------------------------------------------------------

def _fit_scm(
    df: pd.DataFrame,
) -> tuple:
    """
    Fit a simple back-door LAN SCM on the SOURCE data (non-Fanno observations).

    DAG:  Z → X,  Z → Y,  S → Y,  X → Y
    Variables (standardised, zero mean in source):
        Z = PC1 of (Z_elevation, Z_precip, Z_mean_temp)
        S = synoptic_event (1–4, standardised)
        X = canopy_site_dens_pct (standardised)
        Y = do_mgl (standardised)

    Target SCM uses same W + source residual noise for endogenous vars (X, Y)
    + Fanno Creek root-node (Z, S) distribution — no target Y leakage.

    Returns
    -------
    source_scm, target_scm, interventions, scaler_params : dict
    """
    df = df.copy()
    df["site_id"] = df["curbid"].astype("int64")

    src = df[df["watershed"] != "Fanno Creek"].copy()
    tgt = df[df["watershed"] == "Fanno Creek"].copy()
    print(f"\nSCM fitting  —  source: {len(src)} obs, target (Fanno): {len(tgt)} obs")

    Z_raw_cols = ["Z_elevation", "Z_precip", "Z_mean_temp"]

    # ------------------------------------------------------------------
    # 1. Standardise raw Z covariates using SOURCE statistics
    # ------------------------------------------------------------------
    mu_Z_raw  = src[Z_raw_cols].mean().values      # (3,)
    std_Z_raw = src[Z_raw_cols].std(ddof=0).values  # (3,)
    std_Z_raw = np.where(std_Z_raw == 0, 1.0, std_Z_raw)

    Z_raw_src_std = (src[Z_raw_cols].values - mu_Z_raw) / std_Z_raw
    Z_raw_tgt_std = (tgt[Z_raw_cols].values - mu_Z_raw) / std_Z_raw

    # ------------------------------------------------------------------
    # 2. PCA (1 component) fitted on SOURCE standardised Z
    # ------------------------------------------------------------------
    cov = Z_raw_src_std.T @ Z_raw_src_std / len(Z_raw_src_std)
    evals, evecs = np.linalg.eigh(cov)
    # largest eigenvalue last (eigh returns ascending)
    pc1 = evecs[:, -1]  # (3,) — first principal component loadings

    Z_src_pc = Z_raw_src_std @ pc1   # (n_src,) — PC1 scores, source
    Z_tgt_pc = Z_raw_tgt_std @ pc1   # (n_tgt,) — PC1 scores, target (in source coords)

    # ------------------------------------------------------------------
    # 3. Standardise S, X, Y using SOURCE statistics
    # ------------------------------------------------------------------
    S_src_raw = src["synoptic_event"].astype(float).values
    X_src_raw = src["X_canopy_pct"].values
    Y_src_raw = src["do_mgl"].values

    mu_S_src,  std_S_src  = S_src_raw.mean(),  S_src_raw.std(ddof=0)
    mu_X_src,  std_X_src  = X_src_raw.mean(),  X_src_raw.std(ddof=0)
    mu_Y_src,  std_Y_src  = Y_src_raw.mean(),  Y_src_raw.std(ddof=0)

    # Z is already PC1; standardise to zero mean, unit std in source
    mu_Z_src,  std_Z_src  = Z_src_pc.mean(), Z_src_pc.std(ddof=0)
    std_Z_src = max(std_Z_src, 1e-8)

    Z_src = (Z_src_pc - mu_Z_src) / std_Z_src
    S_src = (S_src_raw - mu_S_src) / std_S_src
    X_src = (X_src_raw - mu_X_src) / std_X_src
    Y_src = (Y_src_raw - mu_Y_src) / std_Y_src

    # Fanno Creek in the SOURCE standardised coordinate system
    S_tgt_raw  = tgt["synoptic_event"].astype(float).values
    Z_tgt = (Z_tgt_pc - mu_Z_src) / std_Z_src      # source coords
    S_tgt = (S_tgt_raw - mu_S_src) / std_S_src

    # ------------------------------------------------------------------
    # 4. OLS regressions (no intercept — all vars zero-mean in source)
    # ------------------------------------------------------------------
    def ols(A, b):
        """Return OLS coefficients (no intercept) and residuals."""
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        resid = b - A @ coef
        return coef, resid

    # X ~ Z
    A_X = Z_src.reshape(-1, 1)
    coef_X, resid_X = ols(A_X, X_src)
    w_ZX = float(coef_X[0])

    # Y ~ Z + S + X
    A_Y = np.column_stack([Z_src, S_src, X_src])
    coef_Y, resid_Y = ols(A_Y, Y_src)
    w_ZY, w_SY, w_XY = float(coef_Y[0]), float(coef_Y[1]), float(coef_Y[2])

    print(f"  W[Z→X] = {w_ZX:.4f}")
    print(f"  W[Z→Y] = {w_ZY:.4f}   W[S→Y] = {w_SY:.4f}   W[X→Y] = {w_XY:.4f}")

    # ------------------------------------------------------------------
    # 5. Build W matrix  (var_names = ["Z", "S", "X", "Y"])
    # ------------------------------------------------------------------
    d = 4
    W = np.zeros((d, d))
    # indices: Z=0, S=1, X=2, Y=3
    W[0, 2] = w_ZX   # Z → X
    W[0, 3] = w_ZY   # Z → Y
    W[1, 3] = w_SY   # S → Y
    W[2, 3] = w_XY   # X → Y

    # ------------------------------------------------------------------
    # 6. Source noise
    # ------------------------------------------------------------------
    noise_std_src = np.array([
        1.0,                      # Z: exogenous, std=1 by construction (standardised)
        1.0,                      # S: exogenous, std=1 by construction
        float(resid_X.std(ddof=1)),  # X: mechanism residual
        float(resid_Y.std(ddof=1)),  # Y: mechanism residual
    ])
    noise_mean_src = np.zeros(d)

    # ------------------------------------------------------------------
    # 7. Target noise (Fanno Creek root-node distribution; same mechanism)
    #
    #    Source p(y, x, s, z | source) → W is estimated here.
    #    Target contributes p*(z) only → shift through Z (and S) root nodes.
    #    X and Y noise unchanged (mechanism invariance).
    # ------------------------------------------------------------------
    mu_Z_tgt_sc  = float(Z_tgt.mean())
    std_Z_tgt_sc = float(Z_tgt.std(ddof=0))
    std_Z_tgt_sc = max(std_Z_tgt_sc, 1e-8)

    mu_S_tgt_sc  = float(S_tgt.mean())
    std_S_tgt_sc = float(S_tgt.std(ddof=0))
    std_S_tgt_sc = max(std_S_tgt_sc, 1e-8)

    noise_mean_tgt = np.array([mu_Z_tgt_sc, mu_S_tgt_sc, 0.0, 0.0])
    noise_std_tgt  = np.array([
        std_Z_tgt_sc,            # Fanno Z distribution (encodes target covariate shift)
        std_S_tgt_sc,            # Fanno S distribution
        noise_std_src[2],        # source X residual (mechanism unchanged)
        noise_std_src[3],        # source Y residual (mechanism unchanged)
    ])

    print(f"\n  Source noise std (Z, S, X, Y): {noise_std_src.round(4)}")
    print(f"  Target noise mean (Z, S, X, Y): {noise_mean_tgt.round(4)}")
    print(f"  Target noise std  (Z, S, X, Y): {noise_std_tgt.round(4)}")

    # ------------------------------------------------------------------
    # 8. Intervention values in standardised X units
    # ------------------------------------------------------------------
    interventions = [{}]
    for x_raw in TCC_INTERVENTION_VALUES:
        x_std = (x_raw - mu_X_src) / std_X_src
        interventions.append({"X": float(x_std)})

    print(f"\n  Interventions (standardised X):")
    for iv in interventions:
        if iv:
            x_raw_approx = float(list(iv.values())[0]) * std_X_src + mu_X_src
            print(f"    do(X={list(iv.values())[0]:.4f})  ≈  TCC {x_raw_approx:.1f}%")

    # ------------------------------------------------------------------
    # 9. Scaler / PCA parameters (saved in metadata for reproducibility)
    # ------------------------------------------------------------------
    scaler_params = {
        "Z_raw_cols": Z_raw_cols,
        "Z_raw_mu":   mu_Z_raw.tolist(),
        "Z_raw_std":  std_Z_raw.tolist(),
        "pca_pc1_loadings": pc1.tolist(),
        "Z_pc_mu_src": float(mu_Z_src),
        "Z_pc_std_src": float(std_Z_src),
        "S_mu_src": float(mu_S_src), "S_std_src": float(std_S_src),
        "X_mu_src": float(mu_X_src), "X_std_src": float(std_X_src),
        "Y_mu_src": float(mu_Y_src), "Y_std_src": float(std_Y_src),
        "interventions_raw_tcc": TCC_INTERVENTION_VALUES,
        "interventions_std_X":   [
            float((x - mu_X_src) / std_X_src) for x in TCC_INTERVENTION_VALUES
        ],
        "target_Z_mean_in_source_coords": float(mu_Z_tgt_sc),
        "target_Z_std_in_source_coords":  float(std_Z_tgt_sc),
        "R2_X_from_Z":   float(1 - resid_X.var() / X_src.var()),
        "R2_Y_from_ZSX": float(1 - resid_Y.var() / Y_src.var()),
        "W": W.tolist(),
        "noise_std_source": noise_std_src.tolist(),
        "noise_mean_target": noise_mean_tgt.tolist(),
        "noise_std_target": noise_std_tgt.tolist(),
    }

    print(f"\n  R² X~Z:     {scaler_params['R2_X_from_Z']:.4f}")
    print(f"  R² Y~Z+S+X: {scaler_params['R2_Y_from_ZSX']:.4f}")

    return W, noise_mean_src, noise_std_src, noise_mean_tgt, noise_std_tgt, interventions, scaler_params


# ---------------------------------------------------------------------------
# Save bundles
# ---------------------------------------------------------------------------

def _save_bundles(
    W, noise_mean_src, noise_std_src,
    noise_mean_tgt, noise_std_tgt,
    interventions, out_root: Path
) -> None:
    """Build and pickle source and target SCMBundles."""
    import lan_scm

    var_names = ["Z", "S", "X", "Y"]

    source_scm = lan_scm.LANSCM(
        W=W,
        noise_mean=noise_mean_src,
        noise_cov=np.diag(noise_std_src ** 2),
        var_names=var_names,
    )

    # Target SCM:  same mechanism W, source X/Y residual noise,
    # Fanno Creek Z/S root-node distribution  (no target Y leakage).
    target_scm = lan_scm.LANSCM(
        W=W,
        noise_mean=noise_mean_tgt,
        noise_cov=np.diag(noise_std_tgt ** 2),
        var_names=var_names,
    )

    bundle_src = source_scm.bundle(interventions, n=1000, seed=0)
    bundle_tgt = target_scm.bundle(interventions, n=1000, seed=0)

    joblib.dump(bundle_src, out_root / "bundle.pkl")
    joblib.dump(bundle_tgt, out_root / "bundle_target.pkl")
    print(f"  Saved bundle.pkl          (source SCM, d=4, {len(interventions)} interventions)")
    print(f"  Saved bundle_target.pkl   (EVALUATION-ONLY — Fanno Creek diagnostic)")


# ---------------------------------------------------------------------------
# Build metadata JSON
# ---------------------------------------------------------------------------

def _build_metadata(
    df: pd.DataFrame,
    cleaning_report: list[dict],
    scaler_params: dict,
) -> dict:
    """Assemble the full metadata JSON."""
    obs_by_ws    = df["watershed"].value_counts().to_dict()
    sites_by_ws  = df.groupby("watershed")["curbid"].nunique().to_dict()
    obs_by_season = (
        df["synoptic_event"]
        .map(SEASON_MAP)
        .value_counts()
        .to_dict()
    )

    return {
        "raw_sources": {
            "edi_url": EDI_URL,
            "usgs_url": USGS_URL,
            "usgs_item_id": "65ccd274d34ef4b119cb3b50",
        },
        "column_map": {
            "nlcd_2021_ttc_overwater_%": "X_canopy / X  (NLCD 2021 overwater TCC from USGS; EDI canopy_site_dens_pct is all NA; matches analysis.R)",
            "do_mgl": "Y_do / Y",
            "temp_c": "W_temp / W",
            "synoptic_event": "season_num (1=Summer,2=Fall,3=Winter,4=Spring)",
            "season": "season_name (Dry/Wet from EDI)",
            "collection_date": "date",
            "watershed": "watershed",
            "curbid": "site_id",
            "prism_elev_m": "Z_elevation",
            "prism_ppt_mm": "Z_precip",
            "(prism_atemp_max_C + prism_atemp_min_C)/2": "Z_mean_temp",
        },
        "static_vars": [
            "X_canopy", "Z_elevation", "Z_precip", "Z_mean_temp",
        ],
        "repeated_vars": [
            "Y_do", "W_temp",
        ],
        "cleaning_steps": cleaning_report,
        "n_obs_final": int(len(df)),
        "counts_by_watershed": {k: int(v) for k, v in obs_by_ws.items()},
        "sites_by_watershed": {k: int(v) for k, v in sites_by_ws.items()},
        "counts_by_season": {k: int(v) for k, v in obs_by_season.items()},
        "lan_scm": {
            "description": "Simple back-door DAG fitted on source (non-Fanno) data",
            "var_names": ["Z", "S", "X", "Y"],
            "dag_edges": ["Z→X", "Z→Y", "S→Y", "X→Y"],
            "note_mediation": (
                "Fig. 3b mediation graph (Z, S, X, W, Y): "
                "Stage 1 transport estimates in portland_fig3b_estimates.json; "
                "Stage 2 restricted linear-Gaussian SCM approximation in bundle_mediation.pkl."
            ),
            "note_target_bundle": (
                "bundle_target.pkl is EVALUATION-ONLY. In the simple transport setup "
                "the source provides p(y,x,s,z|source) and the target provides p*(z) only. "
                "The target SCM encodes the Fanno Creek covariate shift through the "
                "root-node Z/S distribution; mechanism W and X/Y residual noise are "
                "unchanged (mechanism invariance assumption)."
            ),
        },
        "standardisation": scaler_params,
    }


# ---------------------------------------------------------------------------
# Fig. 3b — Stage 1: Paper-faithful transport estimator
# ---------------------------------------------------------------------------

def _compute_fig3b_transport(
    df: pd.DataFrame,
    scaler_params: dict,
    n_mc: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Stage 1: Paper-faithful Monte Carlo transport estimator for Fig. 3b.

    Implements the paper's identified formula for the extended graph (mediation + confounding):
        p*(y | do(x)) = Σ_{w,z} p*(z) p(w|x,z,source) Σ_{x'} p(x'|z,source) p(y|w,x',z,source)

    Three SOURCE identification regressions (these operationalize the identification
    formula — they are NOT the structural equations of the SCM):
        X ~ Z          →  p(x' | z, source)   [x' is the free variable in Y's regression]
        W ~ X + Z      →  p(w | x_int, z, source)  [x fixed at intervention]
        Y ~ W + X + Z  →  p(y | w, x', z, source)  [X absorbs confounded X→Y path]

    Target contributes p*(z) = Fanno Creek empirical Z distribution ONLY.
    Target Y is never used here (no leakage).
    """
    sp = scaler_params
    Z_raw_cols = sp["Z_raw_cols"]
    mu_Z_raw   = np.array(sp["Z_raw_mu"])
    std_Z_raw  = np.array(sp["Z_raw_std"])
    pc1        = np.array(sp["pca_pc1_loadings"])
    mu_Z_src   = sp["Z_pc_mu_src"]; std_Z_src = sp["Z_pc_std_src"]
    mu_S_src   = sp["S_mu_src"];    std_S_src = sp["S_std_src"]
    mu_X_src   = sp["X_mu_src"];    std_X_src = sp["X_std_src"]
    mu_Y_src   = sp["Y_mu_src"];    std_Y_src = sp["Y_std_src"]

    src = df[df["watershed"] != "Fanno Creek"].copy()
    tgt = df[df["watershed"] == "Fanno Creek"].copy()

    def _std_Z(raw_df):
        z = (raw_df[Z_raw_cols].values - mu_Z_raw) / std_Z_raw
        return (z @ pc1 - mu_Z_src) / std_Z_src

    Z_src = _std_Z(src)
    Z_tgt = _std_Z(tgt)
    X_src = (src["X_canopy_pct"].values - mu_X_src) / std_X_src
    Y_src = (src["do_mgl"].values - mu_Y_src) / std_Y_src

    W_src_raw = src["temp_c"].values
    mu_W_src  = W_src_raw.mean()
    std_W_src = max(W_src_raw.std(ddof=0), 1e-8)
    W_src = (W_src_raw - mu_W_src) / std_W_src

    def ols(A, b):
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return coef, b - A @ coef

    # Three identification regressions (front-door-style identification formula for mediation+confounding)
    coef_XZ,   resid_XZ   = ols(Z_src.reshape(-1, 1), X_src)       # X ~ Z
    coef_WXZ,  resid_WXZ  = ols(np.c_[X_src, Z_src], W_src)        # W ~ X + Z
    coef_YWXZ, resid_YWXZ = ols(np.c_[W_src, X_src, Z_src], Y_src) # Y ~ W + X + Z

    resid_std_X = float(resid_XZ.std(ddof=1))
    resid_std_W = float(resid_WXZ.std(ddof=1))
    resid_std_Y = float(resid_YWXZ.std(ddof=1))

    a_XZ = float(coef_XZ[0])
    a_WX, a_WZ = float(coef_WXZ[0]), float(coef_WXZ[1])
    a_YW, a_YX, a_YZ = float(coef_YWXZ[0]), float(coef_YWXZ[1]), float(coef_YWXZ[2])

    print(f"\n  Fig. 3b identification regressions (source):")
    print(f"    X ~ Z:         coef_Z = {a_XZ:.4f},  resid_std = {resid_std_X:.4f}")
    print(f"    W ~ X + Z:     coef_X = {a_WX:.4f},  coef_Z = {a_WZ:.4f},  resid_std = {resid_std_W:.4f}")
    print(f"    Y ~ W + X + Z: coef_W = {a_YW:.4f},  coef_X = {a_YX:.4f},  coef_Z = {a_YZ:.4f},  resid_std = {resid_std_Y:.4f}")

    rng = np.random.default_rng(seed)

    estimates = {
        "tcc_values": TCC_INTERVENTION_VALUES,
        "source_ey_do_x_mgl": [],
        "transported_ey_do_x_mgl": [],
        "note": (
            "source = do-formula evaluated using source Z distribution; "
            "transported = do-formula evaluated using Fanno Creek Z distribution (p*(z)). "
            "Target Y is not used."
        ),
        "identification_regressions": {
            "X_from_Z":   {"coef_Z": a_XZ, "resid_std": resid_std_X},
            "W_from_XZ":  {"coef_X": a_WX, "coef_Z": a_WZ, "resid_std": resid_std_W},
            "Y_from_WXZ": {"coef_W": a_YW, "coef_X": a_YX, "coef_Z": a_YZ, "resid_std": resid_std_Y},
            "note": (
                "These regressions implement the paper's identification formula, "
                "NOT the structural equations. Y ~ W + X + Z absorbs the confounded X↔Y "
                "path; structurally there is no direct X→Y edge in Fig. 3b."
            ),
        },
    }

    print(f"\n  Monte Carlo transport estimates (n_mc={n_mc}):")
    for x_raw in TCC_INTERVENTION_VALUES:
        x_int = (x_raw - mu_X_src) / std_X_src

        # Transported: Z drawn from Fanno Creek (target Z distribution = p*(z))
        Z_mc_t  = rng.choice(Z_tgt, size=n_mc, replace=True)
        W_mc_t  = a_WX * x_int + a_WZ * Z_mc_t + rng.normal(0, resid_std_W, n_mc)
        Xp_mc_t = a_XZ * Z_mc_t + rng.normal(0, resid_std_X, n_mc)   # marginalised X'
        Y_mc_t  = a_YW * W_mc_t + a_YX * Xp_mc_t + a_YZ * Z_mc_t + rng.normal(0, resid_std_Y, n_mc)
        ey_t_mgl = float(Y_mc_t.mean()) * std_Y_src + mu_Y_src

        # Source-only: Z drawn from source distribution
        Z_mc_s  = rng.choice(Z_src, size=n_mc, replace=True)
        W_mc_s  = a_WX * x_int + a_WZ * Z_mc_s + rng.normal(0, resid_std_W, n_mc)
        Xp_mc_s = a_XZ * Z_mc_s + rng.normal(0, resid_std_X, n_mc)
        Y_mc_s  = a_YW * W_mc_s + a_YX * Xp_mc_s + a_YZ * Z_mc_s + rng.normal(0, resid_std_Y, n_mc)
        ey_s_mgl = float(Y_mc_s.mean()) * std_Y_src + mu_Y_src

        estimates["source_ey_do_x_mgl"].append(round(ey_s_mgl, 4))
        estimates["transported_ey_do_x_mgl"].append(round(ey_t_mgl, 4))
        print(f"    TCC={x_raw}%:  source = {ey_s_mgl:.4f} mg/L,  "
              f"transported = {ey_t_mgl:.4f} mg/L")

    return estimates


# ---------------------------------------------------------------------------
# Fig. 3b — Stage 2: Restricted linear-Gaussian SCM approximation
# ---------------------------------------------------------------------------

def _fit_scm_mediation(df: pd.DataFrame, scaler_params: dict) -> tuple:
    """
    Stage 2: Restricted linear-Gaussian SCM APPROXIMATION of Fig. 3b.

    DAG (d=5): Z→X, Z→W, Z→Y, X→W, W→Y, S→Y  (no direct X→Y)
    var_names = ["Z", "S", "X", "W", "Y"],  indices Z=0, S=1, X=2, W=3, Y=4

    Structural equations fitted by OLS on source data using structural parents only
    (not the identification formula). Cov(U_X, U_Y) estimated from structural residuals
    as a proxy for the hidden confounder L.

    IMPORTANT: Y ~ W + Z + S is the structural equation (structural parents of Y).
               Y ~ W + X + Z is the IDENTIFICATION regression (in Stage 1 above).
    These are different — the identification formula absorbs the X↔Y confounded path.

    Target SCM: same W, same mechanism noise. PRIMARY shift = Fanno Creek Z (paper's p*(z)).
    Season S also updated — an engineering choice for benchmark packaging, not the core
    transport assumption. No target Y leakage.

    Returns
    -------
    W5, noise_mean_src5, Sigma_src, noise_mean_tgt5, Sigma_tgt, interventions_med, scaler_params_med
    """
    sp = scaler_params
    Z_raw_cols = sp["Z_raw_cols"]
    mu_Z_raw   = np.array(sp["Z_raw_mu"])
    std_Z_raw  = np.array(sp["Z_raw_std"])
    pc1        = np.array(sp["pca_pc1_loadings"])
    mu_Z_src   = sp["Z_pc_mu_src"]; std_Z_src = sp["Z_pc_std_src"]
    mu_S_src   = sp["S_mu_src"];    std_S_src = sp["S_std_src"]
    mu_X_src   = sp["X_mu_src"];    std_X_src = sp["X_std_src"]
    mu_Y_src   = sp["Y_mu_src"];    std_Y_src = sp["Y_std_src"]

    src = df[df["watershed"] != "Fanno Creek"].copy()
    tgt = df[df["watershed"] == "Fanno Creek"].copy()
    print(f"\nMediation SCM (Fig. 3b) — source: {len(src)} obs, target: {len(tgt)} obs")

    def _std_Z(raw_df):
        z = (raw_df[Z_raw_cols].values - mu_Z_raw) / std_Z_raw
        return (z @ pc1 - mu_Z_src) / std_Z_src

    Z_src = _std_Z(src)
    Z_tgt = _std_Z(tgt)
    S_src = (src["synoptic_event"].astype(float).values - mu_S_src) / std_S_src
    S_tgt = (tgt["synoptic_event"].astype(float).values - mu_S_src) / std_S_src
    X_src = (src["X_canopy_pct"].values - mu_X_src) / std_X_src
    Y_src = (src["do_mgl"].values - mu_Y_src) / std_Y_src

    W_src_raw = src["temp_c"].values
    mu_W_src  = W_src_raw.mean()
    std_W_src = max(W_src_raw.std(ddof=0), 1e-8)
    W_src     = (W_src_raw - mu_W_src) / std_W_src

    def ols(A, b):
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return coef, b - A @ coef

    # Structural equations (structural parents only — NOT the identification formula)
    coef_X, U_X = ols(Z_src.reshape(-1, 1), X_src)     # X ~ Z
    w_ZX = float(coef_X[0])

    coef_W, U_W = ols(np.c_[X_src, Z_src], W_src)      # W ~ X + Z
    w_XW, w_ZW = float(coef_W[0]), float(coef_W[1])

    coef_Y, U_Y = ols(np.c_[W_src, Z_src, S_src], Y_src)  # Y ~ W + Z + S (structural, no X→Y)
    w_WY, w_ZY, w_SY = float(coef_Y[0]), float(coef_Y[1]), float(coef_Y[2])

    print(f"  Structural coefficients:")
    print(f"    W[Z→X] = {w_ZX:.4f}")
    print(f"    W[X→W] = {w_XW:.4f}   W[Z→W] = {w_ZW:.4f}")
    print(f"    W[W→Y] = {w_WY:.4f}   W[Z→Y] = {w_ZY:.4f}   W[S→Y] = {w_SY:.4f}")
    print(f"    W[X→Y] = 0.0000  (no direct edge; effect mediated through W)")

    # 5×5 structural matrix (var_names = ["Z", "S", "X", "W", "Y"])
    W5 = np.zeros((5, 5))
    W5[0, 2] = w_ZX   # Z → X
    W5[0, 3] = w_ZW   # Z → W
    W5[0, 4] = w_ZY   # Z → Y
    W5[2, 3] = w_XW   # X → W
    W5[3, 4] = w_WY   # W → Y
    W5[1, 4] = w_SY   # S → Y
    # W5[2, 4] = 0    (no direct X→Y — structural assumption)

    # Noise covariance: diagonal + Cov(U_X, U_Y) off-diagonal [indices 2 and 4]
    var_Z   = float(np.var(Z_src, ddof=1))
    var_S   = float(np.var(S_src, ddof=1))
    var_U_X = float(np.var(U_X,   ddof=1))
    var_U_W = float(np.var(U_W,   ddof=1))
    var_U_Y = float(np.var(U_Y,   ddof=1))
    cov_XY  = float(np.cov(U_X, U_Y, ddof=1)[0, 1])

    Sigma_src = np.diag([var_Z, var_S, var_U_X, var_U_W, var_U_Y])
    Sigma_src[2, 4] = cov_XY
    Sigma_src[4, 2] = cov_XY

    eigs = np.linalg.eigvalsh(Sigma_src)
    print(f"\n  Cov(U_X, U_Y) = {cov_XY:.4f}  (proxy for hidden confounder L)")
    print(f"  Sigma_src eigenvalues: {eigs.round(4)}")
    if not np.all(eigs > 0):
        print("  WARNING: Sigma_src not positive-definite — zeroing Cov(U_X, U_Y)")
        Sigma_src[2, 4] = Sigma_src[4, 2] = 0.0

    # Target noise: PRIMARY = Fanno Z (paper's p*(z)); season updated as engineering choice
    var_Z_tgt = float(np.var(Z_tgt, ddof=1))
    var_S_tgt = float(np.var(S_tgt, ddof=1))
    mu_Z_tgt  = float(Z_tgt.mean())
    mu_S_tgt  = float(S_tgt.mean())

    Sigma_tgt = Sigma_src.copy()
    Sigma_tgt[0, 0] = var_Z_tgt   # Fanno Creek Z (primary shift, per paper's p*(z))
    Sigma_tgt[1, 1] = var_S_tgt   # Fanno season mix (implementation choice — see docstring)
    # Indices 2,3,4 and Cov(U_X,U_Y) unchanged — mechanism invariance

    noise_mean_src5 = np.zeros(5)
    noise_mean_tgt5 = np.array([mu_Z_tgt, mu_S_tgt, 0., 0., 0.])

    # Interventions: same standardised TCC values as back-door bundle
    interventions_med = [{}]
    for x_raw in TCC_INTERVENTION_VALUES:
        interventions_med.append({"X": float((x_raw - mu_X_src) / std_X_src)})

    scaler_params_med = {
        "W_mu_src": float(mu_W_src), "W_std_src": float(std_W_src),
        "W5": W5.tolist(),
        "cov_UX_UY": cov_XY,
        "R2_X_from_Z":   float(1 - np.var(U_X, ddof=1) / np.var(X_src, ddof=1)),
        "R2_W_from_XZ":  float(1 - np.var(U_W, ddof=1) / np.var(W_src, ddof=1)),
        "R2_Y_from_WZS": float(1 - np.var(U_Y, ddof=1) / np.var(Y_src, ddof=1)),
    }
    print(f"\n  R² X~Z:     {scaler_params_med['R2_X_from_Z']:.4f}")
    print(f"  R² W~X+Z:   {scaler_params_med['R2_W_from_XZ']:.4f}")
    print(f"  R² Y~W+Z+S: {scaler_params_med['R2_Y_from_WZS']:.4f}")

    return (W5, noise_mean_src5, Sigma_src,
            noise_mean_tgt5, Sigma_tgt,
            interventions_med, scaler_params_med)


def _save_bundles_mediation(
    W5, noise_mean_src5, Sigma_src,
    noise_mean_tgt5, Sigma_tgt,
    interventions_med, out_root: Path
) -> None:
    """Build and pickle source and target mediation SCMBundles (non-diagonal noise_cov)."""
    import lan_scm

    var_names = ["Z", "S", "X", "W", "Y"]

    scm_med_src = lan_scm.LANSCM(
        W=W5,
        noise_mean=noise_mean_src5,
        noise_cov=Sigma_src,    # non-diagonal: Cov(U_X, U_Y) ≠ 0
        var_names=var_names,
    )
    scm_med_tgt = lan_scm.LANSCM(
        W=W5,
        noise_mean=noise_mean_tgt5,
        noise_cov=Sigma_tgt,
        var_names=var_names,
    )

    bundle_med_src = scm_med_src.bundle(interventions_med, n=1000, seed=0)
    bundle_med_tgt = scm_med_tgt.bundle(interventions_med, n=1000, seed=0)

    joblib.dump(bundle_med_src, out_root / "bundle_mediation.pkl")
    joblib.dump(bundle_med_tgt, out_root / "bundle_mediation_target.pkl")
    print(f"  Saved bundle_mediation.pkl          (source SCM, d=5, {len(interventions_med)} interventions)")
    print(f"  Saved bundle_mediation_target.pkl   (EVALUATION-ONLY — Fanno Creek diagnostic)")


# ---------------------------------------------------------------------------
# Sanity plot
# ---------------------------------------------------------------------------

def _plot_sanity(df: pd.DataFrame, out_root: Path) -> None:
    """Scatterplot of canopy vs DO by watershed (mirrors paper Fig. 2)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        watersheds = sorted(df["watershed"].unique())
        colors = cm.tab10(np.linspace(0, 1, len(watersheds)))
        color_map = dict(zip(watersheds, colors))

        fig, ax = plt.subplots(figsize=(9, 5))
        for ws in watersheds:
            sub = df[df["watershed"] == ws]
            lw = 2.5 if ws == "Fanno Creek" else 0.8
            ax.scatter(
                sub["X_canopy_pct"], sub["do_mgl"],
                label=ws, color=color_map[ws],
                s=18, alpha=0.7, linewidths=0,
            )
            # regression line per watershed
            x = sub["X_canopy_pct"].values
            y = sub["do_mgl"].values
            if len(x) > 1:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, m * xs + b, color=color_map[ws], lw=lw, alpha=0.85)

        ax.set_xlabel("Tree Canopy Cover (%)")
        ax.set_ylabel("Dissolved Oxygen (mg/L)")
        ax.set_title("Portland Watersheds — canopy vs dissolved oxygen (sanity check)")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        fig.savefig(out_root / "sanity_canopy_vs_do.png", dpi=120)
        plt.close()
        print(f"  Saved sanity_canopy_vs_do.png")
    except Exception as exc:
        print(f"  [SKIP] Sanity plot failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the Portland benchmark package from raw EDI + USGS data."
    )
    p.add_argument(
        "--out_root", default="data",
        help="Root directory for output (default: data/)",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force re-download of raw data even if cached files exist.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    out_root = Path(args.out_root) / "portland"

    if args.refresh:
        for p in (RAW_EDI_PATH, RAW_USGS_PATH):
            if p.exists():
                p.unlink()
                print(f"  Deleted cache: {p}")
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Portland Benchmark Builder")
    print("=" * 60)

    # 1. Download raw data
    edi  = _download_edi()
    usgs = _download_usgs()

    # 2. Merge
    print("\nMerging on curbid …")
    df_merged = _merge(edi, usgs)

    # 3. Clean
    print("\nCleaning …")
    df, cleaning_report = _clean(df_merged)

    # 4. CSV outputs
    print("\nSaving CSV files …")
    _build_csvs(df, out_root)

    # 5. Splits JSON
    print("\nBuilding splits …")
    splits = _build_splits(df)
    (out_root / "portland_splits.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )
    main_split = splits["main"]
    print(f"  Main split: source={main_split['source_n_obs']} obs, "
          f"target={main_split['target_n_obs']} obs")
    print(f"  Rotated splits: {len(splits['rotated'])} watersheds")

    # 6. Fit LAN SCM
    print("\nFitting LAN SCM …")
    W, noise_mean_src, noise_std_src, noise_mean_tgt, noise_std_tgt, interventions, scaler_params = (
        _fit_scm(df)
    )

    # 7. Save bundles (Fig. 3a — simple back-door DAG)
    print("\nSaving bundles …")
    _save_bundles(
        W, noise_mean_src, noise_std_src,
        noise_mean_tgt, noise_std_tgt,
        interventions, out_root
    )

    # 8. Fig. 3b Stage 1 — paper-faithful transport estimator
    print("\nComputing Fig. 3b transport estimates (Stage 1) …")
    fig3b_estimates = _compute_fig3b_transport(df, scaler_params)
    (out_root / "portland_fig3b_estimates.json").write_text(
        json.dumps(fig3b_estimates, indent=2), encoding="utf-8"
    )
    print(f"  Saved portland_fig3b_estimates.json")

    # 9. Fig. 3b Stage 2 — restricted linear-Gaussian SCM approximation
    print("\nFitting mediation SCM (Fig. 3b, Stage 2) …")
    W5, nm_src5, Sig_src, nm_tgt5, Sig_tgt, ivs_med, sp_med = _fit_scm_mediation(df, scaler_params)
    print("\nSaving mediation bundles …")
    _save_bundles_mediation(W5, nm_src5, Sig_src, nm_tgt5, Sig_tgt, ivs_med, out_root)

    # 10. Metadata JSON
    print("\nSaving metadata …")
    metadata = _build_metadata(df, cleaning_report, scaler_params)
    (out_root / "portland_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"  Saved portland_metadata.json")

    # 11. Sanity plot
    print("\nGenerating sanity plot …")
    _plot_sanity(df, out_root)

    print("\n" + "=" * 60)
    print(f"Done.  All outputs in: {out_root.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
