#!/usr/bin/env python3
"""Generate the ATE directional-misspecification sweep figure.

Reads evaluation CSVs from seven misspecified-delta runs and the published
delta=0.5 (ate_dir) run, producing a two-panel PDF at r_train = 0.2.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

# ── Style — shared with generate_final.py ─────────────────────────────
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})
try:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{amssymb}"
    )
    fig_test = plt.figure()
    fig_test.text(0.5, 0.5, r"$\delta$")
    fig_test.savefig("/dev/null", format="png")
    plt.close(fig_test)
except Exception:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["text.latex.preamble"] = ""

sns.set_style("whitegrid")

C_DIR = "#EE854A"
C_STD = "#4878CF"
C_FAIL = "#cc0000"
C_GREY = "0.55"

# ── Constants ─────────────────────────────────────────────────────────
ETA = 0.2
TRUE_SHIFT = 0.5

# Seven sweep deltas; delta=0.5 comes from the published ate_dir run.
SWEEP_LABELS = {
    "dm0.5": -0.5, "d0.0": 0.0, "d0.3": 0.3, "d0.4": 0.4,
    "d0.6": 0.6, "d0.7": 0.7, "d1.0": 1.0,
}
ATE_DIR_CONFIG = "ate_gaussian_entrywise_subfamily_directional"


def _is_baseline(m):
    return "baseline" in str(m).lower()


def _load_eval(edir):
    e = pd.read_csv(edir / "evaluation.csv")
    q = pd.read_csv(edir / "evaluation_queries.csv")
    return e, q


# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="ATE misspecification-sweep figure.")
parser.add_argument(
    "--results_dir", type=Path,
    default=Path("results_production/ate_misspec"),
    help="Directory containing the seven sweep result subdirectories.")
parser.add_argument(
    "--ate_dir", type=Path,
    default=Path("results_production/ate_dir") / ATE_DIR_CONFIG,
    help="Directory containing the published delta=0.5 results.")
parser.add_argument(
    "--output", type=Path,
    default=Path(__file__).resolve().parent / "ate_delta_misspec.pdf",
    help="Output PDF path.")
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────
eval_frames, q_frames = [], []

for label, delta in SWEEP_LABELS.items():
    edir = args.results_dir / f"ate_misspec_{label}"
    e, q = _load_eval(edir)
    e["delta"] = delta
    q["delta"] = delta
    eval_frames.append(e)
    q_frames.append(q)

# delta=0.5 from ate_dir
e05, q05 = _load_eval(args.ate_dir)
e05["delta"] = 0.5
q05["delta"] = 0.5
eval_frames.append(e05)
q_frames.append(q05)

eval_all = pd.concat(eval_frames, ignore_index=True)
q_all = pd.concat(q_frames, ignore_index=True)

deltas_sorted = sorted(set(SWEEP_LABELS.values()) | {0.5})

# ── T1: transport ratio at r=0.2 ─────────────────────────────────────
e02 = eval_all[
    np.isclose(eval_all["eps"], ETA) & np.isclose(eval_all["eta"], ETA)]

ratios, ratio_lo, ratio_hi = [], [], []
for d in deltas_sorted:
    t = e02[(e02["delta"] == d)
            & (~e02["method"].apply(_is_baseline))]["target_loss"]
    b = e02[(e02["delta"] == d)
            & (e02["method"].apply(_is_baseline))]["target_loss"]
    per_fold = t.values / b.values
    ratios.append(np.mean(per_fold))
    ratio_lo.append(np.min(per_fold))
    ratio_hi.append(np.max(per_fold))

ratios = np.array(ratios)
ratio_lo = np.array(ratio_lo)
ratio_hi = np.array(ratio_hi)

# ── T3: directional widths at r=0.2 ──────────────────────────────────
q02 = q_all[np.isclose(q_all["eps"], ETA) & np.isclose(q_all["eta"], ETA)]
q02 = q02[~q02["method"].apply(_is_baseline)]

# do(X=1)
doX1 = q02[(q02["iota"] == 2) & (q02["node"] == 1)]
dir_widths, dir_w_lo, dir_w_hi, dir_covs = [], [], [], []
for d in deltas_sorted:
    sub = doX1[doX1["delta"] == d]
    w = sub["dir_width"].values
    dir_widths.append(np.mean(w))
    dir_w_lo.append(np.min(w))
    dir_w_hi.append(np.max(w))
    dir_covs.append(sub["dir_covered"].all())

dir_widths = np.array(dir_widths)
dir_w_lo = np.array(dir_w_lo)
dir_w_hi = np.array(dir_w_hi)
dir_covs = np.array(dir_covs)

# do(X=0) — expected constant across delta
doX0 = q02[(q02["iota"] == 1) & (q02["node"] == 1)]
doX0_widths = []
for d in deltas_sorted:
    sub = doX0[doX0["delta"] == d]
    doX0_widths.append(np.mean(sub["dir_width"].values))
doX0_widths = np.array(doX0_widths)
doX0_mean = np.mean(doX0_widths)

# Derived law: dir_width = 2*(transport + mechanism + environment)
#            = 4(|delta| + eta) + 2*eta
ds = np.array(deltas_sorted)
eff_line = 4 * (np.abs(ds) + ETA) + 2 * ETA

# ── Figure ────────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(
    1, 2, figsize=(13, 5), sharex=True,
    gridspec_kw={"wspace": 0.30},
)

cover_lo = TRUE_SHIFT - ETA   # 0.3
cover_hi = TRUE_SHIFT + ETA   # 0.7

for ax in (ax_l, ax_r):
    ax.axvspan(cover_lo, cover_hi, color=C_DIR, alpha=0.10, zorder=0)
    ax.axvline(0.0, color=C_FAIL, ls=":", lw=1.0, alpha=0.35, zorder=0)

# -- Left panel: transport ratio (log y) --
ax_l.axhline(1.0, color=C_GREY, ls="--", lw=1.2, zorder=1)
ax_l.text(ds[-1] + 0.03, 1.0, "identity", fontsize=13, color=C_GREY,
          va="bottom", ha="right")

yerr_lo = ratios - ratio_lo
yerr_hi = ratio_hi - ratios
ax_l.errorbar(ds, ratios, yerr=[yerr_lo, yerr_hi],
              fmt="o-", color=C_DIR, ms=7, lw=1.8, capsize=3, zorder=3,
              markeredgecolor="white", markeredgewidth=0.5)

idx05 = list(ds).index(0.5)
ax_l.plot(0.5, ratios[idx05], "D", color=C_STD, ms=10, zorder=4,
          markeredgecolor="white", markeredgewidth=0.8)

idx03 = list(ds).index(0.3)
ax_l.annotate(r"$\tau_{YX}$ closest to true shift",
              xy=(0.3, ratios[idx03]),
              xytext=(-0.42, 4e-3),
              fontsize=12, color="0.30",
              arrowprops=dict(arrowstyle="->", color="0.30", lw=1.0))

ax_l.set_yscale("log")
ax_l.set_ylabel(r"Transport error")
ax_l.set_ylim(5e-4, 30)

# -- Right panel: directional width --
ax_r.plot(ds, eff_line, ls="--", color=C_GREY, lw=1.5, zorder=1,
          label=r"$4(|\delta|{+}\eta)+2\eta$")

ax_r.axhline(doX0_mean, color=C_STD, ls="-", lw=1.5, alpha=0.7, zorder=2,
             label=r"$\mathbb{E}[Y \mid do(X{=}0)]$")

cov_mask = dir_covs
fail_mask = ~dir_covs

yerr_lo_w = dir_widths - dir_w_lo
yerr_hi_w = dir_w_hi - dir_widths
ax_r.errorbar(ds[cov_mask], dir_widths[cov_mask],
              yerr=[yerr_lo_w[cov_mask], yerr_hi_w[cov_mask]],
              fmt="o", color=C_DIR, ms=7, lw=1.8, capsize=3, zorder=3,
              markeredgecolor="white", markeredgewidth=0.5,
              label=r"$\mathbb{E}[Y \mid do(X{=}1)]$ (covers)")
if fail_mask.any():
    ax_r.errorbar(ds[fail_mask], dir_widths[fail_mask],
                  yerr=[yerr_lo_w[fail_mask], yerr_hi_w[fail_mask]],
                  fmt="o", color=C_FAIL, ms=8, lw=1.8, capsize=3, zorder=4,
                  markeredgecolor="white", markeredgewidth=0.5,
                  label="coverage fails")
    for i in np.where(fail_mask)[0]:
        ax_r.annotate("coverage\nfails",
                      xy=(ds[i], dir_widths[i]),
                      xytext=(ds[i] + 0.22, dir_widths[i] - 0.55),
                      fontsize=12, color=C_FAIL, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color=C_FAIL, lw=1.2))

ax_r.set_ylabel(r"Directional width")

# -- Shared x-axis --
for ax in (ax_l, ax_r):
    ax.set_xticks(ds)
    ax.set_xticklabels([f"${d:g}$" for d in ds], fontsize=13)

fig.supxlabel(r"$\delta$", fontsize=20, y=-0.02)

# -- Legend --
legend_elements = [
    Line2D([0], [0], marker="o", color=C_DIR, lw=1.8, ms=7,
           markeredgecolor="white", markeredgewidth=0.5,
           label=r"$\mathbb{E}[Y \mid do(X{=}1)]$ (covers)"),
    Line2D([0], [0], marker="o", color=C_FAIL, lw=0, ms=8,
           markeredgecolor="white", markeredgewidth=0.5,
           label="coverage fails"),
    Line2D([0], [0], color=C_STD, lw=1.5, alpha=0.7,
           label=r"$\mathbb{E}[Y \mid do(X{=}0)]$"),
    Line2D([0], [0], marker="D", color=C_STD, lw=0, ms=9,
           markeredgecolor="white", markeredgewidth=0.8,
           label=r"$\delta = 0.5$ (true shift)"),
    Line2D([0], [0], ls="--", color=C_GREY, lw=1.5,
           label=r"$4(|\delta|{+}\eta)+2\eta$"),
    Patch(facecolor=C_DIR, alpha=0.10, edgecolor="none",
          label=r"$[\delta{-}\eta,\,\delta{+}\eta] \ni 0.5$"),
]

fig.legend(handles=legend_elements, loc="lower center",
           ncol=3, frameon=True, fontsize=13,
           bbox_to_anchor=(0.5, -0.15))

fig.savefig(args.output)
plt.close(fig)
print(f"Saved {args.output}")

# ── Caption data ──────────────────────────────────────────────────────
print("\n--- Caption data ---")
header = (f"{'delta':>6s}  {'ratio':>8s}  {'dir_w doX1':>10s}  "
          f"{'4(|d|+n)+2n':>12s}  {'dir_w doX0':>10s}  {'dir_cov doX1':>12s}")
print(header)
for i, d in enumerate(ds):
    print(f"{d:+6.1f}  {ratios[i]:8.4f}  {dir_widths[i]:10.4f}  "
          f"{eff_line[i]:12.4f}  {doX0_widths[i]:10.4f}  "
          f"{'covers' if dir_covs[i] else 'FAILS'}")
print(f"\ndo(X=0) mean dir_width = {doX0_mean:.4f}")
print(f"Residual: max|dir_w_doX1 - 4(|d|+eta)-2eta| = "
      f"{np.max(np.abs(dir_widths - eff_line)):.2e}")

print("\nCoverage failures on ATE queries at r_train=0.2:")
any_fail = False
for d in ds:
    for iota, ql in [(1, "do(X=0)"), (2, "do(X=1)")]:
        sub = q02[(q02["delta"] == d) & (q02["iota"] == iota)
                   & (q02["node"] == 1)]
        frac = sub["dir_covered"].mean()
        if frac < 1.0:
            any_fail = True
            n_cov = int(sub["dir_covered"].sum())
            print(f"  delta={d:+.1f}  {ql}:  {frac:.2f} "
                  f"({n_cov}/{len(sub)} folds)")
if not any_fail:
    print("  (none)")
