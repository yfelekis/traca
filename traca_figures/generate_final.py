#!/usr/bin/env python3
"""Generate finalized main-text figures for TraCA paper.

Naming conventions (display vs code):
  Variant axis:     Symmetric / Offset   (code dirs: ate_sym / ate_dir)
  Certificate axis: General / Directional (code columns: std_* / dir_*)

Output (all in traca_figures/):
  ate_heatmaps.pdf           — 2x3 heatmap (sym/offset x obs_distance/gen_width/dir_width)
  ate_width_curves.pdf       — width vs eps (log y), vacuity line, two panels per query
  atce_brackets.pdf          — 3-panel nested bands at COMBOS (eps_test=0.5)
  atce_tightness.pdf         — 3-panel utilisation bars
  atce_coverage_crossing.pdf — 2-panel coverage vs training radius
  portland_coverage.pdf      — coverage vs eps
  portland_bracket_eps2.pdf  — nested bands at eps=2 with inset

Run:  conda run -n traca-run python traca_figures/generate_final.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
    fig_test = plt.figure()
    fig_test.text(0.5, 0.5, r'$\mathbb{E}[Y \mid do(X{=}0)]$')
    fig_test.savefig('/dev/null', format='png')
    plt.close(fig_test)
except Exception:
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = ''

sns.set_style('whitegrid')

OUT = ROOT / 'traca_figures'
OUT.mkdir(exist_ok=True)
RESULTS = ROOT / 'results_production'

# ── Colors & constants ────────────────────────────────────────────
# Code constant names (C_STD, C_DIR) match CSV column prefixes;
# display labels use "General" / "Directional".
C_STD = '#4878CF'     # General certificate color
C_DIR = '#EE854A'     # Directional certificate color
C_TRUTH = 'black'
C_PUSHED = '#2ca02c'
C_FAIL = '#cc0000'

# W₂ between source and Fanno Creek target exogenous distributions restricted
# to shifted coordinates (Z, S).  When ε ≥ this value the Gelbrich ball contains
# the actual Fanno shift, so the certificate should cover the true target query.
FANNO_W2 = 1.165
ATE_QS = 1.05         # mean |target_value| across ATE query family
ATE_VACUITY = 10 * ATE_QS  # 10.5
PORTLAND_QS = 0.1632  # mean |target_value| across Portland query family
PORTLAND_VACUITY = 10 * PORTLAND_QS  # 1.63
PORTLAND_DIR_COVERAGE_ONSET = 2.0  # first r_train where directional covers all queries

ATE_QF = [(1, 1), (2, 1)]
ATCE_QF = {(1, 2), (2, 2)}
PORTLAND_QF = [(1, 3), (2, 3), (3, 3), (4, 3)]

ATE_QLABELS = {
    1: r'$\mathbb{E}[Y \mid do(X{=}0)]$',
    2: r'$\mathbb{E}[Y \mid do(X{=}1)]$',
}
ATCE_QLABELS = {
    (1, 2): r'$\mathbb{E}[Y \mid do(X{=}0)]$',
    (2, 2): r'$\mathbb{E}[Y \mid do(X{=}1)]$',
}
PORTLAND_LABELS = {
    1: r'$do(X{=}45\%)$',
    2: r'$do(X{=}50\%)$',
    3: r'$do(X{=}55\%)$',
    4: r'$do(X{=}60\%)$',
}

# ATCE bracket/tightness combos: (eps_train, eps_test)
# eps_test=0.5 preserves three-panel <,=,> structure with both-informative matched cell
COMBOS = [(0.2, 0.5), (0.5, 0.5), (1.0, 0.5)]
# Column titles for brackets and tightness (unified r notation)
COMBO_TITLES = [
    r'$r_{\mathrm{train}}{=}0.2 < r_{\mathrm{test}}{=}0.5$',
    r'$r_{\mathrm{train}}{=}r_{\mathrm{test}}{=}0.5$',
    r'$r_{\mathrm{train}}{=}1.0 > r_{\mathrm{test}}{=}0.5$',
]


# ── Data loading ──────────────────────────────────────────────────

def _filter_qf(df, qf_pairs):
    s = set(qf_pairs)
    mask = df.apply(lambda r: (int(r['iota']), int(r['node'])) in s, axis=1)
    return df[mask].copy()


def load_ate(variant):
    """variant: 'sym' or 'dir' (= offset in display)"""
    if variant == 'sym':
        path = 'ate_sym/ate_gaussian_entrywise_subfamily'
    else:
        path = 'ate_dir/ate_gaussian_entrywise_subfamily_directional'
    df = pd.read_csv(RESULTS / path / 'evaluation_queries.csv')
    df = df[~df['method'].str.contains('baseline', na=False)]
    return _filter_qf(df, ATE_QF)


def load_atce_r2():
    subdir = 'atce_subfamily/atce_gaussian_z_entrywise_subfamily'
    df_main = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
    df_q = pd.read_csv(RESULTS / subdir / 'radius_eval_queries.csv')
    vi = df_main[['variant', 'eps_train', 'eta_train']].drop_duplicates()
    df_q = df_q.merge(vi, on='variant', how='left').dropna(subset=['eps_train'])
    mask = df_q.apply(
        lambda r: (int(r['iota']), int(r['node'])) in ATCE_QF, axis=1)
    df_main = df_main[~df_main['variant'].str.contains('baseline', na=False)].copy()
    df_q_filt = df_q[mask & ~df_q['variant'].str.contains('baseline', na=False)].copy()
    return df_main, df_q_filt


def load_portland_queries():
    df = pd.read_csv(RESULTS / 'portland/portland_backdoor_gaussian_qrestricted/evaluation_queries.csv')
    df = df[~df['method'].str.contains('baseline', na=False)]
    return _filter_qf(df, PORTLAND_QF)


# ═══════════════════════════════════════════════════════════════════
# Figure 1: ATE heatmaps — 2 rows (sym/offset) x 3 cols
# ═══════════════════════════════════════════════════════════════════

def ate_heatmaps():
    sym_q = load_ate('sym')
    dir_q = load_ate('dir')

    def make_pivot(df, col):
        avg = df.groupby(['eps', 'eta'])[col].mean().reset_index()
        piv = avg.pivot(index='eps', columns='eta', values=col)
        piv = piv.sort_index(ascending=True)
        piv.columns = [f'{c:.1f}' for c in piv.columns]
        piv.index = [f'{r:.1f}' for r in piv.index]
        return piv

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    col_labels = ['Obs. distance', 'General width', 'Directional width']

    for row_idx, (df, variant_label) in enumerate([(sym_q, 'Symmetric'),
                                                    (dir_q, 'Offset')]):
        piv_obs = make_pivot(df, 'obs_distance')
        ax = axes[row_idx, 0]
        sns.heatmap(piv_obs, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                    cbar_kws={'shrink': 0.8}, annot_kws={'size': 13})
        ax.invert_yaxis()
        ax.set_xlabel(r'$\eta$ (train)')
        ax.set_ylabel(f'{variant_label}\n' + r'$\varepsilon$ (train)')

        eps_vals = list(piv_obs.index)
        eta_vals = list(piv_obs.columns)
        ei = eps_vals.index('0.2')
        ej = eta_vals.index('0.2')
        ax.add_patch(plt.Rectangle((ej, ei), 1, 1, fill=False,
                                   edgecolor='red', lw=3))

        piv_sw = make_pivot(df, 'std_width')
        ax = axes[row_idx, 1]
        sns.heatmap(piv_sw, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'shrink': 0.8}, annot_kws={'size': 13})
        ax.invert_yaxis()
        ax.set_xlabel(r'$\eta$ (train)')
        ax.set_ylabel(r'$\varepsilon$ (train)')

        piv_dw = make_pivot(df, 'dir_width')
        ax = axes[row_idx, 2]
        sns.heatmap(piv_dw, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'shrink': 0.8}, annot_kws={'size': 13})
        ax.invert_yaxis()
        ax.set_xlabel(r'$\eta$ (train)')
        ax.set_ylabel(r'$\varepsilon$ (train)')

    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].text(0.5, 1.08, label, transform=axes[0, col_idx].transAxes,
                              ha='center', va='bottom', fontsize=18, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / 'ate_heatmaps.pdf')
    plt.close(fig)
    print('  ate_heatmaps.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 2: ATE width curves — log-y width vs eps, two panels
# ═══════════════════════════════════════════════════════════════════

def ate_width_curves():
    sym = load_ate('sym')
    dir_ = load_ate('dir')

    sym_diag = sym[np.isclose(sym['eps'], sym['eta']) & (sym['eps'] > 1e-9)]
    dir_diag = dir_[np.isclose(dir_['eps'], dir_['eta']) & (dir_['eps'] > 1e-9)]

    def avg_widths(df):
        return df.groupby(['eps', 'iota']).agg(
            sw=('std_width', 'mean'), dw=('dir_width', 'mean'),
        ).reset_index()

    sa = avg_widths(sym_diag)
    da = avg_widths(dir_diag)

    queries = [
        (1, ATE_QLABELS[1]),
        (2, ATE_QLABELS[2]),
    ]

    # 4 curves: variant × certificate
    # Color = certificate (blue=General, orange=Directional)
    # Linestyle = variant (solid=Symmetric, dashed=Offset)
    curve_specs = [
        (sa, 'sw', C_STD, '-',  'o', 'Sym. General'),
        (sa, 'dw', C_DIR, '-',  'o', 'Sym. Directional'),
        (da, 'sw', C_STD, '--', 's', 'Off. General'),
        (da, 'dw', C_DIR, '--', 's', 'Off. Directional'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, (iota, qlabel) in zip(axes, queries):
        for data, col, color, ls, marker, label in curve_specs:
            sub = data[data['iota'] == iota].sort_values('eps')
            ax.plot(sub['eps'].values, sub[col].values,
                    color=color, ls=ls, marker=marker, ms=7, lw=2.2,
                    label=label, zorder=3)

        # Vacuity threshold
        ax.axhline(ATE_VACUITY, color='gray', ls=':', lw=2, alpha=0.7,
                   label=f'Vacuity ($10 \\times$ QS $= {ATE_VACUITY:.1f}$)',
                   zorder=2)

        ax.set_yscale('log')
        ax.set_xlabel(r'$r_{\mathrm{train}}$')
        ax.set_xticks([0.2, 0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(['0.2', '0.5', '1.0', '2.0', '4.0'])

        ax.set_title(qlabel, fontsize=18, pad=10)

    axes[0].set_ylabel('Certificate width')

    # Horizontal legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=13, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / 'ate_width_curves.pdf')
    plt.close(fig)
    print('  ate_width_curves.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 2b: ATE intervals at eps=eta=0.2 — coverage & informativeness
# ═══════════════════════════════════════════════════════════════════

ATE_TRUTHS = {1: 0.300, 2: 1.800}   # E[Y|do(X=0)], E[Y|do(X=1)]


def ate_intervals_diagonal():
    sym = load_ate('sym')
    dir_ = load_ate('dir')

    # Filter: eps=eta=0.2, node=1 (Y), non-baseline — already QF-filtered
    def at_02(df):
        return df[np.isclose(df['eps'], 0.2) & np.isclose(df['eta'], 0.2)]

    sym_02 = at_02(sym)
    dir_02 = at_02(dir_)

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))

    variants = [('Symmetric', sym_02), ('Offset', dir_02)]
    queries = [
        (1, ATE_QLABELS[1]),
        (2, ATE_QLABELS[2]),
    ]

    for row_idx, (vlabel, data) in enumerate(variants):
        for col_idx, (iota, qlabel) in enumerate(queries):
            ax = axes[row_idx, col_idx]
            sub = data[data['iota'] == iota]
            truth = ATE_TRUTHS[iota]

            # Mean intervals across folds
            slo = sub['std_lo'].mean()
            shi = sub['std_hi'].mean()
            dlo = sub['dir_lo'].mean()
            dhi = sub['dir_hi'].mean()
            phi = sub['Phi_pushed'].mean()

            # Coverage fractions
            n_folds = len(sub)
            std_cov = int(sub['std_covered'].sum())
            dir_cov = int(sub['dir_covered'].sum())

            # ── bands ──
            # General (blue, wide)
            ax.fill_betweenx([-0.4, 0.4], slo, shi, color=C_STD, alpha=0.20,
                             label='General' if row_idx == 0 and col_idx == 0 else None)
            ax.plot([slo, slo], [-0.4, 0.4], color=C_STD, lw=1.2, alpha=0.5)
            ax.plot([shi, shi], [-0.4, 0.4], color=C_STD, lw=1.2, alpha=0.5)

            # Directional (orange, nested)
            ax.fill_betweenx([-0.3, 0.3], dlo, dhi, color=C_DIR, alpha=0.35,
                             label='Directional' if row_idx == 0 and col_idx == 0 else None)
            ax.plot([dlo, dlo], [-0.3, 0.3], color=C_DIR, lw=1.2, alpha=0.7)
            ax.plot([dhi, dhi], [-0.3, 0.3], color=C_DIR, lw=1.2, alpha=0.7)

            # Truth line
            ax.axvline(truth, color=C_TRUTH, ls='--', lw=2.2, zorder=5,
                       label='Truth' if row_idx == 0 and col_idx == 0 else None)

            # Interval centre
            ax.scatter([phi], [0], marker='|', color=C_PUSHED, s=200, zorder=6,
                       linewidths=2,
                       label='Interval centre' if row_idx == 0 and col_idx == 0 else None)

            # ── gap annotation where truth misses directional ──
            miss = 0.0
            if truth > dhi:
                miss = truth - dhi
            elif truth < dlo:
                miss = dlo - truth
            if miss > 0.01:
                edge = dhi if truth > dhi else dlo
                y_ann = -0.42
                ax.annotate(
                    '', xy=(edge, y_ann), xytext=(truth, y_ann),
                    arrowprops=dict(arrowstyle='<->', color=C_FAIL, lw=1.8,
                                   shrinkA=0, shrinkB=0))
                ax.text((edge + truth) / 2, y_ann - 0.09,
                        f'gap$={miss:.2f}$', ha='center', va='top',
                        color=C_FAIL, fontsize=13, fontweight='bold')

            # (coverage fractions available: std_cov, dir_cov — not displayed)

            # ── axes ──
            ax.set_yticks([])
            ax.set_ylim(-0.55, 0.6)

            xmin = min(slo, truth) - 0.12 * (shi - slo)
            xmax = max(shi, truth) + 0.12 * (shi - slo)
            ax.set_xlim(xmin, xmax)

            if row_idx == 1:
                ax.set_xlabel('Query value')

    # Row labels
    for row_idx, vlabel in enumerate(['Symmetric', 'Offset']):
        axes[row_idx, 0].set_ylabel(vlabel, fontsize=18, fontweight='bold')

    # Column labels
    for col_idx, (iota, qlabel) in enumerate(queries):
        axes[0, col_idx].text(0.5, 1.08, qlabel, transform=axes[0, col_idx].transAxes,
                              ha='center', va='bottom', fontsize=17)

    # Horizontal legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=16, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(OUT / 'ate_intervals_diagonal.pdf')
    plt.close(fig)
    print('  ate_intervals_diagonal.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 2c: ATE diagonal sweep — nested shaded areas along eps=eta
# ═══════════════════════════════════════════════════════════════════

def ate_diagonal_shaded():
    """Nested shaded areas: general & directional intervals vs truth along diagonal."""
    sym = load_ate('sym')
    dir_ = load_ate('dir')

    # Diagonal slice: eps == eta, exclude baseline and eps=0
    def diag(df):
        d = df[np.isclose(df['eps'], df['eta']) & (df['eps'] > 1e-9)]
        return d.groupby(['eps', 'iota']).agg(
            slo=('std_lo', 'mean'), shi=('std_hi', 'mean'),
            dlo=('dir_lo', 'mean'), dhi=('dir_hi', 'mean'),
            target=('target_value', 'first'),
            phi=('Phi_pushed', 'mean'),
        ).reset_index()

    sa = diag(sym)
    da = diag(dir_)

    queries = [(1, ATE_QLABELS[1]), (2, ATE_QLABELS[2])]
    variants = [('Symmetric', sa, '-'), ('Offset', da, '--')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for row_idx, (vlabel, data, ls) in enumerate(variants):
        for col_idx, (iota, qlabel) in enumerate(queries):
            ax = axes[row_idx, col_idx]
            sub = data[data['iota'] == iota].sort_values('eps')
            eps = sub['eps'].values
            truth = sub['target'].iloc[0]

            # General band (blue)
            ax.fill_between(eps, sub['slo'].values, sub['shi'].values,
                            color=C_STD, alpha=0.20,
                            label='General' if row_idx == 0 and col_idx == 0 else None)
            ax.plot(eps, sub['slo'].values, color=C_STD, lw=1, alpha=0.5)
            ax.plot(eps, sub['shi'].values, color=C_STD, lw=1, alpha=0.5)

            # Directional band (orange, nested)
            ax.fill_between(eps, sub['dlo'].values, sub['dhi'].values,
                            color=C_DIR, alpha=0.35,
                            label='Directional' if row_idx == 0 and col_idx == 0 else None)
            ax.plot(eps, sub['dlo'].values, color=C_DIR, lw=1.2, alpha=0.7)
            ax.plot(eps, sub['dhi'].values, color=C_DIR, lw=1.2, alpha=0.7)

            # Truth
            ax.axhline(truth, color=C_TRUTH, ls='--', lw=2, zorder=5,
                       label='Truth' if row_idx == 0 and col_idx == 0 else None)

            # Interval centre
            ax.plot(eps, sub['phi'].values, color=C_PUSHED, marker='|', ms=8,
                    lw=0, zorder=6,
                    label='Interval centre' if row_idx == 0 and col_idx == 0 else None)

            # Red X markers where directional fails to cover truth
            for _, r in sub.iterrows():
                if truth < r['dlo'] or truth > r['dhi']:
                    ax.scatter([r['eps']], [truth], marker='x', color=C_FAIL,
                               s=100, zorder=7, linewidths=2.5,
                               label='Dir. miss' if row_idx == 0 and col_idx == 0 else None)

            ax.set_ylabel('Query value')
            if row_idx == 1:
                ax.set_xlabel(r'$r_{\mathrm{train}}$')
            ax.set_xticks([0.2, 0.5, 1.0, 2.0, 4.0])
            ax.set_xticklabels(['0.2', '0.5', '1.0', '2.0', '4.0'])

    # Row / column labels
    for row_idx, vlabel in enumerate(['Symmetric', 'Offset']):
        axes[row_idx, 0].text(-0.18, 0.5, vlabel, transform=axes[row_idx, 0].transAxes,
                              ha='center', va='center', fontsize=18, fontweight='bold',
                              rotation=90)
    for col_idx, (_, qlabel) in enumerate(queries):
        axes[0, col_idx].text(0.5, 1.06, qlabel, transform=axes[0, col_idx].transAxes,
                              ha='center', va='bottom', fontsize=17)

    # Horizontal legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Deduplicate (Dir. miss label may appear multiple times)
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    fig.legend(unique_h, unique_l, loc='lower center', ncol=len(unique_l),
               fontsize=16, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(OUT / 'ate_diagonal_shaded.pdf')
    plt.close(fig)
    print('  ate_diagonal_shaded.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 3: ATCE brackets — 3-panel nested bands (eps_test=0.5)
# ═══════════════════════════════════════════════════════════════════

def atce_brackets():
    _, df_q = load_atce_r2()

    qkeys = sorted(ATCE_QF)
    nq = len(qkeys)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for col, (et, etest) in enumerate(COMBOS):
        ax = axes[col]
        bdf = df_q[np.isclose(df_q['eps_train'], et)
                    & np.isclose(df_q['rho_test'], etest)].copy()

        if bdf.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        bdf['std_hw'] = np.sqrt(bdf['std_cert_q'].clip(0))
        bdf['dir_hw'] = bdf['dir_cert'].abs()

        agg = bdf.groupby(['iota', 'node']).agg(
            phi=('Phi_pushed', 'mean'),
            std_hw=('std_hw', 'mean'),
            dir_hw=('dir_hw', 'mean'),
        ).reset_index()

        for idx, (iota, node) in enumerate(qkeys):
            row = agg[(agg['iota'] == iota) & (agg['node'] == node)]
            if row.empty:
                continue
            r = row.iloc[0]

            std_lo = r.phi - r.std_hw
            std_hi = r.phi + r.std_hw
            dir_lo = r.phi - r.dir_hw
            dir_hi = r.phi + r.dir_hw

            ax.fill_betweenx([idx - 0.3, idx + 0.3], std_lo, std_hi,
                             color=C_STD, alpha=0.2,
                             label='General' if idx == 0 and col == 0 else None)
            ax.plot([std_lo, std_lo], [idx - 0.3, idx + 0.3],
                    color=C_STD, lw=1, alpha=0.5)
            ax.plot([std_hi, std_hi], [idx - 0.3, idx + 0.3],
                    color=C_STD, lw=1, alpha=0.5)

            ax.fill_betweenx([idx - 0.2, idx + 0.2], dir_lo, dir_hi,
                             color=C_DIR, alpha=0.35,
                             label='Directional' if idx == 0 and col == 0 else None)
            ax.plot([dir_lo, dir_lo], [idx - 0.2, idx + 0.2],
                    color=C_DIR, lw=1, alpha=0.7)
            ax.plot([dir_hi, dir_hi], [idx - 0.2, idx + 0.2],
                    color=C_DIR, lw=1, alpha=0.7)

            sub = bdf[(bdf['iota'] == iota) & (bdf['node'] == node)]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sub))
            ax.scatter(sub['target_val'], idx + jitter, color='C3',
                       s=4, alpha=0.15, zorder=5,
                       label='Sampled targets' if idx == 0 and col == 0 else None)

            ax.scatter([r.phi], [idx], color='black', s=80, marker='|',
                       zorder=6, linewidths=1.5,
                       label='Interval centre' if idx == 0 and col == 0 else None)

        ax.set_yticks(range(nq))
        if col == 0:
            ax.set_yticklabels([ATCE_QLABELS[k] for k in qkeys], fontsize=17)
        ax.set_xlabel('Query value')

    # Column titles above panels
    for col, title in enumerate(COMBO_TITLES):
        axes[col].set_title(title, fontsize=18, pad=10)

    # Horizontal legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=16, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / 'atce_brackets.pdf')
    plt.close(fig)
    print('  atce_brackets.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 4: ATCE tightness (utilisation) — 3-panel bars (eps_test=0.5)
# ═══════════════════════════════════════════════════════════════════

def atce_tightness():
    _, df_q = load_atce_r2()

    qkeys = sorted(ATCE_QF)
    nq = len(qkeys)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

    for col, (et, etest) in enumerate(COMBOS):
        ax = axes[col]
        bdf = df_q[np.isclose(df_q['eps_train'], et)
                    & np.isclose(df_q['rho_test'], etest)].copy()

        if bdf.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        bdf['std_hw'] = np.sqrt(bdf['std_cert_q'].clip(0))
        bdf['dir_hw'] = bdf['dir_cert'].abs()
        bdf['gap_abs'] = (bdf['target_val'] - bdf['Phi_pushed']).abs()
        bdf['u_std'] = np.where(
            bdf['std_hw'] > 1e-10, bdf['gap_abs'] / bdf['std_hw'], 0.0)
        bdf['u_dir'] = np.where(
            bdf['dir_hw'] > 1e-10, bdf['gap_abs'] / bdf['dir_hw'], 0.0)

        agg = bdf.groupby(['iota', 'node']).agg(
            u_std=('u_std', 'mean'), u_dir=('u_dir', 'mean'),
        ).reset_index()

        x = np.arange(nq)
        w = 0.32
        u_s = [float(agg.loc[(agg['iota'] == i) & (agg['node'] == n),
                              'u_std'].values[0]) for i, n in qkeys]
        u_d = [float(agg.loc[(agg['iota'] == i) & (agg['node'] == n),
                              'u_dir'].values[0]) for i, n in qkeys]

        ax.bar(x - w / 2, u_s, w, color=C_STD, alpha=0.7,
               label='General' if col == 0 else None)
        ax.bar(x + w / 2, u_d, w, color=C_DIR, alpha=0.7,
               label='Directional' if col == 0 else None)
        ax.axhline(1.0, color='grey', ls='--', lw=1, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([ATCE_QLABELS[k] for k in qkeys],
                           fontsize=16, rotation=15, ha='right')

        if col == 0:
            ax.set_ylabel(r'Mean utilisation $\bar{u}$')

    # Column titles with ρ notation (both ε and η swept together)
    for col, title in enumerate(COMBO_TITLES):
        axes[col].set_title(title, fontsize=18, pad=10)

    # Horizontal legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=16, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / 'atce_tightness.pdf')
    plt.close(fig)
    print('  atce_tightness.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 5: ATCE coverage crossing — 2-panel (general / directional)
# ═══════════════════════════════════════════════════════════════════

def atce_coverage_crossing():
    _, df_q = load_atce_r2()

    rtest_vals = [0.5, 1.0, 2.0]
    cset = plt.cm.Set1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for i, rtest in enumerate(rtest_vals):
        sub = df_q[np.isclose(df_q['rho_test'], rtest)]
        c = sub.groupby('eps_train').agg(
            std_cov=('std_covered', 'mean'),
            dir_cov=('dir_covered', 'mean'),
        ).reset_index().sort_values('eps_train')
        color = cset(i / max(len(rtest_vals) - 1, 1))

        ax1.plot(c['eps_train'], c['std_cov'], 'o-', color=color, lw=2,
                 ms=6, label=r'$r_{\mathrm{test}}$' + f'$={rtest}$')
        ax2.plot(c['eps_train'], c['dir_cov'], 's-', color=color, lw=2,
                 ms=6, label=r'$r_{\mathrm{test}}$' + f'$={rtest}$')

        ax1.axvline(rtest, color=color, ls=':', lw=1.2, alpha=0.6)
        ax2.axvline(rtest, color=color, ls=':', lw=1.2, alpha=0.6)

    for ax in [ax1, ax2]:
        ax.axhline(1.0, color='grey', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel(r'$r_{\mathrm{train}}$')
        ax.set_ylabel('Coverage fraction')
        ax.legend(fontsize=15)
        ax.set_ylim(-0.05, 1.1)

    # Panel titles above instead of in-panel boxes
    ax1.set_title('General certificate', fontsize=17, pad=8)
    ax2.set_title('Directional certificate', fontsize=17, pad=8)

    fig.tight_layout()
    fig.savefig(OUT / 'atce_coverage_crossing.pdf')
    plt.close(fig)
    print('  atce_coverage_crossing.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 6: ATCE error vs rho_test — DRO crossover
# ═══════════════════════════════════════════════════════════════════

def atce_error_vs_rtest():
    import re
    df_main, _ = load_atce_r2()

    # Extract r_train from variant string
    def _extract_rtrain(v):
        m = re.search(r'eps_([0-9.]+)_eta', v)
        return float(m.group(1)) if m else None

    df_main['r_train'] = df_main['variant'].apply(_extract_rtrain)

    # Baseline (tau=I) — pick eps=0 variant (all identical)
    bl_all = pd.read_csv(RESULTS / 'atce_subfamily/atce_gaussian_z_entrywise_subfamily'
                         / 'radius_eval.csv')
    bl = bl_all[bl_all['variant'].str.contains('baseline', na=False)].copy()
    bl['r_train'] = bl['variant'].apply(_extract_rtrain)
    bl_agg = bl.groupby('rho_test')['err_subfamily'].mean().reset_index()

    # Mean err_subfamily over folds and k, per (r_train, rho_test)
    agg = df_main.groupby(['r_train', 'rho_test'])['err_subfamily'].mean().reset_index()

    # Distinct r_train values (skip 0.0 — same as baseline)
    rtrains = sorted([r for r in agg['r_train'].unique() if r > 1e-9])
    rho_tests = sorted(agg['rho_test'].unique())

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Color map for r_train lines
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(rtrains) - 1, 1)) for i in range(len(rtrains))]

    # Baseline
    ax.plot(bl_agg['rho_test'], bl_agg['err_subfamily'], 'k--', lw=2.2, ms=7,
            marker='s', label=r'Baseline ($\tau{=}I$)', zorder=4)

    # One line per r_train
    for rt, color in zip(rtrains, colors):
        sub = agg[agg['r_train'] == rt].sort_values('rho_test')
        ax.plot(sub['rho_test'], sub['err_subfamily'], '-o', color=color,
                lw=1.8, ms=6, label=r'$r_{\mathrm{train}}{=}' + f'{rt:.1f}$', zorder=3)

    ax.set_xlabel(r'$r_{\mathrm{test}}$')
    ax.set_ylabel('Transport error')
    ax.set_yscale('log')

    # Legend below figure, horizontal
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray',
               fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT / 'atce_error_vs_rtest.pdf')
    plt.close(fig)
    print('  atce_error_vs_rtest.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 7: Portland coverage vs eps
# ═══════════════════════════════════════════════════════════════════

def portland_coverage():
    df = load_portland_queries()

    per_fold = df.groupby(['eps', 'fold']).agg(
        sc=('std_covered', 'mean'),
        dc=('dir_covered', 'mean'),
    ).reset_index()

    agg = per_fold.groupby('eps').agg(
        sc_m=('sc', 'mean'),
        dc_m=('dc', 'mean'),
    ).reset_index()
    nz = agg[agg['eps'] > 1e-9].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    xmax = nz['eps'].max() + 0.5
    ax.axvspan(FANNO_W2, xmax, alpha=0.07, color='green', zorder=0)
    ax.axvline(FANNO_W2, color='gray', ls='--', lw=1.5,
               label=f'Fanno $W_2 = {FANNO_W2}$')

    ax.plot(nz['eps'], nz['dc_m'], 'o-', color=C_DIR, lw=2.2, ms=8,
            label='Directional', zorder=3)
    ax.plot(nz['eps'], nz['sc_m'], 's--', color=C_STD, lw=2.2, ms=8,
            label='General', zorder=3)

    ax.set_xlabel(r'$r_{\mathrm{train}}$')
    ax.set_ylabel('Coverage fraction')
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(0, xmax)
    ax.legend(loc='center right', framealpha=0.9, edgecolor='gray', fontsize=14)

    fig.tight_layout()
    fig.savefig(OUT / 'portland_coverage.pdf')
    plt.close(fig)
    print('  portland_coverage.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 8: Portland bracket at eps=0.2 — two panels (general + directional)
# ═══════════════════════════════════════════════════════════════════

def portland_bracket():
    df = load_portland_queries()
    sub = df[np.isclose(df['eps'], 0.2)]

    avg = sub.groupby('iota').agg(
        target=('target_value', 'mean'),
        pushed=('Phi_pushed', 'mean'),
        slo=('std_lo', 'mean'), shi=('std_hi', 'mean'),
        dlo=('dir_lo', 'mean'), dhi=('dir_hi', 'mean'),
    ).reset_index().sort_values('iota')

    n = len(avg)
    labels = [PORTLAND_LABELS.get(int(i), str(i)) for i in avg['iota']]

    fig, (ax_gen, ax_dir) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, (_, row) in enumerate(avg.iterrows()):
        # Left panel: General (blue) + nested Directional (orange)
        ax_gen.fill_betweenx([idx - 0.35, idx + 0.35], row['slo'], row['shi'],
                             color=C_STD, alpha=0.2,
                             label='General' if idx == 0 else None)
        ax_gen.plot([row['slo'], row['slo']], [idx - 0.35, idx + 0.35],
                    color=C_STD, lw=1, alpha=0.5)
        ax_gen.plot([row['shi'], row['shi']], [idx - 0.35, idx + 0.35],
                    color=C_STD, lw=1, alpha=0.5)

        ax_gen.fill_betweenx([idx - 0.25, idx + 0.25], row['dlo'], row['dhi'],
                             color=C_DIR, alpha=0.35,
                             label='Directional' if idx == 0 else None)
        ax_gen.plot([row['dlo'], row['dlo']], [idx - 0.25, idx + 0.25],
                    color=C_DIR, lw=1, alpha=0.7)
        ax_gen.plot([row['dhi'], row['dhi']], [idx - 0.25, idx + 0.25],
                    color=C_DIR, lw=1, alpha=0.7)

        ax_gen.scatter([row['target']], [idx], marker='D', color=C_TRUTH, s=55,
                       zorder=5, label='Truth' if idx == 0 else None)
        ax_gen.scatter([row['pushed']], [idx], marker='|', color=C_PUSHED, s=120,
                       zorder=5, linewidths=2,
                       label='Interval centre' if idx == 0 else None)

        # Right panel: Directional zoom only
        ax_dir.fill_betweenx([idx - 0.3, idx + 0.3], row['dlo'], row['dhi'],
                             color=C_DIR, alpha=0.35)
        ax_dir.plot([row['dlo'], row['dlo']], [idx - 0.3, idx + 0.3],
                    color=C_DIR, lw=1, alpha=0.7)
        ax_dir.plot([row['dhi'], row['dhi']], [idx - 0.3, idx + 0.3],
                    color=C_DIR, lw=1, alpha=0.7)
        ax_dir.scatter([row['target']], [idx], marker='D', color=C_TRUTH, s=55,
                       zorder=5)
        ax_dir.scatter([row['pushed']], [idx], marker='|', color=C_PUSHED, s=120,
                       zorder=5, linewidths=1.5)

    ax_gen.set_yticks(range(n))
    ax_gen.set_yticklabels(labels, fontsize=18)
    ax_gen.set_xlabel('Query value')
    ax_gen.set_title('Full view (General + Directional)', fontsize=20, pad=10)

    ax_dir.set_xlabel('Query value')
    ax_dir.set_title('Directional (zoom)', fontsize=20, pad=10)

    # Legend below figure, horizontal
    handles, labels_leg = ax_gen.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=len(labels_leg),
               fontsize=16, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT / 'portland_bracket.pdf')
    plt.close(fig)
    print('  portland_bracket.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 9: Portland width curves — certificate width vs r_train
# ═══════════════════════════════════════════════════════════════════

def portland_width_curves():
    df = load_portland_queries()
    nz = df[df['eps'] > 1e-9]

    agg = nz.groupby('eps').agg(
        sw=('std_width', 'mean'),
        dw=('dir_width', 'mean'),
    ).reset_index().sort_values('eps')

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(agg['eps'], agg['sw'], 's-', color=C_STD, lw=2.2, ms=8,
            label='General')
    ax.plot(agg['eps'], agg['dw'], 'o-', color=C_DIR, lw=2.2, ms=8,
            label='Directional')

    ax.axhline(PORTLAND_VACUITY, color='gray', ls=':', lw=2, alpha=0.7,
               label=f'Vacuity ($10 \\times$ QS $= {PORTLAND_VACUITY:.2f}$)')

    ax.set_xlabel(r'$r_{\mathrm{train}}$')
    ax.set_ylabel('Mean certificate width')
    ax.set_yscale('log')

    ax.legend(framealpha=0.9, edgecolor='gray', fontsize=13)

    fig.tight_layout()
    fig.savefig(OUT / 'portland_width_curves.pdf')
    plt.close(fig)
    print('  portland_width_curves.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 10: Portland shaded area — intervals vs r_train
# ═══════════════════════════════════════════════════════════════════

def portland_shaded():
    df = load_portland_queries()
    nz = df[df['eps'] > 1e-9]

    agg = nz.groupby(['eps', 'iota']).agg(
        slo=('std_lo', 'mean'), shi=('std_hi', 'mean'),
        dlo=('dir_lo', 'mean'), dhi=('dir_hi', 'mean'),
        target=('target_value', 'first'),
        phi=('Phi_pushed', 'mean'),
    ).reset_index()

    queries = [(1, PORTLAND_LABELS[1]),
               (2, PORTLAND_LABELS[2]),
               (3, PORTLAND_LABELS[3]),
               (4, PORTLAND_LABELS[4])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    for panel_idx, (iota, qlabel) in enumerate(queries):
        row_idx, col_idx = divmod(panel_idx, 2)
        ax = axes[row_idx, col_idx]
        sub = agg[agg['iota'] == iota].sort_values('eps')
        eps = sub['eps'].values
        truth = sub['target'].iloc[0]

        # General band (blue)
        ax.fill_between(eps, sub['slo'].values, sub['shi'].values,
                        color=C_STD, alpha=0.20,
                        label='General' if panel_idx == 0 else None)
        ax.plot(eps, sub['slo'].values, color=C_STD, lw=1, alpha=0.5)
        ax.plot(eps, sub['shi'].values, color=C_STD, lw=1, alpha=0.5)

        # Directional band (orange, nested)
        ax.fill_between(eps, sub['dlo'].values, sub['dhi'].values,
                        color=C_DIR, alpha=0.35,
                        label='Directional' if panel_idx == 0 else None)
        ax.plot(eps, sub['dlo'].values, color=C_DIR, lw=1.2, alpha=0.7)
        ax.plot(eps, sub['dhi'].values, color=C_DIR, lw=1.2, alpha=0.7)

        # Truth
        ax.axhline(truth, color=C_TRUTH, ls='--', lw=2, zorder=5,
                   label='Truth' if panel_idx == 0 else None)

        # Interval centre
        ax.plot(eps, sub['phi'].values, color=C_PUSHED, marker='|', ms=8,
                lw=0, zorder=6,
                label='Interval centre' if panel_idx == 0 else None)

        # Coverage onset vertical line
        ax.axvline(PORTLAND_DIR_COVERAGE_ONSET, color=C_DIR, ls='--', lw=1.5, alpha=0.6,
                   label='Dir. covers all' if panel_idx == 0 else None)

        if row_idx == 1:
            ax.set_xlabel(r'$r_{\mathrm{train}}$')
        ax.set_title(qlabel, fontsize=17, pad=8)
        if col_idx == 0:
            ax.set_ylabel('Query value')

    # Horizontal legend below
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=len(labels_leg),
               fontsize=14, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT / 'portland_shaded.pdf')
    plt.close(fig)
    print('  portland_shaded.pdf')


# ═══════════════════════════════════════════════════════════════════
# Figure 11: Portland obs_distance vs r_train
# ═══════════════════════════════════════════════════════════════════

def portland_obs_distance():
    df = load_portland_queries()
    nz = df[df['eps'] > 1e-9]

    # obs_distance per (eps, fold) — take first per group (same across iotas)
    per_fold = nz.groupby(['eps', 'fold'])['obs_distance'].first().reset_index()
    agg = per_fold.groupby('eps').agg(
        mean=('obs_distance', 'mean'),
        std=('obs_distance', 'std'),
    ).reset_index().sort_values('eps')

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(agg['eps'], agg['mean'], yerr=agg['std'],
                fmt='o-', color=C_STD, lw=2.2, ms=8, capsize=5)

    ax.set_xlabel(r'$r_{\mathrm{train}}$')
    ax.set_ylabel(r'Observational distance ($W_2^2$)')

    fig.tight_layout()
    fig.savefig(OUT / 'portland_obs_distance.pdf')
    plt.close(fig)
    print('  portland_obs_distance.pdf')



# ═══════════════════════════════════════════════════════════════════
# LiLuCaS constants and loader
# ═══════════════════════════════════════════════════════════════════

import re as _re

LILUCAS_QF = [(1, 3), (2, 3), (3, 3), (4, 3)]
LILUCAS_QLABELS = {
    (1, 3): r'$\mathbb{E}[\mathrm{LC} \mid do(\mathrm{Sm}{=}0)]$',
    (2, 3): r'$\mathbb{E}[\mathrm{LC} \mid do(\mathrm{Sm}{=}1)]$',
    (3, 3): r'$\mathbb{E}[\mathrm{LC} \mid do(\mathrm{Ge}{=}0)]$',
    (4, 3): r'$\mathbb{E}[\mathrm{LC} \mid do(\mathrm{Ge}{=}1)]$',
}
LILUCAS_QLABELS_SHORT = {
    (1, 3): 'do(Sm=0)', (2, 3): 'do(Sm=1)',
    (3, 3): 'do(Ge=0)', (4, 3): 'do(Ge=1)',
}

LILUCAS_CONFIGS = {
    'ew_sym': ('Symmetric',
               'lilucas_ew_sub/lilucas_light_entrywise_subfamily'),
    'ew_dir': ('Offset',
               'lilucas_dir_sub/lilucas_light_entrywise_subfamily_directional'),
    'gau_sym': ('Gaussian',
                'lilucas_gew_sub/lilucas_light_gaussian_entrywise_subfamily'),
}

LILUCAS_COMBOS = [(0.2, 0.5), (0.5, 0.5), (1.0, 0.5)]
LILUCAS_COMBO_TITLES = [
    r'$r_{\mathrm{train}}{=}0.2 < r_{\mathrm{test}}{=}0.5$',
    r'$r_{\mathrm{train}}{=}r_{\mathrm{test}}{=}0.5$',
    r'$r_{\mathrm{train}}{=}1.0 > r_{\mathrm{test}}{=}0.5$',
]


def _extract_rtrain(v):
    m = _re.search(r'eps_([0-9.]+)_eta', v)
    return float(m.group(1)) if m else None


def load_lilucas(config_key):
    """Return (df_main, df_q) for a LiLuCaS config, filtered to subfamily queries."""
    _, subdir = LILUCAS_CONFIGS[config_key]
    df_main = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
    df_q = pd.read_csv(RESULTS / subdir / 'radius_eval_queries.csv')

    qf_set = set(LILUCAS_QF)
    mask = df_q.apply(lambda r: (int(r['iota']), int(r['node'])) in qf_set, axis=1)
    df_q = df_q[mask].copy()

    vi = df_main[['variant', 'eps_train', 'eta_train']].drop_duplicates()
    df_q = df_q.merge(vi, on='variant', how='left')

    df_main = df_main[~df_main['variant'].str.contains('baseline', na=False)]
    df_q = df_q[~df_q['variant'].str.contains('baseline', na=False)]

    return df_main, df_q


# ═══════════════════════════════════════════════════════════════════
# 1. ew-dir vs ew-sym transport comparison
# ═══════════════════════════════════════════════════════════════════

def lilucas_ew_transport():
    """1x2 error comparison (Symmetric vs Offset) + combined DRO table."""
    fig, (ax_sym, ax_off) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for ax, key, title in [(ax_sym, 'ew_sym', 'Symmetric'),
                           (ax_off, 'ew_dir', 'Offset')]:
        _, subdir = LILUCAS_CONFIGS[key]
        df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
        df['r_train'] = df['variant'].apply(_extract_rtrain)

        bl = df[df['variant'].str.contains('baseline', na=False)]
        tr = df[~df['variant'].str.contains('baseline', na=False)]

        bl_agg = bl.groupby('rho_test')['err_subfamily'].mean().reset_index()
        tr_agg = tr.groupby(['r_train', 'rho_test'])['err_subfamily'].mean().reset_index()

        rtrains = sorted([r for r in tr_agg['r_train'].unique() if r > 1e-9])
        cmap = plt.cm.viridis
        colors = [cmap(i / max(len(rtrains) - 1, 1)) for i in range(len(rtrains))]

        ax.plot(bl_agg['rho_test'], bl_agg['err_subfamily'], 'k--', lw=2.2, ms=7,
                marker='s', label=r'Baseline ($\tau{=}I$)', zorder=4)

        for rt, c in zip(rtrains, colors):
            sub = tr_agg[tr_agg['r_train'] == rt].sort_values('rho_test')
            ax.plot(sub['rho_test'], sub['err_subfamily'], '-o', color=c,
                    lw=1.8, ms=6,
                    label=r'$r_{\mathrm{train}}{=}' + f'{rt:.1f}$', zorder=3)

        ax.set_xlabel(r'$r_{\mathrm{test}}$')
        ax.set_yscale('log')
        ax.set_title(title, fontsize=18, pad=8)

    ax_sym.set_ylabel('Transport error')

    handles, labels = ax_sym.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray',
               fontsize=13)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT / 'lilucas_ew_transport_comparison.pdf')
    plt.close(fig)
    print('  lilucas_ew_transport_comparison.pdf')

    _lilucas_ew_dro_table()


def _lilucas_ew_dro_table():
    """Side-by-side DRO table: Symmetric | Offset, bold tied-best per column."""
    from scipy import stats as _stats

    test_regimes = [0.2, 4.0]

    def _k_err(d, rtest):
        return d[np.isclose(d['rho_test'], rtest)].groupby('k')['err_subfamily'].mean()

    # Load both configs — keep per-k series for paired tests
    variant_data = {}
    for key, vlabel in [('ew_sym', 'Symmetric'), ('ew_dir', 'Offset')]:
        _, subdir = LILUCAS_CONFIGS[key]
        df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
        df['r_train'] = df['variant'].apply(_extract_rtrain)
        bl = df[df['variant'].str.contains('baseline', na=False)].copy()
        tr = df[~df['variant'].str.contains('baseline', na=False)].copy()
        rtrains = sorted([r for r in tr['r_train'].unique() if r > 1e-9])

        # Store per-k series, means, stds
        k_series = {}   # (rtest,) -> series for baseline; (rtest, rt) -> series for trained
        bl_means, bl_stds = {}, {}
        for rtest in test_regimes:
            be = _k_err(bl[np.isclose(bl['r_train'], 0.2)], rtest)
            bl_means[rtest] = be.mean()
            bl_stds[rtest] = be.std(ddof=1)
            k_series[(rtest,)] = be

        rows = {}
        for rt in rtrains:
            rows[rt] = {}
            for rtest in test_regimes:
                te = _k_err(tr[np.isclose(tr['r_train'], rt)], rtest)
                m, s = te.mean(), te.std(ddof=1)
                rows[rt][rtest] = dict(m=m, s=s)
                k_series[(rtest, rt)] = te
                print(f'    {vlabel} r={rt:.1f} rtest={rtest}: err={m:.3f}')

        variant_data[key] = dict(bl_means=bl_means, bl_stds=bl_stds,
                                 rows=rows, rtrains=rtrains, k_series=k_series)

    rtrains = variant_data['ew_sym']['rtrains']

    # For each column, find the minimum and bold all entries not significantly
    # worse than the minimum (paired t-test, p > 0.05).
    col_bold = {}  # (vk, rtest, row_label) -> bool
    for vk in ['ew_sym', 'ew_dir']:
        vd = variant_data[vk]
        for rtest in test_regimes:
            # Collect all (label, mean, k_series_key)
            entries = [('baseline', vd['bl_means'][rtest], (rtest,))]
            for rt in rtrains:
                entries.append((rt, vd['rows'][rt][rtest]['m'], (rtest, rt)))
            # Find the argmin
            best_idx = min(range(len(entries)), key=lambda i: entries[i][1])
            best_series = vd['k_series'][entries[best_idx][2]]
            # Test each entry against the best
            for i, (label, m, ks_key) in enumerate(entries):
                if i == best_idx:
                    col_bold[(vk, rtest, label)] = True
                else:
                    this_series = vd['k_series'][ks_key]
                    sk = best_series.index.intersection(this_series.index)
                    _, p = _stats.ttest_rel(this_series.loc[sk].values,
                                            best_series.loc[sk].values)
                    col_bold[(vk, rtest, label)] = (p > 0.05)

    def _fmt_err(m, s, bold):
        if bold:
            return f'$\\mathbf{{{m:.3f} \\pm {s:.3f}}}$'
        return f'${m:.3f} \\pm {s:.3f}$'

    lines = [
        r'% DRO crossover -- LiLuCaS Symmetric vs Offset (side-by-side)',
        r'% Bold = not significantly worse than column best (paired t, p>0.05)',
        '',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{LiLuCaS transport error under symmetric and offset selectors.',
        r'Each cell shows mean $\pm$ std over $n{=}100$ perturbation seeds $k$',
        r'(fold-averaged). Bold marks entries not significantly worse than the',
        r'column best (paired $t$-test, $p > 0.05$).',
        r'The baselines differ because the offset selector centres',
        r'its ambiguity set at $\delta{=}0.5$.}',
        r'\label{tab:dro-lilucas-ew}',
        r'\footnotesize',
        r'\begin{tabular}{l cc cc}',
        r'\toprule',
        r'& \multicolumn{2}{c}{Symmetric}',
        r'& \multicolumn{2}{c}{Offset} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
        r'$r_{\mathrm{train}}$ & $r_{\mathrm{test}}{=}0.2$'
        r' & $r_{\mathrm{test}}{=}4.0$'
        r' & $r_{\mathrm{test}}{=}0.2$'
        r' & $r_{\mathrm{test}}{=}4.0$ \\',
        r'\midrule',
    ]

    # Baseline row
    bl_cells = [r'Baseline ($\tau{=}I$)']
    for key in ['ew_sym', 'ew_dir']:
        vd = variant_data[key]
        for rtest in test_regimes:
            m, s = vd['bl_means'][rtest], vd['bl_stds'][rtest]
            bold = col_bold[(key, rtest, 'baseline')]
            bl_cells.append(_fmt_err(m, s, bold))
    lines.append('  ' + ' & '.join(bl_cells) + r' \\')

    # TraCA rows
    for rt in rtrains:
        cells = ['$r_{\\mathrm{train}}{=}' + f'{rt:.1f}$']
        for key in ['ew_sym', 'ew_dir']:
            vd = variant_data[key]
            for rtest in test_regimes:
                r = vd['rows'][rt][rtest]
                bold = col_bold[(key, rtest, rt)]
                cells.append(_fmt_err(r['m'], r['s'], bold))
        lines.append('  ' + ' & '.join(cells) + r' \\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    (OUT / 'lilucas_ew_dro_comparison.tex').write_text('\n'.join(lines) + '\n')
    print('  lilucas_ew_dro_comparison.tex')


# ═══════════════════════════════════════════════════════════════════
# Common bracket + tightness helpers
# ═══════════════════════════════════════════════════════════════════

def _lilucas_brackets_impl(config_key, filename, combos, combo_titles):
    """Bracket plot with per-panel diagnostics: category, coverage, per-query widths."""
    display_name, _ = LILUCAS_CONFIGS[config_key]
    _, df_q = load_lilucas(config_key)

    qs = df_q['target_val'].abs().mean()
    vacuity = 10 * qs
    qkeys = LILUCAS_QF
    nq = len(qkeys)

    ncols = len(combos)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    print(f'\n  [{display_name}] QS={qs:.4f}, vacuity={vacuity:.4f}')

    for col, (et, etest) in enumerate(combos):
        ax = axes[col]
        bdf = df_q[np.isclose(df_q['eps_train'], et)
                    & np.isclose(df_q['rho_test'], etest)].copy()

        if bdf.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            print(f'    Panel {col} (r_train={et}, r_test={etest}): NO DATA')
            continue

        bdf['std_hw'] = np.sqrt(bdf['std_cert_q'].clip(0))
        bdf['dir_hw'] = bdf['dir_cert'].abs()

        # Coverage across all (k, fold, query) in this panel
        gen_cov = bdf['std_covered'].mean()
        dir_cov = bdf['dir_covered'].mean()

        agg = bdf.groupby(['iota', 'node']).agg(
            phi=('Phi_pushed', 'mean'),
            std_hw=('std_hw', 'mean'),
            dir_hw=('dir_hw', 'mean'),
        ).reset_index()

        # Per-query widths
        q_std_w, q_dir_w = {}, {}
        for iota, node in qkeys:
            row = agg[(agg['iota'] == iota) & (agg['node'] == node)]
            if not row.empty:
                q_std_w[(iota, node)] = 2 * row.iloc[0].std_hw
                q_dir_w[(iota, node)] = 2 * row.iloc[0].dir_hw

        mean_std_w = np.mean(list(q_std_w.values()))
        mean_dir_w = np.mean(list(q_dir_w.values()))
        std_inf = mean_std_w < vacuity
        dir_inf = mean_dir_w < vacuity

        if std_inf and dir_inf:
            ratio = mean_std_w / mean_dir_w if mean_dir_w > 1e-10 else float('inf')
            cat = f'(a) both informative, {ratio:.1f}x'
        elif dir_inf:
            cat = f'(b) dir informative / gen vacuous'
        else:
            cat = f'(c) both vacuous'

        print(f'    Panel {col} (r_train={et}, r_test={etest}): {cat}')
        print(f'      gen_width={mean_std_w:.3f}, dir_width={mean_dir_w:.3f}')
        print(f'      gen_coverage={gen_cov:.4f}, dir_coverage={dir_cov:.4f}')
        for k in qkeys:
            if k in q_std_w:
                sw, dw = q_std_w[k], q_dir_w[k]
                rq = sw / dw if dw > 1e-10 else float('inf')
                print(f'        {LILUCAS_QLABELS_SHORT[k]}: '
                      f'gen_w={sw:.3f}, dir_w={dw:.3f}, ratio={rq:.1f}x')

        # Draw brackets
        for idx, (iota, node) in enumerate(qkeys):
            row = agg[(agg['iota'] == iota) & (agg['node'] == node)]
            if row.empty:
                continue
            r = row.iloc[0]
            slo, shi = r.phi - r.std_hw, r.phi + r.std_hw
            dlo, dhi = r.phi - r.dir_hw, r.phi + r.dir_hw

            ax.fill_betweenx([idx - 0.3, idx + 0.3], slo, shi,
                             color=C_STD, alpha=0.2,
                             label='General' if idx == 0 and col == 0 else None)
            ax.plot([slo] * 2, [idx - 0.3, idx + 0.3], color=C_STD, lw=1, alpha=0.5)
            ax.plot([shi] * 2, [idx - 0.3, idx + 0.3], color=C_STD, lw=1, alpha=0.5)

            ax.fill_betweenx([idx - 0.2, idx + 0.2], dlo, dhi,
                             color=C_DIR, alpha=0.35,
                             label='Directional' if idx == 0 and col == 0 else None)
            ax.plot([dlo] * 2, [idx - 0.2, idx + 0.2], color=C_DIR, lw=1, alpha=0.7)
            ax.plot([dhi] * 2, [idx - 0.2, idx + 0.2], color=C_DIR, lw=1, alpha=0.7)

            sub = bdf[(bdf['iota'] == iota) & (bdf['node'] == node)]
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sub))
            ax.scatter(sub['target_val'], idx + jitter, color='C3', s=4, alpha=0.15,
                       zorder=5,
                       label='Sampled targets' if idx == 0 and col == 0 else None)

            ax.scatter([r.phi], [idx], color='black', s=80, marker='|',
                       zorder=6, linewidths=1.5,
                       label='Interval centre' if idx == 0 and col == 0 else None)

        ax.set_yticks(range(nq))
        if col == 0:
            ax.set_yticklabels([LILUCAS_QLABELS[k] for k in qkeys], fontsize=15)
        ax.set_xlabel('Query value')

    for col, t in enumerate(combo_titles):
        axes[col].set_title(t, fontsize=18, pad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=16, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(f'  {filename}')


def _lilucas_tightness_impl(config_key, filename, combos, combo_titles):
    """Tightness (utilisation) bar chart."""
    _, df_q = load_lilucas(config_key)
    qkeys = LILUCAS_QF
    nq = len(qkeys)

    ncols = len(combos)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for col, (et, etest) in enumerate(combos):
        ax = axes[col]
        bdf = df_q[np.isclose(df_q['eps_train'], et)
                    & np.isclose(df_q['rho_test'], etest)].copy()
        if bdf.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        bdf['std_hw'] = np.sqrt(bdf['std_cert_q'].clip(0))
        bdf['dir_hw'] = bdf['dir_cert'].abs()
        bdf['gap_abs'] = (bdf['target_val'] - bdf['Phi_pushed']).abs()
        bdf['u_std'] = np.where(
            bdf['std_hw'] > 1e-10, bdf['gap_abs'] / bdf['std_hw'], 0.0)
        bdf['u_dir'] = np.where(
            bdf['dir_hw'] > 1e-10, bdf['gap_abs'] / bdf['dir_hw'], 0.0)

        agg = bdf.groupby(['iota', 'node']).agg(
            u_std=('u_std', 'mean'), u_dir=('u_dir', 'mean'),
        ).reset_index()

        x = np.arange(nq)
        w = 0.32
        u_s = [float(agg.loc[(agg['iota'] == i) & (agg['node'] == n),
                              'u_std'].values[0]) for i, n in qkeys]
        u_d = [float(agg.loc[(agg['iota'] == i) & (agg['node'] == n),
                              'u_dir'].values[0]) for i, n in qkeys]

        ax.bar(x - w / 2, u_s, w, color=C_STD, alpha=0.7,
               label='General' if col == 0 else None)
        ax.bar(x + w / 2, u_d, w, color=C_DIR, alpha=0.7,
               label='Directional' if col == 0 else None)
        ax.axhline(1.0, color='grey', ls='--', lw=1, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([LILUCAS_QLABELS[k] for k in qkeys],
                           fontsize=14, rotation=20, ha='right')
        if col == 0:
            ax.set_ylabel(r'Mean utilisation $\bar{u}$')

    for col, t in enumerate(combo_titles):
        axes[col].set_title(t, fontsize=18, pad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=16, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(f'  {filename}')


# ═══════════════════════════════════════════════════════════════════
# 2. ew-sym brackets + tightness
# ═══════════════════════════════════════════════════════════════════

def lilucas_ewsym_brackets():
    _lilucas_brackets_impl('ew_sym', 'lilucas_ewsym_brackets.pdf',
                           LILUCAS_COMBOS, LILUCAS_COMBO_TITLES)


def lilucas_ewsym_tightness():
    _lilucas_tightness_impl('ew_sym', 'lilucas_ewsym_tightness.pdf',
                            LILUCAS_COMBOS, LILUCAS_COMBO_TITLES)


# ═══════════════════════════════════════════════════════════════════
# 3. gau-sym error + DRO + brackets + tightness
# ═══════════════════════════════════════════════════════════════════

def lilucas_gau_error():
    """Error vs r_test + DRO table for Gaussian-Symmetric."""
    _, subdir = LILUCAS_CONFIGS['gau_sym']
    df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
    df['r_train'] = df['variant'].apply(_extract_rtrain)

    bl = df[df['variant'].str.contains('baseline', na=False)]
    tr = df[~df['variant'].str.contains('baseline', na=False)]

    bl_agg = bl.groupby('rho_test')['err_subfamily'].mean().reset_index()
    tr_agg = tr.groupby(['r_train', 'rho_test'])['err_subfamily'].mean().reset_index()

    rtrains = sorted([r for r in tr_agg['r_train'].unique() if r > 1e-9])
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(rtrains) - 1, 1)) for i in range(len(rtrains))]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(bl_agg['rho_test'], bl_agg['err_subfamily'], 'k--', lw=2.2, ms=7,
            marker='s', label=r'Baseline ($\tau{=}I$)', zorder=4)

    for rt, c in zip(rtrains, colors):
        sub = tr_agg[tr_agg['r_train'] == rt].sort_values('rho_test')
        ax.plot(sub['rho_test'], sub['err_subfamily'], '-o', color=c,
                lw=1.8, ms=6,
                label=r'$r_{\mathrm{train}}{=}' + f'{rt:.1f}$', zorder=3)

    ax.set_xlabel(r'$r_{\mathrm{test}}$')
    ax.set_ylabel('Transport error')
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray',
               fontsize=13)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT / 'lilucas_gau_error.pdf')
    plt.close(fig)
    print('  lilucas_gau_error.pdf')

    _lilucas_gau_dro_table()


def _lilucas_gau_dro_table():
    """DRO table for Gaussian-Symmetric, bold tied-best per column."""
    from scipy import stats as _stats

    _, subdir = LILUCAS_CONFIGS['gau_sym']
    df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
    df['r_train'] = df['variant'].apply(_extract_rtrain)

    bl = df[df['variant'].str.contains('baseline', na=False)].copy()
    tr = df[~df['variant'].str.contains('baseline', na=False)].copy()

    test_regimes = [0.2, 4.0]
    rtrains = sorted([r for r in tr['r_train'].unique() if r > 1e-9])

    def _k_err(d, rtest):
        return d[np.isclose(d['rho_test'], rtest)].groupby('k')['err_subfamily'].mean()

    # Store per-k series for paired tests
    k_series = {}
    bl_means, bl_stds = {}, {}
    for rtest in test_regimes:
        be = _k_err(bl[np.isclose(bl['r_train'], 0.2)], rtest)
        bl_means[rtest] = be.mean()
        bl_stds[rtest] = be.std(ddof=1)
        k_series[(rtest,)] = be

    rows = {}
    for rt in rtrains:
        rows[rt] = {}
        for rtest in test_regimes:
            te = _k_err(tr[np.isclose(tr['r_train'], rt)], rtest)
            m, s = te.mean(), te.std(ddof=1)
            rows[rt][rtest] = dict(m=m, s=s)
            k_series[(rtest, rt)] = te
            print(f'    r={rt:.1f} rtest={rtest}: err={m:.3f}')

    # Tied-bold: bold column min AND entries not significantly worse (paired t, p>0.05)
    col_bold = {}
    for rtest in test_regimes:
        entries = [('baseline', bl_means[rtest], (rtest,))]
        for rt in rtrains:
            entries.append((rt, rows[rt][rtest]['m'], (rtest, rt)))
        best_idx = min(range(len(entries)), key=lambda i: entries[i][1])
        best_series = k_series[entries[best_idx][2]]
        for i, (label, m, ks_key) in enumerate(entries):
            if i == best_idx:
                col_bold[(rtest, label)] = True
            else:
                this_series = k_series[ks_key]
                sk = best_series.index.intersection(this_series.index)
                _, p = _stats.ttest_rel(this_series.loc[sk].values,
                                        best_series.loc[sk].values)
                col_bold[(rtest, label)] = (p > 0.05)
                print(f'    rtest={rtest} {label} vs best: p={p:.4f} -> '
                      f'{"BOLD" if p > 0.05 else "not bold"}')

    def _fmt_err(m, s, bold):
        if bold:
            return f'$\\mathbf{{{m:.3f} \\pm {s:.3f}}}$'
        return f'${m:.3f} \\pm {s:.3f}$'

    lines = [
        r'% DRO crossover -- LiLuCaS Gaussian',
        r'% Bold = not significantly worse than column best (paired t, p>0.05)',
        '',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{LiLuCaS (Gaussian): transport error at small and large test',
        r'perturbations. Each cell shows mean $\pm$ std over $n{=}100$ perturbation',
        r'seeds $k$ (fold-averaged). Bold marks entries not significantly worse than the',
        r'column best (paired $t$-test, $p > 0.05$).}',
        r'\label{tab:dro-lilucas-gau}',
        r'\small',
        r'\begin{tabular}{l c c}',
        r'\toprule',
        r'& $r_{\mathrm{test}} = 0.2$',
        r'& $r_{\mathrm{test}} = 4.0$ \\',
        r'\cmidrule(lr){2-2} \cmidrule(lr){3-3}',
        r'Method & Error & Error \\',
        r'\midrule',
    ]

    # Baseline row
    bl_cells = [r'Baseline ($\tau{=}I$)']
    for rtest in test_regimes:
        m, s = bl_means[rtest], bl_stds[rtest]
        bold = col_bold[(rtest, 'baseline')]
        bl_cells.append(_fmt_err(m, s, bold))
    lines.append('  ' + ' & '.join(bl_cells) + r' \\')

    # TraCA rows
    for rt in rtrains:
        cells = ['$r_{\\mathrm{train}} = ' + f'{rt:.1f}$']
        for rtest in test_regimes:
            r = rows[rt][rtest]
            bold = col_bold[(rtest, rt)]
            cells.append(_fmt_err(r['m'], r['s'], bold))
        lines.append('  ' + ' & '.join(cells) + r' \\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    (OUT / 'lilucas_gau_dro.tex').write_text('\n'.join(lines) + '\n')
    print('  lilucas_gau_dro.tex')


def lilucas_gau_brackets():
    _lilucas_brackets_impl('gau_sym', 'lilucas_gau_brackets.pdf',
                           LILUCAS_COMBOS, LILUCAS_COMBO_TITLES)


def lilucas_gau_tightness():
    _lilucas_tightness_impl('gau_sym', 'lilucas_gau_tightness.pdf',
                            LILUCAS_COMBOS, LILUCAS_COMBO_TITLES)


def _lilucas_width_curves_impl(config_key, filename):
    """Certificate width vs r_train (log-y) with vacuity threshold."""
    display_name, subdir = LILUCAS_CONFIGS[config_key]
    df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
    tr = df[~df['variant'].str.contains('baseline', na=False)]
    tr = tr[tr['eps_train'] > 1e-9]

    # QS from queries CSV (consistent with brackets)
    _, df_q = load_lilucas(config_key)
    qs = df_q['target_val'].abs().mean()
    vacuity = 10 * qs

    # Family-average widths per r_train (widths don't depend on rho_test)
    agg = tr.groupby('eps_train').agg(
        sw=('mean_width', 'mean'),
        dw=('mean_dir_width', 'mean'),
    ).reset_index().sort_values('eps_train')

    rtrains = agg['eps_train'].values
    gen_w = agg['sw'].values
    dir_w = agg['dw'].values

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(rtrains, gen_w, 's-', color=C_STD, lw=2.2, ms=8, label='General')
    ax.plot(rtrains, dir_w, 'o-', color=C_DIR, lw=2.2, ms=8, label='Directional')

    ax.axhline(vacuity, color='gray', ls=':', lw=2, alpha=0.7,
               label=f'Vacuity ($10 \\times$ QS $= {vacuity:.1f}$)')

    ax.set_xlabel(r'$r_{\mathrm{train}}$')
    ax.set_ylabel('Mean certificate width')
    ax.set_yscale('log')
    ax.set_xticks(rtrains)
    ax.set_xticklabels([f'{r:.1f}' for r in rtrains])

    ax.legend(framealpha=0.9, edgecolor='gray', fontsize=13)

    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(f'  {filename}')

    # Report crossings
    print(f'    [{display_name}] QS={qs:.4f}, vacuity={vacuity:.1f}')
    for i, rt in enumerate(rtrains):
        gv = 'informative' if gen_w[i] < vacuity else 'VACUOUS'
        dv = 'informative' if dir_w[i] < vacuity else 'VACUOUS'
        print(f'    r_train={rt:.1f}  gen={gen_w[i]:.2f} ({gv})  '
              f'dir={dir_w[i]:.2f} ({dv})')


def lilucas_gau_width_curves():
    _lilucas_width_curves_impl('gau_sym', 'lilucas_gau_width_curves.pdf')


def lilucas_ewsym_width_curves():
    _lilucas_width_curves_impl('ew_sym', 'lilucas_ewsym_width_curves.pdf')


def lilucas_all():
    """Generate all LiLuCaS figures."""
    print('\n--- LiLuCaS: Empirical Symmetric vs Offset ---')
    lilucas_ew_transport()
    print('\n--- LiLuCaS: Symmetric brackets + tightness ---')
    lilucas_ewsym_brackets()
    lilucas_ewsym_tightness()
    print('\n--- LiLuCaS: Gaussian ---')
    lilucas_gau_error()
    lilucas_gau_brackets()
    lilucas_gau_tightness()
    print('\n--- LiLuCaS: Width curves ---')
    lilucas_gau_width_curves()
    lilucas_ewsym_width_curves()





# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print('Generating finalized main-text figures...')
    ate_heatmaps()
    ate_width_curves()
    ate_intervals_diagonal()
    ate_diagonal_shaded()
    atce_brackets()
    atce_tightness()
    atce_coverage_crossing()
    atce_error_vs_rtest()
    portland_coverage()
    portland_bracket()
    portland_width_curves()
    portland_shaded()
    portland_obs_distance()
    lilucas_all()
    print(f'\nDone. {len(list(OUT.glob("*.pdf")))} PDFs in {OUT}/')


if __name__ == '__main__':
    main()
