#!/usr/bin/env python3
"""Generate linear-y-scale variants of all log-y figures.

For each figure that uses a log y-axis in generate_final.py, this script
produces a linear-scale version with the same data and styling, saved with
a _linear suffix.  The originals are not modified.

Run:  conda run -n traca-run python traca_figures/generate_linear_variants.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import re as _re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global style (same as generate_final.py) ─────────────────────
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
RESULTS = ROOT / 'results_production'

# ── Colors & constants ───────────────────────────────────────────
C_STD = '#4878CF'
C_DIR = '#EE854A'
C_TRUTH = 'black'

ATE_QS = 1.05
ATE_VACUITY = 10 * ATE_QS
PORTLAND_QS = 0.1632
PORTLAND_VACUITY = 10 * PORTLAND_QS

ATE_QF = [(1, 1), (2, 1)]
ATCE_QF = {(1, 2), (2, 2)}
PORTLAND_QF = [(1, 3), (2, 3), (3, 3), (4, 3)]
LILUCAS_QF = [(1, 3), (2, 3), (3, 3), (4, 3)]

ATE_QLABELS = {
    1: r'$\mathbb{E}[Y \mid do(X{=}0)]$',
    2: r'$\mathbb{E}[Y \mid do(X{=}1)]$',
}

LILUCAS_CONFIGS = {
    'ew_sym': ('Symmetric',
               'lilucas_ew_sub/lilucas_light_entrywise_subfamily'),
    'ew_dir': ('Offset',
               'lilucas_dir_sub/lilucas_light_entrywise_subfamily_directional'),
    'gau_sym': ('Gaussian',
                'lilucas_gew_sub/lilucas_light_gaussian_entrywise_subfamily'),
}


# ── Data loading (same as generate_final.py) ─────────────────────

def _filter_qf(df, qf_pairs):
    s = set(qf_pairs)
    mask = df.apply(lambda r: (int(r['iota']), int(r['node'])) in s, axis=1)
    return df[mask].copy()


def load_ate(variant):
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


def load_lilucas(config_key):
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


def _extract_rtrain(v):
    m = _re.search(r'eps_([0-9.]+)_eta', v)
    return float(m.group(1)) if m else None


# ══════════════════════════════════════════════════════════════════
# 1. ate_width_curves_linear.pdf
# ══════════════════════════════════════════════════════════════════

def ate_width_curves_linear():
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

    queries = [(1, ATE_QLABELS[1]), (2, ATE_QLABELS[2])]

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

        ax.axhline(ATE_VACUITY, color='gray', ls=':', lw=2, alpha=0.7,
                   label=f'Vacuity ($10 \\times$ QS $= {ATE_VACUITY:.1f}$)',
                   zorder=2)

        # LINEAR scale
        ax.set_xlabel(r'$r_{\mathrm{train}}$')
        ax.set_xticks([0.2, 0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(['0.2', '0.5', '1.0', '2.0', '4.0'])
        ax.set_title(qlabel, fontsize=18, pad=10)

    axes[0].set_ylabel('Certificate width')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=13, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / 'ate_width_curves_linear.pdf')
    plt.close(fig)
    print('  ate_width_curves_linear.pdf')


# ══════════════════════════════════════════════════════════════════
# 2. atce_error_vs_rtest_linear.pdf
# ══════════════════════════════════════════════════════════════════

def atce_error_vs_rtest_linear():
    df_main, _ = load_atce_r2()

    df_main['r_train'] = df_main['variant'].apply(_extract_rtrain)

    bl_all = pd.read_csv(RESULTS / 'atce_subfamily/atce_gaussian_z_entrywise_subfamily'
                         / 'radius_eval.csv')
    bl = bl_all[bl_all['variant'].str.contains('baseline', na=False)].copy()
    bl['r_train'] = bl['variant'].apply(_extract_rtrain)
    bl_agg = bl.groupby('rho_test')['err_subfamily'].mean().reset_index()

    agg = df_main.groupby(['r_train', 'rho_test'])['err_subfamily'].mean().reset_index()

    rtrains = sorted([r for r in agg['r_train'].unique() if r > 1e-9])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(rtrains) - 1, 1)) for i in range(len(rtrains))]

    ax.plot(bl_agg['rho_test'], bl_agg['err_subfamily'], 'k--', lw=2.2, ms=7,
            marker='s', label=r'Baseline ($\tau{=}I$)', zorder=4)

    for rt, color in zip(rtrains, colors):
        sub = agg[agg['r_train'] == rt].sort_values('rho_test')
        ax.plot(sub['rho_test'], sub['err_subfamily'], '-o', color=color,
                lw=1.8, ms=6, label=r'$r_{\mathrm{train}}{=}' + f'{rt:.1f}$', zorder=3)

    ax.set_xlabel(r'$r_{\mathrm{test}}$')
    ax.set_ylabel('Transport error')
    # LINEAR scale

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray',
               fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT / 'atce_error_vs_rtest_linear.pdf')
    plt.close(fig)
    print('  atce_error_vs_rtest_linear.pdf')


# ══════════════════════════════════════════════════════════════════
# 3. portland_width_curves_linear.pdf
# ══════════════════════════════════════════════════════════════════

def portland_width_curves_linear():
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
    # LINEAR scale

    ax.legend(framealpha=0.9, edgecolor='gray', fontsize=13)

    fig.tight_layout()
    fig.savefig(OUT / 'portland_width_curves_linear.pdf')
    plt.close(fig)
    print('  portland_width_curves_linear.pdf')


# ══════════════════════════════════════════════════════════════════
# 4. lilucas_ew_transport_comparison_linear.pdf
# ══════════════════════════════════════════════════════════════════

def lilucas_ew_transport_linear():
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
        # LINEAR scale
        ax.set_title(title, fontsize=18, pad=8)

    ax_sym.set_ylabel('Transport error')

    handles, labels = ax_sym.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray',
               fontsize=13)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT / 'lilucas_ew_transport_comparison_linear.pdf')
    plt.close(fig)
    print('  lilucas_ew_transport_comparison_linear.pdf')


# ══════════════════════════════════════════════════════════════════
# 5. lilucas_gau_error_linear.pdf
# ══════════════════════════════════════════════════════════════════

def lilucas_gau_error_linear():
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
    # LINEAR scale

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='gray',
               fontsize=13)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(OUT / 'lilucas_gau_error_linear.pdf')
    plt.close(fig)
    print('  lilucas_gau_error_linear.pdf')


# ══════════════════════════════════════════════════════════════════
# 6 & 7. lilucas_gau_width_curves_linear.pdf
#         lilucas_ewsym_width_curves_linear.pdf
# ══════════════════════════════════════════════════════════════════

def _lilucas_width_curves_linear_impl(config_key, filename):
    display_name, subdir = LILUCAS_CONFIGS[config_key]
    df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
    tr = df[~df['variant'].str.contains('baseline', na=False)]
    tr = tr[tr['eps_train'] > 1e-9]

    _, df_q = load_lilucas(config_key)
    qs = df_q['target_val'].abs().mean()
    vacuity = 10 * qs

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
    # LINEAR scale
    ax.set_xticks(rtrains)
    ax.set_xticklabels([f'{r:.1f}' for r in rtrains])

    ax.legend(framealpha=0.9, edgecolor='gray', fontsize=13)

    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(f'  {filename}')


def lilucas_gau_width_curves_linear():
    _lilucas_width_curves_linear_impl('gau_sym', 'lilucas_gau_width_curves_linear.pdf')


def lilucas_ewsym_width_curves_linear():
    _lilucas_width_curves_linear_impl('ew_sym', 'lilucas_ewsym_width_curves_linear.pdf')


# ══════════════════════════════════════════════════════════════════
# Data range report
# ══════════════════════════════════════════════════════════════════

def report_ranges():
    """Print the data ranges to help assess linear-scale usability."""
    print('\n=== Dynamic range report ===\n')

    # ATE width curves
    sym = load_ate('sym')
    dir_ = load_ate('dir')
    sym_diag = sym[np.isclose(sym['eps'], sym['eta']) & (sym['eps'] > 1e-9)]
    dir_diag = dir_[np.isclose(dir_['eps'], dir_['eta']) & (dir_['eps'] > 1e-9)]
    all_widths = pd.concat([
        sym_diag.groupby(['eps', 'iota']).agg(sw=('std_width', 'mean'), dw=('dir_width', 'mean')).reset_index(),
        dir_diag.groupby(['eps', 'iota']).agg(sw=('std_width', 'mean'), dw=('dir_width', 'mean')).reset_index(),
    ])
    mn = min(all_widths['dw'].min(), all_widths['sw'].min())
    mx = max(all_widths['dw'].max(), all_widths['sw'].max())
    print(f'ATE width curves: min={mn:.3f}, max={mx:.1f}, ratio={mx/mn:.0f}x')

    # ATCE error vs rtest
    df_main, _ = load_atce_r2()
    df_main['r_train'] = df_main['variant'].apply(_extract_rtrain)
    bl_all = pd.read_csv(RESULTS / 'atce_subfamily/atce_gaussian_z_entrywise_subfamily'
                         / 'radius_eval.csv')
    bl = bl_all[bl_all['variant'].str.contains('baseline', na=False)]
    all_err = pd.concat([df_main['err_subfamily'], bl['err_subfamily']])
    agg = df_main.groupby(['r_train', 'rho_test'])['err_subfamily'].mean()
    bl_agg = bl.groupby('rho_test')['err_subfamily'].mean()
    all_agg = pd.concat([agg, bl_agg])
    print(f'ATCE error vs rtest: min={all_agg.min():.4f}, max={all_agg.max():.1f}, '
          f'ratio={all_agg.max()/max(all_agg.min(), 1e-10):.0f}x')

    # Portland width
    df = load_portland_queries()
    nz = df[df['eps'] > 1e-9]
    agg_p = nz.groupby('eps').agg(sw=('std_width', 'mean'), dw=('dir_width', 'mean')).reset_index()
    mn = min(agg_p['dw'].min(), agg_p['sw'].min())
    mx = max(agg_p['dw'].max(), agg_p['sw'].max())
    print(f'Portland width: min={mn:.4f}, max={mx:.1f}, ratio={mx/mn:.0f}x')

    # LiLuCaS ew transport
    for key, label in [('ew_sym', 'LiLuCaS-ew-sym'), ('ew_dir', 'LiLuCaS-ew-dir')]:
        _, subdir = LILUCAS_CONFIGS[key]
        df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
        df['r_train'] = df['variant'].apply(_extract_rtrain)
        bl = df[df['variant'].str.contains('baseline', na=False)]
        tr = df[~df['variant'].str.contains('baseline', na=False)]
        bl_agg = bl.groupby('rho_test')['err_subfamily'].mean()
        tr_agg = tr.groupby(['r_train', 'rho_test'])['err_subfamily'].mean()
        all_agg = pd.concat([bl_agg, tr_agg])
        print(f'{label} error: min={all_agg.min():.4f}, max={all_agg.max():.1f}, '
              f'ratio={all_agg.max()/max(all_agg.min(), 1e-10):.0f}x')

    # LiLuCaS gau error
    _, subdir = LILUCAS_CONFIGS['gau_sym']
    df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
    df['r_train'] = df['variant'].apply(_extract_rtrain)
    bl = df[df['variant'].str.contains('baseline', na=False)]
    tr = df[~df['variant'].str.contains('baseline', na=False)]
    bl_agg = bl.groupby('rho_test')['err_subfamily'].mean()
    tr_agg = tr.groupby(['r_train', 'rho_test'])['err_subfamily'].mean()
    all_agg = pd.concat([bl_agg, tr_agg])
    print(f'LiLuCaS-gau error: min={all_agg.min():.4f}, max={all_agg.max():.1f}, '
          f'ratio={all_agg.max()/max(all_agg.min(), 1e-10):.0f}x')

    # LiLuCaS width curves
    for key, label in [('gau_sym', 'LiLuCaS-gau'), ('ew_sym', 'LiLuCaS-ew-sym')]:
        _, subdir = LILUCAS_CONFIGS[key]
        df = pd.read_csv(RESULTS / subdir / 'radius_eval.csv')
        tr = df[~df['variant'].str.contains('baseline', na=False)]
        tr = tr[tr['eps_train'] > 1e-9]
        agg = tr.groupby('eps_train').agg(sw=('mean_width', 'mean'), dw=('mean_dir_width', 'mean')).reset_index()
        mn = min(agg['dw'].min(), agg['sw'].min())
        mx = max(agg['dw'].max(), agg['sw'].max())
        print(f'{label} width: min={mn:.3f}, max={mx:.1f}, ratio={mx/mn:.0f}x')


# ══════════════════════════════════════════════════════════════════

def main():
    report_ranges()

    print('\n--- Generating linear-scale variants ---\n')
    ate_width_curves_linear()
    atce_error_vs_rtest_linear()
    portland_width_curves_linear()
    lilucas_ew_transport_linear()
    lilucas_gau_error_linear()
    lilucas_gau_width_curves_linear()
    lilucas_ewsym_width_curves_linear()
    print('\nDone. 7 linear-scale PDFs generated.')


if __name__ == '__main__':
    main()
