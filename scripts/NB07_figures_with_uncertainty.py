#!/usr/bin/env python3
"""
NB07_figures_with_uncertainty.py
=================================
Step 1.7: Regenerate the snow-harmonisation bar charts and three-sector
time-series figures with bootstrap confidence intervals overlaid.

PhD Thesis -- Xinlong Liu, IMAS, University of Tasmania

Purpose:
  - Extend NB04's figure-generation pipeline to consume the bootstrap CSVs
    written by NB05 and visualise sampling uncertainty on the manuscript's
    two most analytically important figure types:
      (a) snow-harmonisation bar charts (h_fi and h_sc): add 95% CI whiskers
          to each Case A/B/C/D bar so that the reduction percentages are
          presented with their statistical uncertainty;
      (b) three-sector monthly time-series: add 1-sigma inter-product spread
          envelopes around each product's monthly trace, plus year-bootstrap
          climatological CIs on the multi-year mean overlaid in a separate
          panel.

  - The original NB04 figures are preserved unchanged; this script writes new
    figures with a "_with_ci" suffix to allow side-by-side comparison.

Inputs:
  - Bootstrap CSVs from NB05:
      bootstrap_attribution_ci.csv
      bootstrap_climatology_ci.csv
      bootstrap_monthly_interproduct_spread.csv
  - Original sector-mean monthly CSVs (from NB02):
      sector_means_monthly_CS2.csv
      sector_means_monthly_ENV.csv

Outputs (in output/figures/C3_extended_freeboard/):
  - fig_snow_harmonisation_with_ci.jpg/.pdf      (h_fi bars + CI)
  - fig_snow_harmonisation_hsc_with_ci.jpg/.pdf  (h_sc bars + CI)
  - fig_timeseries_3sectors_with_ci.jpg/.pdf     (time series + envelope)

Run on Gadi:
  nohup bash -c 'source ~/cryo2ice_env/bin/activate && \\
  python3 NB07_figures_with_uncertainty.py' > NB07_output.txt 2>&1 &

  Expected runtime: ~1-2 minutes.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# ==============================================================================
# Configuration (matched to NB04 v5)
# ==============================================================================
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

BASE_HARMON  = '/g/data/gv90/xl1657/phd/M1_workspace/processed/snow_harmonisation'
BASE_SECTORS = '/g/data/gv90/xl1657/phd/M1_workspace/processed/sector_means'
BASE_FIGURES = '/g/data/gv90/xl1657/phd/M1_workspace/output/figures/C3_extended_freeboard'
os.makedirs(BASE_FIGURES, exist_ok=True)

CS2_PRODUCTS = {
    'LEGOS_I_CS2':   {'label': 'LEGOS I',    'color': '#e41a1c', 'marker': 'o'},
    'LEGOS_II_CS2':  {'label': 'LEGOS II',   'color': '#ff7f00', 'marker': 's'},
    'CCI_CS2':       {'label': 'CCI',        'color': '#377eb8', 'marker': '^'},
    'CSAO_CS2':      {'label': 'CSAO',       'color': '#4daf4a', 'marker': 'D'},
    'CryoTEMPO_CS2': {'label': 'Cryo-TEMPO', 'color': '#984ea3', 'marker': 'v'},
}

HARMON_COLORS = {'A': '#4a4a4a', 'B': '#e7298a', 'C': '#7570b3', 'D': '#1b9e77'}
HARMON_LABELS = {
    'A': 'Case A (Native)',
    'B': 'Case B (Fix $\\rho_s$)',
    'C': 'Case C (Fix $h_s$)',
    'D': 'Case D (Fix both)',
}

SECTOR_DISPLAY = {
    'Western_Weddell': 'Western Weddell Sea',
    'Indian': 'Indian Ocean',
    'Ross': 'Ross Sea',
}

FOCUS_SECTORS = ['Western_Weddell', 'Indian', 'Ross']
CS2_YEARS = list(range(2013, 2019))
ENV_YEARS = list(range(2003, 2012))
WINTER_MONTHS = list(range(5, 11))
SEPARATOR = '=' * 70


# ==============================================================================
# CSV loading helpers
# ==============================================================================
def load_csv(filename, base_dir):
    filepath = os.path.join(base_dir, filename)
    rows = []
    if not os.path.exists(filepath):
        print(f'  WARNING: file not found: {filepath}')
        return rows
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                try:
                    row[key] = float(val)
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def save_figure(fig, base_path, dpi=300):
    fig.savefig(base_path, dpi=dpi, bbox_inches='tight')
    name, ext = os.path.splitext(base_path)
    pdf_path = name + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'  Saved {base_path}')
    print(f'  Saved {pdf_path}')


# ==============================================================================
# Figure 1: Snow-harmonisation bars with 95% CI whiskers
# ==============================================================================
def fig_harmonisation_with_ci():
    print(f'\n{SEPARATOR}')
    print('  Figure: Snow-harmonisation bars with bootstrap CI whiskers')
    print(SEPARATOR)

    bootstrap_rows = load_csv('bootstrap_attribution_ci.csv', BASE_HARMON)
    if not bootstrap_rows:
        print('  ERROR: bootstrap_attribution_ci.csv is empty or missing.')
        print('  Run NB05_bootstrap_uncertainty.py first.')
        return

    for var, var_full, ylabel in [
        ('hfi', 'sea-ice freeboard ($h_{fi}$)',
         'Inter-product sea-ice freeboard ($h_{fi}$) spread (m)'),
        ('hsc', 'speed correction ($h_{sc}$)',
         'Inter-product speed correction ($h_{sc}$) spread (m)'),
    ]:
        var_rows = [r for r in bootstrap_rows if r['variable'] == var]
        if not var_rows:
            print(f'  WARNING: no rows for variable {var}')
            continue

        sectors = sorted(set(r['sector'] for r in var_rows))
        sector_labels = [s.replace('_', '\n') for s in sectors]

        # Build per-(sector, case) lookup of mean and CI
        lookup = {}
        for r in var_rows:
            lookup[(r['sector'], r['case'])] = {
                'spread_mean': r['spread_mean'],
                'spread_lo': r['spread_lo'],
                'spread_hi': r['spread_hi'],
                'reduction_mean': r['reduction_mean'],
                'reduction_lo': r['reduction_lo'],
                'reduction_hi': r['reduction_hi'],
            }

        x = np.arange(len(sectors))
        width = 0.19
        fig, ax = plt.subplots(figsize=(18, 8.5))

        for k, case in enumerate(['A', 'B', 'C', 'D']):
            offset = (k - 1.5) * width
            heights = [lookup[(s, case)]['spread_mean'] for s in sectors]
            lo_vals = [lookup[(s, case)]['spread_lo'] for s in sectors]
            hi_vals = [lookup[(s, case)]['spread_hi'] for s in sectors]
            err_lower = [
                max(h - lo, 0.0) for h, lo in zip(heights, lo_vals)
            ]
            err_upper = [
                max(hi - h, 0.0) for h, hi in zip(heights, hi_vals)
            ]
            err = np.array([err_lower, err_upper])

            bars = ax.bar(
                x + offset, heights, width,
                label=HARMON_LABELS[case],
                color=HARMON_COLORS[case],
                edgecolor='black', linewidth=0.5,
                yerr=err, capsize=4, ecolor='black',
                error_kw={'elinewidth': 1.2, 'capthick': 1.2},
            )

            # Annotate Case A bars with the mean, others with mean + reduction
            for i, (bar, val) in enumerate(zip(bars, heights)):
                sector = sectors[i]
                if case == 'A':
                    label_text = f'{val:.3f}'
                else:
                    red = lookup[(sector, case)]['reduction_mean']
                    red_lo = lookup[(sector, case)]['reduction_lo']
                    red_hi = lookup[(sector, case)]['reduction_hi']
                    label_text = (
                        f'{val:.3f}\n'
                        f'\u2212{red:.0f}%\n'
                        f'[{red_lo:.0f},{red_hi:.0f}]'
                    )

                y_pos = lookup[(sector, case)]['spread_hi'] + 0.001
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y_pos,
                    label_text,
                    ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color=HARMON_COLORS[case],
                )

        ax.set_xlabel('Sector', fontsize=24)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_title(
            f'Snow-Harmonisation: Reduction in Inter-Product {var_full} Spread '
            f'(95% CI from year-level bootstrap, $n=1000$)',
            fontsize=22, fontweight='bold',
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sector_labels, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.legend(fontsize=18, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        max_hi = max(
            lookup[(s, c)]['spread_hi']
            for s in sectors for c in ['A', 'B', 'C', 'D']
            if np.isfinite(lookup[(s, c)]['spread_hi'])
        )
        ax.set_ylim(0, max_hi * 1.4)
        plt.tight_layout()

        suffix = '' if var == 'hfi' else f'_{var}'
        save_figure(
            fig,
            os.path.join(BASE_FIGURES, f'fig_snow_harmonisation{suffix}_with_ci.jpg'),
            dpi=300,
        )
        plt.close(fig)


# ==============================================================================
# Figure 2: Three-sector monthly time series with inter-product spread envelope
# ==============================================================================
def fig_timeseries_with_envelope():
    print(f'\n{SEPARATOR}')
    print('  Figure: Three-sector time series with inter-product spread envelope')
    print(SEPARATOR)

    cs2_rows = load_csv('sector_means_monthly_CS2.csv', BASE_SECTORS)
    env_rows = load_csv('sector_means_monthly_ENV.csv', BASE_SECTORS)
    spread_rows = load_csv(
        'bootstrap_monthly_interproduct_spread.csv', BASE_HARMON)
    if not cs2_rows or not env_rows or not spread_rows:
        print('  ERROR: missing one or more required input CSVs.')
        return

    # Index inter-product spread by (era, year, month, sector, variable)
    spread_lookup = {}
    for r in spread_rows:
        key = (r['era'], int(r['year']), int(r['month']),
               r['sector'], r['variable'])
        spread_lookup[key] = {
            'mean': r['mean'], 'min': r['min'], 'max': r['max'],
            'std': r['std'], 'n_products': int(r['n_products']),
        }

    variables = [
        ('hfr', 'Radar freeboard, $h_{fr}$ (m)'),
        ('hfi', 'Sea-ice freeboard, $h_{fi}$ (m)'),
    ]

    for var, ylabel in variables:
        n_rows = len(FOCUS_SECTORS)
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(20, 4.5 * n_rows), sharey='row')

        for row_i, sector in enumerate(FOCUS_SECTORS):
            sector_display = SECTOR_DISPLAY.get(sector, sector.replace('_', ' '))

            # --- Envisat panel (left) ---
            ax_env = axes[row_i, 0]
            for product, props in [
                ('LEGOS_ENV', {'label': 'LEGOS', 'color': '#e41a1c', 'marker': 'o'}),
                ('CCI_ENV',   {'label': 'CCI',   'color': '#377eb8', 'marker': '^'}),
            ]:
                xs, ys = [], []
                for r in env_rows:
                    if r['product'] == product and r['sector'] == sector:
                        try:
                            year = int(r['year'])
                            month = int(r['month'])
                            val = float(r[f'{var}_mean'])
                        except (ValueError, TypeError, KeyError):
                            continue
                        if not np.isfinite(val):
                            continue
                        xs.append(year + (month - 1) / 12.0)
                        ys.append(val)
                if xs:
                    order = np.argsort(xs)
                    xs = np.array(xs)[order]
                    ys = np.array(ys)[order]
                    ax_env.plot(
                        xs, ys, color=props['color'], marker=props['marker'],
                        linewidth=1.2, markersize=5, label=props['label'],
                        alpha=0.85,
                    )

            # Envelope: shade the inter-product (min, max) range per month
            env_year_to_xy = defaultdict(list)
            for year in ENV_YEARS:
                for month in WINTER_MONTHS:
                    key = ('ENV', year, month, sector, var)
                    if key in spread_lookup:
                        info = spread_lookup[key]
                        if info['n_products'] >= 2:
                            x_pos = year + (month - 1) / 12.0
                            env_year_to_xy['x'].append(x_pos)
                            env_year_to_xy['min'].append(info['min'])
                            env_year_to_xy['max'].append(info['max'])
            if env_year_to_xy['x']:
                order = np.argsort(env_year_to_xy['x'])
                xa = np.array(env_year_to_xy['x'])[order]
                ya_min = np.array(env_year_to_xy['min'])[order]
                ya_max = np.array(env_year_to_xy['max'])[order]
                ax_env.fill_between(
                    xa, ya_min, ya_max, color='gray', alpha=0.20,
                    label='Inter-product range', zorder=0,
                )

            ax_env.set_title(f'{sector_display} (Envisat)', fontsize=20)
            ax_env.grid(True, alpha=0.3)
            ax_env.set_xlabel('Year', fontsize=18)
            if row_i == 0:
                ax_env.legend(fontsize=14, loc='upper right', ncol=2)

            # --- CryoSat-2 panel (right) ---
            ax_cs2 = axes[row_i, 1]
            for product, props in CS2_PRODUCTS.items():
                xs, ys = [], []
                for r in cs2_rows:
                    if r['product'] == product and r['sector'] == sector:
                        try:
                            year = int(r['year'])
                            month = int(r['month'])
                            val = float(r[f'{var}_mean'])
                        except (ValueError, TypeError, KeyError):
                            continue
                        if not np.isfinite(val):
                            continue
                        xs.append(year + (month - 1) / 12.0)
                        ys.append(val)
                if xs:
                    order = np.argsort(xs)
                    xs = np.array(xs)[order]
                    ys = np.array(ys)[order]
                    ax_cs2.plot(
                        xs, ys, color=props['color'], marker=props['marker'],
                        linewidth=1.2, markersize=5, label=props['label'],
                        alpha=0.85,
                    )

            cs2_year_to_xy = defaultdict(list)
            for year in CS2_YEARS:
                for month in WINTER_MONTHS:
                    key = ('CS2', year, month, sector, var)
                    if key in spread_lookup:
                        info = spread_lookup[key]
                        if info['n_products'] >= 2:
                            x_pos = year + (month - 1) / 12.0
                            cs2_year_to_xy['x'].append(x_pos)
                            cs2_year_to_xy['min'].append(info['min'])
                            cs2_year_to_xy['max'].append(info['max'])
            if cs2_year_to_xy['x']:
                order = np.argsort(cs2_year_to_xy['x'])
                xa = np.array(cs2_year_to_xy['x'])[order]
                ya_min = np.array(cs2_year_to_xy['min'])[order]
                ya_max = np.array(cs2_year_to_xy['max'])[order]
                ax_cs2.fill_between(
                    xa, ya_min, ya_max, color='gray', alpha=0.20,
                    label='Inter-product range', zorder=0,
                )

            ax_cs2.set_title(f'{sector_display} (CryoSat-2)', fontsize=20)
            ax_cs2.grid(True, alpha=0.3)
            ax_cs2.set_xlabel('Year', fontsize=18)
            if row_i == 0:
                ax_cs2.legend(fontsize=14, loc='upper right', ncol=2)

            if row_i == 0:
                fig.text(
                    0.04, 0.5, ylabel, va='center', rotation='vertical',
                    fontsize=22)

        plt.tight_layout(rect=[0.05, 0, 1, 1])
        save_figure(
            fig,
            os.path.join(BASE_FIGURES, f'fig_timeseries_3sectors_{var}_with_ci.jpg'),
            dpi=300,
        )
        plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f'\nNB07_figures_with_uncertainty.py')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    fig_harmonisation_with_ci()
    fig_timeseries_with_envelope()

    print(f'\n{SEPARATOR}')
    print('  GENERATED FIGURES (with uncertainty)')
    print(SEPARATOR)
    import glob
    for f in sorted(glob.glob(
            os.path.join(BASE_FIGURES, '*_with_ci*'))):
        size_kb = os.path.getsize(f) / 1024
        print(f'  {os.path.basename(f)}: {size_kb:.0f} KB')

    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('Done.')


if __name__ == '__main__':
    main()
