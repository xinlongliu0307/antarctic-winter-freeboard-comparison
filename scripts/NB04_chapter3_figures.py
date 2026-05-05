#!/usr/bin/env python3
"""
NB04_chapter3_figures.py (v5 — caption-matched text sizes + tighter maps)
==========================================================================
Generate all Chapter 3 figures with publication-quality formatting.

Changes from v4:
  - ALL font sizes scaled up ~1.7x so that when figures are rendered at
    \linewidth in the thesis (~6 inches from ~15-18 inch originals),
    figure text matches caption text size (~10-11pt rendered).
  - Map extent tightened from 50°S to 55°S, cropping empty open ocean
    and making the sea-ice zone fill more of each subplot.
  - Map subplot height increased from 4 to 5 inches per row.
  - Colorbar tick label size increased via cbar.ax.tick_params.

Run on Gadi:
  nohup bash -c 'source ~/cryo2ice_env/bin/activate && \
  python3 NB04_chapter3_figures.py' > NB04_output_v5.txt 2>&1 &
"""

import os, glob, numpy as np, csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime

# ==============================================================================
# GLOBAL FONT SIZES — v5: scaled to match ~10-11pt caption at thesis rendering
# Figures are ~15-18" wide, displayed at ~6" → ~2.5-3x shrink factor
# So 22pt in figure → ~8-9pt rendered, 26pt → ~10pt rendered
# ==============================================================================
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
    print('  Cartopy available — using Antarctic Polar Stereographic projection')
except ImportError:
    HAS_CARTOPY = False
    print('  Cartopy not available — using simple grid projection')

# ==============================================================================
# Configuration
# ==============================================================================
BASE_REGRIDDED = '/g/data/gv90/xl1657/phd/M1_workspace/processed/regridded'
BASE_MASKS     = '/g/data/gv90/xl1657/phd/M1_workspace/processed/masks'
BASE_SECTORS   = '/g/data/gv90/xl1657/phd/M1_workspace/processed/sector_means'
BASE_HARMON    = '/g/data/gv90/xl1657/phd/M1_workspace/processed/snow_harmonisation'
BASE_FIGURES   = '/g/data/gv90/xl1657/phd/M1_workspace/output/figures/C3_extended_freeboard'
os.makedirs(BASE_FIGURES, exist_ok=True)

CS2_PRODUCTS = {
    'LEGOS_I_CS2':   {'label': 'LEGOS I',    'color': '#e41a1c', 'marker': 'o'},
    'LEGOS_II_CS2':  {'label': 'LEGOS II',   'color': '#ff7f00', 'marker': 's'},
    'CCI_CS2':       {'label': 'CCI',        'color': '#377eb8', 'marker': '^'},
    'CSAO_CS2':      {'label': 'CSAO',       'color': '#4daf4a', 'marker': 'D'},
    'CryoTEMPO_CS2': {'label': 'Cryo-TEMPO', 'color': '#984ea3', 'marker': 'v'},
}
ENV_PRODUCTS = {
    'LEGOS_ENV': {'label': 'LEGOS', 'color': '#e41a1c', 'marker': 'o'},
    'CCI_ENV':   {'label': 'CCI',   'color': '#377eb8', 'marker': '^'},
}
HARMON_COLORS = {'A': '#4a4a4a', 'B': '#e7298a', 'C': '#7570b3', 'D': '#1b9e77'}
HARMON_LABELS = {
    'A': 'Case A (Native)', 'B': 'Case B (Fix $\\rho_s$)',
    'C': 'Case C (Fix $h_s$)', 'D': 'Case D (Fix both)',
}

VAR_LABELS_FULL = {
    'hfr': 'Radar freeboard, $h_{fr}$ (m)',
    'hfi': 'Sea-ice freeboard, $h_{fi}$ (m)',
    'hsc': 'Speed correction, $h_{sc}$ (m)',
}
VAR_LABELS_CCI = {
    'hfr': 'CCI radar freeboard, $h_{fr}$ (m)',
    'hfi': 'CCI sea-ice freeboard, $h_{fi}$ (m)',
    'hsc': 'CCI speed correction, $h_{sc}$ (m)',
}

SECTOR_DISPLAY = {
    'Western_Weddell': 'Western Weddell Sea',
    'Indian': 'Indian Ocean',
    'Ross': 'Ross Sea',
}

SECTOR_DEFS = [
    ('Western_Weddell', -62.0, -40.0), ('Eastern_Weddell', -40.0, 15.0),
    ('Indian', 15.0, 90.0), ('Western_Pacific', 90.0, 160.0),
    ('Ross', 160.0, -140.0), ('Amundsen_Bellingshausen', -140.0, -62.0),
]
FOCUS_SECTORS = ['Western_Weddell', 'Indian', 'Ross']
CS2_YEARS = list(range(2013, 2019))
ENV_YEARS = list(range(2003, 2012))
WINTER_MONTHS = list(range(5, 11))
SEPARATOR = '=' * 70

# v5: Tighter map extent — crop open ocean, focus on sea-ice zone
MAP_EXTENT = [-180, 180, -90, -55]  # was -50 in v4


def load_csv(filename, base_dir=BASE_SECTORS):
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for key, val in row.items():
                try:
                    row[key] = float(val)
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def compute_multiyear_mean(product, variable, years):
    all_fields = []
    for year in years:
        for month in WINTER_MONTHS:
            npz = os.path.join(BASE_REGRIDDED, f'{product}_{year}{month:02d}.npz')
            if os.path.exists(npz):
                data = np.load(npz)
                if variable in data:
                    all_fields.append(data[variable].astype(np.float64))
                data.close()
    if all_fields:
        with np.errstate(all='ignore'):
            return np.nanmean(np.stack(all_fields), axis=0)
    return np.full((216, 216), np.nan)


def save_figure(fig, base_path, dpi=200):
    fig.savefig(base_path, dpi=dpi, bbox_inches='tight')
    name, ext = os.path.splitext(base_path)
    pdf_path = name + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'  Saved {base_path}')
    print(f'  Saved {pdf_path}')


def add_product_label(ax, label, loc='upper left', fontsize=20):
    if loc == 'upper left':
        x, y, ha, va = 0.03, 0.97, 'left', 'top'
    elif loc == 'upper right':
        x, y, ha, va = 0.97, 0.97, 'right', 'top'
    else:
        x, y, ha, va = 0.03, 0.97, 'left', 'top'
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     alpha=0.85, edgecolor='gray', linewidth=0.5))


def add_sector_lines(ax):
    sector_lons = [-62, -40, 15, 90, 160, -140]
    sector_labels = ['W.Wed', 'E.Wed', 'Indian', 'Pacific', 'Ross', 'A-B']
    sector_label_lons = [-51, -12, 52, 125, 180, -101]
    if HAS_CARTOPY:
        for lon in sector_lons:
            ax.plot([lon, lon], [-90, -55], 'k-', linewidth=0.8, alpha=0.5,
                   transform=ccrs.PlateCarree())
        for label, lon in zip(sector_labels, sector_label_lons):
            ax.text(lon, -56.5, label, fontsize=16, ha='center',
                   transform=ccrs.PlateCarree(), fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            alpha=0.7, edgecolor='none'))


# ==============================================================================
# Fig 3.2: Spatial maps
# ==============================================================================
def fig_spatial_maps():
    print(f'\n{SEPARATOR}')
    print('  Generating Figure 3.2: Spatial maps...')
    print(SEPARATOR)
    ref = np.load(os.path.join(BASE_REGRIDDED, 'CCI_CS2_201305.npz'))
    lat_grid, lon_grid = ref['lat'], ref['lon']
    ref.close()

    for era, products, years, suffix, era_label in [
        ('Envisat', ['LEGOS_ENV', 'CCI_ENV'], ENV_YEARS, 'ENV', 'Envisat'),
        ('CryoSat-2', list(CS2_PRODUCTS.keys()), CS2_YEARS, 'CS2', 'CryoSat-2')
    ]:
        n_prods = len(products)
        vars_info = [
            ('hfr', 0.25, 'Radar freeboard, $h_{fr}$ (m)'),
            ('hfi', 0.30, 'Sea-ice freeboard, $h_{fi}$ (m)'),
            ('hsc', 0.10, 'Speed correction, $h_{sc}$ (m)')
        ]
        if HAS_CARTOPY:
            proj = ccrs.SouthPolarStereo()
            fig, axes = plt.subplots(n_prods, 3, figsize=(18, 5.5 * n_prods),
                                     subplot_kw={'projection': proj})
        else:
            fig, axes = plt.subplots(n_prods, 3, figsize=(18, 4.5 * n_prods))
        if n_prods == 1:
            axes = axes[np.newaxis, :]

        for i, product in enumerate(products):
            prod_info = CS2_PRODUCTS.get(product, ENV_PRODUCTS.get(product, {}))
            prod_label = prod_info.get('label', product)
            for j, (var, vmax, label) in enumerate(vars_info):
                mean_field = compute_multiyear_mean(product, var, years)
                ax = axes[i, j]
                if HAS_CARTOPY:
                    ax.set_extent(MAP_EXTENT, ccrs.PlateCarree())
                    im = ax.pcolormesh(lon_grid, lat_grid, mean_field,
                                       cmap='YlOrRd', vmin=0, vmax=vmax,
                                       transform=ccrs.PlateCarree(), shading='auto')
                    ax.coastlines(resolution='50m', linewidth=0.5)
                    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
                    for lat_val in [-60, -70, -80]:
                        ax.text(180, lat_val, f'{lat_val}\u00b0',
                               transform=ccrs.PlateCarree(), fontsize=16, ha='left')
                    add_sector_lines(ax)
                else:
                    im = ax.pcolormesh(mean_field, cmap='YlOrRd', vmin=0, vmax=vmax)
                    ax.set_xticks([]); ax.set_yticks([])
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(label, fontsize=20)
                cbar.ax.tick_params(labelsize=16)
                add_product_label(ax, prod_label, fontsize=20)
                if i == 0:
                    ax.set_title(label, fontsize=22)

        plt.tight_layout()
        save_figure(fig, os.path.join(BASE_FIGURES, f'fig_spatial_{suffix}.jpg'))
        plt.close(fig)


# ==============================================================================
# Fig 3.3: Difference maps
# ==============================================================================
def fig_difference_maps():
    print(f'\n{SEPARATOR}')
    print('  Generating Figure 3.3: Difference maps...')
    print(SEPARATOR)
    ref = np.load(os.path.join(BASE_REGRIDDED, 'CCI_CS2_201305.npz'))
    lat_grid, lon_grid = ref['lat'], ref['lon']
    ref.close()

    for era, ref_prod, diff_prods, years, suffix, era_label in [
        ('ENV', 'CCI_ENV', ['LEGOS_ENV'], ENV_YEARS, 'ENV', 'Envisat'),
        ('CS2', 'CCI_CS2', ['LEGOS_I_CS2', 'LEGOS_II_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2'],
         CS2_YEARS, 'CS2', 'CryoSat-2')
    ]:
        n_prods = len(diff_prods)
        vars_info = [
            ('hfr', 0.10, '$\\Delta$ Radar freeboard (m)'),
            ('hfi', 0.10, '$\\Delta$ Sea-ice freeboard (m)'),
            ('hsc', 0.05, '$\\Delta$ Speed correction (m)')
        ]
        if HAS_CARTOPY:
            proj = ccrs.SouthPolarStereo()
            fig, axes = plt.subplots(n_prods, 3, figsize=(18, 5.5 * n_prods),
                                     subplot_kw={'projection': proj})
        else:
            fig, axes = plt.subplots(n_prods, 3, figsize=(18, 4.5 * n_prods))
        if n_prods == 1:
            axes = axes[np.newaxis, :]

        for j, (var, vlim, label) in enumerate(vars_info):
            ref_mean = compute_multiyear_mean(ref_prod, var, years)
            for i, product in enumerate(diff_prods):
                prod_mean = compute_multiyear_mean(product, var, years)
                diff_field = prod_mean - ref_mean
                ax = axes[i, j]
                norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
                prod_info = CS2_PRODUCTS.get(product, ENV_PRODUCTS.get(product, {}))
                prod_label = prod_info.get('label', product)
                if HAS_CARTOPY:
                    ax.set_extent(MAP_EXTENT, ccrs.PlateCarree())
                    im = ax.pcolormesh(lon_grid, lat_grid, diff_field,
                                       cmap='RdBu_r', norm=norm,
                                       transform=ccrs.PlateCarree(), shading='auto')
                    ax.coastlines(resolution='50m', linewidth=0.5)
                    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
                    for lat_val in [-60, -70, -80]:
                        ax.text(180, lat_val, f'{lat_val}\u00b0',
                               transform=ccrs.PlateCarree(), fontsize=16, ha='left')
                    add_sector_lines(ax)
                else:
                    im = ax.pcolormesh(diff_field, cmap='RdBu_r', norm=norm)
                    ax.set_xticks([]); ax.set_yticks([])
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(label, fontsize=20)
                cbar.ax.tick_params(labelsize=16)
                add_product_label(ax, f'{prod_label} \u2212 CCI', fontsize=20)
                if i == 0:
                    ax.set_title(label, fontsize=22)

        plt.tight_layout()
        save_figure(fig, os.path.join(BASE_FIGURES, f'fig_diffs_{suffix}.jpg'))
        plt.close(fig)


# ==============================================================================
# Fig 3.4: Time series
# ==============================================================================
def fig_timeseries():
    print(f'\n{SEPARATOR}')
    print('  Generating Figure 3.4: Time series...')
    print(SEPARATOR)
    cs2_rows = load_csv('sector_means_monthly_CS2.csv')
    env_rows = load_csv('sector_means_monthly_ENV.csv')

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

    fig, axes = plt.subplots(3, 3, figsize=(22, 15))
    for col, sector in enumerate(FOCUS_SECTORS):
        sector_display = SECTOR_DISPLAY.get(sector, sector.replace('_', ' '))

        for row, (var, ylabel) in enumerate([
            ('hfr', VAR_LABELS_FULL['hfr']),
            ('hfi', VAR_LABELS_FULL['hfi']),
            ('hsc', VAR_LABELS_FULL['hsc'])
        ]):
            ax = axes[row, col]
            panel_idx = row * 3 + col

            # Envisat era
            for product, props in ENV_PRODUCTS.items():
                for year in ENV_YEARS:
                    yr = [r for r in env_rows if r['product'] == product
                          and r['sector'] == sector and int(r['year']) == year]
                    yr.sort(key=lambda r: int(r['month']))
                    if not yr: continue
                    times = [int(r['year']) + (int(r['month']) - 0.5) / 12 for r in yr]
                    vals = []
                    for r in yr:
                        try: vals.append(float(r[f'{var}_mean']))
                        except: vals.append(np.nan)
                    lbl = props['label'] if year == ENV_YEARS[0] else None
                    ax.plot(times, vals, color=props['color'], marker=props['marker'],
                           markersize=4, linewidth=1.0, label=lbl, alpha=0.8)

            # CryoSat-2 era
            for product, props in CS2_PRODUCTS.items():
                for year in CS2_YEARS:
                    yr = [r for r in cs2_rows if r['product'] == product
                          and r['sector'] == sector and int(r['year']) == year]
                    yr.sort(key=lambda r: int(r['month']))
                    if not yr: continue
                    times = [int(r['year']) + (int(r['month']) - 0.5) / 12 for r in yr]
                    vals = []
                    for r in yr:
                        try: vals.append(float(r[f'{var}_mean']))
                        except: vals.append(np.nan)
                    lbl = props['label'] if year == CS2_YEARS[0] else None
                    ax.plot(times, vals, color=props['color'], marker=props['marker'],
                           markersize=4, linewidth=1.0, label=lbl, alpha=0.8)

            ax.axvspan(2011.9, 2012.9, alpha=0.04, color='gray')
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            if row == 0:
                ax.set_title(sector_display, fontsize=24, fontweight='bold')
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=22)
            if row == 2:
                ax.set_xlabel('Year', fontsize=22)

            ax.text(0.02, 0.95, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=22, fontweight='bold', va='top')

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=18)

    handles, labels, seen = [], [], set()
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l and l not in seen:
                    handles.append(h); labels.append(l); seen.add(l)
    fig.legend(handles, labels, loc='upper center',
              ncol=min(len(labels), 7), fontsize=20, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, os.path.join(BASE_FIGURES, 'fig_timeseries_3sectors.jpg'))
    plt.close(fig)


# ==============================================================================
# Figs 3.5-3.7: Scatter plots
# ==============================================================================
def fig_scatter_plot(variable, fig_num):
    var_label = VAR_LABELS_FULL[variable]
    var_label_cci = VAR_LABELS_CCI[variable]

    print(f'\n{SEPARATOR}')
    print(f'  Generating Figure 3.{fig_num}: Scatter plot \u2014 {var_label}...')
    print(SEPARATOR)
    cs2_rows = load_csv('sector_means_monthly_CS2.csv')
    other_products = ['LEGOS_I_CS2', 'LEGOS_II_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']
    n_cols, n_rows = len(other_products), len(FOCUS_SECTORS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.5 * n_rows))

    for row, sector in enumerate(FOCUS_SECTORS):
        sector_display = SECTOR_DISPLAY.get(sector, sector.replace('_', ' '))
        cci_cs2 = {}
        for r in cs2_rows:
            if r['product'] == 'CCI_CS2' and r['sector'] == sector:
                try:
                    cci_cs2[(int(r['year']), int(r['month']))] = float(r[f'{variable}_mean'])
                except: pass

        for col, product in enumerate(other_products):
            ax = axes[row, col]
            prod_cs2 = {}
            for r in cs2_rows:
                if r['product'] == product and r['sector'] == sector:
                    try:
                        prod_cs2[(int(r['year']), int(r['month']))] = float(r[f'{variable}_mean'])
                    except: pass

            x_vals, y_vals = [], []
            for key in prod_cs2:
                if key in cci_cs2:
                    xv, yv = cci_cs2[key], prod_cs2[key]
                    if np.isfinite(xv) and np.isfinite(yv):
                        x_vals.append(xv); y_vals.append(yv)

            props = CS2_PRODUCTS[product]
            if x_vals:
                ax.scatter(x_vals, y_vals, c=props['color'], marker=props['marker'],
                          s=50, alpha=0.7, edgecolors='none')
                x_arr, y_arr = np.array(x_vals), np.array(y_vals)
                if len(x_arr) > 2:
                    r_val = np.corrcoef(x_arr, y_arr)[0, 1]
                    bias = np.mean(y_arr - x_arr)
                    ax.text(0.05, 0.95,
                           f'r = {r_val:.2f}\nbias = {bias:+.3f} m\nn = {len(x_arr)}',
                           transform=ax.transAxes, fontsize=18, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                all_vals = x_vals + y_vals
                vmin, vmax = min(all_vals), max(all_vals)
                padding = (vmax - vmin) * 0.15
                ax_min = max(vmin - padding, -0.05)
                ax_max = vmax + padding
                ax.set_xlim(ax_min, ax_max); ax.set_ylim(ax_min, ax_max)

            lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
            ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=0.8)

            add_product_label(ax, f'{props["label"]} vs CCI\n{sector_display}',
                            loc='upper left', fontsize=16)

            if row == 0:
                ax.set_title(props['label'], fontsize=22, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{sector_display}\n{var_label}', fontsize=20)
            if row == n_rows - 1:
                ax.set_xlabel(var_label_cci, fontsize=20)
            ax.grid(True, alpha=0.2); ax.set_aspect('equal')
            ax.tick_params(labelsize=16)

    plt.tight_layout()
    save_figure(fig, os.path.join(BASE_FIGURES, f'fig_scatter_{variable}.png'))
    plt.close(fig)


# ==============================================================================
# Fig 3.8: Snow-harmonisation bar chart
# ==============================================================================
def fig_snow_harmonisation():
    print(f'\n{SEPARATOR}')
    print('  Generating Figure 3.8: Snow-harmonisation bar chart...')
    print(SEPARATOR)
    attribution = load_csv('harmonisation_attribution.csv', BASE_HARMON)

    for var, var_full, ylabel in [
        ('hfi', 'sea-ice freeboard ($h_{fi}$)',
         'Inter-product sea-ice freeboard ($h_{fi}$) spread (m)'),
        ('hsc', 'speed correction ($h_{sc}$)',
         'Inter-product speed correction ($h_{sc}$) spread (m)')
    ]:
        var_rows = [r for r in attribution if r['variable'] == var]
        if not var_rows: continue

        sectors = [r['sector'] for r in var_rows]
        sector_labels = [s.replace('_', '\n') for s in sectors]
        spreads = {}
        for case in ['A', 'B', 'C', 'D']:
            spreads[case] = [float(r[f'spread_{case}']) for r in var_rows]

        x = np.arange(len(sectors))
        width = 0.19
        fig, ax = plt.subplots(figsize=(18, 8))

        for k, case in enumerate(['A', 'B', 'C', 'D']):
            offset = (k - 1.5) * width
            bars = ax.bar(x + offset, spreads[case], width,
                         label=HARMON_LABELS[case], color=HARMON_COLORS[case],
                         edgecolor='black', linewidth=0.5)

            for i, (bar, val) in enumerate(zip(bars, spreads[case])):
                base_val = spreads['A'][i]
                pct = (base_val - val) / base_val * 100 if base_val > 0 else 0

                if case == 'A':
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=14,
                           fontweight='bold', color=HARMON_COLORS[case])
                elif val > 0.0005:
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                           f'{val:.3f}\n\u2212{pct:.0f}%', ha='center', va='bottom',
                           fontsize=13, fontweight='bold', color=HARMON_COLORS[case])
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.0008,
                           f'0.000\n\u2212{pct:.0f}%', ha='center', va='bottom',
                           fontsize=13, fontweight='bold', color=HARMON_COLORS[case])

        ax.set_xlabel('Sector', fontsize=24)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_title(
            f'Snow-Harmonisation: Reduction in Inter-Product {var_full} Spread',
            fontsize=24, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sector_labels, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.legend(fontsize=18, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(spreads['A']) * 1.35)
        plt.tight_layout()

        suffix = '' if var == 'hfi' else f'_{var}'
        save_figure(fig, os.path.join(BASE_FIGURES, f'fig_snow_harmonisation{suffix}.jpg'), dpi=300)
        plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    print(f'\nNB04_chapter3_figures.py (v5 \u2014 caption-matched text + tighter maps)')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output directory: {BASE_FIGURES}')

    fig_spatial_maps()
    fig_difference_maps()
    fig_timeseries()
    fig_scatter_plot('hfr', 5)
    fig_scatter_plot('hsc', 6)
    fig_scatter_plot('hfi', 7)
    fig_snow_harmonisation()

    print(f'\n{SEPARATOR}')
    print('  GENERATED FIGURES')
    print(SEPARATOR)
    for f in sorted(glob.glob(os.path.join(BASE_FIGURES, '*'))):
        size_kb = os.path.getsize(f) / 1024
        print(f'  {os.path.basename(f)}: {size_kb:.0f} KB')

    n_pdf = len(glob.glob(os.path.join(BASE_FIGURES, '*.pdf')))
    n_jpg = len(glob.glob(os.path.join(BASE_FIGURES, '*.jpg')))
    n_png = len(glob.glob(os.path.join(BASE_FIGURES, '*.png')))
    print(f'\n  Total: {n_jpg} JPG + {n_png} PNG + {n_pdf} PDF files')
    print(f'  All figures saved to: {BASE_FIGURES}')
    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('Done.')
