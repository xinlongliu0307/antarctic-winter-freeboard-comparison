#!/usr/bin/env python3
"""
NB03_snow_harmonisation.py (v2 — CCI Antarctic density corrected)
==================================================================
Step 1.4: Snow-harmonisation experiments (Cases A-D).

PhD Thesis — Xinlong Liu, IMAS, University of Tasmania

Changes from v1:
  - CCI native Antarctic snow density corrected from Mallett et al. (2020)
    time-dependent parameterisation to FIXED 300 kg/m³, as verified from
    CCI v4.0 L3C netCDF back-calculation (median 300.0-300.5 across all
    12 months of 2018) and confirmed by ATBD Table 2-1 ("fixed/clim").
  - Common density for Cases B and D changed from Mallett to fixed 300 kg/m³
    (the CCI Antarctic value), providing a constant, hemisphere-appropriate
    baseline against which month-dependent (LEGOS, CSAO) and seasonally
    varying (Cryo-TEMPO) density schemes are compared.
  - mallett2020_density() function RETAINED for reference only.

Common reference for Cases C and D: CCI AMSR-E/AMSR2 snow thickness
Common density for Cases B and D: CCI Antarctic fixed value of 300 kg/m³

Run on Gadi:
  nohup bash -c 'source ~/cryo2ice_env/bin/activate && \
  python3 NB03_snow_harmonisation.py' > NB03_output_v2.txt 2>&1 &
"""

import os
import glob
import numpy as np
import csv
from datetime import datetime
from collections import defaultdict

# ==============================================================================
# Configuration
# ==============================================================================
BASE_REGRIDDED = '/g/data/gv90/xl1657/phd/M1_workspace/processed/regridded'
BASE_MASKS     = '/g/data/gv90/xl1657/phd/M1_workspace/processed/masks'
BASE_OUT       = '/g/data/gv90/xl1657/phd/M1_workspace/processed/snow_harmonisation'
os.makedirs(BASE_OUT, exist_ok=True)

CS2_YEARS = range(2013, 2019)
WINTER_MONTHS = range(5, 11)
CS2_PRODUCTS = ['CCI_CS2', 'LEGOS_I_CS2', 'LEGOS_II_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']
SNOW_REF_PRODUCT = 'CCI_CS2'

# CORRECTED: CCI Antarctic uses fixed 300 kg/m³ (ATBD Table 2-1)
COMMON_DENSITY = 300.0  # kg/m³

SECTOR_DEFS = [
    ('Western_Weddell',          -62.0,  -40.0),
    ('Eastern_Weddell',          -40.0,   15.0),
    ('Indian',                    15.0,   90.0),
    ('Western_Pacific',           90.0,  160.0),
    ('Ross',                     160.0, -140.0),
    ('Amundsen_Bellingshausen', -140.0,  -62.0),
]
SECTOR_NAMES = [s[0] for s in SECTOR_DEFS]
SEPARATOR = '=' * 70


# ==============================================================================
# Snow density parameterisations
# ==============================================================================
def mallett2020_density(month):
    """
    Mallett et al. (2020) linear parameterisation — ARCTIC ONLY.
    Retained for reference; NOT used for CCI Antarctic or as common density.
    """
    if month >= 10:
        t = month - 10 + 0.5
    else:
        t = month + 2 + 0.5
    return 6.5 * t + 274.51


def kurtz2012_density(month):
    """LEGOS/CSAO: Kurtz & Markus (2012). May: 320, Jun-Sep: 350, Oct: 340."""
    if month == 5:
        return 320.0
    elif month == 10:
        return 340.0
    else:
        return 350.0


def fons2023_density(month):
    """Cryo-TEMPO: Fons et al. (2023). Summer: 360, Autumn: 350, Winter: 330, Spring: 310."""
    seasonal = {12: 360, 1: 360, 2: 360,
                3: 350, 4: 350, 5: 350,
                6: 330, 7: 330, 8: 330,
                9: 310, 10: 310, 11: 310}
    return float(seasonal.get(month, 330))


def get_native_density(product, month):
    """Return the native snow density for a given product and month."""
    if product in ['CCI_CS2', 'CCI_ENV']:
        return 300.0  # CORRECTED: fixed 300 kg/m³ for Antarctic (ATBD Table 2-1)
    elif product in ['LEGOS_I_CS2', 'LEGOS_II_CS2', 'LEGOS_ENV', 'CSAO_CS2']:
        return kurtz2012_density(month)
    elif product == 'CryoTEMPO_CS2':
        return fons2023_density(month)
    else:
        return 350.0


def compute_hsc(hs, rho_s):
    """Compute h_sc = (c/c_s - 1) * h_s where c_s = c * (1 + 5.1e-4 * rho_s)^(-1.5)."""
    c = 3e8
    cs = c * (1 + 5.1e-4 * rho_s) ** (-1.5)
    return (c / cs - 1) * hs


# ==============================================================================
# Load sector map and common masks
# ==============================================================================
def load_sector_map():
    data = np.load(os.path.join(BASE_MASKS, 'sector_map.npz'), allow_pickle=True)
    sector_map = data['sector_map']
    data.close()
    return sector_map


def load_common_masks():
    data = np.load(os.path.join(BASE_MASKS, 'common_masks_CS2.npz'))
    masks = {}
    for key in data.files:
        year, month = key.split('_')
        masks[(int(year), int(month))] = data[key]
    data.close()
    return masks


# ==============================================================================
# Run Cases A-D
# ==============================================================================
def run_harmonisation():
    print(f'\n{SEPARATOR}')
    print('  Loading sector map and common masks...')
    print(SEPARATOR)

    sector_map = load_sector_map()
    masks = load_common_masks()
    print(f'  Loaded {len(masks)} monthly common masks')

    all_rows = []
    spread_data = defaultdict(list)

    for year in CS2_YEARS:
        for month in WINTER_MONTHS:
            key = (year, month)
            if key not in masks:
                continue

            common_mask = masks[key]

            cci_file = os.path.join(BASE_REGRIDDED, f'{SNOW_REF_PRODUCT}_{year}{month:02d}.npz')
            if not os.path.exists(cci_file):
                continue
            cci_data = np.load(cci_file)
            cci_hs = cci_data['hs']
            cci_data.close()

            # CORRECTED: fixed 300 kg/m³ instead of mallett2020_density(month)
            common_rho = COMMON_DENSITY

            for product in CS2_PRODUCTS:
                npz_file = os.path.join(BASE_REGRIDDED, f'{product}_{year}{month:02d}.npz')
                if not os.path.exists(npz_file):
                    continue

                data = np.load(npz_file)
                hfr = data['hfr']
                hs_native = data['hs']
                data.close()

                native_rho = get_native_density(product, month)

                # Case A: Native h_s, native rho_s
                hsc_A = compute_hsc(hs_native, native_rho)
                hfi_A = hfr + hsc_A

                # Case B: Native h_s, COMMON rho_s (fixed 300 kg/m³)
                hsc_B = compute_hsc(hs_native, common_rho)
                hfi_B = hfr + hsc_B

                # Case C: COMMON h_s (CCI), native rho_s
                hsc_C = compute_hsc(cci_hs, native_rho)
                hfi_C = hfr + hsc_C

                # Case D: COMMON h_s (CCI), COMMON rho_s (fixed 300 kg/m³)
                hsc_D = compute_hsc(cci_hs, common_rho)
                hfi_D = hfr + hsc_D

                case_fields = {
                    'A': (hsc_A, hfi_A),
                    'B': (hsc_B, hfi_B),
                    'C': (hsc_C, hfi_C),
                    'D': (hsc_D, hfi_D),
                }

                for case_label, (hsc_case, hfi_case) in case_fields.items():
                    for sector_idx, sector_name in enumerate(SECTOR_NAMES):
                        mask = common_mask & (sector_map == sector_idx)
                        n_cells = np.sum(mask)
                        if n_cells == 0:
                            continue

                        hfr_vals = hfr[mask]
                        hsc_vals = hsc_case[mask]
                        hfi_vals = hfi_case[mask]

                        hfr_valid = hfr_vals[np.isfinite(hfr_vals)]
                        hsc_valid = hsc_vals[np.isfinite(hsc_vals)]
                        hfi_valid = hfi_vals[np.isfinite(hfi_vals)]

                        if len(hfr_valid) == 0 or len(hsc_valid) == 0:
                            continue

                        row = {
                            'case': case_label,
                            'year': year,
                            'month': month,
                            'product': product,
                            'sector': sector_name,
                            'n_cells': n_cells,
                            'hfr_mean': float(np.mean(hfr_valid)),
                            'hsc_mean': float(np.mean(hsc_valid)),
                            'hfi_mean': float(np.mean(hfi_valid)),
                        }
                        all_rows.append(row)

                        spread_data[(case_label, year, month, sector_name, 'hfr')].append(
                            float(np.mean(hfr_valid)))
                        spread_data[(case_label, year, month, sector_name, 'hsc')].append(
                            float(np.mean(hsc_valid)))
                        spread_data[(case_label, year, month, sector_name, 'hfi')].append(
                            float(np.mean(hfi_valid)))

    print(f'  Computed {len(all_rows)} case-product-sector-month entries')
    return all_rows, spread_data


# ==============================================================================
# Compute spread and attribution
# ==============================================================================
def compute_spread_and_attribution(spread_data):
    print(f'\n{SEPARATOR}')
    print('  Computing inter-product spread per case...')
    print(SEPARATOR)

    spread_rows = []
    for (case, year, month, sector, variable), values in sorted(spread_data.items()):
        if len(values) >= 2:
            spread = max(values) - min(values)
            spread_rows.append({
                'case': case, 'year': year, 'month': month,
                'sector': sector, 'variable': variable,
                'spread': spread, 'n_products': len(values),
            })

    print(f'  Computed {len(spread_rows)} spread entries across all cases')

    mean_spread = defaultdict(list)
    for r in spread_rows:
        key = (r['case'], r['sector'], r['variable'])
        mean_spread[key].append(r['spread'])

    print(f'\n{SEPARATOR}')
    print('  SPREAD SUMMARY AND ATTRIBUTION')
    print(SEPARATOR)

    attribution_rows = []
    for variable in ['hfr', 'hsc', 'hfi']:
        print(f'\n  --- {variable} ---')
        print(f'  {"Sector":<25} {"Case A":>8} {"Case B":>8} {"Case C":>8} '
              f'{"Case D":>8} {"B red%":>8} {"C red%":>8} {"D red%":>8}')

        for sector in SECTOR_NAMES:
            spreads = {}
            for case in ['A', 'B', 'C', 'D']:
                key = (case, sector, variable)
                if key in mean_spread:
                    spreads[case] = np.mean(mean_spread[key])
                else:
                    spreads[case] = np.nan

            A_val = spreads.get('A', np.nan)
            reductions = {}
            for case in ['B', 'C', 'D']:
                if np.isfinite(A_val) and A_val > 0 and np.isfinite(spreads.get(case, np.nan)):
                    reductions[case] = (A_val - spreads[case]) / A_val * 100
                else:
                    reductions[case] = np.nan

            print(f'  {sector:<25} {spreads.get("A", np.nan):8.4f} '
                  f'{spreads.get("B", np.nan):8.4f} {spreads.get("C", np.nan):8.4f} '
                  f'{spreads.get("D", np.nan):8.4f} '
                  f'{reductions.get("B", np.nan):7.1f}% '
                  f'{reductions.get("C", np.nan):7.1f}% '
                  f'{reductions.get("D", np.nan):7.1f}%')

            attribution_rows.append({
                'sector': sector, 'variable': variable,
                'spread_A': spreads.get('A', np.nan),
                'spread_B': spreads.get('B', np.nan),
                'spread_C': spreads.get('C', np.nan),
                'spread_D': spreads.get('D', np.nan),
                'reduction_B_pct': reductions.get('B', np.nan),
                'reduction_C_pct': reductions.get('C', np.nan),
                'reduction_D_pct': reductions.get('D', np.nan),
                'residual_D': spreads.get('D', np.nan),
            })

    return spread_rows, attribution_rows


# ==============================================================================
# Save results
# ==============================================================================
def save_csv(rows, filename):
    if not rows:
        print(f'  WARNING: No data to save for {filename}')
        return
    filepath = os.path.join(BASE_OUT, filename)
    fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved {len(rows)} rows to {filepath}')


# ==============================================================================
# Thesis-ready summary
# ==============================================================================
def print_thesis_summary(attribution_rows):
    print(f'\n{SEPARATOR}')
    print('  THESIS-READY SUMMARY FOR SECTION 3.4.4')
    print(SEPARATOR)

    hfi_rows = [r for r in attribution_rows if r['variable'] == 'hfi']

    print('\n  Key findings for h_fi inter-product spread:')
    for r in hfi_rows:
        print(f'\n  {r["sector"]}:')
        print(f'    Case A (native):         spread = {r["spread_A"]:.4f} m')
        print(f'    Case B (fix rho_s 300):  spread = {r["spread_B"]:.4f} m  '
              f'(reduction: {r["reduction_B_pct"]:.1f}%)')
        print(f'    Case C (fix h_s to CCI): spread = {r["spread_C"]:.4f} m  '
              f'(reduction: {r["reduction_C_pct"]:.1f}%)')
        print(f'    Case D (fix both):       spread = {r["spread_D"]:.4f} m  '
              f'(reduction: {r["reduction_D_pct"]:.1f}%)')

    print(f'\n  -------------------------------------------------------')
    ww = [r for r in hfi_rows if r['sector'] == 'Western_Weddell'][0]
    ross = [r for r in hfi_rows if r['sector'] == 'Ross'][0]
    print(f'  W. Weddell: Case D reduces h_fi spread by {ww["reduction_D_pct"]:.0f}%, '
          f'from {ww["spread_A"]:.3f} m to {ww["spread_D"]:.3f} m.')
    print(f'  Ross:       Case D reduces h_fi spread by {ross["reduction_D_pct"]:.0f}%, '
          f'from {ross["spread_A"]:.3f} m to {ross["spread_D"]:.3f} m.')
    print(f'  -------------------------------------------------------')


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    print(f'\nNB03_snow_harmonisation.py (v2 — CCI Antarctic density corrected)')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Common snow reference: {SNOW_REF_PRODUCT}')
    print(f'Common density: {COMMON_DENSITY} kg/m3 (CCI Antarctic fixed)')

    print(f'\n{SEPARATOR}')
    print('  Snow Density Parameterisations by Month (kg/m3)')
    print(SEPARATOR)
    print(f'  {"Month":<8} {"CCI(fixed)":>12} {"LEGOS/CSAO":>12} {"CryoTEMPO":>12}'
          f'  {"Mallett(Arctic)":>15}')
    for month in WINTER_MONTHS:
        print(f'  {month:<8} {300.0:12.1f} {kurtz2012_density(month):12.1f} '
              f'{fons2023_density(month):12.1f}  {mallett2020_density(month):15.1f}')

    all_rows, spread_data = run_harmonisation()
    spread_rows, attribution_rows = compute_spread_and_attribution(spread_data)

    print(f'\n{SEPARATOR}')
    print('  Saving results...')
    print(SEPARATOR)
    save_csv(all_rows, 'harmonisation_sector_means.csv')
    save_csv(spread_rows, 'harmonisation_spread.csv')
    save_csv(attribution_rows, 'harmonisation_attribution.csv')

    print_thesis_summary(attribution_rows)

    print(f'\n{SEPARATOR}')
    print('  FINAL OUTPUT SUMMARY')
    print(SEPARATOR)
    print(f'  Output directory: {BASE_OUT}')
    print(f'  Case entries:     {len(all_rows)}')
    print(f'  Spread entries:   {len(spread_rows)}')
    print(f'  Attribution:      {len(attribution_rows)} sector-variable combinations')
    print(f'\n  Ready for NB04_chapter3_figures.py')
    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('Done.')
