#!/usr/bin/env python3
"""
NB02_common_mask_and_sectors.py
===============================
Step 1.3: Build common masks and compute sector-mean time series.

PhD Thesis — Xinlong Liu, IMAS, University of Tasmania

Purpose:
  - For each month, build a common mask where ALL products have valid data.
  - Assign each CCI grid cell to one of six sectors using explicit longitude
    boundaries from the thesis (Comiso 2008; Kacimi & Kwok 2020), with a
    latitude cut-off at 50S.
  - Compute sector-mean time series for h_fr, h_fi, h_s, h_sc.
  - Save results as CSV tables for direct use in LaTeX and figures.

Sector definitions (from thesis Section 3.3):
  Western Weddell:        62W -- 40W    (-62 to -40)
  Eastern Weddell:        40W -- 15E    (-40 to  15)
  Indian:                 15E -- 90E    ( 15 to  90)
  Pacific:                90E -- 160E   ( 90 to 160)
  Ross:                  160E -- 140W   (160 to -140, wraps around 180)
  Amundsen-Bellingshausen: 140W -- 62W  (-140 to -62)

Input:  288 regridded .npz files from NB01 in processed/regridded/
Output: CSV files in processed/sector_means/ and common masks in processed/masks/

Run on Gadi:
  python3 NB02_common_mask_and_sectors.py 2>&1 | tee NB02_output.txt
"""

import os
import glob
import numpy as np
import csv
from datetime import datetime

# ==============================================================================
# Configuration
# ==============================================================================
BASE_REGRIDDED = '/g/data/gv90/xl1657/phd/M1_workspace/processed/regridded'
BASE_MASKS     = '/g/data/gv90/xl1657/phd/M1_workspace/processed/masks'
BASE_SECTORS   = '/g/data/gv90/xl1657/phd/M1_workspace/processed/sector_means'

os.makedirs(BASE_MASKS, exist_ok=True)
os.makedirs(BASE_SECTORS, exist_ok=True)

# Study periods
ENV_YEARS = range(2003, 2012)   # 2003-2011
CS2_YEARS = range(2013, 2019)   # 2013-2018
WINTER_MONTHS = range(5, 11)    # May(5) to October(10)

# CryoSat-2 era products (five products on common grid)
CS2_PRODUCTS = ['CCI_CS2', 'LEGOS_I_CS2', 'LEGOS_II_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']

# Envisat era products (two products on common grid)
ENV_PRODUCTS = ['CCI_ENV', 'LEGOS_ENV']

# Sector definitions — explicit longitude boundaries from thesis Section 3.3
# Following Comiso (2008) and Kacimi & Kwok (2020)
SECTOR_DEFS = [
    ('Western_Weddell',          -62.0,  -40.0),
    ('Eastern_Weddell',          -40.0,   15.0),
    ('Indian',                    15.0,   90.0),
    ('Western_Pacific',           90.0,  160.0),
    ('Ross',                     160.0, -140.0),   # wraps around ±180
    ('Amundsen_Bellingshausen', -140.0,  -62.0),
]

SECTOR_NAMES = [s[0] for s in SECTOR_DEFS]
LAT_CUTOFF = -50.0   # study region: 90S to 50S

VARIABLES = ['hfr', 'hfi', 'hs', 'hsc']

SEPARATOR = '=' * 70


# ==============================================================================
# STEP 1: Build sector assignment map from explicit longitude boundaries
# ==============================================================================
def build_sector_map():
    """
    Create a 216x216 sector assignment array using explicit longitude
    boundaries from the thesis (Comiso 2008; Kacimi & Kwok 2020).

    Each grid cell is assigned to a sector based on its longitude if its
    latitude is south of 50S. The Ross sector wraps around the ±180
    meridian and is handled as a special case.

    Returns:
      sector_map: (216, 216) int array, values 0-5 for six sectors,
                  -1 for cells outside the study region
      lat_grid, lon_grid: CCI reference grid coordinates
    """
    print(f'\n{SEPARATOR}')
    print('  Building sector assignment map (longitude-based)...')
    print(SEPARATOR)

    # Load CCI grid coordinates from a regridded file
    cci_file = os.path.join(BASE_REGRIDDED, 'CCI_CS2_201305.npz')
    data = np.load(cci_file)
    lat_grid = data['lat']   # (216, 216)
    lon_grid = data['lon']   # (216, 216)
    data.close()

    # Initialise all cells as excluded
    sector_map = np.full((216, 216), -1, dtype=np.int8)

    # Apply latitude cut-off: only cells south of 50S
    lat_mask = (lat_grid <= LAT_CUTOFF)

    # Assign sectors based on longitude boundaries
    for idx, (name, lon_west, lon_east) in enumerate(SECTOR_DEFS):
        if lon_west < lon_east:
            # Standard case: sector does not cross the ±180 meridian
            sector_mask = lat_mask & (lon_grid >= lon_west) & (lon_grid < lon_east)
        else:
            # Ross sector wraps around ±180: lon >= 160 OR lon < -140
            sector_mask = lat_mask & ((lon_grid >= lon_west) | (lon_grid < lon_east))

        sector_map[sector_mask] = idx

    # Print sector statistics
    print(f'\n  Latitude cut-off: {LAT_CUTOFF} degrees')
    print(f'  Total cells south of {LAT_CUTOFF}S: {np.sum(lat_mask)}')
    print()

    for idx, name in enumerate(SECTOR_NAMES):
        n_cells = np.sum(sector_map == idx)
        lon_west, lon_east = SECTOR_DEFS[idx][1], SECTOR_DEFS[idx][2]

        # Report actual lat/lon ranges of assigned cells
        mask = (sector_map == idx)
        if np.any(mask):
            lat_range = f'[{lat_grid[mask].min():.1f}, {lat_grid[mask].max():.1f}]'
            lon_range = f'[{lon_grid[mask].min():.1f}, {lon_grid[mask].max():.1f}]'
        else:
            lat_range = 'N/A'
            lon_range = 'N/A'

        print(f'  Sector {idx} ({name}):')
        print(f'    Definition: {lon_west}E to {lon_east}E')
        print(f'    Grid cells: {n_cells}')
        print(f'    Lat range:  {lat_range}')
        print(f'    Lon range:  {lon_range}')

    n_assigned = np.sum(sector_map >= 0)
    n_excluded = np.sum(sector_map == -1)
    print(f'\n  Total assigned: {n_assigned} cells')
    print(f'  Excluded (land/ice/north of 50S): {n_excluded} cells')

    # Save sector map
    np.savez_compressed(os.path.join(BASE_MASKS, 'sector_map.npz'),
                        sector_map=sector_map,
                        sector_names=SECTOR_NAMES,
                        sector_defs=np.array([(s[1], s[2]) for s in SECTOR_DEFS]),
                        lat_cutoff=LAT_CUTOFF,
                        lat=lat_grid, lon=lon_grid)
    print('  Sector map saved.')

    return sector_map, lat_grid, lon_grid


# ==============================================================================
# STEP 2: Build common masks per month
# ==============================================================================
def build_common_masks(sector_map, era='CS2'):
    """
    For each month in the study period, build a common mask where ALL
    products have valid h_fr data simultaneously.
    """
    print(f'\n{SEPARATOR}')
    print(f'  Building common masks ({era} era)...')
    print(SEPARATOR)

    if era == 'CS2':
        products = CS2_PRODUCTS
        years = CS2_YEARS
    else:
        products = ENV_PRODUCTS
        years = ENV_YEARS

    masks = {}
    total_months = 0
    total_cells_mean = 0

    for year in years:
        for month in WINTER_MONTHS:
            common_mask = np.ones((216, 216), dtype=bool)

            all_available = True
            for product in products:
                npz_file = os.path.join(BASE_REGRIDDED, f'{product}_{year}{month:02d}.npz')
                if not os.path.exists(npz_file):
                    all_available = False
                    break

                data = np.load(npz_file)
                hfr = data['hfr']
                data.close()

                common_mask &= np.isfinite(hfr)

            if not all_available:
                print(f'    {year}/{month:02d}: MISSING PRODUCT — skipping')
                continue

            common_mask &= (sector_map >= 0)

            n_valid = np.sum(common_mask)
            masks[(year, month)] = common_mask
            total_months += 1
            total_cells_mean += n_valid

            print(f'    {year}/{month:02d}: {n_valid} common valid cells')

    if total_months > 0:
        avg_cells = total_cells_mean / total_months
        print(f'  {era}: {total_months} months, avg {avg_cells:.0f} common cells/month')

    mask_file = os.path.join(BASE_MASKS, f'common_masks_{era}.npz')
    mask_dict = {}
    for (year, month), mask in masks.items():
        mask_dict[f'{year}_{month:02d}'] = mask
    np.savez_compressed(mask_file, **mask_dict)
    print(f'  Common masks saved to {mask_file}')

    return masks


# ==============================================================================
# STEP 3: Compute sector-mean time series
# ==============================================================================
def compute_sector_means(masks, sector_map, era='CS2'):
    """
    For each product, month, sector, and variable, compute the sector mean
    using the common mask.
    """
    print(f'\n{SEPARATOR}')
    print(f'  Computing sector-mean time series ({era} era)...')
    print(SEPARATOR)

    if era == 'CS2':
        products = CS2_PRODUCTS
    else:
        products = ENV_PRODUCTS

    rows = []

    for (year, month), common_mask in sorted(masks.items()):
        for product in products:
            npz_file = os.path.join(BASE_REGRIDDED, f'{product}_{year}{month:02d}.npz')
            if not os.path.exists(npz_file):
                continue

            data = np.load(npz_file)
            fields = {}
            for var in VARIABLES:
                if var in data:
                    fields[var] = data[var]
                else:
                    fields[var] = np.full((216, 216), np.nan)
            data.close()

            for sector_idx, sector_name in enumerate(SECTOR_NAMES):
                mask = common_mask & (sector_map == sector_idx)
                n_cells = np.sum(mask)

                if n_cells == 0:
                    continue

                row = {
                    'era': era,
                    'year': year,
                    'month': month,
                    'product': product,
                    'sector': sector_name,
                    'n_cells': n_cells
                }

                for var in VARIABLES:
                    values = fields[var][mask]
                    valid = np.isfinite(values)
                    if np.sum(valid) > 0:
                        row[f'{var}_mean'] = float(np.mean(values[valid]))
                        row[f'{var}_std']  = float(np.std(values[valid]))
                        row[f'{var}_median'] = float(np.median(values[valid]))
                    else:
                        row[f'{var}_mean'] = np.nan
                        row[f'{var}_std']  = np.nan
                        row[f'{var}_median'] = np.nan

                rows.append(row)

    print(f'  Computed {len(rows)} sector-product-month entries for {era} era.')
    return rows


# ==============================================================================
# STEP 4: Compute multi-year winter means per sector
# ==============================================================================
def compute_multiyear_means(rows, era='CS2'):
    """
    Compute multi-year austral winter means (May-Oct average per year,
    then average across years) for each product and sector.
    """
    print(f'\n{SEPARATOR}')
    print(f'  Computing multi-year winter means ({era} era)...')
    print(SEPARATOR)

    era_rows = [r for r in rows if r['era'] == era]

    from collections import defaultdict
    yearly_means = defaultdict(list)

    for r in era_rows:
        key = (r['product'], r['sector'])
        year = r['year']

        for var in VARIABLES:
            val = r[f'{var}_mean']
            if np.isfinite(val):
                yearly_means[(key, year, var)].append(val)

    winter_yearly = defaultdict(list)
    for (key, year, var), monthly_vals in yearly_means.items():
        if len(monthly_vals) > 0:
            winter_mean = np.mean(monthly_vals)
            winter_yearly[(key, var)].append(winter_mean)

    summary_rows = []
    for (key, var), yearly_vals in sorted(winter_yearly.items()):
        product, sector = key
        if len(yearly_vals) > 0:
            multiyear_mean = np.mean(yearly_vals)
            multiyear_std  = np.std(yearly_vals)
            summary_rows.append({
                'era': era,
                'product': product,
                'sector': sector,
                'variable': var,
                'multiyear_mean': multiyear_mean,
                'multiyear_std': multiyear_std,
                'n_years': len(yearly_vals)
            })

    return summary_rows


# ==============================================================================
# STEP 5: Compute inter-product spread
# ==============================================================================
def compute_spread(rows, era='CS2'):
    """
    Compute inter-product spread (max - min across products) for each
    month, sector, and variable. This is the Case A baseline spread.
    """
    print(f'\n{SEPARATOR}')
    print(f'  Computing inter-product spread ({era} era)...')
    print(SEPARATOR)

    era_rows = [r for r in rows if r['era'] == era]

    from collections import defaultdict
    grouped = defaultdict(list)

    for r in era_rows:
        key = (r['year'], r['month'], r['sector'])
        grouped[key].append(r)

    spread_rows = []
    for (year, month, sector), product_rows in sorted(grouped.items()):
        for var in VARIABLES:
            values = [r[f'{var}_mean'] for r in product_rows if np.isfinite(r[f'{var}_mean'])]
            if len(values) >= 2:
                spread = max(values) - min(values)
                spread_rows.append({
                    'era': era,
                    'year': year,
                    'month': month,
                    'sector': sector,
                    'variable': var,
                    'spread': spread,
                    'n_products': len(values),
                    'max_val': max(values),
                    'min_val': min(values)
                })

    print(f'  Computed {len(spread_rows)} spread entries for {era} era.')
    return spread_rows


# ==============================================================================
# STEP 6: Compute Envisat-to-CryoSat-2 transition step
# ==============================================================================
def compute_cross_mission_step(env_summary, cs2_summary):
    """
    Compute the cross-mission transition step (CS2 mean - ENV mean)
    for LEGOS and CCI, the two products available in both eras.
    """
    print(f'\n{SEPARATOR}')
    print('  Computing Envisat-to-CryoSat-2 transition step...')
    print(SEPARATOR)

    cross_products = {
        'CCI': ('CCI_ENV', 'CCI_CS2'),
        'LEGOS': ('LEGOS_ENV', 'LEGOS_I_CS2'),
    }

    step_rows = []

    for label, (env_prod, cs2_prod) in cross_products.items():
        for sector in SECTOR_NAMES:
            for var in VARIABLES:
                env_vals = [r['multiyear_mean'] for r in env_summary
                           if r['product'] == env_prod and r['sector'] == sector
                           and r['variable'] == var]
                cs2_vals = [r['multiyear_mean'] for r in cs2_summary
                           if r['product'] == cs2_prod and r['sector'] == sector
                           and r['variable'] == var]

                if env_vals and cs2_vals:
                    step = cs2_vals[0] - env_vals[0]
                    step_rows.append({
                        'product': label,
                        'sector': sector,
                        'variable': var,
                        'env_mean': env_vals[0],
                        'cs2_mean': cs2_vals[0],
                        'step': step
                    })
                    print(f'    {label} {sector} {var}: '
                          f'ENV={env_vals[0]:.4f}, CS2={cs2_vals[0]:.4f}, '
                          f'step={step:+.4f} m')

    return step_rows


# ==============================================================================
# STEP 7: Save all results to CSV
# ==============================================================================
def save_csv(rows, filename, fieldnames=None):
    """Save a list of dicts to CSV."""
    if not rows:
        print(f'  WARNING: No data to save for {filename}')
        return

    filepath = os.path.join(BASE_SECTORS, filename)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'  Saved {len(rows)} rows to {filepath}')


# ==============================================================================
# STEP 8: Print summary tables for thesis
# ==============================================================================
def print_summary_tables(cs2_summary, env_summary, spread_rows_cs2):
    """Print formatted summary tables matching Chapter 3 Tables 3.1-3.2."""
    print(f'\n{SEPARATOR}')
    print('  TABLE 3.1: Multi-Year Mean Freeboards (m) — CryoSat-2 Era')
    print(SEPARATOR)

    cs2_prods = ['LEGOS_I_CS2', 'LEGOS_II_CS2', 'CCI_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']

    for var, var_label in [('hfr', 'h_fr'), ('hfi', 'h_fi'), ('hsc', 'h_sc')]:
        print(f'\n  --- {var_label} (m) ---')
        print(f'  {"Sector":<25} ', end='')
        for prod in cs2_prods:
            short = (prod.replace('_CS2', '')
                        .replace('LEGOS_I', 'LEG-I')
                        .replace('LEGOS_II', 'LEG-II')
                        .replace('CryoTEMPO', 'CryoT'))
            print(f'{short:>8}', end='')
        print()

        for sector in SECTOR_NAMES:
            print(f'  {sector:<25} ', end='')
            for prod in cs2_prods:
                vals = [r['multiyear_mean'] for r in cs2_summary
                       if r['product'] == prod and r['sector'] == sector
                       and r['variable'] == var]
                if vals:
                    print(f'{vals[0]:8.3f}', end='')
                else:
                    print(f'{"N/A":>8}', end='')
            print()

    # Envisat era
    print(f'\n{SEPARATOR}')
    print('  TABLE 3.1 (continued): Multi-Year Mean Freeboards (m) — Envisat Era')
    print(SEPARATOR)

    env_prods = ['LEGOS_ENV', 'CCI_ENV']

    for var, var_label in [('hfr', 'h_fr'), ('hfi', 'h_fi'), ('hsc', 'h_sc')]:
        print(f'\n  --- {var_label} (m) ---')
        print(f'  {"Sector":<25} ', end='')
        for prod in env_prods:
            short = prod.replace('_ENV', '')
            print(f'{short:>8}', end='')
        print()

        for sector in SECTOR_NAMES:
            print(f'  {sector:<25} ', end='')
            for prod in env_prods:
                vals = [r['multiyear_mean'] for r in env_summary
                       if r['product'] == prod and r['sector'] == sector
                       and r['variable'] == var]
                if vals:
                    print(f'{vals[0]:8.3f}', end='')
                else:
                    print(f'{"N/A":>8}', end='')
            print()

    # Inter-product spread summary
    print(f'\n{SEPARATOR}')
    print('  CryoSat-2 Era: Mean Monthly Inter-Product Spread (m)')
    print(SEPARATOR)

    from collections import defaultdict

    for var, var_label in [('hfr', 'h_fr'), ('hfi', 'h_fi'), ('hsc', 'h_sc')]:
        print(f'\n  --- {var_label} ---')
        spread_by_sector = defaultdict(list)
        for r in spread_rows_cs2:
            if r['variable'] == var:
                spread_by_sector[r['sector']].append(r['spread'])

        for sector in SECTOR_NAMES:
            vals = spread_by_sector[sector]
            if vals:
                mean_spread = np.mean(vals)
                max_spread  = np.max(vals)
                min_spread  = np.min(vals)
                print(f'  {sector:<25}: mean={mean_spread:.4f}, '
                      f'range=[{min_spread:.4f}, {max_spread:.4f}] m')


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    print(f'\nNB02_common_mask_and_sectors.py')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # Step 1: Build sector map
    sector_map, lat_grid, lon_grid = build_sector_map()

    # Step 2: Build common masks
    masks_cs2 = build_common_masks(sector_map, era='CS2')
    masks_env = build_common_masks(sector_map, era='ENV')

    # Step 3: Compute sector means
    rows_cs2 = compute_sector_means(masks_cs2, sector_map, era='CS2')
    rows_env = compute_sector_means(masks_env, sector_map, era='ENV')

    # Step 4: Compute multi-year winter means
    summary_cs2 = compute_multiyear_means(rows_cs2, era='CS2')
    summary_env = compute_multiyear_means(rows_env, era='ENV')

    # Step 5: Compute inter-product spread
    spread_cs2 = compute_spread(rows_cs2, era='CS2')
    spread_env = compute_spread(rows_env, era='ENV')

    # Step 6: Cross-mission transition step
    step_rows = compute_cross_mission_step(summary_env, summary_cs2)

    # Step 7: Save all results
    print(f'\n{SEPARATOR}')
    print('  Saving all results to CSV...')
    print(SEPARATOR)

    save_csv(rows_cs2, 'sector_means_monthly_CS2.csv')
    save_csv(rows_env, 'sector_means_monthly_ENV.csv')
    save_csv(summary_cs2, 'multiyear_means_CS2.csv')
    save_csv(summary_env, 'multiyear_means_ENV.csv')
    save_csv(spread_cs2, 'interproduct_spread_CS2.csv')
    save_csv(spread_env, 'interproduct_spread_ENV.csv')
    save_csv(step_rows, 'cross_mission_step.csv')

    # Step 8: Print summary tables
    print_summary_tables(summary_cs2, summary_env, spread_cs2)

    # Final summary
    print(f'\n{SEPARATOR}')
    print('  FINAL OUTPUT SUMMARY')
    print(SEPARATOR)
    print(f'  Masks directory:   {BASE_MASKS}')
    print(f'  Sectors directory: {BASE_SECTORS}')
    print(f'  Monthly means:     {len(rows_cs2)} CS2 + {len(rows_env)} ENV entries')
    print(f'  Multi-year means:  {len(summary_cs2)} CS2 + {len(summary_env)} ENV entries')
    print(f'  Spread entries:    {len(spread_cs2)} CS2 + {len(spread_env)} ENV entries')
    print(f'  Cross-mission:     {len(step_rows)} transition step entries')
    print(f'\n  Ready for NB03_snow_harmonisation.py')

    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('Done.')
