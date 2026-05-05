#!/usr/bin/env python3
"""
NB05_bootstrap_uncertainty.py
==============================
Step 1.5: Year-level bootstrap uncertainty quantification for snow-harmonisation
attribution and sector-mean freeboard variables.

PhD Thesis -- Xinlong Liu, IMAS, University of Tasmania

Purpose:
  - Address the expert reviewer's Tier 2 reproducibility requirement (T2.2,
    T2.3) by quantifying sampling uncertainty in:
      (a) inter-product spread reductions for Cases B, C, D relative to
          Case A (used as confidence interval whiskers on the harmonisation
          bar charts in NB04);
      (b) sector-mean h_fi, h_fr, h_sc values per product (used as
          confidence envelopes around the three-sector time-series figure
          in NB04).

  - Bootstrap procedure: year-level resampling with replacement within each
    mission era. In each realisation, winter years are sampled with
    replacement and the May-October monthly structure within each selected
    year is preserved. This avoids treating individual grid cells or
    adjacent months as independent observations and produces an uncertainty
    estimate appropriate for the sector-scale comparisons used in the paper.

  - 1000 bootstrap realisations. Reports the 2.5th and 97.5th percentiles
    of the resulting distribution as the 95% confidence interval.

Inputs:
  - processed/snow_harmonisation/harmonisation_sector_means.csv  (from NB03)
    Columns: case, year, month, product, sector, n_cells, hfr_mean,
             hsc_mean, hfi_mean

Outputs:
  - processed/snow_harmonisation/bootstrap_attribution_ci.csv
    Columns: sector, variable, case, reduction_mean, reduction_lo,
             reduction_hi, spread_mean, spread_lo, spread_hi

  - processed/snow_harmonisation/bootstrap_sectormean_ci.csv
    Columns: era, product, sector, variable, year, month, value_mean,
             value_lo, value_hi

Run on Gadi:
  nohup bash -c 'source ~/cryo2ice_env/bin/activate && \\
  python3 NB05_bootstrap_uncertainty.py' > NB05_output.txt 2>&1 &

  Expected runtime: ~5-10 minutes for 1000 realisations.
"""

import os
import csv
import numpy as np
from datetime import datetime
from collections import defaultdict

# ==============================================================================
# Configuration
# ==============================================================================
BASE_HARMON  = '/g/data/gv90/xl1657/phd/M1_workspace/processed/snow_harmonisation'
BASE_SECTORS = '/g/data/gv90/xl1657/phd/M1_workspace/processed/sector_means'
BASE_OUT     = BASE_HARMON

# Bootstrap configuration
N_BOOTSTRAP = 1000
CI_LO_PCT   = 2.5    # 2.5th percentile
CI_HI_PCT   = 97.5   # 97.5th percentile
RANDOM_SEED = 20260430  # for reproducibility

# Mission eras
CS2_YEARS = list(range(2013, 2019))
ENV_YEARS = list(range(2003, 2012))
WINTER_MONTHS = list(range(5, 11))

CS2_PRODUCTS = ['CCI_CS2', 'LEGOS_I_CS2', 'LEGOS_II_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']
ENV_PRODUCTS = ['CCI_ENV', 'LEGOS_ENV']

CASES = ['A', 'B', 'C', 'D']
VARIABLES = ['hfr', 'hsc', 'hfi']

SECTOR_NAMES = [
    'Western_Weddell', 'Eastern_Weddell', 'Indian',
    'Western_Pacific', 'Ross', 'Amundsen_Bellingshausen',
]

SEPARATOR = '=' * 70


# ==============================================================================
# Load harmonisation sector-means CSV from NB03
# ==============================================================================
def load_harmonisation_csv():
    """
    Load the per-(case, year, month, product, sector) sector-mean values
    written by NB03. Returns a nested dict:
        data[case][(year, month)][product][sector][variable] = value
    """
    filepath = os.path.join(BASE_HARMON, 'harmonisation_sector_means.csv')
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    n_rows = 0

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            case = r['case']
            year = int(r['year'])
            month = int(r['month'])
            product = r['product']
            sector = r['sector']
            try:
                hfr = float(r['hfr_mean'])
                hsc = float(r['hsc_mean'])
                hfi = float(r['hfi_mean'])
            except (ValueError, TypeError):
                continue

            data[case][(year, month)][product][sector] = {
                'hfr': hfr, 'hsc': hsc, 'hfi': hfi,
            }
            n_rows += 1

    print(f'  Loaded {n_rows} rows from harmonisation_sector_means.csv')
    return data


def load_sector_means_csv(filename):
    """
    Load the sector-mean monthly time series written by NB02. Returns:
        rows[era_key] = list of dicts with keys
            year, month, product, sector, hfr_mean, hsc_mean, hfi_mean
    """
    filepath = os.path.join(BASE_SECTORS, filename)
    rows = []
    if not os.path.exists(filepath):
        print(f'  WARNING: file not found: {filepath}')
        return rows

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                row = {
                    'year': int(r['year']),
                    'month': int(r['month']),
                    'product': r['product'],
                    'sector': r['sector'],
                }
                for v in VARIABLES:
                    key = f'{v}_mean'
                    row[v] = float(r[key]) if r.get(key) not in (None, '') else np.nan
                rows.append(row)
            except (ValueError, TypeError, KeyError):
                continue

    print(f'  Loaded {len(rows)} rows from {filename}')
    return rows


# ==============================================================================
# PART A: Bootstrap on inter-product spread and reduction percentages
# ==============================================================================
def compute_spread_for_year_subset(data, case, sector, variable, year_subset):
    """
    Compute inter-product spread for a given (case, sector, variable) using
    only the months whose year is in year_subset. Spread is defined as
    (max - min) across products within each month, then averaged across
    months in year_subset.

    This matches the spread definition used in NB03's compute_spread_and_attribution.
    """
    monthly_spreads = []
    for year in year_subset:
        for month in WINTER_MONTHS:
            ym_key = (year, month)
            if ym_key not in data[case]:
                continue
            product_dict = data[case][ym_key]
            values = []
            for product in CS2_PRODUCTS:
                if product in product_dict and sector in product_dict[product]:
                    val = product_dict[product][sector].get(variable)
                    if val is not None and np.isfinite(val):
                        values.append(val)
            if len(values) >= 2:
                monthly_spreads.append(max(values) - min(values))

    if monthly_spreads:
        return float(np.mean(monthly_spreads))
    return np.nan


def bootstrap_attribution(data):
    """
    For each (sector, variable, case in {B, C, D}):
      - resample CS2_YEARS with replacement N_BOOTSTRAP times
      - compute Case A and Case X spread for the resampled year set
      - compute reduction percentage = (spread_A - spread_X) / spread_A * 100
      - report mean, 2.5th, 97.5th percentiles across bootstrap realisations

    Returns a list of dicts ready for CSV writing.
    """
    print(f'\n{SEPARATOR}')
    print(f'  Part A: Bootstrap on inter-product spread reductions')
    print(f'    {N_BOOTSTRAP} realisations, year-level resampling')
    print(SEPARATOR)

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    n_combinations = len(SECTOR_NAMES) * len(VARIABLES) * len(CASES)
    counter = 0

    for sector in SECTOR_NAMES:
        for variable in VARIABLES:
            # Generate the same bootstrap year sets for all four cases
            bootstrap_year_sets = [
                rng.choice(CS2_YEARS, size=len(CS2_YEARS), replace=True).tolist()
                for _ in range(N_BOOTSTRAP)
            ]

            for case in CASES:
                counter += 1
                spread_realisations = np.full(N_BOOTSTRAP, np.nan)
                for i, year_set in enumerate(bootstrap_year_sets):
                    spread_realisations[i] = compute_spread_for_year_subset(
                        data, case, sector, variable, year_set)

                valid = spread_realisations[np.isfinite(spread_realisations)]
                if len(valid) > 0:
                    spread_mean = float(np.mean(valid))
                    spread_lo = float(np.percentile(valid, CI_LO_PCT))
                    spread_hi = float(np.percentile(valid, CI_HI_PCT))
                else:
                    spread_mean = spread_lo = spread_hi = np.nan

                # Compute reduction percentage relative to Case A
                if case == 'A':
                    reduction_mean = 0.0
                    reduction_lo = 0.0
                    reduction_hi = 0.0
                else:
                    a_realisations = np.full(N_BOOTSTRAP, np.nan)
                    for i, year_set in enumerate(bootstrap_year_sets):
                        a_realisations[i] = compute_spread_for_year_subset(
                            data, 'A', sector, variable, year_set)

                    reduction_realisations = np.full(N_BOOTSTRAP, np.nan)
                    for i in range(N_BOOTSTRAP):
                        a_val = a_realisations[i]
                        x_val = spread_realisations[i]
                        if np.isfinite(a_val) and a_val > 0 and np.isfinite(x_val):
                            reduction_realisations[i] = (a_val - x_val) / a_val * 100

                    valid_red = reduction_realisations[np.isfinite(reduction_realisations)]
                    if len(valid_red) > 0:
                        reduction_mean = float(np.mean(valid_red))
                        reduction_lo = float(np.percentile(valid_red, CI_LO_PCT))
                        reduction_hi = float(np.percentile(valid_red, CI_HI_PCT))
                    else:
                        reduction_mean = reduction_lo = reduction_hi = np.nan

                results.append({
                    'sector': sector,
                    'variable': variable,
                    'case': case,
                    'spread_mean': spread_mean,
                    'spread_lo': spread_lo,
                    'spread_hi': spread_hi,
                    'reduction_mean': reduction_mean,
                    'reduction_lo': reduction_lo,
                    'reduction_hi': reduction_hi,
                })

                if counter % 12 == 0:
                    print(f'    Progress: {counter:3d}/{n_combinations} '
                          f'({sector[:18]:<18} {variable} Case {case})')

    print(f'  Completed {len(results)} (sector, variable, case) combinations')
    return results


# ==============================================================================
# PART B: Bootstrap on sector-mean monthly values
# ==============================================================================
def bootstrap_sector_means(rows, era_label, era_years):
    """
    For each (product, sector, variable, year, month), the sector-mean comes
    from spatial averaging of all common-mask grid cells within that
    (year, month). The bootstrap here resamples those grid cells with
    replacement to estimate the spatial-sampling uncertainty of each monthly
    sector mean.

    NOTE: NB02 already collapses each (year, month) to a single sector-mean
    number. Without re-running the spatial averaging from regridded fields,
    we cannot directly resample grid cells. As a pragmatic alternative that
    serves the time-series figure use case, we report inter-monthly variability
    within each (year, month) -> not meaningful, only one number per cell.

    Therefore, we instead bootstrap the YEAR dimension within each
    (product, sector, variable, month) to provide a confidence envelope on the
    multi-year-mean monthly climatology shown in time-series figures.

    For per-month-per-year confidence envelopes (which is what the time-series
    figure actually shows), the appropriate uncertainty is the inter-product
    spread within that (year, month) cell. This is computed and reported as
    an alternative confidence indicator.

    Returns two output structures:
      (1) climatology_ci: bootstrap CI on multi-year mean for each
          (product, sector, variable, month)
      (2) interproduct_spread: per (year, month, sector, variable),
          inter-product spread metrics
    """
    print(f'\n{SEPARATOR}')
    print(f'  Part B ({era_label}): Bootstrap on sector-mean climatologies')
    print(SEPARATOR)

    # Index rows by (product, sector, variable, month) -> list of (year, value)
    indexed = defaultdict(list)
    for r in rows:
        for v in VARIABLES:
            val = r.get(v)
            if val is not None and np.isfinite(val):
                key = (r['product'], r['sector'], v, r['month'])
                indexed[key].append((r['year'], val))

    rng = np.random.default_rng(RANDOM_SEED + 1)
    climatology_results = []

    for key, year_value_list in indexed.items():
        product, sector, variable, month = key
        years = np.array([yv[0] for yv in year_value_list])
        values = np.array([yv[1] for yv in year_value_list])

        if len(values) < 2:
            mean_val = float(values[0]) if len(values) == 1 else np.nan
            lo_val = hi_val = mean_val
        else:
            n = len(values)
            boot_means = np.full(N_BOOTSTRAP, np.nan)
            for i in range(N_BOOTSTRAP):
                idx = rng.integers(0, n, size=n)
                boot_means[i] = float(np.mean(values[idx]))

            mean_val = float(np.mean(values))  # point estimate
            lo_val = float(np.percentile(boot_means, CI_LO_PCT))
            hi_val = float(np.percentile(boot_means, CI_HI_PCT))

        climatology_results.append({
            'era': era_label,
            'product': product,
            'sector': sector,
            'variable': variable,
            'month': month,
            'n_years': len(values),
            'value_mean': mean_val,
            'value_lo': lo_val,
            'value_hi': hi_val,
        })

    print(f'  Computed climatology CI for {len(climatology_results)} '
          f'(product, sector, variable, month) combinations')

    # Per-monthly inter-product spread (for time-series figure envelopes)
    monthly_spread_results = []
    by_year_month = defaultdict(list)
    for r in rows:
        for v in VARIABLES:
            val = r.get(v)
            if val is not None and np.isfinite(val):
                by_year_month[(r['year'], r['month'], r['sector'], v)].append(
                    (r['product'], val))

    for key, prod_value_list in by_year_month.items():
        year, month, sector, variable = key
        values = np.array([pv[1] for pv in prod_value_list])
        if len(values) >= 2:
            mean_val = float(np.mean(values))
            min_val = float(np.min(values))
            max_val = float(np.max(values))
            std_val = float(np.std(values, ddof=1))
            n_prod = len(values)
        else:
            mean_val = float(values[0]) if len(values) == 1 else np.nan
            min_val = max_val = mean_val
            std_val = 0.0
            n_prod = len(values)

        monthly_spread_results.append({
            'era': era_label,
            'year': year,
            'month': month,
            'sector': sector,
            'variable': variable,
            'n_products': n_prod,
            'mean': mean_val,
            'min': min_val,
            'max': max_val,
            'std': std_val,
        })

    print(f'  Computed inter-product spread for {len(monthly_spread_results)} '
          f'(year, month, sector, variable) combinations')

    return climatology_results, monthly_spread_results


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


def print_attribution_summary(attribution_rows):
    print(f'\n{SEPARATOR}')
    print('  THESIS-READY SUMMARY: Bootstrap CIs on h_fi spread reductions')
    print(SEPARATOR)
    print(f'  {"Sector":<25} {"Case":>5} {"Reduction (%)":>20} '
          f'{"95% CI":>20}')
    print(f'  {"-" * 70}')

    for sector in SECTOR_NAMES:
        for case in ['B', 'C', 'D']:
            row = next(
                (r for r in attribution_rows
                 if r['sector'] == sector
                 and r['variable'] == 'hfi'
                 and r['case'] == case),
                None,
            )
            if row is None:
                continue
            mean_str = f'{row["reduction_mean"]:>7.1f}'
            ci_str = f'[{row["reduction_lo"]:6.1f}, {row["reduction_hi"]:6.1f}]'
            print(f'  {sector:<25} {case:>5} {mean_str:>20} {ci_str:>20}')


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f'\nNB05_bootstrap_uncertainty.py')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Random seed: {RANDOM_SEED}')
    print(f'Bootstrap realisations: {N_BOOTSTRAP}')
    print(f'CI percentiles: [{CI_LO_PCT}, {CI_HI_PCT}]')

    # Part A: bootstrap on harmonisation attribution
    print(f'\n{SEPARATOR}')
    print('  Loading harmonisation_sector_means.csv ...')
    print(SEPARATOR)
    harmon_data = load_harmonisation_csv()

    attribution_rows = bootstrap_attribution(harmon_data)
    save_csv(attribution_rows, 'bootstrap_attribution_ci.csv')
    print_attribution_summary(attribution_rows)

    # Part B: bootstrap on sector-mean monthly time series
    print(f'\n{SEPARATOR}')
    print('  Loading sector-mean monthly time series for both eras ...')
    print(SEPARATOR)

    cs2_rows = load_sector_means_csv('sector_means_monthly_CS2.csv')
    env_rows = load_sector_means_csv('sector_means_monthly_ENV.csv')

    cs2_clim, cs2_spread = bootstrap_sector_means(cs2_rows, 'CS2', CS2_YEARS)
    env_clim, env_spread = bootstrap_sector_means(env_rows, 'ENV', ENV_YEARS)

    save_csv(cs2_clim + env_clim, 'bootstrap_climatology_ci.csv')
    save_csv(cs2_spread + env_spread, 'bootstrap_monthly_interproduct_spread.csv')

    print(f'\n{SEPARATOR}')
    print('  FINAL OUTPUT SUMMARY')
    print(SEPARATOR)
    print(f'  Output directory: {BASE_OUT}')
    print(f'  Files written:')
    for fn in [
        'bootstrap_attribution_ci.csv',
        'bootstrap_climatology_ci.csv',
        'bootstrap_monthly_interproduct_spread.csv',
    ]:
        path = os.path.join(BASE_OUT, fn)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f'    {fn} ({size_kb:.1f} KB)')

    print(f'\n  Ready for NB04 to read these files and add CI whiskers / envelopes')
    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('Done.')


if __name__ == '__main__':
    main()
