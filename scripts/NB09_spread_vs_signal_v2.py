#!/usr/bin/env python3
"""
NB09_spread_vs_signal_v2.py
============================
Inter-product freeboard spread expressed as a fraction of the signals against
which it would ordinarily be interpreted.

MS#1 minor revision -- Xinlong Liu, IMAS, University of Tasmania

Answers Referee 2, general comment 4, and the Results comment on L226.

WHY v2: RECONCILING TWO SPREAD DEFINITIONS
------------------------------------------
v1 of this script reported a Western Weddell h_fi spread of 0.036 m, whereas the
manuscript reports approximately 0.10 m for the same sector and variable. Both
are correct; they are different quantities, and quoting one in a paper that
states the other would be an inconsistency a referee would notice.

NB03 defines spread as

    mean over (year, month) of [ max(products) - min(products) ]

i.e. the RANGE across products at a given month, averaged over the record.

v1 of this script computed

    standard deviation across products of the winter-mean value

which is a STANDARD DEVIATION, and is taken after averaging each product over
the whole record. Two differences at once: range vs standard deviation (for five
approximately normal values the range is roughly 2.3 times the standard
deviation), and instantaneous vs time-averaged. Together these account for the
factor of about three.

Both are informative and they say different things:

  range_monthly   how much the products disagree at any given month. This is
                  the manuscript's convention and the right comparator for
                  "can this signal be detected at all".

  sd_winter       the persistent offset that survives averaging over the whole
                  record. A constant offset displaces absolute freeboard but
                  does not corrupt the shape of a seasonal cycle or a trend.

v2 computes both from the same file, alongside a range-based winter-mean spread
that bridges them, so the decomposition is explicit.

SIGNAL SCALES (both are ranges, so they are dimensionally consistent with
range_monthly):

  seasonal        per product, the range of its monthly climatology across
                  May-October; averaged over products.
  interannual     per product, the range of its annual winter means; averaged
                  over products.

NO NEW ANALYSIS. Reads Case A of the NB03 harmonisation table (harmonisation_sector_means.csv).

Run on Gadi (login node, a few seconds):
    source ~/cryo2ice_env/bin/activate
    cd /g/data/gv90/xl1657/phd/M1_workspace/scripts
    python3 NB09_spread_vs_signal_v2.py 2>&1 | tee ../output/NB09_v3_output.txt

Author: Xinlong Liu | 1 September 2026
"""
import os
import sys

import numpy as np
import pandas as pd

BASE_SECTORS = '/g/data/gv90/xl1657/phd/M1_workspace/processed/sector_means'
OUT_DIR = '/g/data/gv90/xl1657/phd/M1_workspace/processed/trend_sensitivity'

SECTORS = ['Western_Weddell', 'Eastern_Weddell', 'Indian',
           'Western_Pacific', 'Ross', 'Amundsen_Bellingshausen']
LABEL = {'Western_Weddell': 'Western Weddell',
         'Eastern_Weddell': 'Eastern Weddell',
         'Indian': 'Indian',
         'Western_Pacific': 'Western Pacific',
         'Ross': 'Ross',
         'Amundsen_Bellingshausen': 'Amundsen-Bellingshausen'}
MIN_PRODUCTS = 3        # below this, a spread estimate is not meaningful
SEP = '=' * 78


def pick_col(df, prefer):
    for c in (f'{prefer}_mean', f'mean_{prefer}'):
        if c in df.columns:
            return c
    return None


def analyse(df, value_col, era_label):
    rows = []
    for sec in SECTORS:
        g = df[df['sector'] == sec]
        if g.empty:
            continue
        n_prod = g['product'].nunique()
        if n_prod < 2:
            continue

        # --- spread, NB03 convention: range across products per month, averaged
        per_month = g.groupby(['year', 'month'])[value_col]
        range_monthly = float(per_month.agg(lambda x: x.max() - x.min()).mean())

        # --- spread, winter-mean range (bridges the two conventions)
        winter_mean = g.groupby('product')[value_col].mean()
        range_winter = float(winter_mean.max() - winter_mean.min())

        # --- spread, winter-mean standard deviation (v1 convention)
        sd_winter = float(winter_mean.std(ddof=1))

        # --- signal scales, both ranges
        monthly_clim = g.groupby(['product', 'month'])[value_col].mean()
        seasonal = float(monthly_clim.groupby('product')
                                     .agg(lambda x: x.max() - x.min()).mean())
        annual = g.groupby(['product', 'year'])[value_col].mean()
        interannual = float(annual.groupby('product')
                                  .agg(lambda x: x.max() - x.min()).mean())

        def pct(a, b):
            return 100 * a / b if b else np.nan

        rows.append({
            'era': era_label, 'variable': value_col, 'sector': sec,
            'n_products': int(n_prod), 'n_years': int(g['year'].nunique()),
            'reliable': n_prod >= MIN_PRODUCTS,
            'range_monthly_m': range_monthly,
            'range_winter_m': range_winter,
            'sd_winter_m': sd_winter,
            'seasonal_amplitude_m': seasonal,
            'interannual_range_m': interannual,
            'range_monthly_pct_seasonal': pct(range_monthly, seasonal),
            'range_monthly_pct_interannual': pct(range_monthly, interannual),
            'sd_winter_pct_seasonal': pct(sd_winter, seasonal),
            'sd_winter_pct_interannual': pct(sd_winter, interannual),
            'persistent_fraction': pct(range_winter, range_monthly),
        })
    return rows


def report(rows, era_label, value_col):
    if not rows:
        return
    n_prod = rows[0]['n_products']
    flag = '' if rows[0]['reliable'] else \
           f'   [ONLY {n_prod} PRODUCTS -- SPREAD NOT RELIABLE]'
    print(f'\n{SEP}')
    print(f'  {era_label} era, {value_col}{flag}')
    print(SEP)

    print('  Manuscript convention: range across products per month, averaged')
    print(f'  {"Sector":<26}{"spread":>9}{"seasonal":>10}{"  %":>7}'
          f'{"interann":>10}{"  %":>7}')
    print('  ' + '-' * 68)
    for r in rows:
        print(f'  {LABEL[r["sector"]]:<26}'
              f'{r["range_monthly_m"]:9.4f}'
              f'{r["seasonal_amplitude_m"]:10.4f}'
              f'{r["range_monthly_pct_seasonal"]:7.0f}'
              f'{r["interannual_range_m"]:10.4f}'
              f'{r["range_monthly_pct_interannual"]:7.0f}')
    ps = [r['range_monthly_pct_seasonal'] for r in rows]
    pi = [r['range_monthly_pct_interannual'] for r in rows]
    print(f'\n  --> {min(ps):.0f}-{max(ps):.0f}% of the seasonal cycle, '
          f'{min(pi):.0f}-{max(pi):.0f}% of the inter-annual range.')
    if min(ps) > 100 and min(pi) > 100:
        print('      The spread EXCEEDS both signals in EVERY sector.')
    elif min(ps) > 100:
        print('      The spread exceeds the seasonal cycle in every sector.')

    print('\n  Persistent component: winter-mean range as a share of the '
          'monthly range')
    print(f'  {"Sector":<26}{"win range":>11}{"sd(winter)":>12}'
          f'{"persistent %":>14}')
    print('  ' + '-' * 63)
    for r in rows:
        print(f'  {LABEL[r["sector"]]:<26}'
              f'{r["range_winter_m"]:11.4f}'
              f'{r["sd_winter_m"]:12.4f}'
              f'{r["persistent_fraction"]:13.0f}%')
    pf = [r['persistent_fraction'] for r in rows]
    print(f'\n  --> {np.mean(pf):.0f}% of the monthly disagreement persists '
          f'after averaging over the record;')
    print('      the remainder varies month to month and cannot be removed by '
          'a constant offset.')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(SEP)
    print('  NB09 v2: inter-product spread vs seasonal and inter-annual signal')
    print('  Both spread conventions reported. Referee 2, general comment 4.')
    print(SEP)

    all_rows = []
    for era_label, fname in (('CryoSat-2',
                              '../snow_harmonisation/harmonisation_sector_means.csv'),):
        path = os.path.join(BASE_SECTORS, fname)
        if not os.path.exists(path):
            print(f'\n  Skipped {era_label}: {path} not found.')
            continue
        df = pd.read_csv(path)
        if 'case' in df.columns:
            df = df[df['case'] == 'A'].copy()
            print(f'  Filtered to Case A (native snow inputs): {len(df):,} rows')
        prods = sorted(df['product'].unique())
        print(f'\n  {fname}: {len(df):,} rows, {len(prods)} products {prods}')
        if len(prods) < MIN_PRODUCTS:
            print(f'  WARNING: {len(prods)} products only. A spread over fewer '
                  f'than {MIN_PRODUCTS} members is a poor estimator and these')
            print('           ratios should not be quoted alongside the '
                  'CryoSat-2 figures.')
        for var in ('hfi', 'hfr', 'hsc'):
            col = pick_col(df, var)
            if col is None:
                continue
            rows = analyse(df, col, era_label)
            report(rows, era_label, col)
            all_rows.extend(rows)

    if not all_rows:
        sys.exit('\n  No data processed. Check BASE_SECTORS.')

    out = os.path.join(OUT_DIR, 'NB09_spread_vs_signal_v2.csv')
    pd.DataFrame(all_rows).to_csv(out, index=False)

    print(f'\n{SEP}')
    print('  FOR THE MANUSCRIPT')
    print(SEP)
    key = [r for r in all_rows
           if r['era'] == 'CryoSat-2' and r['variable'] == 'hfi_mean']
    if key:
        ps = [r['range_monthly_pct_seasonal'] for r in key]
        pi = [r['range_monthly_pct_interannual'] for r in key]
        print(f'  Quote for h_fi, CryoSat-2 era, manuscript convention:')
        print(f'    {min(ps):.0f}-{max(ps):.0f}% of the seasonal cycle '
              f'amplitude')
        print(f'    {min(pi):.0f}-{max(pi):.0f}% of the inter-annual range')
        print(f'  Cross-check the spread against manuscript L167 '
              f'(approximately 0.10 m, W. Weddell):')
        ww = [r for r in key if r['sector'] == 'Western_Weddell']
        if ww:
            print(f'    computed here: {ww[0]["range_monthly_m"]:.4f} m')
    print(f'\n  Saved: {out}')
    print(SEP + '\n')


if __name__ == '__main__':
    main()
