#!/usr/bin/env python3
"""
NB10_reconcile_spread_sources.py
=================================
Diagnostic: reconcile the inter-product spread computed from two different
NB02/NB03 outputs, which disagree by about 14%.

MS#1 minor revision -- Xinlong Liu, IMAS, University of Tasmania

THE PROBLEM
-----------
Two files should give the same Case A inter-product spread, and do not:

    harmonisation_sector_means.csv   (NB03)  Western Weddell h_fi  0.0985 m
    sector_means_monthly_CS2.csv     (NB02)  Western Weddell h_fi  0.0856 m

Both use the same definition -- the mean over (year, month) of the across-product
range -- so the difference must lie in the inputs, not the arithmetic. The
manuscript quotes "approximately 0.10 m" at L167, which the first supports and
the second does not.

This matters because the response to Referee 2's general comment 4 expresses the
spread as a percentage of the seasonal and inter-annual signal. Whichever file
those percentages come from must be the file the manuscript's existing spread
numbers came from, or the paper will contain two spread values differing by 14%.

WHAT IS TESTED
--------------
Five candidate explanations, in the order they are most likely:

  1. Different product sets (e.g. CSAO h_fi present in one, absent in the other)
  2. Different numbers of (year, month) combinations
  3. Different per-cell sample counts, implying a different mask
  4. Systematic offset in the values themselves for a shared product
  5. Arithmetic: recompute the spread from both files under one definition

Nothing is modified. The script reports and diagnoses only.

Run on Gadi (login node, a few seconds):
    source ~/cryo2ice_env/bin/activate
    cd /g/data/gv90/xl1657/phd/M1_workspace/scripts
    python3 NB10_reconcile_spread_sources.py 2>&1 | tee ../output/NB10_output.txt

Author: Xinlong Liu | 1 September 2026
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = '/g/data/gv90/xl1657/phd/M1_workspace/processed'
PATH_HARM = os.path.join(BASE, 'snow_harmonisation',
                         'harmonisation_sector_means.csv')
PATH_SECT = os.path.join(BASE, 'sector_means', 'sector_means_monthly_CS2.csv')

SECTORS = ['Western_Weddell', 'Eastern_Weddell', 'Indian',
           'Western_Pacific', 'Ross', 'Amundsen_Bellingshausen']
VAR = 'hfi_mean'
SEP = '=' * 78


def spread_mean_range(df, sector, value_col):
    """Mean over (year, month) of the across-product range. NB03 convention."""
    g = df[df['sector'] == sector]
    if g.empty:
        return np.nan, 0
    per = g.groupby(['year', 'month'])[value_col].agg(
        lambda x: x.max() - x.min() if len(x) >= 2 else np.nan)
    return float(per.mean()), int(per.notna().sum())


def main():
    for p in (PATH_HARM, PATH_SECT):
        if not os.path.exists(p):
            sys.exit(f'Missing {p}')

    h = pd.read_csv(PATH_HARM)
    s = pd.read_csv(PATH_SECT)
    hA = h[h['case'] == 'A'].copy()

    print(SEP)
    print('  NB10: reconciling inter-product spread between two sources')
    print(SEP)
    print(f'  harmonisation_sector_means.csv : {len(h):,} rows '
          f'({len(hA):,} in Case A)')
    print(f'  sector_means_monthly_CS2.csv   : {len(s):,} rows')
    print(f'  harmonisation columns: {list(h.columns)}')
    print(f'  sector_means columns : {list(s.columns)}')

    # ---- 1. product sets -------------------------------------------------
    print(f'\n{SEP}\n  1. Product sets\n{SEP}')
    ph, ps = sorted(hA['product'].unique()), sorted(s['product'].unique())
    print(f'  harmonisation Case A : {len(ph)}  {ph}')
    print(f'  sector_means         : {len(ps)}  {ps}')
    only_h, only_s = set(ph) - set(ps), set(ps) - set(ph)
    if only_h or only_s:
        print(f'  DIFFERENT. Only in harmonisation: {sorted(only_h) or "none"}')
        print(f'             Only in sector_means : {sorted(only_s) or "none"}')
        print('  --> This alone can explain the discrepancy: a range across')
        print('      five products exceeds a range across four.')
    else:
        print('  Identical product sets.')

    # ---- 2. temporal coverage -------------------------------------------
    print(f'\n{SEP}\n  2. Temporal coverage\n{SEP}')
    th = set(map(tuple, hA[['year', 'month']].drop_duplicates().values))
    ts = set(map(tuple, s[['year', 'month']].drop_duplicates().values))
    print(f'  harmonisation : {len(th)} (year, month) combinations')
    print(f'  sector_means  : {len(ts)} (year, month) combinations')
    if th != ts:
        print(f'  DIFFERENT. Only in harmonisation: {sorted(th - ts)}')
        print(f'             Only in sector_means : {sorted(ts - th)}')
    else:
        print('  Identical coverage.')

    # ---- 3. sample counts (mask) ----------------------------------------
    print(f'\n{SEP}\n  3. Cell counts per sector (mask check)\n{SEP}')
    if 'n_cells' in hA.columns and 'n_cells' in s.columns:
        print(f'  {"Sector":<26}{"harm mean":>12}{"sect mean":>12}{"ratio":>9}')
        print('  ' + '-' * 59)
        for sec in SECTORS:
            a = hA[hA.sector == sec]['n_cells'].mean()
            b = s[s.sector == sec]['n_cells'].mean()
            r = a / b if b else np.nan
            flag = '' if 0.99 < r < 1.01 else '   <-- DIFFERENT MASK'
            print(f'  {sec:<26}{a:12.1f}{b:12.1f}{r:9.3f}{flag}')
    else:
        print('  n_cells not present in both files; skipped.')

    # ---- 4. shared-product value comparison ------------------------------
    print(f'\n{SEP}\n  4. Values for products present in both\n{SEP}')
    shared = sorted(set(ph) & set(ps))
    if shared and VAR in hA.columns and VAR in s.columns:
        key = ['year', 'month', 'product', 'sector']
        m = hA[key + [VAR]].merge(s[key + [VAR]], on=key,
                                  suffixes=('_harm', '_sect'))
        if len(m):
            d = m[f'{VAR}_harm'] - m[f'{VAR}_sect']
            print(f'  Matched {len(m):,} rows on {shared}')
            print(f'  mean difference : {d.mean():+.6f} m')
            print(f'  max |difference|: {d.abs().max():.6f} m')
            print(f'  rows differing by >1 mm: {(d.abs() > 1e-3).sum():,}')
            if d.abs().max() < 1e-9:
                print('  --> Values are identical. The discrepancy is NOT in '
                      'the values.')
            else:
                print('  --> Values differ. Inspect per product:')
                for p in shared:
                    dp = d[m['product'] == p]
                    print(f'      {p:<18} mean {dp.mean():+.6f}  '
                          f'max |d| {dp.abs().max():.6f}')
        else:
            print('  No rows matched on (year, month, product, sector).')
    else:
        print(f'  No shared products, or {VAR} absent from one file.')

    # ---- 5. spread under one definition ----------------------------------
    print(f'\n{SEP}\n  5. Spread recomputed identically from both files\n{SEP}')
    print(f'  {"Sector":<26}{"harm":>10}{"n":>5}{"sect":>10}{"n":>5}'
          f'{"ratio":>9}')
    print('  ' + '-' * 65)
    rows = []
    for sec in SECTORS:
        a, na = spread_mean_range(hA, sec, VAR)
        b, nb = spread_mean_range(s, sec, VAR)
        r = a / b if b else np.nan
        print(f'  {sec:<26}{a:10.4f}{na:5d}{b:10.4f}{nb:5d}{r:9.3f}')
        rows.append({'sector': sec, 'spread_harmonisation': a, 'n_harm': na,
                     'spread_sector_means': b, 'n_sect': nb, 'ratio': r})

    out = pd.DataFrame(rows)
    path = os.path.join(BASE, 'trend_sensitivity',
                        'NB10_spread_reconciliation.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)

    # ---- verdict ---------------------------------------------------------
    print(f'\n{SEP}\n  VERDICT\n{SEP}')
    med = out['ratio'].median()
    print(f'  Median spread ratio (harmonisation / sector_means): {med:.3f}')
    if abs(med - 1) < 0.02:
        print('  The two files agree. The earlier discrepancy came from the')
        print('  definition used, not the source. Quote either.')
    elif only_h or only_s:
        print('  The product sets differ, which is the most likely cause.')
        print('  Quote the file whose product set matches the manuscript, and')
        print('  state the number of products explicitly in the text.')
    else:
        print('  The files disagree with identical product sets and coverage.')
        print('  Check sections 3 and 4 above: a mask difference or a value')
        print('  offset must account for it. Do NOT quote either number until')
        print('  this is resolved, since L167 depends on it.')
    print(f'\n  Manuscript L167 states approximately 0.10 m for the Western '
          f'Weddell Sea.')
    ww = out[out.sector == 'Western_Weddell'].iloc[0]
    print(f'    harmonisation : {ww.spread_harmonisation:.4f} m')
    print(f'    sector_means  : {ww.spread_sector_means:.4f} m')
    print(f'\n  Saved: {path}')
    print(SEP + '\n')


if __name__ == '__main__':
    main()
