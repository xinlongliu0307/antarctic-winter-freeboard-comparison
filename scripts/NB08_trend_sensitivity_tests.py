#!/usr/bin/env python3
"""
NB08_trend_sensitivity_tests.py
================================
Graduated menu of tests answering Reviewer 1 / Ted Maksym on MS#1:
"quantification of the impact of how you treated the snow ... particularly in
how it might have impacted the observed trends."

PhD Thesis / MS#1 -- Xinlong Liu, IMAS, University of Tasmania

WHY THIS IS A MENU, NOT A SINGLE ANSWER
---------------------------------------
The submitted manuscript reports NO trend analysis. The only trend claim is at
line 768:

    "That magnitude is comparable to inter-annual variations reported in recent
     Antarctic thickness products and is large enough to affect regional
     thickness budgets, trend estimates, and comparisons involving ICESat-2 and
     CryoSat-2."

That is an assertion the paper never demonstrates, and it is almost certainly
what prompted the comment. The required work is therefore bounded by what the
paper claims: either substantiate that sentence or soften it.

Five tiers are implemented below, from cheapest to most elaborate. Each is
self-contained and prints a decision-ready summary. Run them all (about a
minute total) and bring the output to the supervision meeting so the scope can
be chosen from evidence rather than guessed.

ALL TIERS READ EXISTING CSVs. No regridding, no NPZ access, no recomputation
of fields. That is deliberate: a minor revision with editor-only re-review
should not require the pipeline to be rerun.

Run on Gadi (login node, < 1 min):
    source ~/cryo2ice_env/bin/activate
    cd /g/data/gv90/xl1657/phd/M1_workspace/scripts
    python3 NB08_trend_sensitivity_tests.py 2>&1 | tee ../output/NB08_output.txt

Author: Xinlong Liu | 1 September 2026
"""
import os
import sys
import math

import numpy as np
import pandas as pd

BASE = '/g/data/gv90/xl1657/phd/M1_workspace/processed'
HARM = os.path.join(BASE, 'snow_harmonisation')
OUT = os.path.join(BASE, 'trend_sensitivity')
os.makedirs(OUT, exist_ok=True)

SECTORS = ['Western_Weddell', 'Eastern_Weddell', 'Indian',
           'Western_Pacific', 'Ross', 'Amundsen_Bellingshausen']
CASES = ['A', 'B', 'C', 'D']
CASE_LABEL = {
    'A': 'native h_s, native rho_s',
    'B': 'native h_s, common rho_s',
    'C': 'common h_s, native rho_s',
    'D': 'common h_s, common rho_s',
}
SEP = '=' * 78


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════
def ols_trend(years, values):
    """
    Least-squares linear trend with standard error and a two-sided p-value.

    Returns (slope_per_year, stderr, p_value, n). Uses a normal approximation
    for the p-value, which is adequate for the six-year CryoSat-2 record; the
    bootstrap in Tier 4 is the defensible uncertainty estimate.
    """
    m = np.isfinite(years) & np.isfinite(values)
    x, y = np.asarray(years, float)[m], np.asarray(values, float)[m]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return np.nan, np.nan, np.nan, n
    slope = ((x - xm) * (y - ym)).sum() / sxx
    resid = y - (ym + slope * (x - xm))
    if n > 2:
        se = np.sqrt((resid ** 2).sum() / (n - 2) / sxx)
        z = slope / se if se > 0 else np.inf
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    else:
        se, p = np.nan, np.nan
    return slope, se, p, n


def pick_value_col(df, prefer='hfi'):
    """Resolve the value column under either naming convention."""
    for c in (f'{prefer}_mean', f'mean_{prefer}'):
        if c in df.columns:
            return c
    raise KeyError(f'No column for {prefer}. Available: {list(df.columns)}')


def annual_means(df, value_col='hfi_mean'):
    """Collapse monthly sector means to annual means (winter May-Oct)."""
    return (df.groupby(['case', 'product', 'sector', 'year'], as_index=False)
              [value_col].mean())


def load_harmonisation():
    path = os.path.join(HARM, 'harmonisation_sector_means.csv')
    if not os.path.exists(path):
        sys.exit(f'Missing {path}. Run NB03 first.')
    df = pd.read_csv(path)
    print(f'  Loaded {len(df):,} rows from harmonisation_sector_means.csv')
    print(f'  Columns: {list(df.columns)}')
    return df


# ═════════════════════════════════════════════════════════════════════════
# TIER 1 — Does the snow treatment change the sector-mean trend at all?
# ═════════════════════════════════════════════════════════════════════════
def tier1(df, value_col):
    """
    Cheapest possible answer. Average across products within each case, fit a
    linear trend per sector, and report how much the trend moves between
    Case A (native snow) and Case D (fully harmonised snow).

    This is the minimum needed to substantiate or retract the line-768 claim.
    """
    print(f'\n{SEP}\n  TIER 1: sector-mean trend under each snow case ({value_col})\n{SEP}')
    ann = annual_means(df, value_col)
    ens = ann.groupby(['case', 'sector', 'year'], as_index=False)[value_col].mean()

    rows = []
    print(f'  {"Sector":<26} {"A (m/yr)":>10} {"D (m/yr)":>10} '
          f'{"Delta":>9} {"% change":>9}')
    print('  ' + '-' * 68)
    for sec in SECTORS:
        s = {}
        for c in CASES:
            sub = ens[(ens.case == c) & (ens.sector == sec)]
            sl, se, p, n = ols_trend(sub['year'].values, sub[value_col].values)
            s[c] = (sl, se, p, n)
            rows.append({'tier': 1, 'variable': value_col, 'sector': sec,
                         'case': c, 'slope_m_per_yr': sl, 'stderr': se,
                         'p_value': p, 'n_years': n})
        a, d = s['A'][0], s['D'][0]
        pct = 100 * (d - a) / abs(a) if a not in (0, np.nan) and np.isfinite(a) else np.nan
        print(f'  {sec:<26} {a:>10.5f} {d:>10.5f} {d - a:>9.5f} {pct:>8.1f}%')

    print('\n  READ: if |Delta| is small relative to the Case A standard error')
    print('  (Tier 4), the snow treatment does not materially affect the trend')
    print('  and line 768 should be softened rather than defended.')
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# TIER 2 — Does harmonisation reduce the SPREAD of trends between products?
# ═════════════════════════════════════════════════════════════════════════
def tier2(df, value_col):
    """
    The paper's central result is that snow harmonisation reduces inter-product
    SPREAD in freeboard. The natural trend analogue asks whether it also
    reduces inter-product spread in the fitted TREND.

    This is the test most directly parallel to what the paper already does,
    and is my recommendation if one tier must be chosen.
    """
    print(f'\n{SEP}\n  TIER 2: inter-product spread of fitted trends ({value_col})\n{SEP}')
    ann = annual_means(df, value_col)
    rows = []
    print(f'  {"Sector":<26} {"A sd":>9} {"B sd":>9} {"C sd":>9} {"D sd":>9} '
          f'{"D red%":>8}')
    print('  ' + '-' * 74)
    for sec in SECTORS:
        sds = {}
        for c in CASES:
            slopes = []
            for prod in sorted(ann['product'].unique()):
                sub = ann[(ann.case == c) & (ann.sector == sec) &
                          (ann['product'] == prod)]
                sl, _, _, n = ols_trend(sub['year'].values, sub[value_col].values)
                if np.isfinite(sl):
                    slopes.append(sl)
                    rows.append({'tier': 2, 'variable': value_col,
                                 'sector': sec, 'case': c, 'product': prod,
                                 'slope_m_per_yr': sl, 'n_years': n})
            sds[c] = np.std(slopes, ddof=1) if len(slopes) > 1 else np.nan
        red = 100 * (sds['A'] - sds['D']) / sds['A'] if sds['A'] else np.nan
        print(f'  {sec:<26} {sds["A"]:>9.5f} {sds["B"]:>9.5f} '
              f'{sds["C"]:>9.5f} {sds["D"]:>9.5f} {red:>7.1f}%')

    print('\n  READ: a large D-reduction means snow choice drives disagreement')
    print('  between products about the trend, not just about the mean. That is')
    print('  a direct, quantified answer to the comment and reuses the paper\'s')
    print('  own Case A-D framing, so it needs no new methodology.')
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# TIER 3 — Is the snow-driven offset constant in time, or growing?
# ═════════════════════════════════════════════════════════════════════════
def tier3(df, value_col):
    """
    A CONSTANT offset displaces the mean but leaves the trend intact; a
    TIME-VARYING offset does not. This tier fits a trend to the Case A minus
    Case D difference itself.

    If that difference has no significant trend, the honest statement is that
    the snow treatment shifts absolute freeboard but does not bias the trend --
    which is a cleaner and more defensible claim than the current line 768.
    """
    print(f'\n{SEP}\n  TIER 3: is the snow-driven offset time-varying? ({value_col})\n{SEP}')
    ann = annual_means(df, value_col)
    ens = ann.groupby(['case', 'sector', 'year'], as_index=False)[value_col].mean()
    rows = []
    print(f'  {"Sector":<26} {"mean A-D":>10} {"trend in A-D":>14} '
          f'{"p":>7}  verdict')
    print('  ' + '-' * 76)
    for sec in SECTORS:
        a = ens[(ens.case == 'A') & (ens.sector == sec)].set_index('year')[value_col]
        d = ens[(ens.case == 'D') & (ens.sector == sec)].set_index('year')[value_col]
        common = a.index.intersection(d.index)
        diff = (a.loc[common] - d.loc[common])
        sl, se, p, n = ols_trend(common.values.astype(float), diff.values)
        verdict = ('offset is constant' if (np.isfinite(p) and p > 0.05)
                   else 'offset is time-varying')
        print(f'  {sec:<26} {diff.mean():>10.5f} {sl:>14.6f} {p:>7.3f}  {verdict}')
        rows.append({'tier': 3, 'variable': value_col, 'sector': sec,
                     'mean_offset_m': diff.mean(), 'offset_trend_m_per_yr': sl,
                     'stderr': se, 'p_value': p, 'n_years': n})

    print('\n  READ: this is the distinction Reviewer 3 raised on MS#3 and Ted is')
    print('  raising here. If p > 0.05 everywhere, say so plainly: the snow')
    print('  treatment is an offset, not a trend bias.')
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# TIER 4 — Is the trend change larger than the trend uncertainty?
# ═════════════════════════════════════════════════════════════════════════
def tier4(df, value_col, n_boot=2000, seed=20260901):
    """
    A trend difference only matters if it exceeds the uncertainty on the trend
    itself. Bootstrap over years (the same resampling unit NB05 uses) to get a
    CI on the Case A trend, then ask whether the Case D trend falls inside it.
    """
    print(f'\n{SEP}\n  TIER 4: trend change vs trend uncertainty ({value_col})\n{SEP}')
    rng = np.random.default_rng(seed)
    ann = annual_means(df, value_col)
    ens = ann.groupby(['case', 'sector', 'year'], as_index=False)[value_col].mean()
    rows = []
    print(f'  {"Sector":<26} {"A trend":>10} {"95% CI":>22} {"D trend":>10}  inside?')
    print('  ' + '-' * 76)
    for sec in SECTORS:
        a = ens[(ens.case == 'A') & (ens.sector == sec)].sort_values('year')
        d = ens[(ens.case == 'D') & (ens.sector == sec)].sort_values('year')
        yrs = a['year'].values.astype(float)
        va, vd = a[value_col].values, d[value_col].values
        sl_a, _, _, n = ols_trend(yrs, va)
        sl_d, _, _, _ = ols_trend(yrs, vd)

        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(yrs), len(yrs))
            s, _, _, _ = ols_trend(yrs[idx], va[idx])
            if np.isfinite(s):
                boots.append(s)
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        inside = 'yes' if lo <= sl_d <= hi else 'NO'
        print(f'  {sec:<26} {sl_a:>10.5f} [{lo:>9.5f},{hi:>9.5f}] '
              f'{sl_d:>10.5f}  {inside}')
        rows.append({'tier': 4, 'variable': value_col, 'sector': sec,
                     'slope_A': sl_a, 'ci_lo': lo, 'ci_hi': hi,
                     'slope_D': sl_d, 'D_inside_A_CI': inside == 'yes',
                     'n_years': n, 'n_boot': n_boot})

    print('\n  READ: "yes" everywhere means the snow treatment does not shift the')
    print('  trend beyond what six years of data can resolve. That is a complete')
    print('  and honest answer, and it costs no new analysis section.')
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# TIER 5 — Cross-era: does snow treatment affect the Envisat/CryoSat-2 step?
# ═════════════════════════════════════════════════════════════════════════
def tier5():
    """
    The longest 'trend' the study could support spans Envisat (2003-2011) and
    CryoSat-2 (2013-2018). The two eras use different products and a different
    snow treatment, so any cross-era change conflates a real signal with an
    inter-mission step.

    This tier only reports the size of that step from the existing NB02 sector
    means. It is included for completeness and I would NOT recommend it for a
    minor revision: quantifying it properly is a paper, not a comment response.
    """
    print(f'\n{SEP}\n  TIER 5: cross-era step, Envisat vs CryoSat-2 (diagnostic only)\n{SEP}')
    p_cs2 = os.path.join(BASE, 'sector_means', 'sector_means_monthly_CS2.csv')
    p_env = os.path.join(BASE, 'sector_means', 'sector_means_monthly_ENV.csv')
    for p in (p_cs2, p_env):
        if not os.path.exists(p):
            print(f'  Skipped: {p} not found. Adjust the path if NB02 wrote '
                  f'elsewhere.')
            return pd.DataFrame()
    cs2 = pd.read_csv(p_cs2)
    env = pd.read_csv(p_env)
    col = next((c for c in ('hfi_mean', 'mean_hfi') if c in cs2.columns),
               cs2.columns[-1])
    rows = []
    print(f'  {"Sector":<26} {"ENV mean":>10} {"CS2 mean":>10} {"step":>10}')
    print('  ' + '-' * 60)
    for sec in SECTORS:
        e = env[env.sector == sec][col].mean()
        c = cs2[cs2.sector == sec][col].mean()
        print(f'  {sec:<26} {e:>10.5f} {c:>10.5f} {c - e:>10.5f}')
        rows.append({'tier': 5, 'sector': sec, 'env_mean': e,
                     'cs2_mean': c, 'step': c - e})
    print('\n  READ: report this only as a limitation. The step conflates a real')
    print('  change with an inter-mission and inter-snow-treatment difference,')
    print('  and separating them is beyond a minor revision.')
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
def main():
    print(SEP)
    print('  NB08: trend-sensitivity test menu for the MS#1 minor revision')
    print('  Comment: "quantification of the impact of how you treated the')
    print('           snow ... how it might have impacted the observed trends"')
    print(SEP)
    print('\n  NOTE: the submitted manuscript reports no trend analysis. The only')
    print('  trend claim is at line 768. These tiers test that claim; they do')
    print('  not add a trend section.\n')

    df = load_harmonisation()
    v = pick_value_col(df, 'hfi')
    print(f'  Using value column: {v}  (sea-ice freeboard)')

    # Control: radar freeboard carries no snow term, so every case must give
    # an identical trend. Any non-zero Delta here means the case construction
    # is leaking, not that snow affects the trend.
    try:
        v_ctrl = pick_value_col(df, 'hfr')
        print(f'\n{SEP}\n  CONTROL: radar freeboard ({v_ctrl}) must be '
              f'case-invariant\n{SEP}')
        ctrl = tier1(df, v_ctrl)
        d = ctrl[ctrl.case.isin(['A', 'D'])]
        spread = d.groupby('sector')['slope_m_per_yr'].apply(
            lambda x: x.max() - x.min()).max()
        print(f'\n  Max A-to-D trend difference in h_fr: {spread:.3e} m/yr')
        print('  PASS: machinery is sound.' if spread < 1e-12
              else '  FAIL: h_fr should be case-invariant. Check NB03.')
    except KeyError:
        pass

    out = []
    out.append(tier1(df, v))
    out.append(tier2(df, v))
    out.append(tier3(df, v))
    out.append(tier4(df, v))
    out.append(tier5())

    res = pd.concat([o for o in out if len(o)], ignore_index=True)
    path = os.path.join(OUT, 'NB08_trend_sensitivity.csv')
    res.to_csv(path, index=False)

    print(f'\n{SEP}\n  RECOMMENDATION FOR THE MEETING\n{SEP}')
    print('  Tier 2 + Tier 3 together are my suggestion: one paragraph and one')
    print('  small table. Tier 2 shows whether snow choice drives inter-product')
    print('  trend disagreement, reusing the paper\'s own Case A-D framing.')
    print('  Tier 3 settles whether the effect is an offset or a trend bias,')
    print('  which is precisely what Ted asked.')
    print('  Tier 4 is worth adding if Tier 2 shows a large reduction, since it')
    print('  bounds the claim against six years of data.')
    print('  Tier 1 alone is too thin. Tier 5 belongs in Limitations, not here.')
    print(f'\n  Saved: {path}\n')


if __name__ == '__main__':
    main()
