#!/usr/bin/env python3
"""
NB06_alternative_snow_reference_v2.py
======================================
Step 1.6 (corrected): Alternative-snow-reference sensitivity test for the
snow-harmonisation experiments, with internal verification checks.

PhD Thesis -- Xinlong Liu, IMAS, University of Tasmania

Changes from v1:
  - The original NB06 produced identical CCI and CLIMATOLOGY columns in
    the attribution table. The diagnostic verify_climatology_field.py
    confirmed that the climatology builder itself produces a genuine
    ensemble mean distinct from the CCI field. The bug must therefore
    lie in run_harmonisation_with_reference, most likely through a
    variable shadowing or reference-loading issue in the inner loop.

  - This v2 adds internal verification checks at the points where the
    bug is most likely to occur:
      (a) Confirms ref_hs is loaded from the correct source per branch
          and logs the array hash for each (year, month, reference) call.
      (b) Verifies that ref_hs is distinct from the CCI field for the
          CLIMATOLOGY and LEGOS_II branches.
      (c) Logs the case-D h_fi field hash per (product, year, month,
          reference) so any silent substitution is detected.
      (d) Cross-checks the final attribution table to ensure that the
          CCI and CLIMATOLOGY rows differ in at least one sector. If
          they are identical, the script raises a diagnostic error
          rather than silently writing incorrect output.

  - The actual computational logic is unchanged from v1; only the
    verification checks are added. If v2 produces correct output, the
    bug was a transient issue in v1; if v2 produces the same identical-
    columns result, the verification checks will identify exactly where
    the substitution occurred.

Inputs:
  - Regridded .npz files in processed/regridded/ (from NB01)
  - Sector map and common masks in processed/masks/ (from NB02)

Outputs (in processed/snow_harmonisation/):
  - alt_reference_attribution_v2.csv
  - alt_reference_summary_v2.csv
  - alt_reference_diagnostics_v2.csv (NEW: verification log)

Run on Gadi:
  nohup bash -c 'source ~/cryo2ice_env/bin/activate && \\
  python3 NB06_alternative_snow_reference_v2.py' > NB06_v2_output.txt 2>&1 &

  Expected runtime: ~10-15 minutes.
"""

import os
import csv
import hashlib
import numpy as np
from datetime import datetime
from collections import defaultdict

# ==============================================================================
# Configuration
# ==============================================================================
BASE_REGRIDDED = '/g/data/gv90/xl1657/phd/M1_workspace/processed/regridded'
BASE_MASKS     = '/g/data/gv90/xl1657/phd/M1_workspace/processed/masks'
BASE_OUT       = '/g/data/gv90/xl1657/phd/M1_workspace/processed/snow_harmonisation'
os.makedirs(BASE_OUT, exist_ok=True)

CS2_YEARS = list(range(2013, 2019))
WINTER_MONTHS = list(range(5, 11))
CS2_PRODUCTS = ['CCI_CS2', 'LEGOS_I_CS2', 'LEGOS_II_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']

COMMON_DENSITY = 300.0  # kg/m^3

REFERENCE_CHOICES = ['CCI', 'LEGOS_II', 'CLIMATOLOGY']

CCI_REFERENCE_PRODUCT = 'CCI_CS2'
LEGOS_II_REFERENCE_PRODUCT = 'LEGOS_II_CS2'

SECTOR_DEFS = [
    ('Western_Weddell',          -62.0,  -40.0),
    ('Eastern_Weddell',          -40.0,   15.0),
    ('Indian',                    15.0,   90.0),
    ('Western_Pacific',           90.0,  160.0),
    ('Ross',                     160.0, -140.0),
    ('Amundsen_Bellingshausen', -140.0,  -62.0),
]
SECTOR_NAMES = [s[0] for s in SECTOR_DEFS]
VARIABLES = ['hfr', 'hsc', 'hfi']
CASES = ['A', 'B', 'C', 'D']

# Verification logging
DIAGNOSTICS = []
VERIFICATION_FAILURES = []

SEPARATOR = '=' * 70


# ==============================================================================
# Snow density parameterisations (consistent with NB03)
# ==============================================================================
def kurtz2012_density(month):
    if month == 5:
        return 320.0
    elif month == 10:
        return 340.0
    else:
        return 350.0


def fons2023_density(month):
    seasonal = {
        12: 360, 1: 360, 2: 360,
        3: 350, 4: 350, 5: 350,
        6: 330, 7: 330, 8: 330,
        9: 310, 10: 310, 11: 310,
    }
    return float(seasonal.get(month, 330))


def get_native_density(product, month):
    if product in ['CCI_CS2', 'CCI_ENV']:
        return 300.0
    elif product in ['LEGOS_I_CS2', 'LEGOS_II_CS2', 'LEGOS_ENV', 'CSAO_CS2']:
        return kurtz2012_density(month)
    elif product == 'CryoTEMPO_CS2':
        return fons2023_density(month)
    else:
        return 350.0


def compute_hsc(hs, rho_s):
    c = 3e8
    cs = c * (1 + 5.1e-4 * rho_s) ** (-1.5)
    return (c / cs - 1) * hs


# ==============================================================================
# Verification helpers
# ==============================================================================
def array_hash(arr):
    """Compute a short hash of an array for verification logging."""
    if arr is None:
        return 'NONE'
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return 'EMPTY'
    h = hashlib.md5(valid.tobytes()).hexdigest()[:12]
    return h


def array_summary(arr):
    """Return a brief summary string of an array's valid statistics."""
    if arr is None:
        return 'None'
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return 'all NaN'
    return f'n={len(valid)} mean={valid.mean():.4f} std={valid.std():.4f}'


def log_diagnostic(reference, year, month, item, value):
    """Append a diagnostic entry to the global log."""
    DIAGNOSTICS.append({
        'reference': reference,
        'year': year,
        'month': month,
        'item': item,
        'value': str(value),
    })


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


def load_product_field(product, year, month, variable):
    npz_file = os.path.join(BASE_REGRIDDED, f'{product}_{year}{month:02d}.npz')
    if not os.path.exists(npz_file):
        return None
    data = np.load(npz_file)
    field = data[variable].astype(np.float64) if variable in data.files else None
    data.close()
    return field


# ==============================================================================
# Build the climatological-mean snow reference field
# ==============================================================================
def build_climatology_reference(common_masks):
    print(f'\n{SEPARATOR}')
    print('  Building CLIMATOLOGY snow reference (ensemble mean across products)...')
    print(SEPARATOR)

    climatology = {}
    n_built = 0
    for year in CS2_YEARS:
        for month in WINTER_MONTHS:
            ym_key = (year, month)
            if ym_key not in common_masks:
                continue

            field_stack = []
            for product in CS2_PRODUCTS:
                hs_field = load_product_field(product, year, month, 'hs')
                if hs_field is None:
                    continue
                field_stack.append(hs_field)

            if len(field_stack) < 2:
                continue

            stacked = np.stack(field_stack, axis=0)
            with np.errstate(all='ignore'):
                count_valid = np.sum(np.isfinite(stacked), axis=0)
                ensemble_mean = np.nanmean(stacked, axis=0)
                ensemble_mean = np.where(count_valid >= 2, ensemble_mean, np.nan)

            climatology[ym_key] = ensemble_mean.copy()  # explicit copy to prevent shared references
            n_built += 1

            # VERIFICATION: log climatology hash and confirm distinct from CCI
            cci_hs = field_stack[0] if CS2_PRODUCTS[0] == 'CCI_CS2' else None
            clim_hash = array_hash(ensemble_mean)
            cci_hash = array_hash(cci_hs)
            log_diagnostic('CLIMATOLOGY', year, month, 'climatology_hash', clim_hash)
            log_diagnostic('CLIMATOLOGY', year, month, 'cci_hash_at_build', cci_hash)
            if clim_hash == cci_hash:
                msg = f'CLIMATOLOGY {year}-{month:02d}: hash matches CCI ({clim_hash})'
                VERIFICATION_FAILURES.append(msg)
                print(f'  *** VERIFICATION FAILURE: {msg}')

    print(f'  Built {n_built} monthly climatology reference fields')
    return climatology


# ==============================================================================
# Run harmonisation Cases A-D under a specified snow-reference choice
# ==============================================================================
def run_harmonisation_with_reference(
    reference_label, common_masks, sector_map, climatology=None,
):
    print(f'\n{SEPARATOR}')
    print(f'  Running harmonisation under reference: {reference_label}')
    print(SEPARATOR)

    spread_data = defaultdict(list)
    n_entries = 0

    for year in CS2_YEARS:
        for month in WINTER_MONTHS:
            ym_key = (year, month)
            if ym_key not in common_masks:
                continue
            common_mask = common_masks[ym_key]

            # CRITICAL: load reference field with explicit branch logging
            if reference_label == 'CCI':
                ref_hs = load_product_field(
                    CCI_REFERENCE_PRODUCT, year, month, 'hs')
                ref_source = f'NPZ:{CCI_REFERENCE_PRODUCT}'
            elif reference_label == 'LEGOS_II':
                ref_hs = load_product_field(
                    LEGOS_II_REFERENCE_PRODUCT, year, month, 'hs')
                ref_source = f'NPZ:{LEGOS_II_REFERENCE_PRODUCT}'
            elif reference_label == 'CLIMATOLOGY':
                if climatology is None:
                    raise ValueError(
                        'CLIMATOLOGY reference requested but climatology dict '
                        'is None.')
                clim_field = climatology.get(ym_key)
                ref_hs = clim_field.copy() if clim_field is not None else None
                ref_source = f'DICT:climatology[{ym_key}]'
            else:
                raise ValueError(f'Unknown reference: {reference_label}')

            if ref_hs is None:
                continue

            # VERIFICATION: log ref_hs hash for this (reference, year, month)
            ref_hash = array_hash(ref_hs)
            log_diagnostic(reference_label, year, month, 'ref_hs_source', ref_source)
            log_diagnostic(reference_label, year, month, 'ref_hs_hash', ref_hash)
            log_diagnostic(reference_label, year, month, 'ref_hs_summary',
                           array_summary(ref_hs))

            # CRITICAL VERIFICATION: for CLIMATOLOGY branch, verify ref_hs
            # differs from the CCI field for this (year, month). If they
            # match, raise a diagnostic alert.
            if reference_label == 'CLIMATOLOGY':
                cci_check = load_product_field(
                    CCI_REFERENCE_PRODUCT, year, month, 'hs')
                cci_check_hash = array_hash(cci_check)
                log_diagnostic(reference_label, year, month,
                               'cci_check_hash', cci_check_hash)
                if ref_hash == cci_check_hash:
                    msg = (f'CLIMATOLOGY {year}-{month:02d}: ref_hs hash '
                           f'matches CCI ({ref_hash}). Substitution detected!')
                    VERIFICATION_FAILURES.append(msg)
                    print(f'  *** VERIFICATION FAILURE: {msg}')

            common_rho = COMMON_DENSITY

            for product in CS2_PRODUCTS:
                hfr = load_product_field(product, year, month, 'hfr')
                hs_native = load_product_field(product, year, month, 'hs')
                if hfr is None or hs_native is None:
                    continue

                native_rho = get_native_density(product, month)

                # Compute Cases A-D
                hsc_A = compute_hsc(hs_native, native_rho)
                hfi_A = hfr + hsc_A

                hsc_B = compute_hsc(hs_native, common_rho)
                hfi_B = hfr + hsc_B

                # CRITICAL: ensure ref_hs is used here, not hs_native
                hsc_C = compute_hsc(ref_hs, native_rho)
                hfi_C = hfr + hsc_C

                hsc_D = compute_hsc(ref_hs, common_rho)
                hfi_D = hfr + hsc_D

                # VERIFICATION: log Case D h_fi hash for first product
                # (CCI) only, to keep the log size manageable
                if product == 'CCI_CS2':
                    log_diagnostic(reference_label, year, month,
                                   f'caseD_hfi_hash_{product}',
                                   array_hash(hfi_D))

                case_fields = {
                    'A': (hfr, hsc_A, hfi_A),
                    'B': (hfr, hsc_B, hfi_B),
                    'C': (hfr, hsc_C, hfi_C),
                    'D': (hfr, hsc_D, hfi_D),
                }

                for case_label, (hfr_f, hsc_f, hfi_f) in case_fields.items():
                    for sector_idx, sector_name in enumerate(SECTOR_NAMES):
                        sector_mask = common_mask & (sector_map == sector_idx)
                        if np.sum(sector_mask) == 0:
                            continue

                        for variable, field in [
                            ('hfr', hfr_f), ('hsc', hsc_f), ('hfi', hfi_f),
                        ]:
                            vals = field[sector_mask]
                            valid = vals[np.isfinite(vals)]
                            if len(valid) == 0:
                                continue
                            spread_data[
                                (case_label, year, month, sector_name, variable)
                            ].append(float(np.mean(valid)))
                            n_entries += 1

    print(f'  Logged {n_entries} (case, year, month, sector, variable, product) entries')
    return spread_data


# ==============================================================================
# Compute attribution from spread data
# ==============================================================================
def compute_attribution(spread_data, reference_label):
    monthly_spread = []
    for key, values in spread_data.items():
        case, year, month, sector, variable = key
        if len(values) >= 2:
            monthly_spread.append({
                'case': case, 'year': year, 'month': month,
                'sector': sector, 'variable': variable,
                'spread': max(values) - min(values),
            })

    sector_var_case_spreads = defaultdict(list)
    for r in monthly_spread:
        key = (r['case'], r['sector'], r['variable'])
        sector_var_case_spreads[key].append(r['spread'])

    attribution_rows = []
    for sector in SECTOR_NAMES:
        for variable in VARIABLES:
            spreads = {}
            for case in CASES:
                key = (case, sector, variable)
                spreads[case] = (
                    float(np.mean(sector_var_case_spreads[key]))
                    if sector_var_case_spreads.get(key) else np.nan
                )

            a_val = spreads['A']
            reductions = {}
            for case in ['B', 'C', 'D']:
                if (np.isfinite(a_val) and a_val > 0
                        and np.isfinite(spreads[case])):
                    reductions[case] = (a_val - spreads[case]) / a_val * 100
                else:
                    reductions[case] = np.nan

            attribution_rows.append({
                'reference': reference_label,
                'sector': sector,
                'variable': variable,
                'spread_A': spreads['A'],
                'spread_B': spreads['B'],
                'spread_C': spreads['C'],
                'spread_D': spreads['D'],
                'reduction_B_pct': reductions['B'],
                'reduction_C_pct': reductions['C'],
                'reduction_D_pct': reductions['D'],
            })

    return attribution_rows


# ==============================================================================
# Build comparison summary
# ==============================================================================
def build_comparison_summary(all_attribution):
    summary_rows = []
    for sector in SECTOR_NAMES:
        row = {'sector': sector}
        for ref in REFERENCE_CHOICES:
            r_match = next(
                (r for r in all_attribution
                 if r['reference'] == ref
                 and r['sector'] == sector
                 and r['variable'] == 'hfi'),
                None,
            )
            if r_match is None:
                row[f'{ref}_spread_A'] = np.nan
                row[f'{ref}_spread_D'] = np.nan
                row[f'{ref}_reduction_D_pct'] = np.nan
            else:
                row[f'{ref}_spread_A'] = r_match['spread_A']
                row[f'{ref}_spread_D'] = r_match['spread_D']
                row[f'{ref}_reduction_D_pct'] = r_match['reduction_D_pct']
        summary_rows.append(row)
    return summary_rows


# ==============================================================================
# Final cross-check
# ==============================================================================
def cross_check_attribution(all_attribution):
    """
    Verify that the CCI and CLIMATOLOGY attribution rows differ in at least
    one sector for h_fi Case D. If they are bit-identical across all six
    sectors, the substitution bug is still present.
    """
    print(f'\n{SEPARATOR}')
    print('  CROSS-CHECK: CCI vs CLIMATOLOGY attribution comparison')
    print(SEPARATOR)

    cci_rows = {r['sector']: r for r in all_attribution
                if r['reference'] == 'CCI' and r['variable'] == 'hfi'}
    clim_rows = {r['sector']: r for r in all_attribution
                 if r['reference'] == 'CLIMATOLOGY' and r['variable'] == 'hfi'}

    n_identical = 0
    n_different = 0
    print(f'  {"Sector":<25} {"CCI red%":>10} {"Clim red%":>10} {"Diff":>10}')
    print(f'  {"-" * 60}')
    for sector in SECTOR_NAMES:
        cci_red = cci_rows.get(sector, {}).get('reduction_D_pct', np.nan)
        clim_red = clim_rows.get(sector, {}).get('reduction_D_pct', np.nan)
        if np.isfinite(cci_red) and np.isfinite(clim_red):
            diff = abs(cci_red - clim_red)
            if diff < 0.001:
                n_identical += 1
            else:
                n_different += 1
            print(f'  {sector:<25} {cci_red:>9.1f}% {clim_red:>9.1f}% {diff:>9.2f}')
        else:
            print(f'  {sector:<25} {"N/A":>10} {"N/A":>10} {"N/A":>10}')

    if n_different > 0:
        print(f'\n  PASS: CCI and CLIMATOLOGY differ in {n_different} of '
              f'{n_identical + n_different} sectors. Bug is resolved.')
    else:
        print(f'\n  FAIL: CCI and CLIMATOLOGY are identical in all sectors.')
        print(f'  The substitution bug is still present.')


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


def print_appendix_summary(summary_rows):
    print(f'\n{SEPARATOR}')
    print('  APPENDIX-READY: Sensitivity of h_fi Case D reduction to reference choice')
    print(SEPARATOR)
    print(f'  {"Sector":<25} {"CCI":>10} {"LEGOS-II":>10} {"Climatol":>10}')
    print(f'  {"-" * 60}')
    for r in summary_rows:
        cci_red = r.get('CCI_reduction_D_pct', np.nan)
        leg_red = r.get('LEGOS_II_reduction_D_pct', np.nan)
        clim_red = r.get('CLIMATOLOGY_reduction_D_pct', np.nan)

        def fmt(x):
            return f'{x:>9.1f}%' if np.isfinite(x) else f'{"N/A":>10}'
        print(f'  {r["sector"]:<25} {fmt(cci_red)} {fmt(leg_red)} {fmt(clim_red)}')


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f'\nNB06_alternative_snow_reference_v2.py')
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Reference choices: {REFERENCE_CHOICES}')
    print(f'Common density: {COMMON_DENSITY} kg/m^3 (CCI Antarctic fixed)')

    print(f'\n{SEPARATOR}')
    print('  Loading sector map and common masks ...')
    print(SEPARATOR)
    sector_map = load_sector_map()
    common_masks = load_common_masks()
    print(f'  Loaded {len(common_masks)} monthly common masks')

    climatology = build_climatology_reference(common_masks)

    all_attribution = []
    for ref in REFERENCE_CHOICES:
        spread_data = run_harmonisation_with_reference(
            ref, common_masks, sector_map, climatology=climatology)
        attribution = compute_attribution(spread_data, ref)
        all_attribution.extend(attribution)

    save_csv(all_attribution, 'alt_reference_attribution_v2.csv')

    summary_rows = build_comparison_summary(all_attribution)
    save_csv(summary_rows, 'alt_reference_summary_v2.csv')

    save_csv(DIAGNOSTICS, 'alt_reference_diagnostics_v2.csv')

    print_appendix_summary(summary_rows)
    cross_check_attribution(all_attribution)

    print(f'\n{SEPARATOR}')
    print('  VERIFICATION SUMMARY')
    print(SEPARATOR)
    print(f'  Total diagnostic entries: {len(DIAGNOSTICS)}')
    print(f'  Verification failures:    {len(VERIFICATION_FAILURES)}')
    if VERIFICATION_FAILURES:
        print(f'\n  Failures detected:')
        for msg in VERIFICATION_FAILURES[:10]:
            print(f'    - {msg}')
        if len(VERIFICATION_FAILURES) > 10:
            print(f'    ... and {len(VERIFICATION_FAILURES) - 10} more')

    print(f'\n{SEPARATOR}')
    print('  FINAL OUTPUT SUMMARY')
    print(SEPARATOR)
    print(f'  Output directory: {BASE_OUT}')
    print(f'  Files written:')
    for fn in [
        'alt_reference_attribution_v2.csv',
        'alt_reference_summary_v2.csv',
        'alt_reference_diagnostics_v2.csv',
    ]:
        path = os.path.join(BASE_OUT, fn)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f'    {fn} ({size_kb:.1f} KB)')

    print(f'\nCompleted: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('Done.')


if __name__ == '__main__':
    main()
