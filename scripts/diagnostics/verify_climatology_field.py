#!/usr/bin/env python3
"""
verify_climatology_field.py
============================
Focused diagnostic to localise the bug in NB06's climatology builder.

PhD Thesis -- Xinlong Liu, IMAS, University of Tasmania

Purpose:
  The previous diagnostic (verify_regridded_hs.py) confirmed that the
  regridded NPZ files contain genuinely product-specific snow thickness
  fields, ruling out the regridding pipeline as the source of the
  identical CCI/CLIMATOLOGY columns in the NB06 output. The bug must
  therefore lie in either:

    (a) NB06's build_climatology_reference function, which may be
        producing a climatology field that is bit-identical or
        near-identical to the CCI field rather than a genuine ensemble
        mean across all five products; or

    (b) NB06's run_harmonisation_with_reference function, where the
        ref_hs field for the CLIMATOLOGY branch may be silently
        overwritten or substituted with the CCI field.

  This script tests hypothesis (a) directly by reproducing the exact
  build_climatology_reference logic from NB06 for a single representative
  month, then comparing the resulting climatology field cell-by-cell with
  the CCI snow field for the same month. If the two fields are bit-
  identical or near-identical, the bug is in the climatology builder
  itself. If the climatology field is genuinely distinct from the CCI
  field, the bug must be elsewhere in NB06.

Output:
  Printed comparison of the constructed climatology field with the CCI
  snow field for May 2015 (an arbitrary representative winter month).
  Reports:
    - per-cell counts of contributing products at each grid location
    - summary statistics of the climatology field
    - cell-by-cell comparison of climatology vs CCI
    - count of cells where climatology equals CCI exactly
    - count of cells where climatology differs from CCI by more than
      varying thresholds
    - sample of cells where climatology and CCI disagree, to confirm
      the ensemble-mean computation is producing a genuinely distinct
      field at those locations
    - sample of cells where climatology and CCI agree, to identify
      whether the agreement is because only CCI contributed or because
      the ensemble mean coincidentally equals the CCI value

Run on Gadi:
  source ~/cryo2ice_env/bin/activate
  python3 verify_climatology_field.py
"""

import os
import numpy as np

BASE_REGRIDDED = '/g/data/gv90/xl1657/phd/M1_workspace/processed/regridded'
BASE_MASKS     = '/g/data/gv90/xl1657/phd/M1_workspace/processed/masks'

# Test month
TEST_YEAR = 2015
TEST_MONTH = 5

CS2_PRODUCTS = ['CCI_CS2', 'LEGOS_I_CS2', 'LEGOS_II_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']

SEPARATOR = '=' * 70


def load_product_field(product, year, month, variable):
    """Load a single (product, year, month) field; returns None if missing."""
    npz_file = os.path.join(BASE_REGRIDDED, f'{product}_{year}{month:02d}.npz')
    if not os.path.exists(npz_file):
        return None
    data = np.load(npz_file)
    if variable not in data.files:
        data.close()
        return None
    field = data[variable].astype(np.float64)
    data.close()
    return field


def load_common_mask(year, month):
    """Load the common mask for a given (year, month) from NB02 output."""
    masks_file = os.path.join(BASE_MASKS, 'common_masks_CS2.npz')
    if not os.path.exists(masks_file):
        return None
    data = np.load(masks_file)
    key = f'{year}_{month}'
    if key not in data.files:
        data.close()
        return None
    mask = data[key]
    data.close()
    return mask


def build_climatology_reference_replica(year, month):
    """
    Reproduce the exact logic of NB06's build_climatology_reference for a
    single (year, month). Returns the climatology field, the per-cell count
    of valid contributing products, and the stack of individual product
    fields for diagnostic inspection.
    """
    field_stack = []
    product_labels = []
    for product in CS2_PRODUCTS:
        hs_field = load_product_field(product, year, month, 'hs')
        if hs_field is None:
            continue
        field_stack.append(hs_field)
        product_labels.append(product)

    if len(field_stack) < 2:
        return None, None, None, None

    stacked = np.stack(field_stack, axis=0)
    with np.errstate(all='ignore'):
        count_valid = np.sum(np.isfinite(stacked), axis=0)
        ensemble_mean = np.nanmean(stacked, axis=0)
        ensemble_mean = np.where(count_valid >= 2, ensemble_mean, np.nan)

    return ensemble_mean, count_valid, stacked, product_labels


def main():
    print(f'\n{SEPARATOR}')
    print(f'  Climatology-field diagnostic for {TEST_YEAR}-{TEST_MONTH:02d}')
    print(SEPARATOR)

    # Build the climatology field using NB06's exact logic
    clim, count_valid, stacked, labels = build_climatology_reference_replica(
        TEST_YEAR, TEST_MONTH)
    if clim is None:
        print('  ERROR: could not build climatology (insufficient products).')
        return

    print(f'\n  Products contributing to climatology:')
    for i, label in enumerate(labels):
        valid_count = int(np.sum(np.isfinite(stacked[i])))
        print(f'    [{i}] {label:<20} valid cells: {valid_count}')

    # Load the CCI snow field for direct comparison
    cci_hs = load_product_field('CCI_CS2', TEST_YEAR, TEST_MONTH, 'hs')
    if cci_hs is None:
        print('  ERROR: CCI_CS2 hs field not available.')
        return

    # Per-cell count of contributing products
    print(f'\n{SEPARATOR}')
    print('  Per-cell count of contributing products')
    print(SEPARATOR)
    for k in range(0, 6):
        n_cells = int(np.sum(count_valid == k))
        print(f'    Cells with {k} contributing products: {n_cells}')

    # Summary statistics of the climatology field
    print(f'\n{SEPARATOR}')
    print('  Climatology field summary')
    print(SEPARATOR)
    valid = clim[np.isfinite(clim)]
    if len(valid) > 0:
        print(f'    valid cells:  {len(valid)}')
        print(f'    min:          {valid.min():.4f} m')
        print(f'    max:          {valid.max():.4f} m')
        print(f'    mean:         {valid.mean():.4f} m')
        print(f'    std:          {valid.std():.4f} m')

    # Compare climatology to CCI cell-by-cell on the common valid mask
    print(f'\n{SEPARATOR}')
    print('  Climatology vs CCI: cell-by-cell comparison')
    print(SEPARATOR)
    common = np.isfinite(clim) & np.isfinite(cci_hs)
    n_common = int(np.sum(common))
    if n_common == 0:
        print('  ERROR: no common valid cells between climatology and CCI.')
        return

    clim_v = clim[common]
    cci_v = cci_hs[common]
    abs_diff = np.abs(clim_v - cci_v)

    bit_identical = np.array_equal(clim_v, cci_v)
    n_exact = int(np.sum(abs_diff == 0.0))
    n_within_micron = int(np.sum(abs_diff < 1e-6))
    n_within_mm = int(np.sum(abs_diff < 1e-3))
    n_above_mm = int(np.sum(abs_diff >= 1e-3))
    n_above_cm = int(np.sum(abs_diff >= 1e-2))

    print(f'    common valid cells:     {n_common}')
    print(f'    bit-identical arrays:   {bit_identical}')
    print(f'    cells with diff == 0:   {n_exact}  ({100*n_exact/n_common:.1f}%)')
    print(f'    cells with diff < 1e-6: {n_within_micron}  ({100*n_within_micron/n_common:.1f}%)')
    print(f'    cells with diff < 1e-3: {n_within_mm}  ({100*n_within_mm/n_common:.1f}%)')
    print(f'    cells with diff >= 1e-3: {n_above_mm}  ({100*n_above_mm/n_common:.1f}%)')
    print(f'    cells with diff >= 1e-2: {n_above_cm}  ({100*n_above_cm/n_common:.1f}%)')
    print(f'    max diff:               {abs_diff.max():.6f} m')
    print(f'    mean diff:              {abs_diff.mean():.6f} m')

    # Inspect cells where climatology equals CCI exactly: are these cells
    # where only CCI contributed, or where the ensemble mean coincidentally
    # equals the CCI value?
    print(f'\n{SEPARATOR}')
    print('  Cells where climatology == CCI exactly')
    print(SEPARATOR)
    exact_mask = np.zeros_like(clim, dtype=bool)
    exact_mask[common] = (clim[common] == cci_hs[common])
    counts_at_exact = count_valid[exact_mask]
    print(f'    total exact-match cells: {int(np.sum(exact_mask))}')
    if int(np.sum(exact_mask)) > 0:
        for k in range(1, 6):
            n_k = int(np.sum(counts_at_exact == k))
            if n_k > 0:
                print(f'    of which {k}-product cells: {n_k}  '
                      f'({100*n_k/int(np.sum(exact_mask)):.1f}%)')

    # Inspect a small sample of cells where climatology and CCI disagree,
    # to confirm the ensemble mean is producing genuinely distinct values
    print(f'\n{SEPARATOR}')
    print('  Sample of cells where climatology differs from CCI (5 cells)')
    print(SEPARATOR)
    diff_indices = np.argwhere((abs_diff >= 1e-3))
    if len(diff_indices) > 0:
        # Pick five well-separated samples
        n_samples = min(5, len(diff_indices))
        step = max(1, len(diff_indices) // n_samples)
        sample_idx = diff_indices[::step][:n_samples].flatten()

        # Map flat index back to 2D index
        common_idx_2d = np.argwhere(common)
        for j, flat_i in enumerate(sample_idx):
            row, col = common_idx_2d[flat_i]
            print(f'\n    Cell ({row:3d}, {col:3d}):')
            for i, label in enumerate(labels):
                val = stacked[i, row, col]
                if np.isfinite(val):
                    print(f'      {label:<20} hs = {val:.4f} m')
                else:
                    print(f'      {label:<20} hs = NaN')
            print(f'      {"--> climatology":<20} hs = {clim[row, col]:.4f} m')
            print(f'      {"--> CCI":<20} hs = {cci_hs[row, col]:.4f} m')
            print(f'      contributing products: {int(count_valid[row, col])}')

    # Diagnosis
    print(f'\n{SEPARATOR}')
    print('  Diagnosis')
    print(SEPARATOR)
    if bit_identical:
        print('  CONCLUSION: climatology is BIT-IDENTICAL to CCI.')
        print('  The build_climatology_reference function is producing the')
        print('  CCI field, not an ensemble mean. The bug is in the')
        print('  climatology builder itself.')
    elif n_above_mm == 0:
        print('  CONCLUSION: climatology and CCI differ at the sub-mm level only.')
        print('  This is unexpected and suggests a numerical or logic issue in')
        print('  the climatology builder.')
    elif n_above_mm < 0.05 * n_common:
        print(f'  CONCLUSION: climatology and CCI differ at >= 1 mm in only')
        print(f'  {100*n_above_mm/n_common:.1f}% of cells. The climatology is mostly')
        print(f'  driven by CCI. The bug may be that CCI dominates the ensemble')
        print(f'  due to coverage; check the per-cell contributing-product counts')
        print(f'  above to see whether most cells have only CCI as a contributor.')
    else:
        print(f'  CONCLUSION: climatology and CCI differ at >= 1 mm in')
        print(f'  {100*n_above_mm/n_common:.1f}% of cells, which is consistent with')
        print(f'  a genuine ensemble mean. The climatology field itself appears')
        print(f'  correct. The bug must therefore be in how the climatology is')
        print(f'  consumed downstream in run_harmonisation_with_reference,')
        print(f'  for example through a variable shadowing or assignment issue.')

    print()


if __name__ == '__main__':
    main()
