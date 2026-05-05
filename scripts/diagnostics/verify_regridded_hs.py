#!/usr/bin/env python3
"""
verify_regridded_hs.py
=======================
Quick diagnostic to check whether the regridded NPZ files contain
product-specific snow thickness fields, or whether all five CryoSat-2-era
product variants share the same snow field.

PhD Thesis -- Xinlong Liu, IMAS, University of Tasmania

Purpose:
  NB06 produced identical Case D h_fi reduction percentages under the CCI
  and CLIMATOLOGY snow references, which is mathematically suspicious.
  The most likely explanation is that the NB01 regridding pipeline has
  written the same CCI snow climatology into all five product NPZ files
  under the 'hs' key, rather than preserving each product's native snow
  input. This script tests that hypothesis by loading the 'hs' arrays
  from all five products for a single representative month and comparing
  them pairwise.

  If the arrays are bit-identical (or near-identical), the regridding
  pipeline needs review. If the arrays differ, the issue lies in NB06's
  climatology builder.

Output:
  Printed pairwise comparison of 'hs' arrays for May 2015 (an arbitrary
  representative winter month). For each product pair, the script reports:
    - whether the arrays are bit-identical
    - the maximum absolute difference
    - the mean absolute difference
    - the count of cells where the arrays differ by more than 1 mm

Run on Gadi:
  source ~/cryo2ice_env/bin/activate
  python3 verify_regridded_hs.py
"""

import os
import numpy as np

BASE_REGRIDDED = '/g/data/gv90/xl1657/phd/M1_workspace/processed/regridded'

# Test month: May 2015 (a representative winter month with all products available)
TEST_YEAR = 2015
TEST_MONTH = 5

PRODUCTS = ['CCI_CS2', 'LEGOS_I_CS2', 'LEGOS_II_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']
SEPARATOR = '=' * 70


def load_hs(product, year, month):
    """Load the hs array from a regridded NPZ file."""
    filename = f'{product}_{year}{month:02d}.npz'
    filepath = os.path.join(BASE_REGRIDDED, filename)
    if not os.path.exists(filepath):
        return None, f'FILE NOT FOUND: {filepath}'
    data = np.load(filepath)
    if 'hs' not in data.files:
        keys = list(data.files)
        data.close()
        return None, f'No "hs" key. Available keys: {keys}'
    hs = data['hs'].astype(np.float64)
    data.close()
    return hs, None


def summarise_array(hs, label):
    """Print summary statistics of an hs array."""
    valid = hs[np.isfinite(hs)]
    n_total = hs.size
    n_valid = valid.size
    if n_valid == 0:
        print(f'  {label:<20} shape={hs.shape}  ALL NaN')
        return
    print(f'  {label:<20} shape={hs.shape}  '
          f'valid={n_valid}/{n_total}  '
          f'min={valid.min():.4f}  '
          f'max={valid.max():.4f}  '
          f'mean={valid.mean():.4f}  '
          f'std={valid.std():.4f}')


def compare_arrays(a, b, label_a, label_b):
    """Compare two hs arrays cell-by-cell on the common valid mask."""
    valid_mask = np.isfinite(a) & np.isfinite(b)
    n_common = int(np.sum(valid_mask))
    if n_common == 0:
        print(f'  {label_a} vs {label_b}: NO COMMON VALID CELLS')
        return

    a_valid = a[valid_mask]
    b_valid = b[valid_mask]

    bit_identical = np.array_equal(a_valid, b_valid)
    abs_diff = np.abs(a_valid - b_valid)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    n_diff_above_1mm = int(np.sum(abs_diff > 0.001))

    flag = 'BIT-IDENTICAL' if bit_identical else (
        'NEAR-IDENTICAL' if max_diff < 1e-6 else 'DIFFERENT'
    )

    print(f'  {label_a:<14} vs {label_b:<14}  '
          f'common_cells={n_common:>5}  '
          f'max_diff={max_diff:.6f} m  '
          f'mean_diff={mean_diff:.6f} m  '
          f'>1mm: {n_diff_above_1mm:>5}  '
          f'[{flag}]')


def main():
    print(f'\n{SEPARATOR}')
    print(f'  Diagnostic: regridded hs fields for {TEST_YEAR}-{TEST_MONTH:02d}')
    print(SEPARATOR)
    print(f'  Working directory: {BASE_REGRIDDED}\n')

    # Load all five product hs arrays
    arrays = {}
    for product in PRODUCTS:
        hs, err = load_hs(product, TEST_YEAR, TEST_MONTH)
        if err:
            print(f'  {product}: {err}')
            continue
        arrays[product] = hs

    if len(arrays) < 2:
        print('\n  ERROR: fewer than two products loaded; cannot compare.')
        return

    # Print summary statistics per product
    print(f'\n{SEPARATOR}')
    print('  Per-product summary statistics')
    print(SEPARATOR)
    for product, hs in arrays.items():
        summarise_array(hs, product)

    # Pairwise comparison
    print(f'\n{SEPARATOR}')
    print('  Pairwise comparison')
    print(SEPARATOR)
    products_loaded = list(arrays.keys())
    for i, p_a in enumerate(products_loaded):
        for p_b in products_loaded[i + 1:]:
            compare_arrays(arrays[p_a], arrays[p_b], p_a, p_b)

    # Diagnosis summary
    print(f'\n{SEPARATOR}')
    print('  Diagnosis')
    print(SEPARATOR)
    n_pairs = len(products_loaded) * (len(products_loaded) - 1) // 2
    n_bit_identical = 0
    n_near_identical = 0
    for i, p_a in enumerate(products_loaded):
        for p_b in products_loaded[i + 1:]:
            valid_mask = np.isfinite(arrays[p_a]) & np.isfinite(arrays[p_b])
            if int(np.sum(valid_mask)) == 0:
                continue
            a_valid = arrays[p_a][valid_mask]
            b_valid = arrays[p_b][valid_mask]
            if np.array_equal(a_valid, b_valid):
                n_bit_identical += 1
            elif float(np.abs(a_valid - b_valid).max()) < 1e-6:
                n_near_identical += 1

    print(f'  Pairs compared:        {n_pairs}')
    print(f'  Bit-identical pairs:   {n_bit_identical}')
    print(f'  Near-identical pairs:  {n_near_identical}')
    print(f'  Different pairs:       {n_pairs - n_bit_identical - n_near_identical}')

    if n_bit_identical >= n_pairs - 1:
        print('\n  CONCLUSION: hs fields are essentially identical across products.')
        print('  This explains the suspicious NB06 result. The NB01 regridding')
        print('  pipeline likely needs review to ensure each product\'s native')
        print('  snow thickness is preserved, rather than substituting a single')
        print('  common snow field across all products.')
    elif n_bit_identical == 0 and n_near_identical == 0:
        print('\n  CONCLUSION: hs fields differ meaningfully across products.')
        print('  The issue is therefore in NB06\'s climatology builder, not in')
        print('  the regridded inputs. NB06 should be debugged.')
    else:
        print('\n  CONCLUSION: mixed pattern. Some product pairs are identical')
        print('  while others differ. Inspect the per-pair table above to')
        print('  identify which products share snow fields.')

    print()


if __name__ == '__main__':
    main()
