#!/usr/bin/env python3
"""
C2_sens_CT300.py
Sensitivity test: rerun the NB03 harmonisation and the NB05 bootstrap with
the Cryo-TEMPO native snow density set to 300 kg m-3.
NB03 and NB05 are imported unchanged; only the density lookup is replaced.
Writes to processed/snow_harmonisation_CT300 (existing outputs untouched).
"""
import os
import sys
import importlib.util

import pandas as pd

SCRIPTS = '/g/data/gv90/xl1657/phd/M1_workspace/scripts'
PROC = '/g/data/gv90/xl1657/phd/M1_workspace/processed'
BASE = os.path.join(PROC, 'snow_harmonisation')          # existing results
OUT = os.path.join(PROC, 'snow_harmonisation_CT300')     # new results
os.makedirs(OUT, exist_ok=True)


def load(name):
    path = os.path.join(SCRIPTS, name + '.py')
    if not os.path.exists(path):
        sys.exit(f'Missing {path}. Edit the script name at the bottom of load().')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nb3 = load('NB03_snow_harmonisation')
nb5 = load('NB05_bootstrap_uncertainty')

# ---- the only change: Cryo-TEMPO native density = 300 kg m-3 -------------
_native = nb3.get_native_density


def native_ct300(product, month):
    return 300.0 if product == 'CryoTEMPO_CS2' else _native(product, month)


nb3.get_native_density = native_ct300
nb3.BASE_OUT = OUT

# ---- NB03: Cases A-D ------------------------------------------------------
rows, spread_data = nb3.run_harmonisation()
spread_rows, attr_rows = nb3.compute_spread_and_attribution(spread_data)
nb3.save_csv(rows, 'harmonisation_sector_means.csv')
nb3.save_csv(spread_rows, 'harmonisation_spread.csv')
nb3.save_csv(attr_rows, 'harmonisation_attribution.csv')

# ---- NB05: year-level bootstrap, same seed --------------------------------
nb5.BASE_HARMON = OUT
nb5.BASE_OUT = OUT
boot = nb5.bootstrap_attribution(nb5.load_harmonisation_csv())
nb5.save_csv(boot, 'bootstrap_attribution_ci.csv')
nb5.print_attribution_summary(boot)

# ---- comparison with the existing results ---------------------------------
pd.set_option('display.width', 200)
SEP = '=' * 78


def hfi(df):
    return df[df.variable == 'hfi'].set_index('sector')


old = hfi(pd.read_csv(os.path.join(BASE, 'harmonisation_attribution.csv')))
new = hfi(pd.DataFrame(attr_rows))
cmp_full = pd.DataFrame({
    'A old (m)': old.spread_A, 'A new (m)': new.spread_A,
    'D (m)': new.spread_D,
    'B red% old': old.reduction_B_pct, 'B red% new': new.reduction_B_pct,
    'C red% old': old.reduction_C_pct, 'C red% new': new.reduction_C_pct,
    'D red% old': old.reduction_D_pct, 'D red% new': new.reduction_D_pct,
})
print(f'\n{SEP}\n  h_fi spread, full sample: documented densities (old) vs CT = 300 (new)\n{SEP}')
print(cmp_full.round({'A old (m)': 4, 'A new (m)': 4, 'D (m)': 4}).round(1).to_string())


def boot_d(df):
    df = df[(df.variable == 'hfi') & (df.case == 'D')].set_index('sector')
    return df[['reduction_mean', 'reduction_lo', 'reduction_hi']]


b_old = boot_d(pd.read_csv(os.path.join(BASE, 'bootstrap_attribution_ci.csv')))
b_new = boot_d(pd.DataFrame(boot))
cmp_boot = b_old.join(b_new, lsuffix=' old', rsuffix=' new')
print(f'\n{SEP}\n  Case D reduction in h_fi spread (%), bootstrap mean and 95% CI\n{SEP}')
print(cmp_boot.round(1).to_string())

print(f'\n{SEP}\n  FOR THE MANUSCRIPT\n{SEP}')
print(f'  Case D reduction, bootstrap mean across sectors:')
print(f'    documented densities : {b_old.reduction_mean.min():.1f}-'
      f'{b_old.reduction_mean.max():.1f} %   (manuscript: 7-21 %)')
print(f'    Cryo-TEMPO at 300    : {b_new.reduction_mean.min():.1f}-'
      f'{b_new.reduction_mean.max():.1f} %   <-- X-Y for Section 2.4')
print(f'\n  Outputs in {OUT}\n')