#!/usr/bin/env python3
"""
C1_density_check.py
Which snow density, and which correction formula, does each DISTRIBUTED
product behave as? Uses k = h_sc / h_s from the NB02 sector means.
Reads one CSV; writes nothing.
"""
import numpy as np
import pandas as pd

B = '/g/data/gv90/xl1657/phd/M1_workspace/processed/'
d = pd.read_csv(B + 'sector_means/sector_means_monthly_CS2.csv')
d = d[np.isfinite(d.hsc_mean) & (d.hs_mean > 0)].copy()
d['k'] = d.hsc_mean / d.hs_mean


def rho_eq1(k):
    """Density implied if Equation 1 was applied: k = c/c_s - 1."""
    return ((1 + k) ** (2 / 3) - 1) / 0.51 * 1000


def rho_alt(k):
    """Density implied if h_s(1 - c_s/c) was applied: k = 1 - c_s/c."""
    return ((1 / (1 - k)) ** (2 / 3) - 1) / 0.51 * 1000


pd.set_option('display.width', 200)
g = d.groupby(['product', 'month'])['k']
k = g.mean().unstack()

print('\nk = h_sc / h_s (mean over sectors and years)')
print(k.round(4).to_string())

print('\nRange of k across sectors and years (small = a clean constant)')
print(g.agg(lambda x: x.max() - x.min()).unstack().round(4).to_string())

print('\nImplied density (kg m-3) if Equation 1, h_s(c/c_s - 1), was applied')
print(rho_eq1(k).round(0).to_string())

print('\nImplied density (kg m-3) if h_s(1 - c_s/c) was applied')
print(rho_alt(k).round(0).to_string())

print('\nKey, Equation 1:       0.238 (300)  0.246 (310)  0.263 (330)  0.279 (350)')
print('Key, h_s(1 - c_s/c):   0.203 (320)  0.213 (340)  0.218 (350)')