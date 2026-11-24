#!/usr/bin/env python3
"""
P1_offset_trend_pvalues.py
Recompute the Table 6 p-values (trend in the Case A minus Case D offset)
with a Student's t test, n - 2 degrees of freedom, in place of the normal
approximation used in NB08. Reads NB08_trend_sensitivity.csv; writes nothing.
"""
import pandas as pd
from scipy import stats

F = ('/g/data/gv90/xl1657/phd/M1_workspace/processed/'
     'trend_sensitivity/NB08_trend_sensitivity.csv')
OLD = {'Western_Weddell': '$0.047^{*}$', 'Eastern_Weddell': '$<0.001^{*}$',
       'Indian': '$0.771$', 'Western_Pacific': '$0.646$',
       'Ross': '$<0.001^{*}$', 'Amundsen_Bellingshausen': '$0.951$'}

t3 = pd.read_csv(F)
t3 = t3[t3.tier == 3].copy()
t3['t'] = t3.offset_trend_m_per_yr / t3.stderr
t3['dof'] = (t3.n_years - 2).astype(int)
t3['p_t'] = 2 * stats.t.sf(t3.t.abs(), t3.dof)

pd.set_option('display.width', 200)
print('\nOffset trend (Case A minus Case D): normal vs t test')
print(t3[['sector', 'offset_trend_m_per_yr', 'stderr', 't', 'dof',
          'p_value', 'p_t']].to_string(index=False))


def cell(p):
    s = '<0.001' if p < 0.001 else f'{p:.3f}'
    return f'${s}^{{*}}$' if p < 0.05 else f'${s}$'


print('\nTable 6, last column, ready to paste:')
for _, r in t3.iterrows():
    new = cell(r.p_t)
    old = OLD.get(r.sector, '?')
    out = new if new == old else f'\\del{{{old}}}\\add{{{new}}}'
    print(f'  {r.sector:<26} & {out}')

sig = t3[t3.p_t < 0.05].sector.tolist()
print(f'\nSignificant at p < 0.05 with the t test: {len(sig)} of {len(t3)} '
      f'sectors {sig}')
print('Normal approximation in the manuscript: 3 of 6 sectors.\n')