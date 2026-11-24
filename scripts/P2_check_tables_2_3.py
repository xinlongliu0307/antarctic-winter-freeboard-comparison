#!/usr/bin/env python3
"""
P2_check_tables_2_3.py
Regenerate Tables 2 and 3 from the unrounded NB02 multi-year means, rounding
ONCE to 0.01 m, and list every cell that differs from the manuscript.
Reads two CSVs; writes nothing.
"""
import pandas as pd

B = '/g/data/gv90/xl1657/phd/M1_workspace/processed/sector_means/'
m = pd.concat([pd.read_csv(B + 'multiyear_means_CS2.csv'),
               pd.read_csv(B + 'multiyear_means_ENV.csv')])

S = ['Western_Weddell', 'Eastern_Weddell', 'Indian',
     'Western_Pacific', 'Ross', 'Amundsen_Bellingshausen']
P = ['LEGOS_ENV', 'CCI_ENV', 'LEGOS_I_CS2', 'LEGOS_II_CS2',
     'CCI_CS2', 'CSAO_CS2', 'CryoTEMPO_CS2']
R = ['ENV LEGOS-CCI', 'LEGOS I-CCI', 'LEGOS II-CCI', 'CSAO-CCI',
     'CryoTEMPO-CCI']

# Values in the manuscript now (rows = sectors S, columns = products P)
T2 = {
 'hfr': [[.13, .11, .18, .18, .12, .14, .10], [.09, .05, .10, .10, .07, .09, .05],
         [.08, .05, .07, .07, .05, .04, .04], [.12, .10, .12, .12, .09, .09, .09],
         [.10, .06, .09, .09, .07, .09, .06], [.11, .10, .11, .11, .08, .09, .07]],
 'hfi': [[.19, .18, .25, .24, .19, .19, .17], [.12, .09, .14, .14, .11, .12, .09],
         [.09, .08, .09, .09, .07, .07, .07], [.13, .13, .15, .14, .12, .11, .12],
         [.13, .11, .13, .13, .11, .11, .10], [.14, .14, .15, .14, .12, .13, .11]],
 'hsc': [[.06, .07, .07, .05, .07, .05, .07], [.03, .04, .04, .04, .04, .04, .04],
         [.01, .03, .02, .04, .03, .03, .03], [.02, .03, .03, .04, .03, .03, .03],
         [.03, .04, .04, .03, .04, .03, .04], [.03, .04, .04, .04, .04, .03, .04]],
}
# Table 3 (rows = comparisons R, columns = sectors S); "<0.01" entered as 0
T3 = {
 'hfr': [[.02, .04, .03, .02, .03, .01], [.06, .03, .02, .03, .03, .03],
         [.06, .03, .02, .03, .03, .03], [.01, .02, 0, -.01, .02, .02],
         [-.02, -.02, -.01, -.01, -.01, -.01]],
 'hfi': [[0, .03, .01, 0, .02, 0], [.06, .03, .01, .02, .02, .02],
         [.05, .03, .02, .01, .02, .02], [-.01, .01, -.01, -.01, .01, .01],
         [-.02, -.02, -.01, 0, -.01, -.01]],
 'hsc': [[-.02, -.02, -.01, -.01, -.01, -.01], [0, 0, 0, 0, 0, 0],
         [-.02, 0, .01, .01, -.01, 0], [-.02, 0, 0, 0, -.01, -.01],
         [0, 0, 0, 0, 0, 0]],
}


def show(x):
    """Format one Table 3 cell as the manuscript prints it."""
    x = round(float(x), 2)
    return '<0.01' if abs(x) < 0.005 else f'{x:+.2f}'


pd.set_option('display.width', 200)
changes = 0
for v in ['hfr', 'hfi', 'hsc']:
    t = (m[m.variable == v]
         .pivot(index='sector', columns='product', values='multiyear_mean')
         .loc[S, P])
    old2 = pd.DataFrame(T2[v], index=S, columns=P)
    print(f'\n===== Table 2, {v}: unrounded (m) =====')
    print(t.round(4).to_string())
    for s in S:
        for p in P:
            if abs(round(t.loc[s, p], 2) - old2.loc[s, p]) > 1e-9:
                changes += 1
                print(f'  CHANGE Table 2 {v:<4}{s:<26}{p:<15}'
                      f'{old2.loc[s, p]:.2f} -> {t.loc[s, p]:.2f}'
                      f'   (unrounded {t.loc[s, p]:.4f})')

    d = pd.DataFrame({
        R[0]: t.LEGOS_ENV - t.CCI_ENV,
        R[1]: t.LEGOS_I_CS2 - t.CCI_CS2,
        R[2]: t.LEGOS_II_CS2 - t.CCI_CS2,
        R[3]: t.CSAO_CS2 - t.CCI_CS2,
        R[4]: t.CryoTEMPO_CS2 - t.CCI_CS2}).T          # rows R, columns S
    old3 = pd.DataFrame(T3[v], index=R, columns=S)
    print(f'\n===== Table 3, delta {v}: unrounded (m) =====')
    print(d.round(4).to_string())
    for r in R:
        for s in S:
            if abs(round(d.loc[r, s], 2) - old3.loc[r, s]) > 1e-9:
                changes += 1
                print(f'  CHANGE Table 3 d{v:<4}{r:<16}{s:<26}'
                      f'{show(old3.loc[r, s])} -> {show(d.loc[r, s])}'
                      f'   (unrounded {d.loc[r, s]:+.4f})')

print(f'\nTotal cells to change: {changes}\n')