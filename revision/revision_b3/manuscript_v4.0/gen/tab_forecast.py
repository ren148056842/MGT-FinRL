"""Forecast tables for v4.0 from results/forecast_metrics.csv and forecast_ic_diffs.csv."""
import sys, pandas as pd, numpy as np
R = sys.argv[1]; OUT = sys.argv[2]
f = pd.read_csv(R + '/forecast_metrics.csv')
NAMES = [('MGAT_T', 'MGAT'), ('GCN_R', 'GCN'), ('A10_R', 'No cross-stock'), ('MGT_R', 'MGNN'), ('A1_R', 'Graph-free'), ('Ridge_R', 'Ridge')]
def pct(x): return '--' if pd.isna(x) else '%+.1f' % (100 * x)
def z(x, d):
    v = round(float(x), d)
    return 0.0 if v == 0 else v
def ic(r):
    if pd.isna(r.IC2): return '--'
    s = ('%.3f (%.1f)' % (z(r.IC2, 3), z(r.IC2_t, 1)))
    return s.replace('-', '$-$')
def main_table(year, label, cap):
    L = [r'\begin{table*}[!t]\centering\small\color{blue}', r'\caption{' + cap + '}', r'\label{' + label + '}',
         r'\setlength{\tabcolsep}{4pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{lrrrrrrrr}\toprule',
         r' & \multicolumn{4}{c}{Banking} & \multicolumn{4}{c}{Healthcare} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}',
         r'Forecaster & High & Low & Close & IC$_2$ ($t$) & High & Low & Close & IC$_2$ ($t$) \\ \midrule']
    for k, n in NAMES:
        cells = []
        for sec in ['bank', 'healthcare']:
            r = f[(f.sector == sec) & (f.test_year == year) & (f.model == k)].iloc[0]
            cells += [pct(r.RMSE_h1_rel_naive), pct(r.RMSE_l1_rel_naive), pct(r.RMSE_c1_rel_naive), ic(r)]
        L.append(n + ' & ' + ' & '.join(('$-$' + c[1:]) if (c.startswith('-') and c != '--') else c for c in cells) + r' \\')
    L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
    return '\n'.join(L)
open(OUT + '/tab_forecast_2025.tex', 'w').write(main_table(2025, 'tab:forecast_eval',
    r'Forecast accuracy in the evaluation year 2025 (fold 2025: trained on 2018--2023 targets, early-stopped on 2024). High, Low, Close: change in the root mean squared error of the next-day log high, low, and close relative to the previous-day values (\%; negative values are improvements). IC$_2$: mean daily Spearman correlation between the predicted and realised return from the close of $t+1$ to the close of $t+2$, with its $t$-statistic (242 days). Ridge forecasts only this return. Means of three seeds.'))
# appendix: all years
L = [r'\begin{table*}[!t]\centering\small\color{blue}', r'\caption{Forecast accuracy by test year (walk-forward folds). Columns as in Table~\ref{tab:forecast_eval}; 2022--2023 are the development years, 2024 the selection year, and 2025 the evaluation year.}', r'\label{tab:forecast_years}',
     r'\setlength{\tabcolsep}{3.5pt}', r'\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%' + '\n' + r'\begin{tabular}{llrrrrrrrr}\toprule',
     r' & & \multicolumn{4}{c}{Banking} & \multicolumn{4}{c}{Healthcare} \\ \cmidrule(lr){3-6}\cmidrule(lr){7-10}',
     r'Year & Forecaster & High & Low & Close & IC$_2$ ($t$) & High & Low & Close & IC$_2$ ($t$) \\ \midrule']
for y in [2022, 2023, 2024, 2025]:
    for j, (k, n) in enumerate(NAMES):
        cells = []
        for sec in ['bank', 'healthcare']:
            r = f[(f.sector == sec) & (f.test_year == y) & (f.model == k)].iloc[0]
            cells += [pct(r.RMSE_h1_rel_naive), pct(r.RMSE_l1_rel_naive), pct(r.RMSE_c1_rel_naive), ic(r)]
        L.append((str(y) if j == 0 else '') + ' & ' + n + ' & ' + ' & '.join(('$-$' + c[1:]) if (c.startswith('-') and c != '--') else c for c in cells) + r' \\')
    if y != 2025: L.append(r'\midrule')
L += [r'\bottomrule\end{tabular}}', r'\end{table*}']
open(OUT + '/tab_forecast_years.tex', 'w').write('\n'.join(L))
print(open(OUT + '/tab_forecast_2025.tex').read())
