"""LaTeX tables of the v4 forecasting evaluation."""
import sys, pathlib
import numpy as np, pandas as pd
IN, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
d = pd.read_csv(IN / 'forecast_metrics_all.csv')
NAME = {'Naive': 'Previous day', 'Ridge': 'Ridge regression', 'A0_Full_MGT': 'MGT (A0)', 'A1_Transformer_only': 'A1 Transformer only',
        'A10_No_cross_stock': 'A10 No cross-stock propagation', 'B1_GCN_Transformer': 'Multi-GCN-Transformer', 'B2_GAT_Transformer': 'Multi-GAT-Transformer',
        'B3_GNN_GRU': 'Multi-GNN-GRU', 'B4_GCN_GRU': 'Multi-GCN-GRU', 'B5_GAT_GRU': 'Multi-GAT-GRU', 'A2_MGNN_only': 'A2 MGNN only',
        'A3_SH_only': 'A3 SH only', 'A4_QA_only': 'A4 QA only', 'A5_KD_only': 'A5 KD only', 'A6_SH_QA': 'A6 SH+QA (no KD)', 'A7_SH_KD': 'A7 SH+KD (no QA)',
        'A8_QA_KD': 'A8 QA+KD (no SH)', 'A9_Fixed_fusion': 'A9 Fixed fusion', 'I1_alpha_one': r'I1 $\alpha_r(0)=1$', 'I2_alpha_small': r'I2 $\alpha_r(0)=0.1/\rho_r$',
        'I3_alpha_large': r'I3 $\alpha_r(0)=10/\rho_r$'}
SEC = {'bank': 'Banking', 'healthcare': 'Healthcare'}
MAIN = ['Naive', 'Ridge', 'A0_Full_MGT', 'A1_Transformer_only', 'A10_No_cross_stock', 'B1_GCN_Transformer', 'B2_GAT_Transformer', 'B3_GNN_GRU', 'B4_GCN_GRU', 'B5_GAT_GRU']
ABL = ['A0_Full_MGT', 'A2_MGNN_only', 'A3_SH_only', 'A4_QA_only', 'A5_KD_only', 'A6_SH_QA', 'A7_SH_KD', 'A8_QA_KD', 'A9_Fixed_fusion', 'I1_alpha_one', 'I2_alpha_small', 'I3_alpha_large']


def f(x, nd=2, sign=True):
    if pd.isna(x): return '--'
    x = round(float(x), nd) + 0.0
    s = ('%.' + str(nd) + 'f') % x
    if s.startswith('-') and float(s) == 0: s = s[1:]
    return s.replace('-', '$-$') if sign else s


def row(r):
    naive = r.model == 'Naive'
    dm = '--' if naive else '%s (%s)' % (f(r.DM_vs_prevclose, 2), ('%.3f' % r.DM_p) if r.DM_p >= 0.001 else '$<$0.001')
    ic1 = '--' if naive else '%s (%s)' % (f(r.IC, 3), f(r.IC_t, 1))
    ic2 = '--' if naive else '%s (%s)' % (f(r.IC_next, 3), f(r.IC_next_t, 1))
    return ' & '.join([NAME[r.model], f(r.close_rmse_pct, 3), f(r.R2_os_pct, 2), dm, '--' if naive else f(r.dir_acc_pct, 1), ic1, ic2,
                       f(r.high_rmse_ratio_naive, 3), f(r.low_rmse_ratio_naive, 3)])


def table(models, year, caption, label, star=True):
    L = [r'\begin{table*}[!t]' if star else r'\begin{table}[!t]', r'\centering\small\color{blue}', r'\caption{' + caption + '}', r'\label{' + label + '}',
         r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{llrrrrrrrr}\toprule',
         r'Sector & Model & RMSE & $R^2_{\mathrm{OS}}$ & DM ($p$) & Dir. & IC$_1$ ($t$) & IC$_2$ ($t$) & High & Low \\ \midrule']
    for sec in ['bank', 'healthcare']:
        first = True
        for m in models:
            x = d[(d.sector == sec) & (d.test_year.astype(str) == str(year)) & (d.model == m) & (d.kind != 'seed')]
            if len(x) != 1: raise ValueError((sec, m, year, len(x)))
            L.append((SEC[sec] if first else '') + ' & ' + row(x.iloc[0]) + r' \\'); first = False
        L.append(r'\midrule' if sec == 'bank' else r'\bottomrule')
    L += [r'\end{tabular}}', r'\end{table*}' if star else r'\end{table}']
    return '\n'.join(L)


CAP_MAIN = (r'Out-of-sample forecasting results, pooled over the test years 2022, 2023, and 2024 (726 forecast origins per sector; expanding-window folds; '
            r'forecasts averaged over three seeds). RMSE: one-day-ahead closing log return, in \% of price; $R^2_{\mathrm{OS}}$: out-of-sample $R^2$ relative to the previous close '
            r'(a zero-return forecast), in \%; DM: Diebold--Mariano statistic of the squared-error differential against the previous close (positive values favour the previous close) and its $p$-value; '
            r'Dir.: direction accuracy, in \%; IC$_1$ and IC$_2$: mean daily rank information coefficient ($t$-statistic) for the return from $t$ to $t+1$ and from $t+1$ to $t+2$; '
            r'only the latter can be traded by a decision executed at the close of $t+1$. High and Low: RMSE of the one-day-ahead log high and low relative to the previous day\textquotesingle s high and low '
            r'(values below one favour the model; $p<0.001$ in every case).')
(OUT / 'tab_forecast_main.tex').write_text(table(MAIN, 'pooled', CAP_MAIN, 'tab:forecast_eval'), encoding='utf-8')
CAP_ABL = (r'Forecasting results of the MGT ablations (A2--A9) and of the initialisation variants of the relation scalars (I1--I3), pooled over 2022--2024. '
           r'Columns as in Table~\ref{tab:forecast_eval}; A0 is repeated for reference.')
(OUT / 'tab_forecast_ablation.tex').write_text(table(ABL, 'pooled', CAP_ABL, 'tab:forecast_ablation'), encoding='utf-8')
# by year
L = [r'\begin{table*}[!t]', r'\centering\small\color{blue}',
     r'\caption{Forecasting results by test year for the previous day, ridge regression, MGT, and the two graph-free variants. Columns as in Table~\ref{tab:forecast_eval}.}',
     r'\label{tab:forecast_years}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lllrrrrrrrr}\toprule',
     r'Sector & Year & Model & RMSE & $R^2_{\mathrm{OS}}$ & DM ($p$) & Dir. & IC$_1$ ($t$) & IC$_2$ ($t$) & High & Low \\ \midrule']
for sec in ['bank', 'healthcare']:
    for y in [2022, 2023, 2024]:
        first = True
        for m in ['Naive', 'Ridge', 'A0_Full_MGT', 'A1_Transformer_only', 'A10_No_cross_stock']:
            x = d[(d.sector == sec) & (d.test_year.astype(str) == str(y)) & (d.model == m) & (d.kind != 'seed')].iloc[0]
            L.append((SEC[sec] if (y == 2022 and first) else '') + ' & ' + (str(y) if first else '') + ' & ' + row(x) + r' \\'); first = False
        L.append(r'\addlinespace')
    L.append(r'\midrule' if sec == 'bank' else r'\bottomrule')
L += [r'\end{tabular}}', r'\end{table*}']
(OUT / 'tab_forecast_years.tex').write_text('\n'.join(L), encoding='utf-8')
print('tables written')
